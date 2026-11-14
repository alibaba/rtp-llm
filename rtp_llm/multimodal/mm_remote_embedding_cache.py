"""Bounded shared KVCM tier, populated by accepted encoder results.

Remote keys scope local embedding identities to an immutable model namespace.
The local GPU/CPU scheduler and asynchronous eviction policy remain unchanged.
"""

import concurrent.futures
import hashlib
import json
import logging
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
from rtp_llm.multimodal.kvcm.lookup import KvMetaLookupError
from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

# Python 3.10 keeps future wait timeouts separate from built-in IO timeouts.
_TIMEOUT_ERRORS = (TimeoutError, concurrent.futures.TimeoutError)


class MMRemoteEmbeddingCache:
    def __init__(
        self,
        client,
        *,
        max_object_bytes,
        max_inflight_bytes,
        max_pending=8,
        read_timeout_ms=200,
        cuda_device=None,
        client_factory=None,
        namespace="",
        model_identity=None,
        gdr_enabled=None,
        audit_enabled=None,
        request_scoped_inspection=False,
    ):
        if min(max_object_bytes, max_inflight_bytes, max_pending, read_timeout_ms) <= 0:
            raise ValueError("remote embedding cache limits must be positive")
        if client_factory is not None and (
            not isinstance(namespace, str) or not namespace.strip()
        ):
            raise ValueError(
                "MM_REMOTE_CACHE_NAMESPACE must identify immutable model/preprocess revisions"
            )
        self._client = client
        self._factory = client_factory
        self._client_lock = threading.Lock()
        self._client_users = {}
        self.max_object_bytes = max_object_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.max_pending = max_pending
        self.read_timeout_ms = read_timeout_ms
        self.cuda_device = cuda_device
        self._model_identity = model_identity or {}
        scope = json.dumps(
            [namespace, self._model_identity], sort_keys=True, separators=(",", ":")
        ).encode()
        self._key_prefix = "mm-v3:" + hashlib.sha256(scope).hexdigest() + ":"
        self.request_scoped_inspection = request_scoped_inspection
        if gdr_enabled is None:
            gdr_enabled = os.environ.get("MM_REMOTE_CACHE_GDR_ENABLE") == "1"
        self._transport_mode = "gdr" if gdr_enabled else "staging"
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mm-remote-cache"
        )
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = Counter()
        self._audit_enabled = (
            os.environ.get("MM_EMB_REMOTE_AUDIT") == "1"
            if audit_enabled is None
            else audit_enabled
        )
        self._closed = False
        self._bytes = self._tasks = 0
        self._pending_writes = set()
        self._instance = getattr(client, "instance_id", None)
        try:
            self._proc_start_ticks = (
                Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
        except (OSError, IndexError):
            self._proc_start_ticks = None

    @classmethod
    def from_configs(
        cls,
        model_config,
        vit_config,
        *,
        kv_cache_config=None,
        device=None,
        request_scoped_inspection=False,
    ):
        from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient

        namespace = vit_config.mm_remote_cache_namespace
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError(
                "MM_REMOTE_CACHE_NAMESPACE must identify immutable model/preprocess revisions"
            )
        identity = {
            "model_type": model_config.model_type,
            "dtype": str(model_config.compute_dtype),
            "configs": {},
            "input_limits": [
                vit_config.mm_image_max_file_size_kb,
                vit_config.mm_video_max_file_size_kb,
            ],
        }
        for name in (
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
        ):
            path = Path(model_config.ckpt_path) / name
            if path.is_file():
                identity["configs"][name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()

        def raw_factory(**registration):
            kwargs = dict(
                max_object_bytes=vit_config.mm_remote_cache_max_object_bytes,
                call_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
                get_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
                put_timeout_ms=vit_config.mm_remote_cache_write_timeout_ms,
            )
            if kv_cache_config is not None:
                return RtpKvMetaObjectClient.from_kv_cache_config(
                    kv_cache_config, **kwargs, **registration
                )
            return RtpKvMetaObjectClient.from_env(**kwargs, **registration)

        def factory():
            if gdr_enabled:
                from rtp_llm.multimodal.kvcm.registered_transport import (
                    RegisteredObjectTransport,
                )

                return RegisteredObjectTransport(
                    raw_factory,
                    max_object_bytes=vit_config.mm_remote_cache_max_object_bytes,
                    device=device,
                )
            return raw_factory()

        # PACE seals the external GPU registration snapshot at client init.
        # Initialize before readiness, outside a request's read deadline, and
        # keep this registered client for the serving process lifetime. A
        # request timeout must not unregister/recreate the arena after sealing.
        gdr_enabled = vit_config.mm_remote_cache_gdr_enable
        if gdr_enabled is None:
            gdr_enabled = os.environ.get("MM_REMOTE_CACHE_GDR_ENABLE") == "1"
        client = cls._create_client(factory, "gdr") if gdr_enabled else None
        return cls(
            client,
            client_factory=None if gdr_enabled else factory,
            namespace=namespace,
            model_identity=identity,
            max_object_bytes=vit_config.mm_remote_cache_max_object_bytes,
            max_inflight_bytes=vit_config.mm_remote_cache_max_inflight_bytes,
            max_pending=vit_config.mm_remote_cache_max_pending,
            read_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
            cuda_device=device,
            gdr_enabled=gdr_enabled,
            audit_enabled=vit_config.mm_remote_cache_audit,
            request_scoped_inspection=request_scoped_inspection,
        )

    def normalize_cpp_url_input(self, item):
        # QueryConverter's GenerateInput URL path passes torch::empty(1), not
        # image data. The verified Qwen3/VL2/V4.1 IMAGE preprocessors ignore this
        # field and consume url. Never inspect the uninitialized scalar value.
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        tensor = item.tensor
        model_type = self._model_identity.get("model_type")
        is_v41 = model_type in ("deepseek_v41", "deepseek_v41_dspark")
        inline_url = item.url.startswith("data:image/") or (
            is_v41 and item.url.startswith("data:application/octet-stream;base64,")
        )
        if (
            (model_type in ("qwen3_vl", "qwen3_vl_moe", "deepseek_vl_v2") or is_v41)
            and item.mm_type == MMUrlType.IMAGE
            and (inline_url or item.url.startswith(("http://", "https://")))
            and tensor is not None
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and tuple(tensor.shape) == (1,)
        ):
            item.tensor = torch.empty(0, dtype=torch.float32)
            self.record(
                "input_normalized",
                reason="cpp_url_tensor_placeholder",
                mm_type=int(item.mm_type),
                tensor_shape=[1],
                tensor_dtype="torch.float32",
                tensor_device="cpu",
            )

    def key(self, mm_inputs, *, trace_id=None, local_key=None):
        def bypass(reason, **fields):
            self._report_skip("key", reason)
            self.record("cache_bypass", trace_id=trace_id, reason=reason, **fields)
            return None

        # Old work items can group multiple media; until their partitioning is
        # represented in the schema, use the encoder for those groups.
        if len(mm_inputs) != 1:
            return bypass("grouped_inputs", input_count=len(mm_inputs))
        item = mm_inputs[0]
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        if item.mm_type != MMUrlType.IMAGE:
            return bypass("unsupported_media_type", mm_type=int(item.mm_type))
        if item.tensor is not None and item.tensor.numel():
            return bypass(
                "nonempty_tensor",
                tensor_shape=list(item.tensor.shape),
                tensor_dtype=str(item.tensor.dtype),
                tensor_device=str(item.tensor.device),
            )
        if local_key is None:
            if not item.url:
                return bypass("missing_url")
            local_key = item.cache_key()
        key = self._key_prefix + hashlib.sha256(local_key.encode()).hexdigest()
        self.record(
            "cache_admitted",
            key=key,
            trace_id=trace_id,
            identity_source="local_cache_key",
        )
        return key

    @staticmethod
    def _error_context(error):
        fields = {"error_type": type(error).__name__}
        try:
            code = getattr(error, "code", None)
            name = getattr(code, "name", None)
            if isinstance(name, str) and len(name) <= 80:
                fields["error_code"] = name
            elif type(code) is int:
                fields["error_code"] = code
            reason = getattr(error, "reason", None)
            if isinstance(reason, str) and len(reason) <= 80:
                fields["error_reason"] = reason
            inflight = getattr(error, "data_transfer_may_be_inflight", None)
            if type(inflight) is bool:
                fields["data_transfer_may_be_inflight"] = inflight
        except Exception:
            pass  # Diagnostics must not turn cache fallback into request failure.
        return fields

    def record(self, event, *, key=None, trace_id=None, **fields):
        with self._stats_lock:
            self._stats[event] += 1
            count = self._stats[event]
        failure = event in ("read_error", "write_error")
        if not self._audit_enabled and not failure:
            return
        payload = dict(
            event=event,
            key=key,
            trace_id=trace_id,
            timestamp_ns=time.time_ns(),
            pid=os.getpid(),
            proc_start_ticks=self._proc_start_ticks,
            instance=self._instance,
            count=count,
            transport=self._transport_mode,
            **fields,
        )
        try:
            log = logging.warning if failure else logging.info
            log("MM_EMB_REMOTE_JSON %s", json.dumps(payload, sort_keys=True))
        except Exception:
            pass  # Optional evidence logging must not change cache behavior.

    def _audit_payload(self, buffer):
        # The codec always validates its integrity digest. A second full-buffer
        # digest is useful only for explicitly enabled cross-process evidence.
        if not self._audit_enabled:
            return {}
        return {
            "payload_sha256": hashlib.sha256(memoryview(buffer.numpy())).hexdigest()
        }

    def _benchmark(self, event, key, trace_id, **fields):
        if os.environ.get("MM_EMB_BENCHMARK_TIMING") == "1":
            try:
                logging.info(
                    "MM_EMB_BENCHMARK_JSON %s",
                    json.dumps(
                        dict(
                            event=event,
                            key=key,
                            trace_id=trace_id,
                            pid=os.getpid(),
                            timestamp_ns=time.time_ns(),
                            synchronized=False,
                            **fields,
                        ),
                        sort_keys=True,
                    ),
                )
            except Exception:
                pass  # Never prevent IO-task/client release in a finally block.

    @staticmethod
    def _report_metric(metric, value, tags):
        try:
            kmonitor.report(metric, value, tags)
        except Exception:
            # Monitoring must not change cache fallback or prevent budget release.
            logging.warning("MM remote cache metric reporting failed", exc_info=True)

    @classmethod
    def _create_client(cls, factory, transport):
        """Count actual constructor calls, including startup failures."""
        # GDR registration precedes VitEndpointServer, which normally registers
        # metrics. Register here as well so the one startup event is not lost.
        # init is idempotent; monitoring failure must not change cache behavior.
        try:
            kmonitor.init()
        except Exception:
            logging.warning("MM remote cache metric registration failed", exc_info=True)
        started = time.perf_counter()
        outcome = "success"
        try:
            return factory()
        except _TIMEOUT_ERRORS:
            outcome = "timeout"
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            tags = {"transport": transport, "status": outcome}
            cls._report_metric(
                AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_QPS_METRIC, 1, tags
            )
            cls._report_metric(
                AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_ERROR_QPS_METRIC,
                int(outcome != "success"),
                {"transport": transport},
            )
            cls._report_metric(
                GaugeMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_RT_METRIC,
                (time.perf_counter() - started) * 1000,
                tags,
            )

    def report_result_accepted(self, origin, output):
        # An accepted ViT result, not proof of RPC delivery or LLM completion.
        # Origin is retained across local reuse; it is not the current cache tier.
        origin = origin if origin in ("remote", "encoder") else "unknown"
        output = "hashes" if output == "hashes" else "embeddings"
        self._report_metric(
            AccMetrics.VIT_REMOTE_EMB_RESULT_ACCEPTED_QPS_METRIC,
            1,
            {"origin": origin, "output": output},
        )

    def _report_queue_wait(self, operation, queued_at):
        self._report_metric(
            GaugeMetrics.VIT_REMOTE_EMB_QUEUE_WAIT_METRIC,
            (time.perf_counter() - queued_at) * 1000,
            {"operation": operation},
        )

    def _lookup_object_size(self, client, key, trace_id, timeout_ms):
        outcome = "error"
        try:
            with self._observe_stage("read", "metadata"):
                try:
                    size = client.object_size(
                        key, trace_id=trace_id, timeout_ms=timeout_ms
                    )
                except Exception as error:
                    # The typed lookup error is metadata-only; discovery can
                    # recover on the existing client without GPU registration.
                    if not isinstance(error, KvMetaLookupError):
                        self._retire_client(client, error)
                    raise
                if size is None:
                    outcome = "miss"
                elif not 0 < size <= self.max_object_bytes:
                    outcome = "invalid"
                    raise ValueError("remote embedding object exceeds byte limit")
                else:
                    outcome = "hit"
                return size
        except _TIMEOUT_ERRORS:
            outcome = "timeout"
            raise
        finally:
            self._report_metric(
                AccMetrics.VIT_REMOTE_EMB_METADATA_QPS_METRIC, 1, {"status": outcome}
            )

    def _report_operation(self, operation, started, outcome, nbytes=0):
        elapsed_ms = (time.perf_counter() - started) * 1000
        if operation == "read":
            qps = AccMetrics.VIT_REMOTE_EMB_READ_QPS_METRIC
            error_qps = AccMetrics.VIT_REMOTE_EMB_READ_ERROR_QPS_METRIC
            rt = GaugeMetrics.VIT_REMOTE_EMB_READ_RT_METRIC
        else:
            qps = AccMetrics.VIT_REMOTE_EMB_WRITE_QPS_METRIC
            error_qps = AccMetrics.VIT_REMOTE_EMB_WRITE_ERROR_QPS_METRIC
            rt = GaugeMetrics.VIT_REMOTE_EMB_WRITE_RT_METRIC
        tags = {"status": outcome}
        self._report_metric(qps, 1, tags)
        self._report_metric(rt, elapsed_ms, tags)
        # Emit zero for successful calls/misses so a healthy stream has an
        # explicit zero error series. No request IDs, keys or error text as tags.
        self._report_metric(error_qps, int(outcome in ("error", "timeout")), {})
        tags = {"operation": operation}
        accepted_bytes = nbytes if outcome in ("hit", "success") else 0
        self._report_metric(
            AccMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_PER_SECOND_METRIC,
            accepted_bytes,
            tags,
        )
        if accepted_bytes:
            self._report_metric(
                GaugeMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_METRIC, accepted_bytes, tags
            )
            if elapsed_ms > 0:
                self._report_metric(
                    GaugeMetrics.VIT_REMOTE_EMB_EFFECTIVE_SPEED_METRIC,
                    accepted_bytes * 1000 / elapsed_ms / (1024 * 1024),
                    tags,
                )

    @contextmanager
    def _observe_stage(self, operation, stage, nbytes=0):
        """Observe completed SDK work independently of a caller's deadline.

        Stage values are fixed at call sites. Object bytes count fully successful
        SDK calls, including late reads and deduplicated saves. This is logical
        API volume, not actual IO bytes; partial failures are unknown.
        """
        started = time.perf_counter()
        outcome = "success"
        try:
            yield
        except _TIMEOUT_ERRORS:
            outcome = "timeout"
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            tags = {"operation": operation, "stage": stage}
            outcome_tags = dict(tags, status=outcome)
            self._report_metric(
                AccMetrics.VIT_REMOTE_EMB_STAGE_QPS_METRIC, 1, outcome_tags
            )
            self._report_metric(
                GaugeMetrics.VIT_REMOTE_EMB_STAGE_RT_METRIC, elapsed_ms, outcome_tags
            )
            self._report_metric(
                AccMetrics.VIT_REMOTE_EMB_STAGE_ERROR_QPS_METRIC,
                int(outcome != "success"),
                tags,
            )
            if stage == "transfer":
                tags = {"operation": operation}
                completed_bytes = nbytes if outcome == "success" else 0
                self._report_metric(
                    AccMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_PER_SECOND_METRIC,
                    completed_bytes,
                    tags,
                )
                if completed_bytes:
                    self._report_metric(
                        GaugeMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_METRIC,
                        completed_bytes,
                        tags,
                    )
                    if elapsed_ms > 0:
                        self._report_metric(
                            GaugeMetrics.VIT_REMOTE_EMB_SDK_EFFECTIVE_SPEED_METRIC,
                            completed_bytes * 1000 / elapsed_ms / (1024 * 1024),
                            tags,
                        )

    def _report_skip(self, operation, reason):
        self._report_metric(
            AccMetrics.VIT_REMOTE_EMB_SKIP_QPS_METRIC,
            1,
            {"operation": operation, "reason": reason},
        )

    def _get_client(self):
        with self._client_lock:
            if self._closed:
                raise RuntimeError("remote embedding cache is closed")
            if self._client is None:
                if self._factory is None:
                    raise RuntimeError("remote embedding client is unavailable")
                self._client = self._create_client(self._factory, self._transport_mode)
                self._instance = self._client.instance_id
            client = self._client
            state = self._client_users.setdefault(id(client), [client, 0, False])
            state[1] += 1
            return client

    @staticmethod
    def _close_client(client):
        try:
            client.close()
        except Exception:
            logging.warning("MM remote cache client cleanup failed", exc_info=True)

    def _retire_client(self, client, error):
        # A classified metadata failure leaves the connection and registered
        # arena usable. Rebuilding them for every conflict/RPC failure is costly.
        check = getattr(client, "is_transfer_quiescent", None)
        try:
            if callable(check) and check(error) is True:
                return
        except Exception:
            pass
        # Detach a failed generation immediately, but do not close a shared
        # client while another admitted metadata/transfer operation uses it.
        # Waiting for that operation here could occupy both IO pool workers.
        if self._factory is None:
            return
        with self._client_lock:
            if self._client is client:
                self._client = None
            state = self._client_users.get(id(client))
            if state is not None:
                state[2] = True

    def _release_client(self, client):
        closing = None
        with self._client_lock:
            state = self._client_users[id(client)]
            state[1] -= 1
            if state[1] == 0:
                self._client_users.pop(id(client))
                if state[2]:
                    closing = client
        if closing is not None:
            self._close_client(closing)

    def _count(self, name):
        with self._stats_lock:
            self._stats[name] += 1

    def _reserve(self, nbytes):
        with self._lock:
            if (
                self._closed
                or self._tasks >= self.max_pending
                or self._bytes + nbytes > self.max_inflight_bytes
            ):
                self._count("admission_skip")
                return False
            self._tasks += 1
            self._bytes += nbytes
            return True

    def _release(self, nbytes):
        with self._lock:
            self._tasks -= 1
            self._bytes -= nbytes

    def bind_entry(self, entry, key, trace_id):
        if entry is not None and key is not None:
            entry.remote_cache_context = dict(key=key, trace_id=trace_id)

    def submit_result(
        self, key, result, *, trace_id=None, producer_events=(), entry=None
    ):
        """Snapshot an accepted encoder result, then upload owned bytes.

        Admission precedes allocation. Snapshotting happens on the caller while
        it owns the tensors; queued IO never retains model buffers or cache pool
        slots. CPU residency and eviction are unrelated to remote writeback.
        """
        if key is None:
            return
        try:
            inspection = None
            if entry is not None:
                verdict, checked = entry.greennet_snapshot()
                if verdict is not None:
                    if not verdict.passed:
                        return
                    inspection = dict(
                        checked=checked,
                        passed=verdict.passed,
                        code=verdict.code,
                        message=verdict.message,
                        request_scoped=self.request_scoped_inspection,
                    )
            plan = plan_object(
                {"version": 1, "result": result, "greennet": inspection},
                max_bytes=self.max_object_bytes,
            )
        except (ValueError, TypeError, RecursionError):
            self._count("encode_skip")
            self._report_skip("write", "unsupported_payload")
            return
        cost = 3 * plan.nbytes
        with self._lock:
            duplicate = key in self._pending_writes
            if not duplicate:
                self._pending_writes.add(key)
        if duplicate:
            self._report_skip("write", "duplicate_pending")
            return
        if not self._reserve(cost):
            self._report_skip("write", "admission")
            with self._lock:
                self._pending_writes.discard(key)
            return
        started = time.perf_counter()
        self.record(
            "encoder_result_admitted",
            key=key,
            trace_id=trace_id,
            nbytes=plan.nbytes,
            tensors=plan.header["tensors"],
        )

        def done(_):
            with self._lock:
                self._pending_writes.discard(key)
            self._release(cost)

        try:
            with self._observe_stage("write", "pack"):
                # Host event synchronization is also forbidden during global
                # CUDA Graph capture, even for a different producer stream.
                with cuda_graph_gate.operation():
                    for event in producer_events or ():
                        event.synchronize()
                buffer = pack_object(plan)
        except Exception as error:
            self.record(
                "write_error",
                key=key,
                trace_id=trace_id,
                **self._error_context(error),
            )
            self._report_operation("write", started, "error", plan.nbytes)
            done(None)
            return
        # The worker owns only the snapshot and small metadata, not tensors.
        nbytes, tensors = plan.nbytes, plan.header["tensors"]

        def write():
            client = None
            queue_ms = (time.perf_counter() - queued_at) * 1000
            self._report_queue_wait("write", queued_at)
            api_ms = None
            outcome = "error"
            try:
                with self._observe_stage("write", "client_acquire"):
                    client = self._get_client()
                api_started = time.perf_counter()
                with self._observe_stage("write", "transfer", buffer.numel()):
                    try:
                        client.save_one(key, buffer, trace_id=trace_id)
                    except Exception as error:
                        self._retire_client(client, error)
                        raise
                api_ms = (time.perf_counter() - api_started) * 1000
                outcome = "success"
                self._count("write_success")
                self.record(
                    "store_success",
                    key=key,
                    trace_id=trace_id,
                    write_source="encoder_result",
                    **self._audit_payload(buffer),
                    nbytes=buffer.numel(),
                    tensors=tensors,
                )
            except Exception as error:
                outcome = "timeout" if isinstance(error, _TIMEOUT_ERRORS) else "error"
                self.record(
                    "write_error",
                    key=key,
                    trace_id=trace_id,
                    **self._error_context(error),
                )
            finally:
                self._report_operation("write", started, outcome, nbytes)
                self._benchmark(
                    "remote_store_timing",
                    key,
                    trace_id,
                    wall_ms=(time.perf_counter() - started) * 1000,
                    queue_ms=queue_ms,
                    sdk_api_ms=api_ms,
                    status=outcome,
                    scope="snapshot_queue_client_acquire_sdk_save_and_validation",
                )
                if client is not None:
                    self._release_client(client)

        queued_at = time.perf_counter()
        try:
            future = self._executor.submit(write)
        except RuntimeError:
            self._report_skip("write", "executor_closed")
            done(None)
            return
        future.add_done_callback(done)

    @staticmethod
    def _decode_value(value):
        # Older workers wrote the bare result tuple/list without inspection metadata.
        approval = None
        if isinstance(value, dict):
            if (
                set(value) != {"version", "result", "greennet"}
                or type(value["version"]) is not int
                or value["version"] != 1
            ):
                raise ValueError("invalid remote embedding cache schema")
            inspection = value["greennet"]
            if inspection is not None:
                if (
                    not isinstance(inspection, dict)
                    or set(inspection)
                    != {"checked", "passed", "code", "message", "request_scoped"}
                    or any(
                        type(inspection[k]) is not bool
                        for k in ("checked", "passed", "request_scoped")
                    )
                    or type(inspection["code"]) is not int
                    or not isinstance(inspection["message"], str)
                ):
                    raise ValueError("invalid remote GreenNet verdict")
                if not inspection["passed"]:
                    raise ValueError("remote embedding has a rejected GreenNet verdict")
                if inspection["checked"] and not inspection["request_scoped"]:
                    approval = GreenNetVerdict(
                        True, inspection["code"], inspection["message"]
                    )
            value = value["result"]
        if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
            raise ValueError("remote object is not a multimodal result")
        return value, approval

    def load(
        self, key, *, timeout_ms=None, trace_id=None, entry=None, restore_approval=True
    ):
        if key is None:
            return None
        timeout_ms = min(self.read_timeout_ms, timeout_ms or self.read_timeout_ms)
        deadline = time.monotonic() + timeout_ms / 1000
        started = time.perf_counter()
        timing = {}
        payload_nbytes = 0
        outcome = "error"
        if not self._reserve(0):
            self._report_skip("read", "admission")
            return None
        cost = 0
        future = None

        def read():
            nonlocal cost
            self._report_queue_wait("read", queued_at)
            client = None
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote embedding queue deadline expired")
                with self._observe_stage("read", "client_acquire"):
                    client = self._get_client()
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    raise TimeoutError("remote embedding lookup deadline expired")
                api_started = time.perf_counter()
                size = self._lookup_object_size(client, key, trace_id, remaining_ms)
                if size is None:
                    timing["outcome"] = "miss"
                    timing["sdk_api_ms"] = (time.perf_counter() - api_started) * 1000
                    return None
                with self._lock:
                    budget_exceeded = self._bytes + 3 * size > self.max_inflight_bytes
                    if not budget_exceeded:
                        cost = 3 * size
                        self._bytes += cost
                if budget_exceeded:
                    self._count("admission_skip")
                    timing["outcome"] = "admission_skip"
                    self._report_skip("read", "payload_budget")
                    return None
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote embedding transfer deadline expired")
                buffer = torch.empty(size, dtype=torch.uint8)
                with self._observe_stage("read", "transfer", size):
                    try:
                        client.load_one(key, buffer, trace_id=trace_id)
                    except Exception as error:
                        self._retire_client(client, error)
                        raise
                timing["sdk_api_ms"] = (time.perf_counter() - api_started) * 1000
                return buffer
            finally:
                if client is not None:
                    # Close may join native workers even after our last call.
                    # It stays in this admitted IO task, never the request
                    # thread; its task/byte budget remains held until done.
                    self._release_client(client)

        queued_at = time.perf_counter()
        try:
            future = self._executor.submit(read)
            buffer = future.result(timeout=max(0, deadline - time.monotonic()))
            if buffer is None:
                outcome = timing["outcome"]
                if outcome == "miss":
                    self._count("miss")
                    self.record("load_miss", key=key, trace_id=trace_id)
                return None
            with self._observe_stage("read", "restore"):
                value = unpack_object(
                    buffer,
                    max_bytes=self.max_object_bytes,
                    cuda_device=self.cuda_device,
                )
                value, approval = self._decode_value(value)
                plan = plan_object(value, max_bytes=self.max_object_bytes)
                if (
                    entry is not None
                    and restore_approval
                    and not self.request_scoped_inspection
                    and approval is not None
                ):
                    entry.set_greennet_verdict(approval)
            self._count("hit")
            outcome = "hit"
            payload_nbytes = buffer.numel()
            self.record(
                "load_hit",
                key=key,
                trace_id=trace_id,
                **self._audit_payload(buffer),
                nbytes=buffer.numel(),
                tensors=plan.header["tensors"],
            )
            return value
        except _TIMEOUT_ERRORS:
            outcome = "timeout"
            self._count("read_timeout")
            if future is not None:
                future.cancel()
            self.record(
                "read_error", key=key, trace_id=trace_id, error_type="TimeoutError"
            )
            return None
        except Exception as error:
            self.record(
                "read_error",
                key=key,
                trace_id=trace_id,
                **self._error_context(error),
            )
            return None
        finally:
            self._report_operation("read", started, outcome, payload_nbytes)
            self._benchmark(
                "remote_load_timing",
                key,
                trace_id,
                wall_ms=(time.perf_counter() - started) * 1000,
                sdk_api_ms=timing.get("sdk_api_ms"),
                status=outcome,
                scope="caller_queue_client_acquire_sdk_load_unpack_and_validation",
            )
            if future is None or future.done():
                self._release(cost)
            else:
                future.add_done_callback(lambda _: self._release(cost))

    def stats(self):
        with self._lock, self._stats_lock:
            return dict(
                self._stats, inflight_bytes=self._bytes, inflight_tasks=self._tasks
            )

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._client_lock:
            client, self._client = self._client, None
            if client is not None:
                state = self._client_users.get(id(client))
                if state is not None:
                    state[2] = True
                    client = None
        if client is not None:
            self._close_client(client)
