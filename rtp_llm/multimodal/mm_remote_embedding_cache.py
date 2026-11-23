"""Bounded shared KVCM tier, populated by accepted encoder results.

Remote keys scope local embedding identities to an immutable model namespace.
The local GPU/CPU scheduler and asynchronous eviction policy remain unchanged.
"""

import concurrent.futures
import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
from rtp_llm.multimodal.kvcm.lookup import (
    KvMetaLookupError,
    KvMetaLookupTimeout,
)
from rtp_llm.multimodal.kvcm.native import USE_NATIVE_EMB
from rtp_llm.multimodal.kvcm.registered_transport import _safe_error_context
from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)
from rtp_llm.multimodal.vit_metrics import VIT_PHASE_METRICS
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

# Python 3.10 keeps future wait timeouts separate from built-in IO timeouts.
_TIMEOUT_ERRORS = (TimeoutError, concurrent.futures.TimeoutError)
_USE_NATIVE_SNAPSHOT = USE_NATIVE_EMB
_BLOCKING_CUDA_EVENTS = USE_NATIVE_EMB
_RETRY_METADATA_TRANSPORT = os.environ.get("MM_EMB_METADATA_TRANSPORT_RETRY") == "1"
_SLOW_RESTORE_COMPONENTS = (
    "slow_restore_header",
    "slow_restore_layout",
    "slow_restore_cuda_payload",
    "slow_restore_host_resume",
    "slow_restore_python_object",
)


def _is_write_in_progress(error):
    """Recognize a known pre-I/O KVMeta conflict without importing the optional wheel."""
    code = getattr(error, "code", None)
    return (
        type(error).__name__ == "KvMetaObjectClientError"
        and getattr(code, "name", None) == "ER_SERVICE_WRITE_IN_PROGRESS"
        and getattr(error, "operation", None) == "save"
        and getattr(error, "unknown_outcome", True) is False
        and getattr(error, "data_transfer_may_be_inflight", None) is False
    )


def _is_transfer_timeout(error):
    if isinstance(error, _TIMEOUT_ERRORS):
        return True
    return (
        type(error).__name__ == "KvMetaObjectClientError"
        and getattr(error, "operation", None) in ("load", "save")
        and getattr(getattr(error, "code", None), "name", None)
        in ("ER_SDK_TIMEOUT", "ER_SDK_TIMEOUT_BUFFER_SAFE")
    )


def _is_read_miss(error):
    return (
        type(error).__name__ == "KvMetaObjectClientError"
        and getattr(getattr(error, "code", None), "name", None)
        == "ER_SERVICE_NOT_FOUND"
        and getattr(error, "operation", None) == "load"
        and getattr(error, "data_transfer_may_be_inflight", None) is False
    )


def _log_error(marker, error, **context):
    try:
        logging.error(
            "%s %s",
            marker,
            json.dumps(dict(context, **_safe_error_context(error)), sort_keys=True),
        )
    except Exception:
        pass  # A broken diagnostic sink must not affect fallback or ownership.


class MMRemoteEmbeddingCache:
    _GDR_READ_SLOT_WAIT_SECONDS = 0.025
    _METADATA_RETRY_MAX_MS = 10

    def __init__(
        self,
        client,
        *,
        max_object_bytes,
        max_inflight_bytes,
        max_pending=128,
        read_transfer_workers=2,
        write_transfer_workers=2,
        read_timeout_ms=200,
        write_timeout_ms=1000,
        cuda_device=None,
        client_factory=None,
        namespace="",
        model_identity=None,
        gdr_enabled=None,
        gdr_pool=None,
        audit_enabled=None,
        request_scoped_inspection=False,
    ):
        if min(
            max_object_bytes,
            max_inflight_bytes,
            max_pending,
            read_timeout_ms,
            write_timeout_ms,
            read_transfer_workers,
            write_transfer_workers,
        ) <= 0:
            raise ValueError("remote embedding cache limits must be positive")
        if client_factory is not None and (
            not isinstance(namespace, str) or not namespace.strip()
        ):
            raise ValueError(
                "MM_REMOTE_CACHE_NAMESPACE must identify immutable model/preprocess revisions"
            )
        self._client = client
        self._gdr_pool = gdr_pool
        self._direct_payload = getattr(gdr_pool, "direct_payload", False) is True
        self._fixed_gdr_client = (
            client if client is gdr_pool and client_factory is None else None
        )
        self._factory = client_factory
        self._client_lock = threading.Lock()
        self._client_users = {}
        self.max_object_bytes = max_object_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.max_pending = max_pending
        self.read_timeout_ms = read_timeout_ms
        self.write_timeout_ms = write_timeout_ms
        self.cuda_device = cuda_device
        self._model_identity = model_identity or {}
        scope = json.dumps(
            [namespace, self._model_identity], sort_keys=True, separators=(",", ":")
        ).encode()
        # V5 uses the checksum-free codec. During a rolling update V4
        # workers keep their own keys, so neither version reads the old layout.
        self._key_prefix = "mm-v5:" + hashlib.sha256(scope).hexdigest() + ":"
        self.request_scoped_inspection = request_scoped_inspection
        if gdr_enabled is None:
            gdr_enabled = os.environ.get("MM_REMOTE_CACHE_GDR_ENABLE") == "1"
        self._transport_mode = "gdr" if gdr_enabled else "staging"
        self._transfer_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=read_transfer_workers, thread_name_prefix="mm-remote-cache"
        )
        # Admit only as many payloads as can run. A stalled transfer must not
        # leave subsequent hits waiting out their deadline in the executor.
        self._transfer_slots = threading.BoundedSemaphore(read_transfer_workers)
        self._write_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=write_transfer_workers, thread_name_prefix="mm-remote-write"
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

        client_kwargs = dict(
            max_object_bytes=vit_config.mm_remote_cache_max_object_bytes,
            call_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
            get_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
            put_timeout_ms=vit_config.mm_remote_cache_write_timeout_ms,
        )

        def raw_factory(**registration):
            if kv_cache_config is not None:
                return RtpKvMetaObjectClient.from_kv_cache_config(
                    kv_cache_config, **client_kwargs, **registration
                )
            return RtpKvMetaObjectClient.from_env(**client_kwargs, **registration)

        def factory():
            if gdr_enabled:
                from rtp_llm.multimodal.kvcm._config import RtpKvMetaObjectClientConfig
                from rtp_llm.multimodal.kvcm.registered_transport import (
                    RegisteredTransportPool,
                )

                # Resolve discovery before allocating an arena or registering
                # memory. A pure config/VIP failure must not quarantine an MR.
                prepared_config = (
                    RtpKvMetaObjectClientConfig.from_kv_cache_config(
                        kv_cache_config, **client_kwargs
                    )
                    if kv_cache_config is not None
                    else RtpKvMetaObjectClientConfig.from_env(**client_kwargs)
                )

                def registered_factory(**registration):
                    return RtpKvMetaObjectClient._from_config(
                        prepared_config, **registration
                    )

                return RegisteredTransportPool(
                    registered_factory,
                    max_object_bytes=vit_config.mm_remote_cache_max_object_bytes,
                    device=device,
                    slots=gdr_slots,
                    arena_bytes=getattr(
                        vit_config, "mm_remote_cache_gdr_arena_bytes", None
                    ),
                    min_block_bytes=getattr(
                        vit_config, "mm_remote_cache_gdr_min_block_bytes", None
                    ),
                    observe_stage=cls._observe_stage,
                    quarantine_ttl_seconds=getattr(
                        vit_config, "mm_remote_cache_quarantine_ttl_seconds", 300
                    ),
                    observe_hold=cls._observe_buffer_hold,
                    observe_lease=cls._observe_gdr_lease,
                    observe_arena=cls._observe_gdr_arena,
                    observe_acquire_failure=cls._observe_gdr_acquire_failure,
                    observe_lock_hold=cls._observe_gdr_lock_hold,
                    observe_header=cls._report_read_header if USE_NATIVE_EMB else None,
                    direct_payload=True,
                )
            client = raw_factory()
            logging.info(
                "MM_EMB_REMOTE_BACKEND requested_native=%d read=legacy_staging write=legacy_staging",
                USE_NATIVE_EMB,
            )
            return client

        # PACE seals the external GPU registration snapshot at client init.
        # Initialize before readiness, outside a request's read deadline, and
        # keep this registered client for the serving process lifetime. A
        # request timeout must not unregister/recreate the arena after sealing.
        gdr_enabled = vit_config.mm_remote_cache_gdr_enable
        if gdr_enabled is None:
            gdr_enabled = os.environ.get("MM_REMOTE_CACHE_GDR_ENABLE") == "1"
        gdr_slots = getattr(vit_config, "mm_remote_cache_gdr_num", 1)
        default_transfer_workers = gdr_slots if gdr_enabled else 2
        read_transfer_workers = getattr(
            vit_config, "mm_remote_cache_read_transfer_workers", None
        )
        write_transfer_workers = getattr(
            vit_config, "mm_remote_cache_write_transfer_workers", None
        )
        client = cls._create_client(factory, "gdr") if gdr_enabled else None
        return cls(
            client,
            client_factory=None if gdr_enabled else factory,
            namespace=namespace,
            model_identity=identity,
            max_object_bytes=vit_config.mm_remote_cache_max_object_bytes,
            max_inflight_bytes=vit_config.mm_remote_cache_max_inflight_bytes,
            max_pending=vit_config.mm_remote_cache_max_pending,
            read_transfer_workers=(
                default_transfer_workers
                if read_transfer_workers is None
                else read_transfer_workers
            ),
            write_transfer_workers=(
                default_transfer_workers
                if write_transfer_workers is None
                else write_transfer_workers
            ),
            read_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
            write_timeout_ms=vit_config.mm_remote_cache_write_timeout_ms,
            cuda_device=device,
            gdr_enabled=gdr_enabled,
            gdr_pool=client if gdr_enabled else None,
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
        return _safe_error_context(error)

    def record(self, event, *, key=None, trace_id=None, **fields):
        with self._stats_lock:
            self._stats[event] += 1
            count = self._stats[event]
        failure = event in (
            "read_error",
            "read_late_error",
            "read_late_complete",
            "write_error",
        )
        # Keep counters/metrics for every outcome, but never emit normal
        # request logs, even when a legacy audit flag is still configured.
        if not failure:
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
            operation="read" if event.startswith("read_") else "write",
            **fields,
        )
        try:
            logging.error("MM_EMB_REMOTE_JSON %s", json.dumps(payload, sort_keys=True))
        except Exception:
            pass  # Optional evidence logging must not change cache behavior.

    def _benchmark(self, event, key, trace_id, **fields):
        if os.environ.get("MM_EMB_BENCHMARK_TIMING") == "1":
            try:
                logging.debug(
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
        except Exception as error:
            # Monitoring must not change cache fallback or prevent budget release.
            _log_error(
                "MM_EMB_REMOTE_METRIC_ERROR",
                error,
                operation="metric_report",
                metric=str(metric),
            )

    @classmethod
    def _add_inflight(cls, stage, delta):
        cls._report_metric(
            GaugeMetrics.VIT_REMOTE_EMB_INFLIGHT_TASKS_METRIC, delta, {"stage": stage}
        )

    @classmethod
    @contextmanager
    def _observe_inflight(cls, stage):
        cls._add_inflight(stage, 1)
        try:
            yield
        finally:
            cls._add_inflight(stage, -1)

    @classmethod
    def _observe_gdr_lease(cls, for_write, delta, nbytes):
        stage = "write_gdr_lease" if for_write else "read_gdr_lease"
        cls._add_inflight(stage, delta)
        cls._report_metric(
            GaugeMetrics.VIT_REMOTE_EMB_INFLIGHT_BYTES_METRIC, nbytes, {"stage": stage}
        )

    @classmethod
    def _observe_gdr_acquire_failure(cls, for_write, reason):
        cls._report_metric(
            AccMetrics.VIT_REMOTE_EMB_GDR_ACQUIRE_FAILURE_QPS_METRIC,
            1,
            {"operation": "write" if for_write else "read", "reason": reason},
        )

    @classmethod
    def _observe_gdr_lock_hold(cls, for_write, phase, duration_ns):
        cls._report_metric(
            GaugeMetrics.VIT_REMOTE_EMB_GDR_LOCK_HOLD_RT_METRIC,
            duration_ns / 1_000_000,
            {"operation": "write" if for_write else "read", "phase": phase},
        )

    @classmethod
    def _observe_gdr_arena(cls, total, free, largest_free):
        for stage, nbytes in (
            ("total", total),
            ("free", free),
            ("largest_free", largest_free),
        ):
            cls._report_metric(
                GaugeMetrics.VIT_REMOTE_EMB_GDR_ARENA_BYTES_METRIC,
                nbytes,
                {"stage": stage},
            )

    @classmethod
    def _create_client(cls, factory, transport):
        """Count actual constructor calls, including startup failures."""
        # GDR registration precedes VitEndpointServer, which normally registers
        # metrics. Register here as well so the one startup event is not lost.
        # init is idempotent; monitoring failure must not change cache behavior.
        try:
            kmonitor.init()
        except Exception as error:
            _log_error(
                "MM_EMB_REMOTE_METRIC_INIT_ERROR", error, operation="metric_init"
            )
        outcome = "success"
        try:
            return factory()
        except Exception as error:
            outcome = "timeout" if isinstance(error, _TIMEOUT_ERRORS) else "error"
            _log_error(
                "MM_EMB_REMOTE_CLIENT_CREATE_ERROR",
                error,
                operation="client_create",
                transport=transport,
            )
            raise
        finally:
            cls._report_metric(
                AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_QPS_METRIC,
                1,
                {"transport": transport},
            )
            cls._report_metric(
                AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_ERROR_QPS_METRIC,
                int(outcome != "success"),
                {"transport": transport},
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
        elapsed_ms = (time.perf_counter() - queued_at) * 1000
        self._report_metric(
            GaugeMetrics.VIT_REMOTE_EMB_QUEUE_WAIT_METRIC,
            elapsed_ms,
            {"operation": operation},
        )
        return elapsed_ms

    @classmethod
    def _report_header_size(cls, operation, length, copies):
        # Size bins describe only JSON bytes, excluding the fixed prefix and pad.
        # The 1008 boundary is exactly where the existing 1024-byte read spills.
        bounds = (256, 512, 768, 1008, 1536, 2048, 4096, 8192, 16384, 65536)
        band = str(next((upper for upper in bounds if length <= upper), "+Inf"))
        cls._report_metric(
            AccMetrics.VIT_REMOTE_EMB_HEADER_OBSERVATION_QPS_METRIC,
            1,
            {"operation": operation, "json_size_upper": band, "copies": str(copies)},
        )

    @classmethod
    def _report_read_header(cls, length, d2h_bytes, copies):
        cls._report_header_size("read", length, copies)
        cls._report_metric(
            AccMetrics.VIT_REMOTE_EMB_HEADER_D2H_BYTES_PER_SECOND_METRIC,
            d2h_bytes,
            {"copies": str(copies)},
        )

    def _fetch_meta(self, client, key, trace_id, timeout_ms, deadline):
        try:
            return client.get_object_meta(key, trace_id=trace_id, timeout_ms=timeout_ms)
        except Exception as error:
            if (
                not _RETRY_METADATA_TRANSPORT
                or getattr(error, "operation", None) != "get_object_meta"
                or getattr(error, "data_transfer_may_be_inflight", True) is not False
                or getattr(getattr(error, "code", None), "name", None)
                != "ER_INVALID_GRPCSTATUS"
            ):
                raise
            # Exact-object metadata is idempotent. Bound extra delay on the
            # retry while keeping the original request budget on the first call.
            retry_ms = min(
                self._METADATA_RETRY_MAX_MS, int((deadline - time.monotonic()) * 1000)
            )
            if retry_ms > 0:
                try:
                    with self._observe_stage("read", "metadata_retry"):
                        return client.get_object_meta(
                            key, trace_id=trace_id, timeout_ms=retry_ms
                        )
                except Exception:
                    # The retry stage records its outcome separately. Preserve
                    # the original typed transport failure for the caller.
                    pass
            raise

    def _lookup_object_meta(self, client, key, trace_id, timeout_ms, deadline):
        diagnostic_start_ns = time.perf_counter_ns() if VIT_PHASE_METRICS else None
        outcome = "error"
        try:
            with self._observe_inflight("metadata"), self._observe_stage(
                "read", "metadata"
            ):
                try:
                    meta = self._fetch_meta(client, key, trace_id, timeout_ms, deadline)
                except Exception as error:
                    # The typed lookup error is metadata-only; discovery can
                    # recover on the existing client without GPU registration.
                    native_lookup_failure = (
                        getattr(error, "operation", None) == "get_object_meta"
                        and getattr(error, "data_transfer_may_be_inflight", True) is False
                    )
                    if not native_lookup_failure and not isinstance(
                        error, (KvMetaLookupError, KvMetaLookupTimeout)
                    ):
                        self._retire_client(client, error)
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote embedding metadata completed after deadline")
                if meta is None:
                    outcome = "miss"
                elif not 0 < meta.size_bytes <= self.max_object_bytes:
                    outcome = "invalid"
                    raise ValueError("remote embedding object exceeds byte limit")
                else:
                    outcome = "hit"
                return meta
        except _TIMEOUT_ERRORS:
            outcome = "timeout"
            raise
        finally:
            if diagnostic_start_ns is not None:
                self._report_metric(
                    GaugeMetrics.VIT_REMOTE_EMB_METADATA_RT_METRIC,
                    (time.perf_counter_ns() - diagnostic_start_ns) / 1e6,
                    {"status": outcome},
                )
            self._report_outcome(
                "metadata", "success" if outcome in ("hit", "miss") else "error"
            )
            self._report_metric(
                AccMetrics.VIT_REMOTE_EMB_METADATA_QPS_METRIC, 1, {"status": outcome}
            )

    def _report_operation(self, operation, started, outcome, nbytes=0):
        elapsed_ms = (time.perf_counter() - started) * 1000
        if outcome != "admission_skip":
            status = (
                "success"
                if outcome in ("hit", "miss", "success")
                else "skipped" if outcome == "skipped" else "error"
            )
            self._report_outcome(operation, status)
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

    @classmethod
    @contextmanager
    def _observe_stage(cls, operation, stage, nbytes=0):
        """Observe completed SDK work independently of a caller's deadline.

        Stage values are fixed at call sites. Object bytes count fully successful
        SDK calls, including late reads and deduplicated saves. This is logical
        API volume, not actual IO bytes; partial failures are unknown.
        """
        started = time.perf_counter()
        outcome = "success"
        try:
            yield
        except Exception as error:
            outcome = (
                "timeout"
                if _is_transfer_timeout(error)
                else "skipped"
                if stage == "transfer"
                and (
                    (operation == "write" and _is_write_in_progress(error))
                    or (operation == "read" and _is_read_miss(error))
                )
                else "error"
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            cls._report_stage_latency(operation, stage, outcome, elapsed_ms, nbytes)

    @classmethod
    def _report_stage_latency(cls, operation, stage, outcome, elapsed_ms, nbytes=0):
        metric_stage = (
            "metadata_put_start"
            if operation == "write" and outcome == "skipped" and stage == "transfer"
            else stage
        )
        cls._report_metric(
            GaugeMetrics.VIT_REMOTE_EMB_STAGE_RT_METRIC,
            elapsed_ms,
            {"operation": operation, "stage": metric_stage, "status": outcome},
        )
        if stage == "transfer" and outcome != "skipped":
            tags = {"operation": operation}
            completed_bytes = nbytes if outcome == "success" else 0
            cls._report_metric(
                AccMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_PER_SECOND_METRIC,
                completed_bytes,
                tags,
            )
            if completed_bytes:
                cls._report_metric(
                    GaugeMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_METRIC,
                    completed_bytes,
                    tags,
                )
                if elapsed_ms > 0:
                    cls._report_metric(
                        GaugeMetrics.VIT_REMOTE_EMB_SDK_EFFECTIVE_SPEED_METRIC,
                        completed_bytes * 1000 / elapsed_ms / (1024 * 1024),
                        tags,
                    )

    @classmethod
    def _report_outcome(cls, operation, outcome, *, stage="all"):
        # One completed call has exactly one outcome. Miss is a successful
        # lookup; non-executed work is skipped, never a transport failure.
        status = outcome if outcome in ("success", "skipped") else "error"
        cls._report_metric(
            AccMetrics.VIT_REMOTE_EMB_OPERATION_QPS_METRIC,
            1,
            {"operation": operation, "stage": stage, "status": status},
        )

    def _report_skip(self, operation, reason, *, report_outcome=True):
        if report_outcome and operation in ("read", "write"):
            self._report_outcome(operation, "skipped")
        if operation == "write" and reason == "write_in_progress":
            self._report_metric(
                AccMetrics.VIT_REMOTE_EMB_WRITE_IN_PROGRESS_QPS_METRIC, 1, {}
            )
        self._report_metric(
            AccMetrics.VIT_REMOTE_EMB_SKIP_QPS_METRIC,
            1,
            {"operation": operation, "reason": reason},
        )

    def _gdr_pool_quarantined(self):
        client = self._client
        return (
            self._transport_mode == "gdr" and client is not None and client.quarantined
        )

    def _skip_quarantined(self, operation):
        if self._gdr_pool_quarantined():
            self._report_skip(operation, "gdr_quarantined")
            return True
        return False

    @classmethod
    def _observe_buffer_hold(
        cls, operation, event, nbytes, age_seconds, registered_bytes, held_count
    ):
        for kind in ("read", "write"):
            cls._report_metric(
                AccMetrics.VIT_REMOTE_EMB_BUFFER_HOLD_QPS_METRIC,
                int(event == "entered" and operation == kind),
                {"operation": kind},
            )
        cls._report_metric(
            AccMetrics.VIT_REMOTE_EMB_BUFFER_GC_QPS_METRIC,
            int(event in ("gc_success", "gc_error")),
            {},
        )
        cls._report_metric(
            AccMetrics.VIT_REMOTE_EMB_BUFFER_GC_SUCCESS_QPS_METRIC,
            int(event == "gc_success"),
            {},
        )
        cls._report_metric(
            AccMetrics.VIT_REMOTE_EMB_BUFFER_GC_ERROR_QPS_METRIC,
            int(event == "gc_error"),
            {},
        )
        for metric, value in (
            (GaugeMetrics.VIT_REMOTE_EMB_BUFFER_HOLD_BYTES_METRIC, nbytes),
            (
                GaugeMetrics.VIT_REMOTE_EMB_BUFFER_RETAINED_BYTES_METRIC,
                registered_bytes,
            ),
            (GaugeMetrics.VIT_REMOTE_EMB_BUFFER_HOLD_AGE_METRIC, age_seconds),
            (GaugeMetrics.VIT_REMOTE_EMB_BUFFER_HOLD_COUNT_METRIC, held_count),
        ):
            cls._report_metric(metric, value, {})

    def _get_client(self):
        # ViT drains its request workers before stopping the cache. Only
        # replaceable clients need per-operation reference tracking.
        if self._fixed_gdr_client is not None:
            if self._closed:
                raise RuntimeError("remote embedding cache is closed")
            return self._fixed_gdr_client
        with self._client_lock:
            if self._closed:
                raise RuntimeError("remote embedding cache is closed")
            if self._client is None:
                if self._factory is None:
                    raise RuntimeError("remote embedding client is unavailable")
                self._client = self._create_client(self._factory, self._transport_mode)
                self._instance = self._client.instance_id
            client = self._client
            state = self._client_users.get(id(client))
            if state is None:
                self._client_users[id(client)] = [client, 1, False]
            else:
                state[1] += 1
            return client

    @staticmethod
    def _close_client(client):
        try:
            client.close()
        except Exception as error:
            _log_error("MM_EMB_REMOTE_CLEANUP_ERROR", error, operation="close")

    def _retire_client(self, client, error):
        if self._factory is None:
            return
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
        with self._client_lock:
            if self._client is client:
                self._client = None
            state = self._client_users.get(id(client))
            if state is not None:
                state[2] = True

    def _release_client(self, client):
        if client is self._fixed_gdr_client:
            return
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

    @staticmethod
    def _snapshot(plan, producer_streams):
        """Detach from reusable encoder output before returning it to the caller."""
        streams = {stream.device: stream for stream in producer_streams}
        cuda_indices = {}
        for i, tensor in enumerate(plan.tensors):
            if tensor.is_cuda:
                cuda_indices.setdefault(tensor.device, []).append(i)
            else:
                plan.tensors[i] = tensor.detach().clone(
                    memory_format=torch.contiguous_format
                )

        events = []
        native_snapshot = None
        if cuda_indices and _USE_NATIVE_SNAPSHOT:
            import librtp_compute_ops

            native_snapshot = getattr(
                librtp_compute_ops, "snapshot_emb_cuda_tensors", None
            )
        for device, indices in cuda_indices.items():
            with cuda_graph_gate.operation():
                if native_snapshot is not None:
                    producer = streams.get(device)
                    snapshots, ready = native_snapshot(
                        [plan.tensors[i] for i in indices],
                        producer.stream_id if producer is not None else None,
                        _BLOCKING_CUDA_EVENTS,
                    )
                    for i, snapshot in zip(indices, snapshots):
                        plan.tensors[i] = snapshot
                    events.append(ready)
                    continue
                caller = torch.cuda.current_stream(device)
                producer = streams.get(device, caller)
                # Inspection can finish on a different stream. Order later
                # reuse on either stream after the copies, without a host wait.
                if producer != caller:
                    producer.wait_stream(caller)
                with torch.cuda.stream(producer):
                    if _USE_NATIVE_SNAPSHOT:
                        from librtp_compute_ops import snapshot_emb_tensors

                        snapshots = snapshot_emb_tensors(
                            [plan.tensors[i] for i in indices]
                        )
                        for i, snapshot in zip(indices, snapshots):
                            plan.tensors[i] = snapshot
                    else:
                        for i in indices:
                            tensor = plan.tensors[i]
                            plan.tensors[i] = tensor.detach().clone(
                                memory_format=torch.contiguous_format
                            )
                            tensor.record_stream(producer)
                    event = (
                        torch.cuda.Event(blocking=True) if _BLOCKING_CUDA_EVENTS else None
                    )
                    ready = producer.record_event(event)
                if producer != caller:
                    caller.wait_event(ready)
                events.append(ready)
        return events

    def submit_result(
        self, key, result, *, trace_id=None, producer_streams=(), entry=None
    ):
        """Admit an encoder result and enqueue its owned snapshot for upload."""
        if key is None or self._skip_quarantined("write"):
            return
        started = time.perf_counter()
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
            with self._observe_stage("write", "plan_submit"):
                plan = plan_object(
                    {
                        "version": 1,
                        "result": result,
                        "greennet": inspection,
                        "cache_key": key,
                    },
                    max_bytes=self.max_object_bytes,
                    checksum=False,
                )
        except (ValueError, TypeError, RecursionError):
            self._count("encode_skip")
            self._report_skip("write", "unsupported_payload")
            return
        cost = plan.nbytes * (1 if self._direct_payload else 3)
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
        self.record(
            "encoder_result_admitted",
            key=key,
            trace_id=trace_id,
            nbytes=plan.nbytes,
            tensors=plan.header["tensors"],
        )

        self._add_inflight("write_pending", 1)

        def done(_):
            self._add_inflight("write_pending", -1)
            with self._lock:
                self._pending_writes.discard(key)
            self._release(cost)

        try:
            with self._observe_stage("write", "snapshot_submit"):
                ready_events = self._snapshot(plan, producer_streams)
        except Exception as error:
            self.record(
                "write_error",
                key=key,
                trace_id=trace_id,
                stage="pack",
                nbytes=plan.nbytes,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                **self._error_context(error),
            )
            self._report_operation("write", started, "error", plan.nbytes)
            done(None)
            return
        nbytes, tensors = plan.nbytes, plan.header["tensors"]

        def record_success():
            self._count("write_success")
            if USE_NATIVE_EMB and random.getrandbits(4) == 0:
                self._report_header_size("write_plan", len(plan.header_bytes), 0)
            self.record(
                "store_success",
                key=key,
                trace_id=trace_id,
                write_source="encoder_result",
                nbytes=nbytes,
                tensors=tensors,
            )

        def write():
            client = None
            slot = None
            queue_ms = api_ms = None
            outcome = "error"
            failure_stage = "client_acquire"
            try:
                queue_ms = (time.perf_counter() - queued_at) * 1000
                self._report_queue_wait("write", queued_at)
                if time.monotonic() >= queue_deadline:
                    outcome = "skipped"
                    self._report_skip("write", "write_queue_expired")
                    return
                native_write = getattr(type(self._gdr_pool), "try_native_write", None)
                if callable(native_write) and not self._gdr_pool_quarantined():
                    failure_stage = "native_write"
                    result = native_write(
                        self._gdr_pool, plan, ready_events, key, trace_id, queue_deadline
                    )
                    if result is not None:
                        (
                            state,
                            code,
                            inflight,
                            wait_ns,
                            acquire_ns,
                            pack_ns,
                            sdk_ns,
                            _,
                            _,
                            lease_ns,
                        ) = result
                        if wait_ns and ready_events:
                            self._report_stage_latency(
                                "write", "producer_wait", "success", wait_ns / 1e6
                            )
                        if acquire_ns:
                            self._report_stage_latency(
                                "write", "gdr_acquire", "success", acquire_ns / 1e6
                            )
                        if pack_ns:
                            for phase in ("pack", "native_pack_fenced"):
                                self._report_stage_latency(
                                    "write", phase, "success", pack_ns / 1e6
                                )
                        status = state.name
                        if lease_ns:
                            self._report_stage_latency(
                                "write",
                                "gdr_lease_hold",
                                "error" if inflight else "success",
                                lease_ns / 1e6,
                            )
                        if status == "Expired":
                            outcome = "skipped"
                            self._report_skip(
                                "write",
                                "write_producer_wait_expired"
                                if ready_events and not acquire_ns
                                else "write_queue_expired",
                            )
                            return
                        if status in ("ProducerFailure", "CudaFailure"):
                            failure_stage = "producer_wait" if status == "ProducerFailure" else "pack"
                            raise RuntimeError("native EMB CUDA " + failure_stage + " failed")
                        failure_stage = "transfer"
                        api_ms = sdk_ns / 1e6
                        plan.tensors.clear()
                        error = (
                            self._gdr_pool.native_save_error(code, inflight)
                            if status == "SdkFailure"
                            else None
                        )
                        transfer_status = (
                            "success" if error is None
                            else "skipped" if _is_write_in_progress(error)
                            else "timeout" if _is_transfer_timeout(error)
                            else "error"
                        )
                        self._report_stage_latency(
                            "write", "native_save", transfer_status, api_ms
                        )
                        self._report_stage_latency(
                            "write", "transfer", transfer_status, api_ms, nbytes
                        )
                        if error is not None:
                            raise error
                        outcome = "success"
                        record_success()
                        return

                if ready_events and not self._gdr_pool_quarantined():
                    failure_stage = "producer_wait"
                    with self._observe_stage(
                        "write", "producer_wait"
                    ), cuda_graph_gate.operation():
                        for event in ready_events:
                            event.synchronize()
                remaining = queue_deadline - time.monotonic()
                if remaining <= 0:
                    outcome = "skipped"
                    self._report_skip(
                        "write",
                        "write_producer_wait_expired"
                        if ready_events
                        else "write_queue_expired",
                    )
                    return
                if self._skip_quarantined("write"):
                    outcome = "skipped"
                    self.record(
                        "write_skipped",
                        key=key,
                        trace_id=trace_id,
                        reason="gdr_quarantined",
                    )
                    return
                # Producer events belong to the owned snapshot, not an arena.
                # Wait with the same CUDA ordering before leasing scarce GDR
                # storage; pack is the first operation that needs that storage.
                failure_stage = "client_acquire"
                if self._gdr_pool is not None:
                    with self._observe_inflight(
                        "write_gdr_wait"
                    ), self._observe_stage("write", "gdr_acquire"):
                        slot = self._gdr_pool.acquire(
                            nbytes=plan.nbytes, for_write=True, timeout=remaining
                        )
                    if slot is None:
                        outcome = "skipped"
                        if not self._skip_quarantined("write"):
                            reason = (
                                "executor_closed"
                                if self._closed
                                else "write_slot_unavailable"
                            )
                            self._report_skip("write", reason)
                        return
                    if time.monotonic() >= queue_deadline:
                        outcome = "skipped"
                        self._report_skip("write", "write_slot_unavailable")
                        return
                failure_stage = "pack"
                with self._observe_stage("write", "pack"):
                    buffer = (
                        slot.pack(plan)
                        if slot is not None
                        else pack_object(plan, observe_stage=self._observe_stage)
                    )
                # No embedding storage needs to stay live during network IO.
                plan.tensors.clear()
                failure_stage = "client_acquire"
                client = self._get_client()
                api_started = time.perf_counter()
                failure_stage = "transfer"
                with self._observe_stage("write", "transfer", buffer.numel()):
                    try:
                        target = slot if slot is not None else client
                        target.save_one(key, buffer, trace_id=trace_id)
                    except Exception as error:
                        self._retire_client(client, error)
                        raise
                    finally:
                        # save_one has completed the stream work or quarantined
                        # this slot. Reporting and logging only need its size.
                        if slot is not None:
                            self._gdr_pool.release(slot)
                            slot = None
                api_ms = (time.perf_counter() - api_started) * 1000
                outcome = "success"
                record_success()
            except Exception as error:
                if _is_write_in_progress(error):
                    outcome = "skipped"
                    self.record("write_skipped")
                    self._report_skip(
                        "write", "write_in_progress", report_outcome=False
                    )
                else:
                    outcome = (
                        "timeout" if _is_transfer_timeout(error) else "error"
                    )
                    self.record(
                        "write_error",
                        key=key,
                        trace_id=trace_id,
                        stage=failure_stage,
                        nbytes=nbytes,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        **self._error_context(error),
                    )
            finally:
                if slot is not None:
                    self._gdr_pool.release(slot)
                # Pre-transfer skips have already reported their outcome. A
                # rejected SDK save still counts as a completed write attempt.
                if outcome != "skipped" or failure_stage == "transfer":
                    self._report_operation("write", started, outcome, nbytes)
                self._benchmark(
                    "remote_store_timing",
                    key,
                    trace_id,
                    wall_ms=(time.perf_counter() - started) * 1000,
                    queue_ms=queue_ms,
                    sdk_api_ms=api_ms,
                    status=outcome,
                    scope="plan_admission_snapshot_queue_client_acquire_sdk_save_and_validation",
                )
                if client is not None:
                    self._release_client(client)

        def active_write():
            with self._observe_inflight("write_worker"):
                write()

        queued_at = time.perf_counter()
        queue_deadline = time.monotonic() + self.write_timeout_ms / 1000
        try:
            future = self._write_executor.submit(active_write)
        except RuntimeError:
            self._report_skip("write", "executor_closed")
            done(None)
            return
        future.add_done_callback(done)

    @staticmethod
    def _decode_value(value, expected_key=None, *, require_key=False):
        # Older workers wrote the bare result tuple/list without inspection metadata.
        approval = None
        if isinstance(value, dict) and "cache_key" in value:
            if (
                not isinstance(value["cache_key"], str)
                or value["cache_key"] != expected_key
            ):
                raise ValueError("remote embedding cache key mismatch")
            value = dict(value)
            value.pop("cache_key")
        elif require_key:
            raise ValueError("remote embedding cache key binding missing")
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

    def _finish_native_read(self, result, size, key, consumer_stream):
        (
            state, value, code, inflight, acquire_ns, sdk_ns, restore_ns, held_ns,
            _, _, header_size, header_bytes, header_copies, header_ns,
        ) = result
        status = state.name
        error = self._gdr_pool.native_read_error(code, inflight) if status == "SdkFailure" else None
        sdk_timed_out = error is not None and _is_transfer_timeout(error)
        if status == "Expired" or sdk_timed_out:
            stage_status = "timeout"
        elif error is not None and _is_read_miss(error):
            stage_status = "skipped"
        elif status in ("SdkFailure", "CudaFailure", "InvalidObject"):
            stage_status = "error"
        else:
            stage_status = "success"
        if acquire_ns:
            self._report_stage_latency("read", "gdr_acquire", "success", acquire_ns / 1e6)
        if held_ns:
            self._report_stage_latency("read", "gdr_lease_hold", stage_status, held_ns / 1e6)
        if sdk_ns:
            sdk_status = "success"
            if status == "SdkFailure":
                sdk_status = "timeout" if sdk_timed_out else "error"
            self._report_stage_latency(
                "read", "native_load", sdk_status, sdk_ns / 1e6
            )
            self._report_stage_latency(
                "read", "transfer", stage_status, (sdk_ns + restore_ns) / 1e6, size
            )
        if restore_ns:
            self._report_stage_latency(
                "read", "native_object_restore", stage_status, restore_ns / 1e6
            )
        if header_ns:
            self._report_read_header(header_size, header_bytes, header_copies)
            self._report_stage_latency("read", "native_restore_header", stage_status, header_ns / 1e6)
        if status == "Expired":
            raise TimeoutError("registered EMB read expired")
        if error is not None:
            raise error
        if status == "InvalidObject":
            raise ValueError("invalid registered EMB object")
        if status == "CudaFailure":
            raise RuntimeError("native registered EMB restore failed")
        if status == "LayoutFallback":
            # Only the rare unsupported layout takes a host snapshot. The
            # original transfer is already complete and its lease returned.
            with cuda_graph_gate.operation(), (
                torch.cuda.stream(consumer_stream) if consumer_stream is not None else nullcontext()
            ):
                return unpack_object(
                    value, cuda_device=self.cuda_device, max_bytes=self.max_object_bytes,
                    expected_key=key, observe_stage=self._observe_stage,
                )
        return value

    def load(
        self, key, *, timeout_ms=None, trace_id=None, entry=None, restore_approval=True
    ):
        if key is None:
            return None
        if self._closed:
            self._report_skip("read", "executor_closed")
            return None
        timeout_ms = min(self.read_timeout_ms, timeout_ms or self.read_timeout_ms)
        deadline = time.monotonic() + timeout_ms / 1000
        started = time.perf_counter()
        timing = {"stage": "client_acquire", "nbytes": 0}
        payload_nbytes = 0
        outcome = "error"
        cost = 0
        future = None
        caller_timed_out = False

        def read_timings():
            return {
                key: timing[key]
                for key in (
                    "lookup_budget_ms",
                    "metadata_ms",
                    "transfer_queue_ms",
                )
                if key in timing
            }

        def read(meta):
            size = meta.size_bytes
            timing["transfer_queue_ms"] = self._report_queue_wait(
                "read_transfer", transfer_queued_at
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("remote embedding transfer queue deadline expired")
            client = None
            slot = None
            try:
                timing["stage"] = "client_acquire"
                client = self._get_client()
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote embedding transfer deadline expired")
                native_read = getattr(type(self._gdr_pool), "try_native_read", None)
                if self._direct_payload and callable(native_read):
                    timing["stage"] = "transfer"
                    api_started = time.perf_counter()
                    result = native_read(self._gdr_pool, meta, key, trace_id, deadline, consumer_stream)
                    if result is not None:
                        try:
                            value = self._finish_native_read(result, size, key, consumer_stream)
                        except Exception as error:
                            if _is_read_miss(error):
                                timing["outcome"] = "miss"
                                return None
                            self._retire_client(client, error)
                            raise
                        timing["sdk_api_ms"] = (time.perf_counter() - api_started) * 1000
                        return value
                    unwrap = getattr(type(self._gdr_pool), "unwrap_object_meta", None)
                    if callable(unwrap):
                        meta = unwrap(meta)
                if self._gdr_pool is not None:
                    timing["stage"] = "client_acquire"
                    slot_wait = min(
                        self._GDR_READ_SLOT_WAIT_SECONDS,
                        max(0.0, (deadline - time.monotonic()) / 2),
                    )
                    with self._observe_inflight(
                        "read_gdr_wait"
                    ), self._observe_stage("read", "gdr_acquire"):
                        slot = self._gdr_pool.acquire(nbytes=size, timeout=slot_wait)
                    if slot is None:
                        timing["outcome"] = "skipped"
                        self._report_skip(
                            "read", "gdr_slot_unavailable", report_outcome=False
                        )
                        self._count("miss")
                        self.record(
                            "load_miss",
                            key=key,
                            trace_id=trace_id,
                            reason="gdr_slot_unavailable",
                        )
                        return None
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "remote embedding slot acquisition deadline expired"
                        )
                source = slot if slot is not None else client
                timing["stage"] = "transfer"
                api_started = time.perf_counter()
                try:
                    with self._observe_stage("read", "transfer", size):
                        if self._direct_payload:
                            buffer = slot.load_object_from_meta(
                                meta,
                                consumer_stream=consumer_stream,
                                expected_key=key,
                                trace_id=trace_id,
                                deadline=deadline,
                            )
                        else:
                            # CPU views may outlive the transfer. Give them
                            # independent memory in the staging path as before.
                            buffer = torch.empty(size, dtype=torch.uint8)
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "remote embedding buffer preparation deadline expired"
                                )
                            source.load_from_meta(meta, buffer, trace_id=trace_id)
                except Exception as error:
                    if _is_read_miss(error):
                        timing["outcome"] = "miss"
                        return None
                    self._retire_client(client, error)
                    raise
                timing["sdk_api_ms"] = (time.perf_counter() - api_started) * 1000
                return buffer
            finally:
                if slot is not None:
                    # Only the IO worker owns this lease. A caller timeout or
                    # queued-task cancellation cannot release an active buffer;
                    # an SDK-unsafe slot remains quarantined by the transport.
                    self._gdr_pool.release(slot)
                if client is not None:
                    # Close may join native workers even after our last call.
                    # It stays in this admitted IO task, never the request
                    # thread; its task/byte budget remains held until done.
                    self._release_client(client)

        def release_transfer():
            self._add_inflight("read_transfer", -1)
            self._transfer_slots.release()

        def transfer(meta):
            try:
                with self._observe_inflight("read_worker"):
                    return read(meta)
            finally:
                release_transfer()

        def release_cancelled_transfer(task):
            # A cancelled queued task never enters transfer().
            if task.cancelled():
                release_transfer()

        try:
            # The existing ViT request worker calls the native SDK, which
            # releases the GIL and applies the remaining deadline to the RPC.
            # Only metadata hits can reserve payload resources below.
            timing["stage"] = "client_acquire"
            client = self._get_client()
            try:
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    raise TimeoutError("remote embedding lookup deadline expired")
                timing["stage"] = "metadata"
                timing["lookup_budget_ms"] = remaining_ms
                api_started = time.perf_counter()
                try:
                    meta = self._lookup_object_meta(
                        client, key, trace_id, remaining_ms, deadline
                    )
                finally:
                    timing["metadata_ms"] = (time.perf_counter() - api_started) * 1000
                    timing["sdk_api_ms"] = timing["metadata_ms"]
            finally:
                self._release_client(client)
            if time.monotonic() >= deadline:
                raise TimeoutError("remote embedding metadata completed after deadline")
            if meta is None:
                outcome = "miss"
                self._count("miss")
                self.record("load_miss", key=key, trace_id=trace_id)
                return None

            if self._gdr_pool_quarantined():
                outcome = "skipped"
                self._report_skip("read", "gdr_quarantined", report_outcome=False)
                return None

            size = meta.size_bytes
            timing["nbytes"] = size
            reservation = size * (1 if self._direct_payload else 3)
            with self._lock:
                if self._closed:
                    skip_reason = "executor_closed"
                elif self._bytes + reservation > self.max_inflight_bytes:
                    skip_reason = "payload_budget"
                else:
                    skip_reason = None
                    cost = reservation
                    self._tasks += 1
                    self._bytes += cost
            if skip_reason is not None:
                self._count("admission_skip")
                outcome = "admission_skip"
                self._report_skip("read", skip_reason)
                return None

            consumer_stream = None
            if self._direct_payload:
                # Restore on the leased slot stream, but keep the independent
                # allocations protected while the request stream consumes them.
                with self._observe_stage(
                    "read", "consumer_stream"
                ), cuda_graph_gate.operation():
                    consumer_stream = torch.cuda.current_stream(self.cuda_device)
            if not self._transfer_slots.acquire(blocking=False):
                self._count("admission_skip")
                outcome = "admission_skip"
                self._report_skip("read", "transfer_admission")
                return None
            timing["stage"] = "transfer_queue"
            transfer_queued_at = time.perf_counter()
            self._add_inflight("read_transfer", 1)
            try:
                future = self._transfer_executor.submit(transfer, meta)
            except BaseException:
                release_transfer()
                raise
            future.add_done_callback(release_cancelled_transfer)
            result_wait_started = time.perf_counter()
            try:
                buffer = future.result(timeout=max(0, deadline - time.monotonic()))
            finally:
                self._report_queue_wait("read_result", result_wait_started)
            if buffer is None:
                outcome = timing["outcome"]
                if outcome == "miss":
                    self._count("miss")
                    self.record("load_miss", key=key, trace_id=trace_id)
                return None
            with self._observe_stage("read", "restore"):
                timing["stage"] = "restore"
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote embedding read expired before restore")
                value = (
                    buffer
                    if self._direct_payload
                    else unpack_object(
                        buffer,
                        max_bytes=self.max_object_bytes,
                        cuda_device=self.cuda_device,
                        observe_stage=self._observe_stage,
                        expected_key=key,
                    )
                )
                value, approval = self._decode_value(
                    value, key, require_key=self._transport_mode == "gdr"
                )
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote embedding read expired before publication")
                if (
                    entry is not None
                    and restore_approval
                    and not self.request_scoped_inspection
                    and approval is not None
                ):
                    entry.set_greennet_verdict(approval)
            self._count("hit")
            outcome = "hit"
            payload_nbytes = size
            self.record(
                "load_hit",
                key=key,
                trace_id=trace_id,
                nbytes=size,
            )
            return value
        except Exception as error:
            timeout_fields = {}
            if _is_transfer_timeout(error):
                outcome = "timeout"
                self._count("read_timeout")
                if future is not None and timing["stage"] != "restore":
                    # Only caller-side timeouts need a later completion diagnostic.
                    # SDK timeouts raised by this same worker remain its outcome.
                    caller_timed_out = not future.done() or (
                        not future.cancelled() and future.exception() is not error
                    )
                    if caller_timed_out:
                        future.cancel()
                timeout_fields = {"caller_timed_out": caller_timed_out}
            self.record(
                "read_error",
                key=key,
                trace_id=trace_id,
                stage=timing["stage"],
                nbytes=timing["nbytes"],
                timeout_ms=timeout_ms,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                **timeout_fields,
                **read_timings(),
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

            def complete_read(completed):
                try:
                    if caller_timed_out and not completed.cancelled():
                        error = completed.exception()
                        if error is not None:
                            self.record(
                                "read_late_error",
                                key=key,
                                trace_id=trace_id,
                                stage=timing["stage"],
                                nbytes=timing["nbytes"],
                                timeout_ms=timeout_ms,
                                caller_timed_out=True,
                                elapsed_ms=(time.perf_counter() - started) * 1000,
                                **read_timings(),
                                **self._error_context(error),
                            )
                        else:
                            # This is the completion of an already failed read,
                            # not a new hit or a second error-QPS observation.
                            self.record(
                                "read_late_complete",
                                key=key,
                                trace_id=trace_id,
                                stage=timing["stage"],
                                nbytes=timing["nbytes"],
                                timeout_ms=timeout_ms,
                                caller_timed_out=True,
                                elapsed_ms=(time.perf_counter() - started) * 1000,
                                sdk_api_ms=timing.get("sdk_api_ms"),
                                **read_timings(),
                                completion_outcome=timing.get("outcome", "success"),
                            )
                finally:
                    if cost:
                        self._release(cost)

            if future is None or (future.done() and not caller_timed_out):
                if cost:
                    self._release(cost)
            else:
                future.add_done_callback(complete_read)

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
        if self._gdr_pool is not None:
            self._gdr_pool.stop_acquiring()
        self._write_executor.shutdown(wait=True, cancel_futures=True)
        self._transfer_executor.shutdown(wait=True, cancel_futures=True)
        with self._client_lock:
            client, self._client = self._client, None
            if client is not None:
                state = self._client_users.get(id(client))
                if state is not None:
                    state[2] = True
                    client = None
        if client is not None:
            self._close_client(client)
