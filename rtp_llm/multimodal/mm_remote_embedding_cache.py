"""Bounded shared KVCM tier, populated only by detached CPU cache evictions.

Remote identity binds actual image bytes to an immutable model namespace.
The local GPU/CPU scheduler and asynchronous eviction policy remain unchanged.
"""

import base64
import binascii
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

import torch

from rtp_llm.multimodal import emb_deepseek_diagnostics, emb_diagnostics
from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)
from rtp_llm.multimodal.mm_preprocess_result import (
    ImageSourceIdentity,
    is_http_image_input,
)


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
        model_identity=None
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
        self._namespace = namespace
        self._model_identity = model_identity or {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mm-remote-cache"
        )
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = Counter()
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
        cls, model_config, vit_config, *, kv_cache_config=None, device=None
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

        def factory():
            kwargs = dict(
                max_object_bytes=vit_config.mm_remote_cache_max_object_bytes,
                call_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
                get_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
                put_timeout_ms=vit_config.mm_remote_cache_write_timeout_ms,
            )
            if kv_cache_config is not None:
                return RtpKvMetaObjectClient.from_kv_cache_config(
                    kv_cache_config, **kwargs
                )
            return RtpKvMetaObjectClient.from_env(**kwargs)

        return cls(
            None,
            client_factory=factory,
            namespace=namespace,
            model_identity=identity,
            max_object_bytes=vit_config.mm_remote_cache_max_object_bytes,
            max_inflight_bytes=vit_config.mm_remote_cache_max_inflight_bytes,
            max_pending=vit_config.mm_remote_cache_max_pending,
            read_timeout_ms=vit_config.mm_remote_cache_read_timeout_ms,
            cuda_device=device,
        )

    def normalize_cpp_url_input(self, item):
        # QueryConverter's GenerateInput URL path passes torch::empty(1), not
        # image data. The verified Qwen3/VL2 IMAGE preprocessors ignore this
        # field and consume url. Never inspect the uninitialized scalar value.
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        tensor = item.tensor
        if (
            self._model_identity.get("model_type") in ("qwen3_vl", "deepseek_vl_v2")
            and item.mm_type == MMUrlType.IMAGE
            and item.url.startswith("data:image/")
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

    def key(self, mm_inputs, *, trace_id=None, source_identity=None):
        def bypass(reason, **fields):
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
        # The V4.1 renderer wraps immutable image bytes as octet-stream.
        # IMAGE remains the type gate; the MIME label is not part of identity.
        match = re.match(
            r"^data:(?:image/[^;,]+|application/octet-stream);base64,", item.url
        )
        if item.tensor is not None and item.tensor.numel():
            return bypass(
                "nonempty_tensor",
                tensor_shape=list(item.tensor.shape),
                tensor_dtype=str(item.tensor.dtype),
                tensor_device=str(item.tensor.device),
            )
        identity_source = "inline_image_bytes"
        if match is not None:
            try:
                media = base64.b64decode(item.url[match.end() :], validate=True)
            except (binascii.Error, ValueError):
                return bypass("invalid_inline_base64")
            if not media:
                return bypass("empty_inline_image")
            identity = ImageSourceIdentity.from_bytes(media)
            if source_identity is not None and source_identity != identity:
                return bypass("source_identity_mismatch")
        elif is_http_image_input(mm_inputs) and isinstance(
            source_identity, ImageSourceIdentity
        ):
            # Only the trusted CPU preprocessor can supply this sidecar. The
            # public request/RPC contains no digest field. URL strings, headers
            # and signed query parameters are never accepted as content identity.
            identity = source_identity
            identity_source = "downloaded_image_bytes"
        else:
            return bypass("non_inline_image")
        limits = self._model_identity.get("input_limits", [])
        if limits and limits[0] > 0 and identity.nbytes > limits[0] * 1024:
            return bypass(
                "inline_image_too_large" if match else "downloaded_image_too_large",
                media_nbytes=identity.nbytes,
            )
        config = item.mm_preprocess_config
        fields = (
            "width",
            "height",
            "min_pixels",
            "max_pixels",
            "fps",
            "min_frames",
            "max_frames",
            "crop_positions",
            "max_long_side_pixel",
        )
        preprocessing = {name: getattr(config, name, None) for name in fields}
        media_sha256 = identity.sha256
        material = dict(
            schema="rtp-mm-result-v2",
            namespace=self._namespace,
            model=self._model_identity,
            mm_type=int(item.mm_type),
            media_sha256=media_sha256,
            preprocessing=preprocessing,
        )
        encoded = json.dumps(
            material, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        key = "mm-v2:" + hashlib.sha256(encoded).hexdigest()
        self.record(
            "cache_admitted",
            key=key,
            trace_id=trace_id,
            identity_source=identity_source,
            media_sha256=media_sha256,
            media_nbytes=identity.nbytes,
        )
        return key

    def record(self, event, *, key=None, trace_id=None, **fields):
        with self._stats_lock:
            self._stats[event] += 1
            count = self._stats[event]
        payload = dict(
            event=event,
            key=key,
            trace_id=trace_id,
            timestamp_ns=time.time_ns(),
            pid=os.getpid(),
            proc_start_ticks=self._proc_start_ticks,
            instance=self._instance,
            count=count,
            **fields
        )
        logging.info("MM_EMB_REMOTE_JSON %s", json.dumps(payload, sort_keys=True))

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
                            **fields
                        ),
                        sort_keys=True,
                    ),
                )
            except Exception:
                pass  # Never prevent IO-task/client release in a finally block.

    def _get_client(self):
        with self._client_lock:
            if self._closed:
                raise RuntimeError("remote embedding cache is closed")
            if self._client is None:
                if self._factory is None:
                    raise RuntimeError("remote embedding client is unavailable")
                self._client = self._factory()
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

    def _retire_client(self, client):
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

    def observe_entry(self, entry, trace_id):
        if entry is None or not hasattr(entry, "remote_cache_context"):
            return
        context = entry.remote_cache_context
        if context["trace_id"] != trace_id:
            return
        # Only immutable metadata/hashes are retained; no tensor or file is an
        # input to inference. The entry owns this bounded observation context.
        context["qwen_origin"] = emb_diagnostics.eviction_origin(trace_id)
        context["deepseek_origin"] = emb_deepseek_diagnostics.eviction_origin(trace_id)

    def submit_eviction(self, local_key, entry):
        """Admit only a completed CPU eviction, after pool-storage detachment."""
        context = getattr(entry, "remote_cache_context", None)
        if (
            not context
            or entry.pool_owners
            or entry.result is None
            or entry.error is not None
        ):
            return
        key, trace_id = context["key"], context["trace_id"]
        try:
            plan = plan_object(
                entry.result,
                devices=entry.original_devices,
                max_bytes=self.max_object_bytes,
            )
            if any(t.device.type != "cpu" for t in plan.tensors):
                return
        except (ValueError, TypeError, RecursionError):
            self._count("encode_skip")
            return
        cost = 3 * plan.nbytes
        with self._lock:
            if key in self._pending_writes:
                return
            self._pending_writes.add(key)
        if not self._reserve(cost):
            with self._lock:
                self._pending_writes.discard(key)
            return
        self.record(
            "cpu_eviction_admitted",
            key=key,
            trace_id=trace_id,
            generation=entry.generation,
            nbytes=plan.nbytes,
            tensors=plan.header["tensors"],
        )

        def write():
            client = None
            started = time.perf_counter()
            api_ms = None
            outcome = "error"
            try:
                buffer = pack_object(plan)
                emb_diagnostics.eviction_serialized(
                    trace_id, context.get("qwen_origin"), plan, buffer
                )
                emb_deepseek_diagnostics.eviction_serialized(
                    trace_id, context.get("deepseek_origin"), plan, buffer
                )
                client = self._get_client()
                api_started = time.perf_counter()
                client.save_one(key, buffer, trace_id=trace_id)
                api_ms = (time.perf_counter() - api_started) * 1000
                outcome = "success"
                self._count("write_success")
                self.record(
                    "store_success",
                    key=key,
                    trace_id=trace_id,
                    write_source="cpu_eviction",
                    generation=entry.generation,
                    payload_sha256=hashlib.sha256(
                        memoryview(buffer.numpy())
                    ).hexdigest(),
                    nbytes=buffer.numel(),
                    tensors=plan.header["tensors"],
                )
            except Exception as error:
                self.record(
                    "write_error",
                    key=key,
                    trace_id=trace_id,
                    error_type=type(error).__name__,
                )
                if client is not None:
                    self._retire_client(client)
            finally:
                self._benchmark(
                    "remote_store_timing",
                    key,
                    trace_id,
                    wall_ms=(time.perf_counter() - started) * 1000,
                    sdk_api_ms=api_ms,
                    status=outcome,
                    scope="eviction_pack_client_init_sdk_save_and_validation",
                )
                if client is not None:
                    self._release_client(client)

        def done(_):
            with self._lock:
                self._pending_writes.discard(key)
            self._release(cost)

        try:
            future = self._executor.submit(write)
        except RuntimeError:
            done(None)
            return
        future.add_done_callback(done)

    def load(self, key, *, timeout_ms=None, trace_id=None):
        if key is None:
            return None
        timeout_ms = min(self.read_timeout_ms, timeout_ms or self.read_timeout_ms)
        deadline = time.monotonic() + timeout_ms / 1000
        started = time.perf_counter()
        timing = {}
        outcome = "error"
        if not self._reserve(0):
            return None
        cost = 0
        future = None

        def read():
            nonlocal cost
            client = None
            try:
                client = self._get_client()
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    raise TimeoutError("remote embedding lookup deadline expired")
                api_started = time.perf_counter()
                size = client.object_size(
                    key, trace_id=trace_id, timeout_ms=remaining_ms
                )
                if size is None:
                    timing["sdk_api_ms"] = (time.perf_counter() - api_started) * 1000
                    self._count("miss")
                    self.record("load_miss", key=key, trace_id=trace_id)
                    return None
                if not 0 < size <= self.max_object_bytes:
                    raise ValueError("remote embedding object exceeds byte limit")
                with self._lock:
                    if self._bytes + 3 * size > self.max_inflight_bytes:
                        self._count("admission_skip")
                        return None
                    cost = 3 * size
                    self._bytes += cost
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote embedding transfer deadline expired")
                buffer = torch.empty(size, dtype=torch.uint8)
                client.load_one(key, buffer, trace_id=trace_id)
                timing["sdk_api_ms"] = (time.perf_counter() - api_started) * 1000
                return buffer
            except Exception:
                if client is not None:
                    self._retire_client(client)
                raise
            finally:
                if client is not None:
                    # Close may join native workers even after our last call.
                    # It stays in this admitted IO task, never the request
                    # thread; its task/byte budget remains held until done.
                    self._release_client(client)

        try:
            future = self._executor.submit(read)
            buffer = future.result(timeout=max(0, deadline - time.monotonic()))
            if buffer is None:
                outcome = "miss_or_admission_skip"
                return None
            value = unpack_object(
                buffer, max_bytes=self.max_object_bytes, cuda_device=self.cuda_device
            )
            if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
                raise ValueError("remote object is not a multimodal result")
            plan = plan_object(value, max_bytes=self.max_object_bytes)
            emb_diagnostics.serialized(trace_id, plan, buffer, value, loading=True)
            if self._model_identity.get("model_type") == "deepseek_vl_v2":
                emb_deepseek_diagnostics.serialized(
                    trace_id, plan, buffer, loading=True
                )
            self._count("hit")
            outcome = "hit"
            self.record(
                "load_hit",
                key=key,
                trace_id=trace_id,
                payload_sha256=hashlib.sha256(memoryview(buffer.numpy())).hexdigest(),
                nbytes=buffer.numel(),
                tensors=plan.header["tensors"],
            )
            return value
        except concurrent.futures.TimeoutError:
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
                error_type=type(error).__name__,
            )
            return None
        finally:
            self._benchmark(
                "remote_load_timing",
                key,
                trace_id,
                wall_ms=(time.perf_counter() - started) * 1000,
                sdk_api_ms=timing.get("sdk_api_ms"),
                status=outcome,
                scope="caller_queue_client_init_sdk_load_unpack_and_validation",
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
