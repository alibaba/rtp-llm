"""Bounded, request-scoped V4.1 patch uploads on the selected ViT endpoint."""

import re
import threading
import time
from dataclasses import dataclass

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    EmptyPB,
    V41RequestInputsPB,
)


@dataclass(frozen=True)
class _Upload:
    request_id: int
    inputs: V41RequestInputsPB
    metadata: bytes
    size: int
    expires_at: float


class V41InputTransfer:
    """Shared RPC handlers for standalone ViT and the multi-process ViT proxy.

    The proxy resolves uploads before choosing a child, so child failover never
    needs to migrate handles. Entries survive reads until frontend release or TTL.
    There is no eviction of live uploads: admission fails when the budget is full.
    """

    def __init__(self, max_bytes=512 * 1024**2, max_entries=128):
        self._v41_uploads = {}
        self._v41_bytes = 0
        self._v41_max_bytes = max_bytes
        self._v41_max_entries = max_entries
        self._v41_condition = threading.Condition()
        self._v41_reaper = None

    def _remove_upload(self, payload_id):
        entry = self._v41_uploads.pop(payload_id, None)
        if entry is not None:
            self._v41_bytes -= entry.size

    def _expire_uploads(self):
        now = time.monotonic()
        for key, entry in list(self._v41_uploads.items()):
            if entry.expires_at <= now:
                self._remove_upload(key)

    def _reap_uploads(self):
        # One lazily started thread, which exits when the last upload is gone.
        with self._v41_condition:
            while True:
                self._expire_uploads()
                if not self._v41_uploads:
                    self._v41_reaper = None
                    return
                deadline = min(e.expires_at for e in self._v41_uploads.values())
                self._v41_condition.wait(max(0, deadline - time.monotonic()))

    def UploadV41Inputs(self, request, context):
        typed = request.inputs
        if (
            not re.fullmatch(r"[0-9a-f]{32}", request.payload_id)
            or request.ttl_ms <= 0
            or request.ttl_ms > 600000
            or typed.schema_version != 1
            or typed.payload_id
            or not typed.images
            or any(not image.HasField("patches") for image in typed.images)
        ):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "invalid V4.1 direct upload"
            )
        size = typed.ByteSize()
        with self._v41_condition:
            self._expire_uploads()
            if request.payload_id in self._v41_uploads:
                # Never overwrite an existing request, even with the same ID.
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS, "V4.1 upload already exists"
                )
            if (
                len(self._v41_uploads) >= self._v41_max_entries
                or self._v41_bytes + size > self._v41_max_bytes
            ):
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED, "V4.1 upload budget exhausted"
                )
            stored = V41RequestInputsPB()
            stored.CopyFrom(typed)
            metadata = V41RequestInputsPB()
            metadata.CopyFrom(typed)
            for image in metadata.images:
                image.ClearField("patches")
            metadata.payload_id = request.payload_id
            self._v41_uploads[request.payload_id] = _Upload(
                request.request_id,
                stored,
                metadata.SerializeToString(deterministic=True),
                size,
                time.monotonic() + request.ttl_ms / 1000.0,
            )
            self._v41_bytes += size
            if self._v41_reaper is None:
                self._v41_reaper = threading.Thread(
                    target=self._reap_uploads, daemon=True
                )
                self._v41_reaper.start()
            self._v41_condition.notify_all()
        return EmptyPB()

    def ReleaseV41Inputs(self, request, context):
        with self._v41_condition:
            entry = self._v41_uploads.get(request.payload_id)
            if entry is not None and entry.request_id != request.request_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "V4.1 upload request mismatch"
                )
            self._remove_upload(request.payload_id)
            self._v41_condition.notify_all()
        return EmptyPB()

    def resolve_v41_inputs(self, request, context):
        typed = request.v41_inputs
        if not typed.payload_id:
            return request
        if any(image.HasField("patches") for image in typed.images):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "V4.1 reference contains inline patches",
            )
        metadata = typed.SerializeToString(deterministic=True)
        with self._v41_condition:
            self._expire_uploads()
            entry = self._v41_uploads.get(typed.payload_id)
            if entry is None:
                context.abort(
                    grpc.StatusCode.NOT_FOUND, "V4.1 upload expired or was released"
                )
            if entry.request_id != request.request_id or entry.metadata != metadata:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "V4.1 upload metadata mismatch"
                )
            # Copy while holding a reference to the immutable stored payload. A
            # concurrent release cannot invalidate an already admitted read.
        request.v41_inputs.CopyFrom(entry.inputs)
        return request
