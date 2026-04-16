"""Content Addressable Storage client for REAPI v2."""
import logging
import threading
import uuid
import grpc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from . import remote_execution_pb2 as re_pb2
from . import remote_execution_pb2_grpc as re_grpc
from . import bytestream_pb2 as bs_pb2
from . import bytestream_pb2_grpc as bs_grpc
from .merkle import build_merkle_tree, sha256_digest
from .endpoint_info import describe_reapi_endpoint

log = logging.getLogger(__name__)

MAX_BATCH_SIZE = 3 * 1024 * 1024
BYTESTREAM_THRESHOLD = 3 * 1024 * 1024
BYTESTREAM_CHUNK = 2 * 1024 * 1024
GRPC_MAX_MSG_SIZE = 16 * 1024 * 1024
PARALLEL_UPLOADS = 12  # concurrent ByteStream uploads for large files
BATCH_UPLOAD_WORKERS = 12  # concurrent BatchUpdateBlobs RPCs


@dataclass
class UploadProgress:
    """Thread-safe CAS upload progress for live terminal display."""

    total_blobs: int = 0
    total_bytes: int = 0
    uploaded_blobs: int = 0
    uploaded_bytes: int = 0
    skipped_blobs: int = 0
    skipped_bytes: int = 0
    phase: str = "merkle"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def set_totals(self, total_blobs: int, total_bytes: int) -> None:
        with self._lock:
            self.total_blobs = total_blobs
            self.total_bytes = total_bytes

    def set_skipped(self, skipped_blobs: int, skipped_bytes: int) -> None:
        with self._lock:
            self.skipped_blobs = skipped_blobs
            self.skipped_bytes = skipped_bytes

    def add_uploaded(self, n_blobs: int, n_bytes: int) -> None:
        with self._lock:
            self.uploaded_blobs += n_blobs
            self.uploaded_bytes += n_bytes


class CASClient:
    def __init__(
        self,
        endpoint: str,
        metadata: Optional[List[tuple]] = None,
        *,
        batch_upload_workers: int = BATCH_UPLOAD_WORKERS,
        parallel_bytestream: int = PARALLEL_UPLOADS,
    ):
        self.grpc_uri = endpoint
        self.reapi_peer_line = describe_reapi_endpoint("CAS", endpoint)
        log.info("REAPI %s", self.reapi_peer_line)

        addr = endpoint.replace("grpc://", "")
        self._addr = addr
        self.channel = grpc.insecure_channel(
            addr, options=[("grpc.max_send_message_length", GRPC_MAX_MSG_SIZE),
                           ("grpc.max_receive_message_length", GRPC_MAX_MSG_SIZE)])
        self.stub = re_grpc.ContentAddressableStorageStub(self.channel)
        self.bs_stub = bs_grpc.ByteStreamStub(self.channel)
        self.metadata = metadata or []
        self.instance_name = ""
        self._batch_upload_workers = max(1, batch_upload_workers)
        self._parallel_bytestream = max(1, parallel_bytestream)
        self._extra_channels: list = []

    def _new_bs_stub(self):
        """Create a new ByteStream stub on a separate channel (for parallel uploads)."""
        ch = grpc.insecure_channel(
            self._addr, options=[("grpc.max_send_message_length", GRPC_MAX_MSG_SIZE)])
        self._extra_channels.append(ch)
        return bs_grpc.ByteStreamStub(ch)

    def new_bytestream_stub(self) -> bs_grpc.ByteStreamStub:
        """Separate channel for ByteStream.Read (streaming logs) or parallel Write."""
        ch = grpc.insecure_channel(
            self._addr,
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MSG_SIZE),
                ("grpc.max_receive_message_length", GRPC_MAX_MSG_SIZE),
            ],
        )
        self._extra_channels.append(ch)
        return bs_grpc.ByteStreamStub(ch)

    def close(self):
        """Close all gRPC channels (main + extras created for parallel ops)."""
        for ch in self._extra_channels:
            try:
                ch.close()
            except Exception:
                pass
        self._extra_channels.clear()
        try:
            self.channel.close()
        except Exception:
            pass

    def upload_directory(
        self,
        root: Path,
        files: List[str],
        progress: Optional[UploadProgress] = None,
    ) -> re_pb2.Digest:
        """Build Merkle tree and upload missing blobs to CAS. Memory-efficient."""
        if progress:
            progress.set_phase("merkle")

        result = build_merkle_tree(root, files)
        total_size = sum(result.sizes.values())
        total_count = len(result.file_map) + len(result.dir_blobs)
        total_mb = total_size / 1024 / 1024
        log.info("Merkle tree: %d blobs, %.1f MB total", total_count, total_mb)

        all_digests = [re_pb2.Digest(hash=h, size_bytes=s) for h, s in result.sizes.items()]
        if progress:
            progress.set_phase("find_missing")

        missing = self._find_missing(all_digests)
        if not missing:
            log.info("All blobs already in CAS")
            log.info("[CAS_SUMMARY] blobs=%d total_mb=%.1f missing=0 uploaded_mb=0.0", total_count, total_mb)
            if progress:
                progress.set_skipped(total_count, total_size)
                progress.set_totals(0, 0)
                progress.set_phase("done")
            return result.root_digest

        missing_size = sum(result.sizes.get(h, 0) for h in missing)
        log.info("Uploading %d missing blobs (%.1f MB)", len(missing), missing_size / 1024 / 1024)
        log.info(
            "[CAS_SUMMARY] blobs=%d total_mb=%.1f missing=%d uploaded_mb=%.1f",
            total_count,
            total_mb,
            len(missing),
            missing_size / 1024 / 1024,
        )

        skipped_blobs = total_count - len(missing)
        skipped_bytes = total_size - missing_size
        if progress:
            progress.set_skipped(skipped_blobs, skipped_bytes)
            progress.set_totals(len(missing), missing_size)
            progress.set_phase("uploading")

        # Collect BatchUpdateBlobs batches (dir blobs + small files), then upload in parallel
        all_batches: List[List[re_pb2.BatchUpdateBlobsRequest.Request]] = []

        small_batch: List[re_pb2.BatchUpdateBlobsRequest.Request] = []
        small_batch_size = 0
        for h in list(missing):
            if h in result.dir_blobs:
                data = result.dir_blobs[h]
                small_batch.append(re_pb2.BatchUpdateBlobsRequest.Request(
                    digest=re_pb2.Digest(hash=h, size_bytes=len(data)), data=data))
                small_batch_size += len(data)
                missing.discard(h)
                if small_batch_size > MAX_BATCH_SIZE:
                    all_batches.append(small_batch)
                    small_batch, small_batch_size = [], 0
        if small_batch:
            all_batches.append(small_batch)

        large_files: List[tuple] = []
        small_files: List[tuple] = []
        for h in missing:
            path = result.file_map.get(h)
            if path is None:
                continue
            size = result.sizes[h]
            if size > BYTESTREAM_THRESHOLD:
                large_files.append((h, path, size))
            else:
                small_files.append((h, path, size))

        file_batch: List[re_pb2.BatchUpdateBlobsRequest.Request] = []
        file_batch_size = 0
        for h, path, size in small_files:
            data = path.read_bytes()
            file_batch.append(re_pb2.BatchUpdateBlobsRequest.Request(
                digest=re_pb2.Digest(hash=h, size_bytes=size), data=data))
            file_batch_size += len(data)
            if file_batch_size > MAX_BATCH_SIZE:
                all_batches.append(file_batch)
                file_batch, file_batch_size = [], 0
        if file_batch:
            all_batches.append(file_batch)

        def _send_batch_tracked(batch: List[re_pb2.BatchUpdateBlobsRequest.Request]) -> None:
            self._send_batch(batch)
            if progress:
                n_bytes = sum(len(req.data) for req in batch)
                progress.add_uploaded(len(batch), n_bytes)

        if all_batches:
            if len(all_batches) == 1:
                _send_batch_tracked(all_batches[0])
            else:
                workers = min(len(all_batches), self._batch_upload_workers)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(_send_batch_tracked, all_batches))

        if large_files:
            n_workers = min(len(large_files), self._parallel_bytestream)
            log.info("ByteStream uploading %d large files in parallel (workers=%d)",
                     len(large_files), n_workers)

            def _upload_large(args: tuple) -> None:
                h, path, size = args
                digest = re_pb2.Digest(hash=h, size_bytes=size)
                self._bytestream_write_file_parallel(digest, path)
                if progress:
                    progress.add_uploaded(1, size)

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for h, path, size in large_files:
                    log.info("  ByteStream: %s (%.1f MB) %s", h[:12], size / 1024 / 1024, path.name)
                list(pool.map(_upload_large, large_files))

        if progress:
            progress.set_phase("done")

        return result.root_digest

    def upload_blob(self, data: bytes) -> re_pb2.Digest:
        """Upload a single blob (always uploads)."""
        digest = sha256_digest(data)
        if len(data) > BYTESTREAM_THRESHOLD:
            self._bytestream_write(digest, data)
        else:
            self._send_batch([re_pb2.BatchUpdateBlobsRequest.Request(
                digest=re_pb2.Digest(hash=digest.hash, size_bytes=digest.size_bytes), data=data)])
        return digest

    def download_blob(self, digest: re_pb2.Digest) -> bytes:
        """Download a blob from CAS.  Uses ByteStream for large blobs."""
        if digest.size_bytes > BYTESTREAM_THRESHOLD:
            return self._bytestream_read(digest)
        try:
            resp = self.stub.BatchReadBlobs(
                re_pb2.BatchReadBlobsRequest(instance_name=self.instance_name, digests=[digest]),
                metadata=self.metadata)
            for r in resp.responses:
                if r.status.code == 0:
                    return r.data
        except grpc.RpcError:
            pass
        return b""

    def _bytestream_read(self, digest: re_pb2.Digest) -> bytes:
        """Download a blob via ByteStream.Read (symmetric to _bytestream_write)."""
        stub = self.new_bytestream_stub()
        resource_name = f"{self.instance_name}/blobs/{digest.hash}/{digest.size_bytes}"
        chunks: list = []
        try:
            for resp in stub.Read(
                bs_pb2.ReadRequest(resource_name=resource_name, read_offset=0, read_limit=0),
                metadata=self.metadata,
                timeout=300,
            ):
                chunks.append(resp.data)
        except grpc.RpcError as e:
            log.warning("ByteStream.Read failed for %s: %s", digest.hash[:12], e)
            return b""
        return b"".join(chunks)

    def _find_missing(self, digests: List[re_pb2.Digest]) -> Set[str]:
        missing = set()
        for i in range(0, len(digests), 1000):
            batch = digests[i:i + 1000]
            resp = self.stub.FindMissingBlobs(
                re_pb2.FindMissingBlobsRequest(instance_name=self.instance_name, blob_digests=batch),
                metadata=self.metadata)
            missing.update(d.hash for d in resp.missing_blob_digests)
        return missing

    def _bytestream_write_file_parallel(self, digest: re_pb2.Digest, path: Path):
        """Upload a file via ByteStream on a dedicated channel (thread-safe)."""
        stub = self._new_bs_stub()
        uid = uuid.uuid4().hex
        resource_name = f"{self.instance_name}/uploads/{uid}/blobs/{digest.hash}/{digest.size_bytes}"

        def _chunks():
            with open(path, "rb") as f:
                offset = 0
                is_last = False
                while True:
                    data = f.read(BYTESTREAM_CHUNK)
                    if not data:
                        break
                    is_last = len(data) < BYTESTREAM_CHUNK
                    yield bs_pb2.WriteRequest(
                        resource_name=resource_name,
                        write_offset=offset,
                        data=data,
                        finish_write=is_last,
                    )
                    offset += len(data)
                if not is_last:
                    yield bs_pb2.WriteRequest(
                        resource_name=resource_name,
                        write_offset=offset,
                        finish_write=True,
                    )

        stub.Write(_chunks(), metadata=self.metadata)

    def _bytestream_write(self, digest: re_pb2.Digest, data: bytes):
        """Upload in-memory data via ByteStream Write RPC."""
        uid = uuid.uuid4().hex
        resource_name = f"{self.instance_name}/uploads/{uid}/blobs/{digest.hash}/{digest.size_bytes}"

        def _chunks():
            offset = 0
            while offset < len(data):
                end = min(offset + BYTESTREAM_CHUNK, len(data))
                yield bs_pb2.WriteRequest(
                    resource_name=resource_name,
                    write_offset=offset,
                    data=data[offset:end],
                    finish_write=(end == len(data)),
                )
                offset = end

        self.bs_stub.Write(_chunks(), metadata=self.metadata)

    def _send_batch(self, requests):
        resp = self.stub.BatchUpdateBlobs(
            re_pb2.BatchUpdateBlobsRequest(instance_name=self.instance_name, requests=requests),
            metadata=self.metadata)
        for r in resp.responses:
            if r.status.code != 0:
                log.warning("BatchUpdateBlobs failed for %s: code=%d msg=%s",
                            r.digest.hash[:12], r.status.code, r.status.message)
