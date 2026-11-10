"""Private, bounded two-slot refit hash candidate; not integrated into RTP.

This only hashes one already-stable, contiguous source. The caller owns its
GPU/IPC lifetime and the transaction; it must retain the source through ACK.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading


def _pipeline(nbytes, chunk_bytes, submit, view, stream_digest, item_digest, *,
              pool=None, serial_cutoff=256*1024):
    """submit(slot,offset,length) returns an object with synchronize().

    A slot is not submitted again until both digest consumers have completed.
    Callers own transport shutdown on success or failure.
    """
    if nbytes < 0 or chunk_bytes <= 0: raise ValueError('invalid byte count or chunk size')
    count = (nbytes + chunk_bytes - 1) // chunk_bytes
    if not count: return
    tokens = [None, None]

    def schedule(index):
        offset=index*chunk_bytes;length=min(chunk_bytes,nbytes-offset);slot=index%2
        tokens[slot] = (submit(slot,offset,length),length)

    owns_pool=pool is None
    if owns_pool:pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='q36-hash')
    try:
        schedule(0)
        for i in range(count):
            if i+1<count:schedule(i+1)
            slot=i%2;token,length=tokens[slot]
            token.synchronize()
            payload=view(slot,length)
            if length<=serial_cutoff:
                stream_digest.update(payload);item_digest.update(payload);del payload
                continue
            a=pool.submit(stream_digest.update,payload)
            b=pool.submit(item_digest.update,payload)
            # Wait for both even if either raises; never reuse their buffer early.
            error=None
            for future in (a,b):
                try:future.result()
                except BaseException as exc:
                    if error is None:error=exc
            del payload
            if error is not None:raise error
    finally:
        if owns_pool:pool.shutdown(wait=True)


def hash_cuda_source(value, name, source_digest, *, chunk_bytes=4*1024*1024):
    """Hash a CUDA source and append the exact existing full-stream format.

    The source must not be mutated or released while this call runs or before
    its separate native ACK. Two pinned chunks bound the host allocation. This
    function neither advances the export iterator nor performs an IPC push.
    """
    workspace=CudaDigestWorkspace(value.device,chunk_bytes=chunk_bytes,
                                  serial_cutoff=min(256*1024,chunk_bytes))
    try:
        return workspace.hash(value,name,source_digest)
    finally:
        workspace.close()


# If CUDA synchronization itself fails, neither pinned host storage nor the
# source can be proven safe to release. Retain them until process teardown.
_CUDA_FAILURE_QUARANTINE=[]


class CudaDigestWorkspace:
    """Per-worker reusable slots and executor; one active digest at a time."""
    def __init__(self,device,*,chunk_bytes=4*1024*1024,serial_cutoff=256*1024):
        import torch
        if type(chunk_bytes) is not int or not 0<chunk_bytes<=16*1024*1024:
            raise ValueError('invalid bounded chunk size')
        if type(serial_cutoff) is not int or not 0<=serial_cutoff<=chunk_bytes:
            raise ValueError('invalid serial cutoff')
        self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('CUDA device required')
        self.chunk_bytes=chunk_bytes;self.serial_cutoff=serial_cutoff
        self.slots=[torch.empty(chunk_bytes,dtype=torch.uint8,device='cpu',pin_memory=True) for _ in range(2)]
        self.stream=torch.cuda.Stream(device=self.device)
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='q36-hash')
        self.lock=threading.Lock();self.closed=False;self.poisoned=False

    def hash(self,value,name,source_digest):
        import torch
        if not self.lock.acquire(blocking=False):raise RuntimeError('digest workspace already active')
        try:
            if self.closed or self.poisoned:raise RuntimeError('digest workspace is unavailable')
            if value.device!=self.device or not value.is_contiguous():raise ValueError('source device/layout mismatch')
            error=None
            try:
                source_digest.update(name.encode()+b'\0'+str(tuple(value.shape)).encode()+b'\0'+str(value.dtype).encode()+b'\0')
                item=hashlib.sha256();flat=value.detach().reshape(-1).view(torch.uint8);nbytes=flat.numel()
                self.stream.wait_stream(torch.cuda.current_stream(self.device));value.record_stream(self.stream)
                def submit(slot,offset,length):
                    with torch.cuda.stream(self.stream):
                        self.slots[slot][:length].copy_(flat[offset:offset+length],non_blocking=True)
                        event=torch.cuda.Event();event.record(self.stream)
                    return event
                def view(slot,length):return memoryview(self.slots[slot][:length].numpy()).cast('B')
                _pipeline(nbytes,self.chunk_bytes,submit,view,source_digest,item,
                          pool=self.pool,serial_cutoff=self.serial_cutoff)
            except BaseException as exc:
                error=exc;self.poisoned=True;raise
            finally:
                try:self.stream.synchronize()
                except BaseException:
                    self.poisoned=True;_CUDA_FAILURE_QUARANTINE.append((self,value))
                    if error is None:raise
            return item.hexdigest(),nbytes
        finally:self.lock.release()

    def close(self):
        if not self.lock.acquire(blocking=False):raise RuntimeError('digest workspace still active')
        try:
            if not self.closed:
                self.pool.shutdown(wait=True);self.closed=True
        finally:self.lock.release()
