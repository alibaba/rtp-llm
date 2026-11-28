"""Check the ali noGDS native completion pointer before publishing a CUDA file.

0.1.20+ali and 0.3.4+ali return a destination address on success and nullptr on a worker
error; the wheel's Python file copier tests only for a negative value. On
GB200 a failed worker cudaMemcpy can therefore expose unwritten file bytes.
Keep the native fast path. Only a failed extent is transferred once more,
from the same open fd into the same FST buffer, on the owning CUDA device.
"""

import logging
import os


def checked_nogds_loader(base):
    class CheckedLoader(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if kwargs.get("nogds") and not kwargs.get("use_shm"):
                protect_nogds(self)

    return CheckedLoader


def _transfer(fd, gbuf, offset, length, dest_offset, device):
    import torch
    from fastsafetensors import cpp
    from fastsafetensors.dlpack import from_cuda_buffer
    from fastsafetensors.st_types import DType

    from rtp_llm.models_py.fst_range_reader import native_host_reader

    if not (0 <= dest_offset < dest_offset + length <= gbuf.get_length()):
        raise OSError("Invalid noGDS retry extent")
    # Like the FST CUDA file copier, the DLPack tensor borrows the allocation.
    # Keep gbuf alive through the synchronous last transfer; the native FST
    # file factory still owns and frees it after its consumer closes.
    with torch.cuda.device(device.as_str()):
        ptr = gbuf.get_base_address() + dest_offset
        dst = torch.from_dlpack(from_cuda_buffer(ptr, [length], [1], DType.U8, device))
        buf = torch.empty(
            min(16 << 20, length), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        hptr = buf.data_ptr()
        host = cpp.gds_device_buffer(hptr, buf.numel(), False)
        reader = native_host_reader(cpp)
        for start in range(0, length, buf.numel()):
            size = min(buf.numel(), length - start)
            token = reader.submit_read(fd, host, offset + start, size, 0)
            if token < 0 or reader.wait_read(token) != hptr:
                raise OSError(
                    "Native noGDS host read failed while repairing a CUDA extent"
                )
            # Synchronize before the native reader may overwrite the pinned buffer.
            dst[start : start + size].copy_(buf[:size], non_blocking=False)
    return ptr


class _CheckedReader:
    def __init__(self, reader, device, transfer):
        self.reader = reader
        self.device = device
        self.transfer = transfer
        self.pending = {}

    def submit_read(self, fd, gbuf, offset, length, dest_offset):
        token = self.reader.submit_read(fd, gbuf, offset, length, dest_offset)
        if token >= 0:
            self.pending[token] = (fd, gbuf, offset, length, dest_offset)
        return token

    def wait_read(self, token):
        extent = self.pending.pop(token)
        result = self.reader.wait_read(token)
        expected = extent[1].get_base_address() + extent[4]
        if result == expected and result:
            return result
        if result != 0:
            raise OSError("Unexpected native noGDS completion address")
        logging.getLogger(__name__).warning(
            "FST native H2D failed at file offset %d, length %d; retrying this extent on its owning device",
            extent[2],
            extent[3],
        )
        if not expected or self.transfer(*extent, self.device) != expected:
            raise OSError("Unable to complete native noGDS transfer")
        return expected

    def drain(self):
        for token in tuple(self.pending):
            self.reader.wait_read(token)
            del self.pending[token]


class _CheckedCopier:
    def __init__(self, copier, device, framework):
        self.copier, self.device, self.framework = copier, device, framework
        copier.reader = _CheckedReader(copier.reader, device, _transfer)

    def __getattr__(self, name):
        return getattr(self.copier, name)

    def submit_io(self, *args, **kwargs):
        return self.copier.submit_io(*args, **kwargs)

    def wait_io(self, gbuf, *args, **kwargs):
        try:
            return self.copier.wait_io(gbuf, *args, **kwargs)
        except Exception:
            # The file buffer is not published. Join readers which may still
            # use this fd and allocation before releasing the failed extent.
            self.copier.reader.drain()
            if self.copier.fd > 0:
                os.close(self.copier.fd)
                self.copier.fd = 0
            self.framework.free_tensor_memory(gbuf, self.device)
            raise


def protect_nogds(loader):
    # Install per parallel-loader instance, never on the process-global copier
    # registry. FST still assigns files to the same ranks and runs the same
    # ReadSpecs/collectives. Other devices/readers are untouched.
    def wrap(make):
        def create(meta, device, framework, *args, **kwargs):
            copier = make(meta, device, framework, *args, **kwargs)
            return _CheckedCopier(copier, device, framework)

        return create

    producers = getattr(loader, "loaders", None)
    if producers is None:
        producers = (loader.loader,)
    for producer in producers:
        producer.copier_constructor = wrap(producer.copier_constructor)
