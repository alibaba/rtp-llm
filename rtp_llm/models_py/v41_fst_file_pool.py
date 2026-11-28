"""Select CUDA file transfer guards and isolate V4.1's FST staging allocator."""

import inspect

import torch


def cuda_file_loader(base, model_type, device, *, cuda_backend, use_shm, nogds):
    staged = v41_file_staging_loader(
        base, model_type, device, cuda_backend=cuda_backend, use_shm=use_shm
    )
    if device.type == "cuda" and cuda_backend == "Cuda" and nogds and not use_shm:
        from rtp_llm.models_py.v4_fst_nogds import checked_nogds_loader

        # The completion ABI belongs to the native reader, not the model.
        return checked_nogds_loader(staged)
    return staged


def v41_file_staging_loader(base, model_type, device, *, cuda_backend, use_shm):
    if (
        model_type not in ("deepseek_v41", "deepseek_v41_dspark")
        or device.type != "cuda"
        or cuda_backend != "Cuda"
        or use_shm
        or not callable(getattr(base, "_load_single_batch", None))
        or not callable(getattr(base, "_consumer_worker", None))
        or not callable(getattr(torch.cuda, "MemPool", None))
        or not callable(getattr(torch.cuda, "use_mem_pool", None))
    ):
        return base
    get_backend = getattr(torch.cuda, "get_allocator_backend", None)
    if not callable(get_backend):
        return base
    try:
        if (
            get_backend() != "native"
            or "queue_size" not in inspect.signature(base.__init__).parameters
        ):
            return base
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # This is a staging optimization, not an extra FST prerequisite.
        return base

    class StagedFileLoader(base):
        def __init__(self, *args, **kwargs):
            # In FST, zero releases the producer at dequeue; negative releases
            # it *after* fb.close(), for both plain and ReadSpec consumers.
            super().__init__(*args, queue_size=-1, **kwargs)
            self._rtp_file_pool = None

        def _load_single_batch(self, *args, **kwargs):
            # The FIFO producer waits here before allocating the next pool.
            # Ordered producers already acquired the sole q=-1 inflight slot.
            processed = getattr(self, "consumer_processed", None)
            if processed is not None:
                processed.wait()
                if self.stop_event.is_set() or self.error_event.is_set():
                    return
            with torch.cuda.device(device):
                # An entire checkpoint's worth of differently-sized raw files
                # must not share a cache: a larger one cannot reuse a smaller
                # free segment. Only the current rank's one batch stays resident.
                self._rtp_file_pool = torch.cuda.MemPool()
                with torch.cuda.use_mem_pool(self._rtp_file_pool, device):
                    return super()._load_single_batch(*args, **kwargs)

        def _consumer_worker(self):
            try:
                yield from super()._consumer_worker()
            finally:
                # A caller error or GeneratorExit can bypass the wheel's normal
                # post-close notification. Stop before waking the producer so
                # it cannot load another shard while the consumer unwinds.
                self.stop_event.set()
                processed = getattr(self, "consumer_processed", None)
                if processed is not None:
                    processed.set()

    return StagedFileLoader
