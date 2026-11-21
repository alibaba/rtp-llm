"""Bounded, current-device-only snapshots for the V4.1 CUDA FST startup budget."""

import logging

import torch

from rtp_llm.device.device_type import is_cuda

logger = logging.getLogger(__name__)


class V41FstMemoryTrace:
    @classmethod
    def for_load(cls, model_type, requested_method, config):
        if (
            model_type not in ("deepseek_v41", "deepseek_v41_dspark")
            or requested_method != "fastsafetensors"
            or config.force_cpu_load_weights
            or not logger.isEnabledFor(logging.INFO)
        ):
            return None
        target = torch.device(config.device)
        if target.type != "cuda":
            return None
        return cls(
            model_type, target, getattr(config.parallelism_config, "world_rank", None)
        )

    def __init__(self, model_type, target, rank):
        self.model_type = model_type
        self.target = target
        self.rank = rank

    def record(self, stage):
        # Never create a CUDA context, select another device, synchronize
        # streams, clear an allocator, or turn a diagnostic failure into a
        # model load failure. Driver free is device-wide; allocator counts are
        # from this process. The two reads are observations, not an atomic sum.
        idx = None
        try:
            if not torch.cuda.is_initialized():
                reason = "CUDA context not initialized"
            elif not is_cuda():
                return
            else:
                idx = torch.cuda.current_device()
                if self.target.index is None or self.target.index == idx:
                    free, total = torch.cuda.mem_get_info(idx)
                    backend, stats = "unavailable", {}
                    try:
                        backend = torch.cuda.get_allocator_backend()
                        if backend == "native":
                            stats = torch.cuda.memory_stats(idx)
                    except Exception:
                        # Missing allocator stats must not discard the driver
                        # observation (nor be reported as zero live bytes).
                        pass
                    logger.info(
                        "V4.1 FST HBM model=%s rank=%s device=cuda:%d stage=%s "
                        "driver_free_bytes=%d driver_total_bytes=%d torch_allocator=%s "
                        "torch_allocated_bytes=%s torch_reserved_bytes=%s",
                        self.model_type,
                        self.rank,
                        idx,
                        stage,
                        free,
                        total,
                        backend,
                        stats.get("allocated_bytes.all.current", "unavailable"),
                        stats.get("reserved_bytes.all.current", "unavailable"),
                    )
                    return
                reason = "target is not the current CUDA device"
        except Exception as exc:
            # For example an unsupported allocator ABI. Do not log arbitrary
            # exception payloads or substitute CPU RSS / global free for Torch.
            reason = type(exc).__name__
        logger.info(
            "V4.1 FST HBM model=%s rank=%s target=%s current_device=%s stage=%s unavailable=%s",
            self.model_type,
            self.rank,
            self.target,
            idx,
            stage,
            reason,
        )
