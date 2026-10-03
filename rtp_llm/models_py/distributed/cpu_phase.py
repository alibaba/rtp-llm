"""CPU control group for fixed-batch mixed Prefill/Decode scheduling.

The model communicators remain NCCL. A world group covers every participating
EP group, including TP peers; only TP0 contributes local scheduler state.
"""

from datetime import timedelta

import torch.distributed as dist

from rtp_llm.ops import RoleType, SpeculativeType

_group = None


def needs_cpu_phase_group(engine_config):
    return (
        engine_config.runtime_config.use_batch_decode_scheduler
        and engine_config.pd_sep_config.role_type == RoleType.PDFUSION
        and engine_config.parallelism_config.dp_size > 1
        and engine_config.sp_config.type
        in (
            SpeculativeType.MTP,
            SpeculativeType.EAGLE,
            SpeculativeType.EAGLE3,
            SpeculativeType.DSPARK,
        )
    )


def init_cpu_phase_group(engine_config, timeout_seconds=300):
    global _group
    if not needs_cpu_phase_group(engine_config):
        return
    if _group is not None:
        raise RuntimeError("CPU phase group already initialized")
    if not dist.is_initialized():
        raise RuntimeError("model distributed environment must be initialized first")
    if dist.get_world_size() != engine_config.parallelism_config.world_size:
        raise ValueError("CPU phase group world size differs from engine topology")
    # Programmatic PyEnvConfigs leaves this unset; CLI supplies300. Both paths
    # must get the same finite default rather than rejecting a valid config.
    timeout_seconds = 300 if timeout_seconds is None else timeout_seconds
    if timeout_seconds <= 0:
        raise ValueError("CPU phase agreement requires a finite positive timeout")
    import librtp_compute_ops

    # All world ranks call new_group in identical startup order. This is never
    # initialized for production PD-separated workers or the normal scheduler.
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))
    try:
        librtp_compute_ops.register_cpu_phase_group(group)
    except Exception:
        dist.destroy_process_group(group)
        raise
    _group = group


def destroy_cpu_phase_group():
    """Call only after the engine has stopped, not while agreement is in flight.

    Gloo collective cancellation is not assumed. A lost peer is bounded by the
    configured group timeout, including shutdown while a collective is pending.
    """
    global _group
    if _group is None:
        return
    import librtp_compute_ops

    librtp_compute_ops.clear_cpu_phase_group()
    dist.destroy_process_group(_group)
    _group = None
