"""Forward-owned CP8 permission for the existing immutable fake-route pattern.

Only synthetic balanced Prefill uses this adapter. The native mapping, the
existing mask/pack kernel's weights, real routing and Decode stay unchanged.
The symmetric buffer is still prepared by the original CP8 collective path.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class CPForwardMegaBuffer:
    engine: object
    buffer: object
    capacity: int
    bound: int
    generation: int
    epoch: int
    device: torch.device
    stream: tuple
    binding: tuple
    group: object


def _device_supported(device):
    from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

    return (device.type == "cuda" and torch.cuda.is_available() and is_ppu()
            and torch.cuda.get_device_capability(device) == (8, 9)
            and not torch.cuda.is_current_stream_capturing())


def _stream_key(device):
    stream = torch.cuda.current_stream(device)
    return stream.device, int(stream.cuda_stream)


def _binding(engine):
    return (engine.backend, engine.precision, engine.ep_size, engine.ep_rank,
            engine.num_routed_experts, engine.hidden_size, engine.inter_dim,
            getattr(engine, "top_k", 6), engine._fake_balance_config,
            getattr(engine, "_fake_balance_policy", "rtp"))


def prepare(engine, buffer, device, rows):
    """Wrap an already prepared uniform CP buffer; never allocate or agree."""
    engine._cp_fake_active = None
    if (buffer is None or torch.is_grad_enabled()
            or getattr(engine, "_fake_balance_config", None) != (True, 6, 0, 1)
            or getattr(engine, "_fake_balance_policy", "rtp") != "rtp"
            or (engine.backend, engine.precision, engine.ep_size, engine.num_routed_experts,
                engine.hidden_size, engine.inter_dim) != ("megamoe", "w4a4", 8, 384, 5120, 2304)
            or getattr(engine, "top_k", 6) != 6 or type(rows) is not int
            or not 0 < rows <= engine._mega_capacity <= (2**31 - 1) // 6
            or buffer is not engine._mega_buffer or engine._mega_device != device
            or not _device_supported(device)):
        return buffer
    epoch = getattr(engine, "_cp_fake_epoch", 0) + 1
    receipt = CPForwardMegaBuffer(engine, buffer, engine._mega_capacity, rows,
                                 engine._mega_generation, epoch, device, _stream_key(device),
                                 _binding(engine), engine.group)
    engine._cp_fake_epoch, engine._cp_fake_active = epoch, receipt
    engine._cp_fake_evidence = dict(epoch=epoch, bound=rows, capacity=receipt.capacity,
                                    generation=receipt.generation, calls=0, pattern_builds=0, slots=None)
    return receipt


def validate(engine, receipt, rows, device):
    if (type(receipt) is not CPForwardMegaBuffer or receipt.engine is not engine
            or getattr(engine, "_cp_fake_active", None) is not receipt
            or receipt.epoch != getattr(engine, "_cp_fake_epoch", None)
            or receipt.buffer is not engine._mega_buffer or receipt.capacity != engine._mega_capacity
            or receipt.generation != engine._mega_generation or receipt.device != device
            or engine._mega_device != device or type(rows) is not int
            or not 0 < rows <= receipt.bound <= receipt.capacity
            or getattr(engine, "_fake_balance_config", None) != (True, 6, 0, 1)
            or receipt.binding != _binding(engine) or receipt.group is not engine.group
            or receipt.stream != _stream_key(device)):
        raise RuntimeError("stale CP Prefill fake-route buffer preparation")


def try_balance(engine, ids, weights, receipt):
    from rtp_llm.models_py.modules.dsv41.ppu import moe

    if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
        return None
    validate(engine, receipt, ids.shape[0], ids.device)
    if (torch.is_grad_enabled() or ids.ndim != 2 or ids.shape[1] not in (6, 8)
            or ids.shape != weights.shape or ids.dtype != torch.int64 or weights.dtype != torch.float32
            or weights.device != ids.device or not ids.is_contiguous() or not weights.is_contiguous()
            or ids.is_neg() or ids.is_conj() or weights.is_neg() or weights.is_conj()
            or not _device_supported(ids.device)):
        return None
    try:
        kernels = moe._get_triton_kernels()
    except ImportError:
        return None
    operation = getattr(kernels, "mask_fake_routes", None)
    if operation is None:
        return None
    configuration = engine.num_routed_experts, engine.ep_size, engine._fake_balance_config
    pattern = engine._prefill_fake_pattern
    if pattern is None:
        if engine._prefill_fake_pattern_generation is not None:
            return None
        pattern = engine._build_prefill_fake_pattern(receipt.capacity, ids.device)
        engine._cp_fake_evidence["pattern_builds"] += 1
    elif (pattern.configuration != configuration or pattern.capacity != receipt.capacity
          or pattern.generation != receipt.generation
          or engine._prefill_fake_pattern_generation != receipt.generation
          or pattern.ids.device != ids.device
          or tuple(moe._fake_pattern_tensor_key(value) for value in (pattern.ids, pattern.weights)) != pattern.keys):
        return None
    output_ids, output_weights = torch.empty_like(ids), torch.empty_like(weights)
    operation[(kernels.triton.cdiv(ids.numel(), 256),)](
        ids, pattern.ids, output_ids, output_weights, ids.shape[0], BLOCK=256, SLOTS=ids.shape[1],
    )
    engine._cp_fake_evidence["calls"] += 1
    engine._cp_fake_evidence["slots"] = ids.shape[1]
    return output_ids, output_weights
