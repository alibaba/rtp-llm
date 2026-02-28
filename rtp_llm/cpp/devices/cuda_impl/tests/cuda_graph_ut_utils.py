"""Shared utilities for CUDA graph unit tests."""

from typing import Optional

import torch

from rtp_llm.ops.compute_ops import PyModelInputs, get_scalar_type


def _print_py_model_inputs_full(
    m: PyModelInputs, label: str = "py_model_inputs"
) -> None:
    """Print full contents of PyModelInputs (mirrors CudaGraphRunner.cc prepareInputs debug block)."""

    def sizes(t: Optional[torch.Tensor]) -> str:
        if t is None or (
            hasattr(t, "numel") and t.numel() == 0 and not hasattr(t, "shape")
        ):
            return "undef"
        if hasattr(t, "shape"):
            return ",".join(str(x) for x in t.shape)
        return "undef"

    def print_tensor_int(
        name: str, t: Optional[torch.Tensor], max_vals: int = 64
    ) -> None:
        if t is None:
            print(f"  attention_inputs.{name}: defined=False sizes=undef")
            return
        defined = True
        sizes_str = sizes(t)
        print(
            f"  attention_inputs.{name}: defined={defined} sizes=[{sizes_str}]", end=""
        )
        if hasattr(t, "numel") and t.numel() > 0:
            cpu = t.cpu() if t.is_cuda else t
            if cpu.dtype in (torch.int32, torch.int64, torch.long):
                n = min(cpu.numel(), max_vals)
                vals = cpu.flatten()[:n].tolist()
                print(f" values({n})= {vals}", end="")
                if cpu.numel() > max_vals:
                    print(" ... (truncated)", end="")
        print()

    print(f"[{label}] full dump:")
    print(
        f"  input_ids: defined={m.input_ids is not None} sizes=[{sizes(m.input_ids)}]"
    )
    if m.input_ids is not None and m.input_ids.numel() > 0:
        ids_cpu = m.input_ids.cpu()
        n = min(ids_cpu.numel(), 2048)
        vals = ids_cpu.flatten()[:n].tolist()
        print(
            f"    values({n})= {vals}"
            + (" ... (truncated)" if ids_cpu.numel() > 2048 else "")
        )
    print(
        f"  input_hiddens: defined={m.input_hiddens is not None} sizes=[{sizes(m.input_hiddens)}]"
    )
    a = m.attention_inputs
    print(
        f"  attention_inputs (scalars): is_prefill={a.is_prefill} is_s_padded={a.is_s_padded} "
        f"is_cuda_graph={getattr(a, 'is_cuda_graph', 'N/A')} "
        f"context_total_kv_length={getattr(a, 'context_total_kv_length', 0)} "
        f"total_tokens={getattr(a, 'total_tokens', 0)}"
    )
    print_tensor_int("input_lengths", getattr(a, "input_lengths", None), 32)
    print_tensor_int("sequence_lengths", getattr(a, "sequence_lengths", None), 32)
    print_tensor_int("prefix_lengths", getattr(a, "prefix_lengths", None), 32)
    print_tensor_int("cu_seqlens", getattr(a, "cu_seqlens", None), 32)
    print_tensor_int("cu_kv_seqlens", getattr(a, "cu_kv_seqlens", None), 32)
    print_tensor_int(
        "decode_cu_seqlens_host", getattr(a, "decode_cu_seqlens_host", None), 32
    )
    print_tensor_int("padding_offset", getattr(a, "padding_offset", None), 256)
    print(
        f"  attention_inputs.kv_cache_block_id_host: defined={a.kv_cache_block_id_host is not None} sizes=[{sizes(a.kv_cache_block_id_host)}]"
    )
    print(
        f"  attention_inputs.kv_cache_block_id_device: defined={a.kv_cache_block_id_device is not None} sizes=[{sizes(a.kv_cache_block_id_device)}]"
    )
    print_tensor_int("prefix_lengths_d", getattr(a, "prefix_lengths_d", None), 32)
    print_tensor_int(
        "sequence_lengths_plus_1_d", getattr(a, "sequence_lengths_plus_1_d", None), 32
    )
    print_tensor_int("input_lengths_d", getattr(a, "input_lengths_d", None), 32)
    print_tensor_int("decode_cu_seqlens_d", getattr(a, "decode_cu_seqlens_d", None), 32)
    dtype_obj = getattr(a, "dtype", None)
    if dtype_obj is not None:
        try:
            dtype_str = str(get_scalar_type(dtype_obj))
        except Exception:
            name_attr = getattr(dtype_obj, "name", None)
            dtype_str = name_attr() if callable(name_attr) else str(dtype_obj)
    else:
        dtype_str = "None"
    print(f"  attention_inputs.dtype: {dtype_str}")
    cs = getattr(a, "cache_store_inputs", None)
    pf = getattr(a, "prefill_cuda_graph_copy_params", None)
    print(
        f"  attention_inputs.cache_store_inputs: {'set' if cs is not None else 'nullopt'}"
    )
    print(
        f"  attention_inputs.prefill_cuda_graph_copy_params: {'set' if pf is not None else 'nullopt'}"
    )
