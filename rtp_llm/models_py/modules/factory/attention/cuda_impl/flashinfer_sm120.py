"""Narrow FlashInfer FMHAv2 specialization for SM120 encoder attention."""

import functools
from dataclasses import replace

import torch


@functools.cache
def _get_sm120_bert_module():
    """Build the V2-equivalent 64x32, non-granular FMHA specialization."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.attention import modules
    from flashinfer.jit.attention.fmha_v2 import fmha_library
    from flashinfer.jit.attention.fmha_v2.utils import InputLayout, encode_name
    from flashinfer.jit.core import gen_jit_spec
    from flashinfer.jit.utils import write_if_different

    uri = "rtp_llm_sm120_bert_fmha_v2_bf16_64x32_v2warps"
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)

    spec = fmha_library.generate_kernel_spec(
        sm=120,
        head_size=64,
        dtype="bf16",
        return_softmax=False,
        enable_attn_logit_softcapping=False,
        alibi=False,
        input_layout=InputLayout.PACKED_QKV,
        output_dtype="bf16",
    )
    # SM120 defaults to the generic granular-tiled kernel. Match the legacy V2
    # runner's 64x32 no-loop specialization, including its 4x1 warp layout.
    spec = replace(
        spec,
        tiled=0,
        kv_loop_step=32,
        ldgsts_q=True,
        ldgsts_k=True,
        ldgsts_v=True,
        alibi=False,
        return_softmax_stats=False,
    )
    if not fmha_library.is_kernel_spec_valid(spec):
        raise RuntimeError("invalid SM120 Vision-BERT FMHAv2 kernel spec")

    fname, lname, kname = encode_name(spec)
    kernel_path = gen_directory / fname
    kernel_code = fmha_library.get_kernel_code(spec, kname, lname)
    # FlashInfer's Ampere-style no-loop template collapses all four warps into
    # WARPS_N. V2 used four vertical warps for this tile; changing only the tile
    # size produces incorrect results, so keep both choices coupled here.
    warp_layout = "    1,\n    4 * 1,"
    if kernel_code.count(warp_layout) < 2:
        raise RuntimeError("unsupported FlashInfer FMHAv2 no-loop warp layout")
    kernel_code = kernel_code.replace(warp_layout, "    4,\n    1,")
    write_if_different(kernel_path, kernel_code)
    api_path = gen_directory / "fmha_v2_api.h"
    write_if_different(
        api_path, fmha_library.get_api_code([(spec, fname, lname, kname)])
    )

    csrc_dir = jit_env.FLASHINFER_CSRC_DIR
    run_source = (csrc_dir / "fmha_v2_run.cu").read_text()
    old_dispatch = "false, false, false, false, false, false, false, force_fp32_acc,"
    if run_source.count(old_dispatch) != 1:
        raise RuntimeError("unsupported FlashInfer fmha_v2_run.cu dispatch layout")
    run_source = run_source.replace(
        old_dispatch,
        "false, false, false, false, false, false, true, force_fp32_acc,",
    )
    run_path = gen_directory / "fmha_v2_run.cu"
    write_if_different(run_path, run_source)

    binding_path = gen_directory / "fmha_v2_jit_binding.cu"
    write_if_different(binding_path, (csrc_dir / "fmha_v2_jit_binding.cu").read_text())

    nvcc_flags = modules.current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    nvcc_flags.extend(
        [
            f"-I{csrc_dir / 'fmha_v2'}",
            f"-I{gen_directory}",
            f"-I{jit_env.FLASHINFER_INCLUDE_DIR}",
            "-Wno-deprecated-gpu-targets",
        ]
    )
    return gen_jit_spec(
        uri,
        [kernel_path, run_path, binding_path],
        extra_cuda_cflags=nvcc_flags,
    ).build_and_load()


def sm120_bert_fmha_v2_prefill(
    qkv: torch.Tensor,
    workspace_buffer: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: float,
    batch_size: int,
    cum_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Run packed BF16 padding-mask attention without ALiBi or softmax save."""
    out = torch.empty(
        (qkv.shape[0], qkv.shape[2], qkv.shape[3]),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    _get_sm120_bert_module().run(
        qkv,
        qkv,
        qkv,
        out,
        workspace_buffer,
        workspace_buffer.numel() * workspace_buffer.element_size(),
        None,
        0,
        seq_lens,
        cum_seq_lens,
        cum_seq_lens,
        "packed_qkv",
        max_q_len,
        max_kv_len,
        batch_size,
        "padding",
        1.0,
        bmm1_scale,
        1.0,
        -1,
        0,
        False,
        0.0,
        0.0,
        None,
        None,
    )
    return out
