import functools
import os

import torch

from ....mhc.norm_fn_kernel import _mhc_pre_norm_fn_fwd_mul, round_to_tf32
from ....mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@functools.cache
def _compute_num_split(block_k: int, k: int, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // max(grid_size, 1)
    num_block_k = _ceil_div(k, block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _requested_backend() -> str:
    requested = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    if requested in ("", "auto"):
        # Experiment branch: enable DeepGEMM by default to validate DSV4
        # greedy/golden semantics under the full SM100 smoke suite. This is
        # intentionally hard: DeepGEMM/JIT failures must surface directly.
        return "deepgemm"
    aliases = {
        "dg": "deepgemm",
        "tilelang": "tilelang_single",
        "single": "tilelang_single",
    }
    return aliases.get(requested, requested)


def _run_tilelang_single_gemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_mult3: int,
    mhc_hidden_size: int,
) -> int:
    fn = round_to_tf32(fn)
    fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size)
    fwd_mul_kernel(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul[:1].view(-1, 1, mhc_mult3),
        gemm_out_sqrsum[:1].view(-1, 1),
    )
    return 1


def _run_deepgemm_splitk_gemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    n_splits: int,
) -> None:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(
        residual_flat,
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    *,
    backend: str | None = None,
    prenorm_gemm=None,
    prenorm_output: str = "partials",
    stabilize_atomic: bool = False,
    prenorm_partials=None,
    norm_weight=None,
    norm_eps=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if norm_weight is not None:
        if (
            norm_eps is None
            or norm_eps <= 0
            or norm_weight.shape != (hidden_size,)
            or norm_weight.dtype != torch.bfloat16
            or norm_weight.device != residual.device
            or not norm_weight.is_contiguous()
        ):
            raise ValueError(
                "HC fused norm requires dense BF16 weights and positive epsilon"
            )
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    backend = _requested_backend() if backend is None else backend
    if prenorm_output not in ("partials", "reduced"):
        raise ValueError("HC prenorm_output must be partials or reduced")
    if prenorm_output == "reduced" and prenorm_gemm is None:
        raise ValueError("Reduced HC output requires an explicit prenorm adapter")
    deepgemm_backend = backend in ("deepgemm", "deepgemm_deterministic")
    block_k = 64
    block_m = 64
    n_splits = (
        _compute_num_split(block_k, mhc_hidden_size, _ceil_div(num_tokens, block_m))
        if deepgemm_backend or backend == "tilelang_splitk"
        else 1
    )

    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    if prenorm_partials is not None:
        if backend != "deepgemm_deterministic":
            raise ValueError("Exposed HC partials require deterministic prenorm")
        gemm_out_mul, gemm_out_sqrsum = prenorm_partials(
            residual_flat.view(num_tokens, mhc_hidden_size), fn_flat
        )
        n_splits = gemm_out_mul.shape[0]
        assert n_splits > 0
        assert gemm_out_mul.shape == (n_splits, num_tokens, mhc_mult3)
        assert gemm_out_sqrsum.shape == (n_splits, num_tokens)
        assert all(
            tensor.dtype == torch.float32
            and tensor.device == residual.device
            and tensor.is_contiguous()
            for tensor in (gemm_out_mul, gemm_out_sqrsum)
        )
    else:
        # CUDA exposes one plane per split. A selected adapter may instead
        # reduce internally; its declared output contract also drives the
        # consumer's reduction, independently of the producer's split count.
        output_splits = (
            1
            if prenorm_output == "reduced" or backend == "tilelang_single"
            else n_splits
        )
        gemm_out_mul = torch.empty(
            output_splits,
            num_tokens,
            mhc_mult3,
            dtype=torch.float32,
            device=residual.device,
        )
        gemm_out_sqrsum = torch.empty(
            output_splits, num_tokens, dtype=torch.float32, device=residual.device
        )
        if deepgemm_backend:
            if prenorm_gemm is not None:
                run_gemm = prenorm_gemm
            elif backend == "deepgemm_deterministic":
                raise ValueError(
                    "Deterministic HC requires an explicit prenorm adapter"
                )
            else:
                run_gemm = _run_deepgemm_splitk_gemm
            run_gemm(
                residual_flat.view(num_tokens, mhc_hidden_size),
                fn_flat,
                gemm_out_mul,
                gemm_out_sqrsum,
                n_splits,
            )
            n_splits = output_splits
        elif backend == "tilelang_single":
            n_splits = _run_tilelang_single_gemm(
                residual_flat,
                fn_flat,
                gemm_out_mul,
                gemm_out_sqrsum,
                mhc_mult3,
                mhc_hidden_size,
            )
            gemm_out_mul = gemm_out_mul[:1]
            gemm_out_sqrsum = gemm_out_sqrsum[:1]
        elif backend == "tilelang_splitk":
            raise RuntimeError(
                "DSV4_MHC_PRE_GEMM_BACKEND=tilelang_splitk is not wired in this "
                "RTP TileKernels snapshot; use deepgemm or tilelang_single."
            )
        else:
            raise ValueError(
                "Unsupported DSV4_MHC_PRE_GEMM_BACKEND="
                f"{backend!r}; expected deepgemm, deepgemm_deterministic, "
                "tilelang_splitk, or tilelang_single."
            )

    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
        # The deterministic reduction needs no BF16 canonicalization. Keep
        # these small vectors in FP32, matching SGLang's HC/Sinkhorn semantics.
        # Retain the legacy atomic backend's behavior for explicit A/B runs.
        stabilize_mixes=stabilize_atomic,
        stabilize_comb=stabilize_atomic,
        parallel_reduction=prenorm_partials is not None,
        fuse_norm=norm_weight is not None,
        norm_eps=norm_eps if norm_eps is not None else 1e-6,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight if norm_weight is not None else layer_input.view(-1)[:0],
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input
