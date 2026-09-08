"""PPU HC uses the bound prenorm kernel and an explicit PDL policy."""

import importlib

import torch
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit


class PpuHCUnit(TileLangHCUnit):
    def __init__(
        self, *args, options, tp_size=1, tp_rank=0, allow_graph=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self._allow_graph = allow_graph
        self._backend = options.get(
            "DSV4_MHC_PRE_GEMM_BACKEND", "deepgemm_deterministic"
        )
        if self._backend not in ("deepgemm", "deepgemm_deterministic"):
            raise ValueError(
                "PPU module HC requires an explicitly supported prenorm backend"
            )
        if allow_graph and (tp_size != 1 or self._backend != "deepgemm_deterministic"):
            raise ValueError("PPU HC Graph requires TP1 and deterministic prenorm")
        if (
            options.get("DSV4_MHC_POST_BACKEND", "tilelang") != "tilelang"
            or options.get("DSV4_MHC_POST_PDL", "0") != "0"
        ):
            raise ValueError("PPU module HC requires TileLang POST with PDL disabled")
        # Prepare the vendored runtime before importing its JIT kernels.
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

        prefix = "rtp_llm.models_py.3rdparty.tile_kernels"
        self._pre_kernel = importlib.import_module(
            prefix + ".modeling.mhc.ops.pre_big_fuse"
        ).mhc_pre_big_fuse
        self._post_kernel = importlib.import_module(
            prefix + ".mhc.post_kernel"
        ).mhc_post_fwd
        if self._backend == "deepgemm_deterministic":
            from .ppu_hc_prenorm import tf32_hc_prenorm_gemm

            self._prenorm_gemm = tf32_hc_prenorm_gemm
        else:
            module = importlib.import_module(prefix + ".modeling.mhc.ops.pre_big_fuse")
            self._prenorm_gemm = module._run_deepgemm_splitk_gemm

    def _pre_operator(
        self,
        residual,
        fn,
        scale,
        base,
        *,
        norm_eps,
        pre_eps,
        sinkhorn_eps,
        sinkhorn_iters,
        hc_mult,
    ):
        if torch.cuda.is_current_stream_capturing() and not self._allow_graph:
            raise RuntimeError("This PPU HC instance does not allow Graph capture")
        if residual.numel() == 0:
            leading = residual.shape[:-2]
            return (
                residual.new_empty((*leading, self.dim)),
                residual.new_empty((*leading, hc_mult, 1), dtype=torch.float32),
                residual.new_empty((*leading, hc_mult, hc_mult), dtype=torch.float32),
            )
        post, comb, layer_input = self._pre_kernel(
            residual,
            fn,
            scale,
            base,
            rms_eps=norm_eps,
            mhc_pre_eps=pre_eps,
            mhc_sinkhorn_eps=sinkhorn_eps,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_iters,
            backend=self._backend,
            prenorm_gemm=self._prenorm_gemm,
        )
        return layer_input, post, comb

    def _post_operator(self, x, residual, post, comb, *, hc_mult, out=None):
        if residual.numel() == 0:
            return residual if out is None else out
        return self._post_kernel(x, residual, post, comb, out=out, enable_pdl=False)
