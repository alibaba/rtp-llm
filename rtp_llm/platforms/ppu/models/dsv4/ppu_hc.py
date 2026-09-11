"""PPU HC uses the bound prenorm kernel and an explicit PDL policy."""

import importlib

import torch

from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit


class PpuHCUnit(TileLangHCUnit):
    def __init__(
        self,
        *args,
        options,
        tp_size=1,
        tp_rank=0,
        allow_graph=False,
        fuse_prenorm=False,
        fuse_norm=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self._allow_graph = allow_graph
        self._fuse_norm = fuse_norm
        self._prenorm_partials = None
        self._backend = options.get(
            "DSV4_MHC_PRE_GEMM_BACKEND", "deepgemm_deterministic"
        )
        if self._backend not in ("deepgemm", "deepgemm_deterministic"):
            raise ValueError(
                "PPU module HC requires an explicitly supported prenorm backend"
            )
        if allow_graph and (tp_size != 1 or self._backend != "deepgemm_deterministic"):
            raise ValueError("PPU HC Graph requires TP1 and deterministic prenorm")
        if fuse_norm and not allow_graph:
            raise ValueError("Fused HC norm requires the PPU Decode instance")
        if fuse_prenorm:
            if (
                not allow_graph
                or tp_size != 1
                or self._backend != "deepgemm_deterministic"
            ):
                raise ValueError("Fused HC reduction requires TP1 deterministic Decode")
            from .ppu_hc_prenorm import tf32_hc_prenorm_partials

            self._prenorm_partials = tf32_hc_prenorm_partials
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

    def pre_norm(self, x, norm, *, tp_size, tp_rank, dbg_tag=None):
        if not self._fuse_norm:
            return super().pre_norm(
                x, norm, tp_size=tp_size, tp_rank=tp_rank, dbg_tag=dbg_tag
            )
        from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

        leading = self._check_residual_shape(x, "pre_norm input")
        weight = norm.weight.data
        if (
            tp_size != 1
            or tp_rank != 0
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or weight.shape != (self.dim,)
            or weight.dtype != torch.bfloat16
            or weight.device != x.device
            or not weight.is_contiguous()
        ):
            raise ValueError("Fused HC norm requires contiguous BF16 TP1 inputs")
        if x.numel() == 0:
            return (
                x.new_empty((*leading, self.dim)),
                x.new_empty((*leading, self.hc_mult, 1), dtype=torch.float32),
                x.new_empty(
                    (*leading, self.hc_mult, self.hc_mult), dtype=torch.float32
                ),
            )
        if dbg_tag is not None:
            from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

            for name, tensor in (
                ("input", x),
                ("fn", self.fn),
                ("scale", self.scale),
                ("base", self.base),
                ("norm_weight", weight),
            ):
                _rt.record_if_level(2, f"{dbg_tag}_fused_norm_{name}", tensor)
        with torch.inference_mode(), record_function_range(
            f"dsv4.hc.L{self.layer_id:02d}.{self.name}.pre_norm"
        ):
            post, comb, y = self._pre_kernel(
                x,
                self.fn,
                self.scale,
                self.base,
                rms_eps=self.norm_eps,
                mhc_pre_eps=self.hc_eps,
                mhc_sinkhorn_eps=self.hc_eps,
                mhc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                backend=self._backend,
                prenorm_gemm=self._prenorm_gemm,
                prenorm_output="reduced",
                stabilize_atomic=self._backend == "deepgemm",
                prenorm_partials=self._prenorm_partials,
                norm_weight=weight,
                norm_eps=norm.variance_epsilon,
            )
        self._check_pre_output(leading, y, post, comb)
        if dbg_tag is not None:
            for name, tensor in (("y", y), ("post", post), ("comb", comb)):
                _rt.record_if_level(2, f"{dbg_tag}_fused_norm_{name}", tensor)
        return y, post, comb

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
            prenorm_output="reduced",
            stabilize_atomic=self._backend == "deepgemm",
            prenorm_partials=self._prenorm_partials,
        )
        return layer_input, post, comb

    def _post_operator(self, x, residual, post, comb, *, hc_mult, out=None):
        if residual.numel() == 0:
            return residual if out is None else out
        return self._post_kernel(x, residual, post, comb, out=out, enable_pdl=False)
