"""PPU HC uses the bound prenorm kernel and an explicit PDL policy."""

import importlib
from functools import partial

import torch

from rtp_llm.models_py.modules.dsv4.ppu.hc.tilelang_impl import TileLangHCUnit


class PpuHCUnit(TileLangHCUnit):
    def __init__(
        self,
        *args,
        options,
        tp_size=1,
        tp_rank=0,
        allow_graph=False,
        row_gemv_stream_pool=None,
        fuse_prenorm=False,
        fuse_norm=False,
        fuse_moe_sum=False,
        fuse_pre_quant=False,
        fuse_post=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self._allow_graph = allow_graph
        self._fuse_norm = fuse_norm
        self._rowwise_prenorm = (
            allow_graph and options.get("DSV4_PPU_DECODE_HC_ROWWISE") == "1"
        )
        self._fuse_moe_sum = fuse_moe_sum
        self._fuse_pre_quant = fuse_pre_quant
        self._fuse_post = fuse_post
        # Prefill-only defaults; explicit zero retains the original operators.
        self._prefill_quant = (
            not allow_graph and options.get("DSV4_PPU_PREFILL_HC_QUANT", "1") == "1"
        )
        self._prefill_post = (
            not allow_graph
            and tp_size in (4, 8)
            and options.get("DSV4_PPU_PREFILL_HC_POST", "1") == "1"
        )
        self._prenorm_partials = None
        self._backend = options.get(
            "DSV4_MHC_PRE_GEMM_BACKEND", "deepgemm_deterministic"
        )
        if self._backend not in ("deepgemm", "deepgemm_deterministic", "fp32"):
            raise ValueError(
                "PPU module HC requires an explicitly supported prenorm backend"
            )
        # The fused kernels contract replicated full-width inputs: the block
        # keeps the residual hidden dimension replicated at any TP size, the
        # loader keeps fn/base/scale and the RMSNorm weight replicated, and
        # the deterministic prenorm GEMM reduces one complete K=hc*dim plane
        # per rank. Every rank therefore computes bit-identical HC outputs
        # from identical inputs, so TP>1 needs no extra collective here.
        # pre_norm still fails closed on non-replicated activations/weights.
        if allow_graph and self._backend not in ("deepgemm_deterministic", "fp32"):
            raise ValueError("PPU HC Graph requires deterministic prenorm")
        if self._backend == "fp32":
            from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_fp32 import (
                fp32_hc_prenorm_partials,
            )

            self._prenorm_partials = partial(
                fp32_hc_prenorm_partials,
                row_gemv_stream_pool=row_gemv_stream_pool,
            )
            self._fuse_moe_sum = self._fuse_pre_quant = False
        elif fuse_prenorm:
            if not allow_graph or self._backend != "deepgemm_deterministic":
                raise ValueError("Fused HC reduction requires deterministic Decode")
            from rtp_llm.models_py.modules.dsv4.ppu.hc.ppu_hc_prenorm import (
                tf32_hc_prenorm_partials,
            )

            self._prenorm_partials = tf32_hc_prenorm_partials
        if (
            options.get("DSV4_MHC_POST_BACKEND", "tilelang") != "tilelang"
            or options.get("DSV4_MHC_POST_PDL", "0") != "0"
        ):
            raise ValueError("PPU module HC requires TileLang POST with PDL disabled")
        # Prepare the vendored runtime before importing its JIT kernels.
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

        prefix = "rtp_llm.models_py.kernels.ppu.hc"
        self._pre_kernel = importlib.import_module(
            prefix + ".pre_big_fuse"
        ).mhc_pre_big_fuse
        self._post_kernel = importlib.import_module(
            prefix + ".post_kernel"
        ).mhc_post_fwd
        if self._backend == "fp32":
            from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_fp32 import (
                fp32_hc_prenorm_gemm,
            )

            self._prenorm_gemm = fp32_hc_prenorm_gemm
            self._post_kernel = partial(self._post_kernel, reference_order=True)
        elif self._backend == "deepgemm_deterministic":
            from rtp_llm.models_py.modules.dsv4.ppu.hc.ppu_hc_prenorm import (
                tf32_hc_prenorm_gemm,
            )

            self._prenorm_gemm = tf32_hc_prenorm_gemm
        else:
            module = importlib.import_module(prefix + ".pre_big_fuse")
            self._prenorm_gemm = module._run_deepgemm_splitk_gemm

    def pre_norm(self, x, norm, *, tp_size, tp_rank, dbg_tag=None, hc_prepare_q3=False):
        if not self._fuse_norm:
            from rtp_llm.models_py.modules.dsv4.ppu.tp_norm import tp_rms_norm

            y, post, comb = self.pre(x, dbg_tag=dbg_tag)
            shape = y.shape
            y = tp_rms_norm(
                norm,
                y.reshape(-1, shape[-1]),
                tp_size=tp_size,
                tp_rank=tp_rank,
                global_dim=self.dim,
            ).view(shape)
            return y, post, comb
        from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

        leading = self._check_residual_shape(x, "pre_norm input")
        weight = norm.weight.data
        if (
            tp_size <= 0
            or not 0 <= tp_rank < tp_size
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or weight.shape != (self.dim,)
            or weight.dtype != torch.bfloat16
            or weight.device != x.device
            or not weight.is_contiguous()
        ):
            raise ValueError("Fused HC norm requires replicated contiguous BF16 inputs")
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
            prenorm_partials = self._prenorm_partials
            if self._backend == "fp32":
                if self._rowwise_prenorm or (x.ndim == 4 and x.shape[1] == 6):
                    prenorm_partials = partial(
                        prenorm_partials, rowwise=True, prepare_q3=hc_prepare_q3
                    )
                else:
                    from rtp_llm.models_py.modules.dsv4.ppu.prefill.request_rows import (
                        prefill_hc_pair,
                        request_row_lengths,
                    )

                    lengths = request_row_lengths(
                        x.numel() // (self.hc_mult * self.dim)
                    )
                    if lengths:
                        prenorm_partials = partial(
                            self._request_prenorm_partials, lengths=lengths
                        )
                    elif tp_size == 8 and x.numel() == 32768 * 16384:
                        pair = prefill_hc_pair()
                        if pair is not None:
                            prenorm_partials = partial(
                                self._pair_prenorm_partials, pair=pair
                            )
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
                prenorm_partials=prenorm_partials,
                norm_weight=None if self._backend == "fp32" else weight,
                norm_eps=norm.variance_epsilon,
            )
            if self._backend == "fp32":
                from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_fp32 import (
                    fp32_hc_rms_norm,
                )

                y = fp32_hc_rms_norm(
                    y,
                    weight,
                    norm.variance_epsilon,
                    rowwise=self._rowwise_prenorm or (x.ndim == 4 and x.shape[1] == 6),
                    prepare_q3=hc_prepare_q3,
                )
        self._check_pre_output(leading, y, post, comb)
        if dbg_tag is not None:
            for name, tensor in (("y", y), ("post", post), ("comb", comb)):
                _rt.record_if_level(2, f"{dbg_tag}_fused_norm_{name}", tensor)
        return y, post, comb

    def _request_prenorm_partials(self, x, weight, *, lengths):
        chunks = [self._prenorm_partials(part, weight) for part in x.split(lengths)]
        dots, squares = zip(*chunks)
        return torch.cat(dots, dim=1), torch.cat(squares, dim=1)

    def _pair_prenorm_partials(self, x, weight, *, pair):
        group, half = pair
        dot, squares = self._prenorm_partials(x.narrow(0, half * 16384, 16384), weight)
        packed = torch.cat((dot[0], squares[0, :, None]), dim=1)
        gathered = packed.new_empty((32768, 25))
        torch.distributed.all_gather_into_tensor(gathered, packed, group=group)
        return gathered[None, :, :-1].contiguous(), gathered[None, :, -1].contiguous()

    def prefill_fast_pre_norm(self, x, norm, *, tp_size, tp_rank):
        """Use the fused HC/RMSNorm boundary from Block's prefill fast path.

        Returning ``None`` keeps the generic separate-norm path available for
        controlled A/B runs without changing the cached HC PRE callables.
        """
        if not self._fuse_norm:
            return None
        return self.pre_norm(x, norm, tp_size=tp_size, tp_rank=tp_rank)

    def prefill_fast_pre_norm_quantized(self, x, norm, *, tp_size, tp_rank):
        if (
            not (
                self._prefill_quant
                and self._fuse_norm
                and self._backend == "deepgemm_deterministic"
                and tp_size == self.tp_size == 4
                and tp_rank == self.tp_rank
                and 0 <= tp_rank < tp_size
                and x.ndim == 3
                and x.shape[1:] == (4, 4096)
                and 1024 <= x.shape[0] <= 65536
                and x.is_cuda
                and x.dtype == torch.bfloat16
                and x.is_contiguous()
                and norm.weight.shape == (4096,)
                and norm.weight.dtype == torch.bfloat16
                and norm.weight.is_contiguous()
                and norm.weight.device == x.device
                and norm.variance_epsilon > 0
            )
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_pre_quant import (
            hc_pre_norm_quantized,
        )

        rows = x.shape[0]
        reduced = torch.empty((1, rows, 24), dtype=torch.float32, device=x.device)
        squares = torch.empty((1, rows), dtype=torch.float32, device=x.device)
        self._prenorm_gemm(x.view(rows, 16384), self.fn, reduced, squares)
        return hc_pre_norm_quantized(
            reduced,
            squares,
            self.scale,
            self.base,
            x,
            norm.weight.data,
            self.norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
            norm.variance_epsilon,
            column_scales=False,
        )

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

    def can_fuse_moe_sum(self, x, residual, post, comb):
        from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_sum import supports_hc_sum

        return self._fuse_moe_sum and supports_hc_sum(x, residual, post, comb)

    def post_moe_sum(self, routed, shared, residual, post, comb):
        from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_sum import hc_post_sum
        from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

        with record_function_range(
            f"dsv4.hc.L{self.layer_id:02d}.{self.name}.post_sum"
        ):
            return hc_post_sum(routed, shared, residual, post, comb)

    def _post_operator(self, x, residual, post, comb, *, hc_mult, out=None):
        if residual.numel() == 0:
            return residual if out is None else out
        if self._fuse_post:
            from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_sum import (
                hc_post,
                supports_hc_post,
            )

            if supports_hc_post(x, residual, post, comb, out):
                return hc_post(x, residual, post, comb, out=out)
        return self._post_kernel(
            x,
            residual,
            post,
            comb,
            out=out,
            enable_pdl=False,
            prefill_tuned=self._prefill_post
            and (
                self.tp_size == 4 or torch.cuda.get_device_name(x.device) == "ZW-M890P"
            ),
        )

    def can_fuse_pre_quant(self, x, norm):
        return (
            self._fuse_pre_quant
            and self._fuse_norm
            and self._prenorm_partials is not None
            and self.tp_size == 1
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.ndim in (3, 4)
            and x.shape[-2:] == (4, 4096)
            and 0 < x.numel() // 16384 <= 512
            and norm.weight.shape == (4096,)
            and norm.weight.is_contiguous()
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.device == x.device
            and norm.variance_epsilon > 0
        )

    def pre_norm_quantized(self, x, norm):
        if not self.can_fuse_pre_quant(x, norm):
            return None
        from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_pre_quant import (
            hc_pre_norm_quantized,
        )
        from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

        with torch.inference_mode(), record_function_range(
            f"dsv4.hc.L{self.layer_id:02d}.{self.name}.pre_norm_quant"
        ):
            partials, squares = self._prenorm_partials(x.view(-1, 16384), self.fn)
            return hc_pre_norm_quantized(
                partials,
                squares,
                self.scale,
                self.base,
                x,
                norm.weight.data,
                self.norm_eps,
                self.hc_eps,
                self.hc_sinkhorn_iters,
                norm.variance_epsilon,
            )
