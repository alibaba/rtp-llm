"""Explicit CUDA13/SM100 FP8 block32 linears for V4.1 components.

Activations cross the official BF16 -> group32 FP8/UE8M0 boundary. Checkpoint
32x32 weight scales are losslessly expanded and packed once for DeepGEMM.
"""

import logging
from functools import partial

import torch
from torch import nn


def is_supported(values: torch.Tensor) -> bool:
    return (
        values.is_cuda
        and torch.version.cuda is not None
        and torch.version.cuda.split(".")[0] == "13"
        and torch.cuda.get_device_capability(values.device)[0] == 10
    )


def quantize_block32(
    values: torch.Tensor, *, packed: bool = False, swizzled: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 and raw, DeepGEMM-packed or F8_128x4 group32 scales."""
    if (
        values.ndim != 2
        or values.dtype != torch.bfloat16
        or not values.is_contiguous()
        or values.shape[1] == 0
        or values.shape[1] % 32
    ):
        raise ValueError("V4.1 activations must be contiguous BF16 [M,K] with K%32=0")
    if packed and swizzled:
        raise ValueError("select only one V4.1 packed activation scale layout")
    if (packed or swizzled) and values.shape[1] % 128:
        raise ValueError("packed V4.1 activation scales require K%128=0")
    from rtp_llm.models_py.modules.dsv41._linear_triton import (
        quantize_block32_vector_kernel,
    )

    rows, columns = values.shape
    encoded = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    if swizzled:
        scales = torch.empty(
            ((rows + 127) // 128) * 128 * (columns // 32),
            dtype=torch.uint8,
            device=values.device,
        )
    elif packed:
        scales = torch.empty(
            (columns // 128, ((rows + 3) // 4) * 4),
            dtype=torch.int32,
            device=values.device,
        ).transpose(0, 1)[:rows]
    else:
        scales = torch.empty(
            (rows, columns // 32), dtype=torch.float32, device=values.device
        )
    if rows:
        packs = 16 if values.numel() >= 1024 * 1024 else 4
        stored_rows = ((rows + 127) // 128) * 128 if swizzled else rows
        blocks = (stored_rows * columns + packs * 128 - 1) // (packs * 128)
        quantize_block32_vector_kernel[(blocks,)](
            values,
            encoded,
            scales,
            rows,
            K=columns,
            PACKED=packed,
            PACKS=packs,
            SWIZZLED=swizzled,
            num_warps=4,
        )
    return encoded, scales


class V41Block32Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        if (
            weight.ndim != 2
            or weight.dtype != torch.float8_e4m3fn
            or not weight.is_contiguous()
            or min(weight.shape) <= 0
            or weight.shape[0] % 32
            or weight.shape[1] % 32
        ):
            raise ValueError(
                "V4.1 dense weight must be contiguous E4M3 [N,K], N/K%32=0"
            )
        self.out_features, self.in_features = weight.shape
        self._packed_activations = self.in_features % 128 == 0
        if (
            scale.shape != (self.out_features // 32, self.in_features // 32)
            or scale.dtype != torch.float8_e8m0fnu
            or scale.device != weight.device
            or not scale.is_contiguous()
        ):
            raise ValueError(
                "V4.1 dense scales must be the checkpoint UE8M0 32x32 grid"
            )
        scales = scale.float()
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scales)
        flashinfer_shape = (
            (self.out_features, self.in_features) == (32768, 1280)
            and torch.cuda.get_device_capability(weight.device) == (10, 3)
            and self._packed_activations
        )
        self._use_cudnn = flashinfer_shape
        self._use_cute = False
        if flashinfer_shape:
            import cutlass
            import flashinfer
            from packaging.version import Version

            self._use_cute = (
                Version(flashinfer.__version__) == Version("0.6.18")
                and Version(cutlass.__version__) == Version("4.6.2")
            )
        self.register_buffer("_weight_scale_packed", None, persistent=False)
        self.register_buffer("_weight_scale_swizzled", None, persistent=False)
        self._cudnn_plans = {}
        self._cute_runner = None
        self._flashinfer_gemm = None
        if self._use_cudnn or self._use_cute:
            from flashinfer.gemm import gemm_base

            self._flashinfer_gemm = gemm_base
        self._refresh_weight_scale_packed()
        self.register_load_state_dict_post_hook(self._refresh_weight_scale_packed)

    def _refresh_weight_scale_packed(self, _module=None, _incompatible_keys=None):
        import deep_gemm

        packed_scales = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(
            self.weight_scale.repeat_interleave(32, dim=0)
        )
        previous = self._weight_scale_packed
        if previous is not None and previous.device == packed_scales.device:
            previous.copy_(packed_scales)
        else:
            self._weight_scale_packed = packed_scales
        if self._use_cudnn or self._use_cute:
            import flashinfer

            swizzled = flashinfer.block_scale_interleave(
                self.weight_scale.to(torch.float8_e8m0fnu)
                .view(torch.uint8)
                .repeat_interleave(32, dim=0)
            )
            previous = self._weight_scale_swizzled
            if previous is not None and previous.device == swizzled.device:
                previous.copy_(swizzled)
            else:
                self._weight_scale_swizzled = swizzled

    def _use_swizzled_gemm(self, rows):
        return (self._use_cute and 1 <= rows <= 8) or (
            self._use_cudnn and rows == 2048
        )

    def _run_quantized_gemm(self, encoded, scales, output):
        if scales.dtype == torch.uint8:
            gemm_base = self._flashinfer_gemm

            if self._use_cute and encoded.shape[0] <= 8:
                if self._cute_runner is None:
                    self._cute_runner = gemm_base._cute_dsl_gemm_mxfp8_runner(
                        10, 3, True, torch.bfloat16
                    )
                self._cute_runner.forward(
                    [
                        encoded,
                        self.weight.t(),
                        scales,
                        self._weight_scale_swizzled,
                        torch.bfloat16,
                        output,
                        None,
                    ]
                )
                return

            stream = torch.cuda.current_stream(encoded.device)
            key = (encoded.device, encoded.shape[0], stream.cuda_stream)
            prepared = self._cudnn_plans.get(key)
            if prepared is None:
                import cudnn

                a = encoded.unsqueeze(0)
                b = self.weight.t().unsqueeze(0)
                out = output.unsqueeze(0)
                if hasattr(gemm_base, "build_cudnn_gemm_mxfp8_graph"):
                    graph = gemm_base.build_cudnn_gemm_mxfp8_graph(
                        a.shape,
                        a.stride(),
                        cudnn.data_type.FP8_E4M3,
                        b.shape,
                        b.stride(),
                        cudnn.data_type.FP8_E4M3,
                        32,
                        cudnn.data_type.BFLOAT16,
                        encoded.device,
                    )
                else:
                    graph = gemm_base._get_cudnn_mxfp8_gemm_graph(
                        a, b, torch.bfloat16, out
                    )
                workspace = torch.empty(
                    graph.get_workspace_size(),
                    dtype=torch.uint8,
                    device=encoded.device,
                )
                uids = tuple(
                    value.value
                    for value in (
                        gemm_base.UIDs.A_UID,
                        gemm_base.UIDs.B_UID,
                        gemm_base.UIDs.BLOCK_DESCALE_A_UID,
                        gemm_base.UIDs.BLOCK_DESCALE_B_UID,
                        gemm_base.UIDs.O_UID,
                    )
                )
                prepared = self._cudnn_plans[key] = (graph._execute, workspace, uids)
            # Plans depend on shape and dtype; bindings stay current after Graph
            # capture or a state-dict update of the persistent weight buffers.
            execute, workspace, (a_uid, b_uid, sa_uid, sb_uid, out_uid) = prepared
            execute(
                {
                    a_uid: encoded.data_ptr(),
                    b_uid: self.weight.data_ptr(),
                    sa_uid: scales.data_ptr(),
                    sb_uid: self._weight_scale_swizzled.data_ptr(),
                    out_uid: output.data_ptr(),
                },
                workspace.data_ptr(),
                gemm_base._get_cudnn_handle(encoded.device, stream),
            )
            return
        import deep_gemm

        packed = self._weight_scale_packed is not None
        deep_gemm.fp8_gemm_nt(
            (encoded, scales),
            (self.weight, self._weight_scale_packed if packed else self.weight_scale),
            output,
            recipe=(1, 1 if packed else 32, 32),
            disable_ue8m0_cast=False,
        )

    @torch.inference_mode()
    def forward(self, values: torch.Tensor, *, out=None) -> torch.Tensor:
        if (
            values.ndim < 2
            or values.shape[-1] != self.in_features
            or values.dtype != torch.bfloat16
            or values.device != self.weight.device
            or not values.is_contiguous()
        ):
            raise ValueError(
                "V4.1 linear input must be contiguous BF16 on its weight GPU"
            )
        shape = (*values.shape[:-1], self.out_features)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=values.device)
        elif (
            out.shape != shape
            or out.dtype != torch.bfloat16
            or out.device != values.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "V4.1 linear output shape, dtype, device or layout mismatch"
            )
        rows = values.numel() // self.in_features
        if rows:
            swizzled = self._use_swizzled_gemm(rows)
            encoded, scales = quantize_block32(
                values.view(rows, self.in_features),
                packed=self._packed_activations and not swizzled,
                swizzled=swizzled,
            )
            self._run_quantized_gemm(encoded, scales, out.view(rows, self.out_features))
        return out


@torch.inference_mode()
def warmup_block32_linears(model: nn.Module, *, max_rows: int) -> None:
    """Prepare each distinct V4.1 dense shape using its selected scale layout."""
    from rtp_llm.utils.warmup import model_warm_up_enabled

    if not model_warm_up_enabled():
        return
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("V4.1 dense warmup requires a positive row budget")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("V4.1 dense warmup cannot run inside CUDA Graph capture")
    from rtp_llm.models_py.modules.dsv4 import dsv4_kernel_jit_warmup as warmup

    linears = {}
    for module in model.modules():
        if isinstance(module, V41Block32Linear):
            key = (module.weight.device, module.out_features, module.in_features)
            linears.setdefault(key, module)

    def prepare():
        for (device, n, k), linear in linears.items():
            grid = warmup._generate_dense_gemm_warmup_m_grid(
                max_m=max_rows,
                n_value=n,
                k_value=k,
                kind="fp8",
                num_sms=warmup._get_deep_gemm_num_sms(device),
                uses_deepjit=True,
            )
            logging.info(
                "[V41 Block32] startup dense n=%d k=%d max_rows=%d device=%s rows=%s",
                n,
                k,
                max_rows,
                device,
                grid,
            )
            for rows in grid:
                values = torch.ones((rows, k), dtype=torch.bfloat16, device=device)
                warmup._run_deepgemm_warmup_launch_with_retry(
                    "V41 Block32",
                    f"n={n} k={k} rows={rows}",
                    partial(linear, values),
                    device=device,
                )
            torch.cuda.synchronize(device)

    warmup._run_deepgemm_warmup_launches_serialized("V41 Block32", prepare)
