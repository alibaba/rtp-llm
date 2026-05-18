"""Load-time FP8-to-FP4 (UE8M0) quantization for GLM-5 MegaMoE weights.

Mirrors ``OnlineMegaMoEFp4Weight`` but starts from FP8 per-block weights
(float8_e4m3fn + float32 scale) instead of BF16. Converts FP8 → BF16 → FP4
at load time so the FP8 tensor can be released immediately, avoiding the
double-storage cost of holding FP8 until ``MegaMoeFusedWrapper.__init__``.

The conversion uses ``convert_fp8_weights_to_fp4`` from ``quant_layouts.py``
which matches ``GLM5MegaMoE.setup_weights_from_fp8`` bit for bit.

Triggered when ``MOE_STRATEGY=mega_moe`` and the checkpoint is FP8 per-block
quantized. Selected explicitly from the ``PerBlockFp8Weight._postprocess``
path via environment variable detection, or wired from model layer code.
"""

from typing import Any, Optional

import torch

from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.utils.model_weight import W, identity

FP4_BLOCK = 32
FP8_BLOCK = 128

_MEGA_MOE_KERNEL_NAMES = (W.moe_w1, W.moe_w2)


def _scale_name_for(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported mega_moe kernel name: {name}")


def _fp8_scale_name_for(name: str) -> str:
    """Return the FP8 per-block scale name used as input."""
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported mega_moe kernel name: {name}")


def _convert_fp8_moe_to_fp4(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: int = FP4_BLOCK,
):
    """Convert stacked FP8 MoE weights [E, N, K] to FP4 + UE8M0 scale.

    Dequantizes FP8 per-block → BF16, then requantizes to FP4 (int8 packed)
    with UE8M0 scale factors suitable for DeepGEMM mega_moe.

    Handles two scale layouts:
      - Per-row: [E, N, K//128] — one scale per 128 elements along K
      - 2D-block: [E, N//128, K//128] — one scale per 128×128 block

    Returns:
        packed: int8 [E, N, K // 2]
        sf: float32 [E, N, K // block_size]
    """
    from deep_gemm.utils import per_token_cast_to_fp4

    assert (
        weight_fp8.dim() == 3
    ), f"expected 3D weight_fp8, got {tuple(weight_fp8.shape)}"
    E, N, K = weight_fp8.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    device = weight_fp8.device

    # Flatten any trailing dimensions in scale
    weight_scale = (
        weight_scale.reshape(E, -1) if weight_scale.dim() > 3 else weight_scale
    )
    if weight_scale.dim() == 2:
        weight_scale = weight_scale.unsqueeze(1)

    # Detect scale layout
    if weight_scale.shape == (E, N, K // FP8_BLOCK):
        # Per-row scale: [E, N, K//128]
        w_float = weight_fp8.float()
        scale_expanded = weight_scale.unsqueeze(-1).expand(
            E, N, K // FP8_BLOCK, FP8_BLOCK
        )
        w_float = (
            w_float.view(E, N, K // FP8_BLOCK, FP8_BLOCK) * scale_expanded
        ).reshape(E, N, K)
    elif weight_scale.shape == (E, N // FP8_BLOCK, K // FP8_BLOCK):
        # 2D-block scale: [E, N//128, K//128] — expand to per-row first
        # Each scale covers a 128×128 block
        scale_per_row = weight_scale.repeat_interleave(FP8_BLOCK, dim=1)[:, :N, :]
        w_float = weight_fp8.float()
        scale_expanded = scale_per_row.unsqueeze(-1).expand(
            E, N, K // FP8_BLOCK, FP8_BLOCK
        )
        w_float = (
            w_float.view(E, N, K // FP8_BLOCK, FP8_BLOCK) * scale_expanded
        ).reshape(E, N, K)
        del scale_per_row
    else:
        # Try to reshape to per-row layout
        expected_elements = E * N * (K // FP8_BLOCK)
        if weight_scale.numel() == expected_elements:
            weight_scale = weight_scale.reshape(E, N, K // FP8_BLOCK)
            w_float = weight_fp8.float()
            scale_expanded = weight_scale.unsqueeze(-1).expand(
                E, N, K // FP8_BLOCK, FP8_BLOCK
            )
            w_float = (
                w_float.view(E, N, K // FP8_BLOCK, FP8_BLOCK) * scale_expanded
            ).reshape(E, N, K)
        else:
            raise ValueError(
                f"Cannot interpret scale shape {tuple(weight_scale.shape)} "
                f"for weight shape [E={E}, N={N}, K={K}]"
            )

    w_bf16 = w_float.to(torch.bfloat16)
    del w_float
    if "scale_expanded" in dir():
        del scale_expanded

    packed = torch.empty((E, N, K // 2), dtype=torch.int8, device=device)
    sf = torch.empty((E, N, K // block_size), dtype=torch.float32, device=device)
    for i in range(E):
        packed[i], sf[i] = per_token_cast_to_fp4(
            w_bf16[i], use_ue8m0=True, gran_k=block_size
        )
    del w_bf16
    return packed, sf


class OnlineMegaMoEFp8ToFp4Weight(CompositeWeight):
    """Load-time FP8→FP4+UE8M0 quantizer for stacked MoE weights
    (``moe_w1`` = gate||up, ``moe_w2`` = down).

    Takes FP8 per-block quantized weights (float8_e4m3fn kernel + float32 scale)
    and produces FP4 packed int8 + UE8M0 float32 scale at load time.

    This class does **not** subclass ``QuantWeight`` — it is selected explicitly
    by the model layer wiring when ``MOE_STRATEGY=mega_moe`` and the checkpoint
    is FP8 per-block quantized.
    """

    moe_weight_list = list(_MEGA_MOE_KERNEL_NAMES)

    def __init__(
        self,
        src_kernel_info: MoeAtomicWeight,
        src_scale_info: MoeAtomicWeight,
        block_size: int = FP4_BLOCK,
        **kwargs: Any,
    ):
        """Initialize from existing FP8 kernel and scale sub-weights.

        Args:
            src_kernel_info: MoeAtomicWeight for the FP8 kernel (moe_w1 or moe_w2)
            src_scale_info: MoeAtomicWeight for the FP8 per-block scale (moe_s1 or moe_s2)
            block_size: FP4 quantization block size (default 32)
        """
        if src_kernel_info.name not in _MEGA_MOE_KERNEL_NAMES:
            raise ValueError(
                f"OnlineMegaMoEFp8ToFp4Weight only wraps {_MEGA_MOE_KERNEL_NAMES}, "
                f"got {src_kernel_info.name}"
            )

        kernel = MoeAtomicWeight(
            name=src_kernel_info.name,
            weights=src_kernel_info.weights,
            process_fun=src_kernel_info.process_fun,
            data_type=torch.float8_e4m3fn,
            config=src_kernel_info.config,
            stacked_ckpt_keys=getattr(src_kernel_info, "stacked_ckpt_keys", False),
        )
        fp8_scale = MoeAtomicWeight(
            name=_fp8_scale_name_for(src_kernel_info.name) + "_fp8_input",
            weights=src_scale_info.weights,
            process_fun=src_scale_info.process_fun,
            data_type=torch.float32,
            config=src_scale_info.config,
            stacked_ckpt_keys=getattr(src_scale_info, "stacked_ckpt_keys", False),
        )
        out_scale = MoeAtomicWeight(
            name=_scale_name_for(src_kernel_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_kernel_info.config,
        )

        sub_weights = {
            kernel.name: kernel,
            fp8_scale.name: fp8_scale,
            out_scale.name: out_scale,
        }
        super().__init__(
            sub_weights,
            name=src_kernel_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.fp8_scale = fp8_scale
        self.out_scale = out_scale
        self._block_size = block_size

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        names = self.kernel.get_tensor_names(layer_id, load_config)
        names |= self.fp8_scale.get_tensor_names(layer_id, load_config)
        return names

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        scale_dict = self.fp8_scale._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        weight_fp8 = kernel_dict[self.kernel.name]
        weight_scale = scale_dict[self.fp8_scale.name]

        packed, sf = _convert_fp8_moe_to_fp4(weight_fp8, weight_scale, self._block_size)
        del weight_fp8, weight_scale

        return {
            self.kernel.name: packed,
            self.out_scale.name: sf,
        }

    def _split(self, tensor, load_config: LoadConfig):
        split_kernel = self.kernel._split(
            {self.kernel.name: tensor[self.kernel.name]}, load_config
        )
        split_scale = self.out_scale._split(
            {self.out_scale.name: tensor[self.out_scale.name]}, load_config
        )
        out = {}
        out.update(split_kernel)
        out.update(split_scale)
        return out

    def _postprocess(self, tensor, device: str, load_config: LoadConfig):
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.out_scale.name: tensor[self.out_scale.name],
        }
