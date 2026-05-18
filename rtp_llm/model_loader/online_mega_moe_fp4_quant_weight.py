"""Load-time NVFP4 (UE8M0) quantization for GLM-5 MegaMoE weights.

Mirrors the structure of ``OnlineModelOptFp4MoeWeight`` but produces the
FP4 layout consumed by DeepGEMM ``fp8_fp4_mega_moe``:

    packed weight: int8 ``[E, N, K // 2]`` (two E2M1 nibbles per byte)
    block scale  : float32 ``[E, N, K // FP4_BLOCK]`` (UE8M0 exponent in float)

The block size is fixed at ``FP4_BLOCK = 32`` and the cast uses
``deep_gemm.utils.per_token_cast_to_fp4(use_ue8m0=True)`` so this class
matches ``GLM5MegaMoE.setup_weights_from_bf16`` bit for bit. Doing the work
here lets the BF16 tensor be released as soon as it lands on the GPU,
instead of staying resident until ``MegaMoeFusedWrapper.__init__`` runs.

Triggered explicitly from the GLM-5 model layer wiring (when
``MOE_STRATEGY=mega_moe``); not registered with
``QuantWeight.create()`` because the BF16 ckpt path has no ``quant_config``.
"""

from typing import Any, Optional

import torch

from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.utils.model_weight import W, identity

FP4_BLOCK = 32


def quantize_moe_weight_to_fp4_ue8m0(
    weight: torch.Tensor,
    block_size: int = FP4_BLOCK,
):
    """Quantize a stacked MoE weight ``[E, N, K]`` (BF16/FP16/FP32) to NVFP4
    + UE8M0 scale, matching the per-expert loop in
    ``GLM5MegaMoE.setup_weights_from_bf16``.

    Returns:
        packed: int8 ``[E, N, K // 2]``
        sf    : float32 ``[E, N, K // block_size]``
    """
    from deep_gemm.utils import per_token_cast_to_fp4

    assert weight.dim() == 3, f"expected 3D, got {tuple(weight.shape)}"
    E, N, K = weight.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    device = weight.device

    packed = torch.empty((E, N, K // 2), dtype=torch.int8, device=device)
    sf = torch.empty((E, N, K // block_size), dtype=torch.float32, device=device)
    for i in range(E):
        packed_i, sf_i = per_token_cast_to_fp4(
            weight[i], use_ue8m0=True, gran_k=block_size
        )
        packed[i].copy_(packed_i)
        sf[i].copy_(sf_i)
    return packed, sf


_MEGA_MOE_KERNEL_NAMES = (W.moe_w1, W.moe_w2)


def _scale_name_for(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported mega_moe kernel name: {name}")


class OnlineMegaMoEFp4Weight(CompositeWeight):
    """Load-time FP4+UE8M0 quantizer for the two stacked MoE weights
    (``moe_w1`` = gate||up, ``moe_w2`` = down).

    This class does **not** subclass ``QuantWeight`` because the GLM-5 BF16
    checkpoint runs without a ``quant_config``; it is selected explicitly by
    the GLM-5 model layer wiring when ``MOE_STRATEGY=mega_moe``.
    """

    moe_weight_list = list(_MEGA_MOE_KERNEL_NAMES)

    def __init__(
        self,
        src_weight_info: MoeAtomicWeight,
        block_size: int = FP4_BLOCK,
        **kwargs: Any,
    ):
        if src_weight_info.name not in _MEGA_MOE_KERNEL_NAMES:
            raise ValueError(
                f"OnlineMegaMoEFp4Weight only wraps {_MEGA_MOE_KERNEL_NAMES}, "
                f"got {src_weight_info.name}"
            )

        kernel = MoeAtomicWeight(
            name=src_weight_info.name,
            weights=src_weight_info.weights,
            process_fun=src_weight_info.process_fun,
            data_type=None,  # follow load_config.compute_dtype (BF16)
            config=src_weight_info.config,
            stacked_ckpt_keys=getattr(src_weight_info, "stacked_ckpt_keys", False),
        )
        scale = MoeAtomicWeight(
            name=_scale_name_for(src_weight_info.name),
            weights=[],  # synthesised in _load_raw_tensor
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        sub_weights = {kernel.name: kernel, scale.name: scale}
        super().__init__(
            sub_weights,
            name=src_weight_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.scale = scale
        self._block_size = block_size

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        # Only the BF16 kernel reads tensors from the checkpoint; scale is
        # produced on the fly.
        return self.kernel.get_tensor_names(layer_id, load_config)

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
        weight = kernel_dict[self.kernel.name]
        # Quantize on whatever device the BF16 tensor already lives on
        # (force_cpu_load_weights=1 → CPU; otherwise GPU).
        packed, sf = quantize_moe_weight_to_fp4_ue8m0(weight, self._block_size)
        del weight
        kernel_dict[self.kernel.name] = packed
        kernel_dict[self.scale.name] = sf
        return kernel_dict

    def _split(self, tensor, load_config: LoadConfig):
        # Both packed kernel and scale follow the same MoE expert split
        # (axis 0). Delegate to each sub-weight's split logic.
        split_kernel = self.kernel._split(
            {self.kernel.name: tensor[self.kernel.name]}, load_config
        )
        split_scale = self.scale._split(
            {self.scale.name: tensor[self.scale.name]}, load_config
        )
        out = {}
        out.update(split_kernel)
        out.update(split_scale)
        return out

    def _postprocess(self, tensor, device: str, load_config: LoadConfig):
        # No layout transform here: the deepgemm SF layout + mega-moe
        # transform require the (l1, l2) pair together and run inside
        # MegaMoeFusedWrapper after the symmetric-memory buffer is ready.
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.scale.name: tensor[self.scale.name],
        }
