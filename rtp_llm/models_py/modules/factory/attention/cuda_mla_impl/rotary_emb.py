from typing import Any

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.base_rotary_embedding_op import (
    BaseRotaryEmbeddingOp,
)


class MlaRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    """Rotary positional embedding for Multi-Latent Attention (MLA).

    This operator only applies RoPE to query and key tensors.
    KV cache writing is handled separately by MlaKVCacheWriteOp.
    """

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        token_per_block: int,
        is_neox_style: bool,
    ) -> None:
        super().__init__(head_size, cos_sin_cache, token_per_block, is_neox_style)
        self.params: Any = None

    def set_params(self, params: Any) -> None:
        """Set parameters for RoPE application.

        Args:
            params: FlashInferMlaAttnParams containing position IDs
        """
        self.params = params

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> None:
        """
        Apply RoPE to query and key tensors in-place.

        Args:
            query: Query tensor to apply RoPE to
            key: Key tensor for RoPE (will be unsqueezed for compatibility)
        """
        # Apply RoPE to Q and K (MLA requires key.unsqueeze(1) for dim compatibility)
        self._apply_rope(query, key.unsqueeze(1), self.params)
