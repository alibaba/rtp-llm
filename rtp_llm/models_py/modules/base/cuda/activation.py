"""CUDA-specific activation function implementations."""

import torch

from rtp_llm.models_py.modules.base.common.activation import SiluAndMulBase


class FusedSiluAndMul(SiluAndMulBase):
    """CUDA implementation of silu_and_mul using flashinfer."""

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Perform SiLU activation and element-wise multiplication using CUDA kernel.

        Args:
            output: Output tensor to write result to
            gate_up: Input tensor with concatenated gate and up projections
        """
        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        # The flashinfer act_and_mul kernel uses 16-byte vector accesses whose
        # offsets scale with d. The pre-upstream-#5013 wheels still pinned for
        # cu129 / legacy / PPU (0.6.9+e87d610f, 0.6.9+a532b824, 0.6.8.post1,
        # 0.6.0) fault with cudaErrorMisalignedAddress when d is not a multiple
        # of the vector width (e.g. the Qwen2.5-VL vision MLP, d = 3420).
        # Keep those shapes on a plain torch silu * up.
        vec_size = 16 // gate_up.element_size()
        if d % vec_size == 0:
            # Lazy import: this module is also imported where flashinfer is
            # absent (e.g. ppu/rocm), so importing it must not require flashinfer.
            import flashinfer

            return flashinfer.activation.silu_and_mul(gate_up, out=output)
        hidden = gate_up[..., :d]
        # Reuse output as the sigmoid buffer; bit-identical to torch.mul(hidden, torch.sigmoid(hidden), out=output).
        torch.sigmoid(hidden, out=output)
        output.mul_(hidden)
        output.mul_(gate_up[..., d:])
        return output
