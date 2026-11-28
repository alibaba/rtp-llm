"""Shared Qwen4 MTP projection used by both parameter-loading paths."""

import torch
from torch import nn

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)


class Qwen4ExpMTPInputProjection(nn.Module):
    """Fuse one token embedding into every target hyper-connection branch."""

    def __init__(
        self,
        embedding_gamma: torch.Tensor,
        hidden_gamma: torch.Tensor,
        fc_embedding: nn.Module,
        fc_hidden: nn.Module,
        hidden_size: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.embedding_gamma = embedding_gamma
        self.hidden_gamma = hidden_gamma
        self.fc_embedding = fc_embedding
        self.fc_hidden = fc_hidden
        self.hidden_size = int(hidden_size)
        if int(hidden_gamma.numel()) % self.hidden_size:
            raise ValueError(
                "qwen4_exp MTP hidden norm width must be divisible by hidden_size"
            )
        self.hc_mult = int(hidden_gamma.numel()) // self.hidden_size
        self.hc_hidden_size = self.hc_mult * self.hidden_size
        self.norm_eps = float(norm_eps)

    def forward(
        self, inputs_embeds: torch.Tensor, input_hiddens: torch.Tensor
    ) -> torch.Tensor:
        if inputs_embeds.dim() != 2 or input_hiddens.dim() != 2:
            raise RuntimeError(
                "qwen4_exp MTP input projection expects packed 2-D tensors"
            )
        if int(inputs_embeds.shape[0]) != int(input_hiddens.shape[0]):
            raise RuntimeError(
                "qwen4_exp MTP embedding/hidden row counts differ: "
                f"{inputs_embeds.shape[0]} != {input_hiddens.shape[0]}"
            )
        if int(inputs_embeds.shape[-1]) != self.hidden_size:
            raise RuntimeError(
                "qwen4_exp MTP embedding width mismatch: expected "
                f"{self.hidden_size}, got {inputs_embeds.shape[-1]}"
            )
        if int(input_hiddens.shape[-1]) != self.hc_hidden_size:
            raise RuntimeError(
                "qwen4_exp MTP target hidden width mismatch: expected "
                f"{self.hc_hidden_size}, got {input_hiddens.shape[-1]}"
            )
        if tuple(self.embedding_gamma.shape) != (self.hidden_size,) or tuple(
            self.hidden_gamma.shape
        ) != (self.hc_hidden_size,):
            raise RuntimeError("qwen4_exp MTP pre-FC norm weight shape is invalid")

        embedding = None
        if inputs_embeds.is_cuda and torch.version.hip is None:
            from rtp_llm.models_py.kernels.cuda.qwen4_mtp_norm import (
                try_mtp_embedding_rms_norm,
            )

            embedding = try_mtp_embedding_rms_norm(
                inputs_embeds, self.embedding_gamma, self.norm_eps
            )
        if embedding is None:
            embedding = exact_head_rms_norm(
                inputs_embeds, self.embedding_gamma, self.norm_eps
            )
        # The MTP pre-FC norm spans the entire HC stream, before the branch
        # reshape (SGLang Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared).
        # Main-model HC norms remain branch-local.
        hidden = exact_head_rms_norm(input_hiddens, self.hidden_gamma, self.norm_eps)
        hidden = hidden.unflatten(-1, (self.hc_mult, self.hidden_size))

        # fc_embedding/fc_hidden are hidden_size-wide matrices.  Apply them to
        # every branch independently; pooling the target stream before either
        # projection destroys the released head's trained branch semantics.
        projected_embedding = self.fc_embedding(embedding).unsqueeze(-2)
        projected_hidden = self.fc_hidden(hidden.reshape(-1, self.hidden_size)).reshape(
            int(hidden.shape[0]), self.hc_mult, self.hidden_size
        )
        return (projected_embedding + projected_hidden).flatten(-2)
