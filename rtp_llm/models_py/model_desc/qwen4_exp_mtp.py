"""Python model descriptor for the Qwen4-Exp MTP draft."""

from typing import Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.modules import LinearFactory
from rtp_llm.models_py.modules.qwen4_exp.gated_residual import grouped_rms_norm
from rtp_llm.models_py.modules.qwen4_exp.norm import exact_head_rms_norm
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs
from rtp_llm.utils.model_weight import W


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

        embedding = exact_head_rms_norm(
            inputs_embeds, self.embedding_gamma, self.norm_eps
        )
        hidden = grouped_rms_norm(
            input_hiddens,
            self.hidden_gamma,
            self.hidden_size,
            self.norm_eps,
        ).unflatten(-1, (self.hc_mult, self.hidden_size))

        # fc_embedding/fc_hidden are hidden_size-wide matrices.  Apply them to
        # every branch independently; pooling the target stream before either
        # projection destroys the released head's trained branch semantics.
        projected_embedding = self.fc_embedding(embedding).unsqueeze(-2)
        projected_hidden = self.fc_hidden(hidden.reshape(-1, self.hidden_size)).reshape(
            int(hidden.shape[0]), self.hc_mult, self.hidden_size
        )
        return (projected_embedding + projected_hidden).flatten(-2)


class Qwen4ExpMTPModel(Qwen4ExpModel):
    """One-layer draft model with target-hidden input fusion and QSA."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        if self.layer_num != 1:
            raise RuntimeError("qwen4_exp MTP model descriptor requires one layer")
        if getattr(model_config, "enable_qwen4_ple", False):
            raise RuntimeError("qwen4_exp MTP model descriptor does not support PLE")
        self.mtp_input_projection = Qwen4ExpMTPInputProjection(
            weights.get_global_weight(W.multi_tokens_predict_enorm),
            weights.get_global_weight(W.multi_tokens_predict_hnorm),
            LinearFactory.create_linear_from_weights(
                weights.global_weights,
                W.qwen4_mtp_fc_embedding_w,
                hw_kernel_config=py_hw_kernel_config,
            ),
            LinearFactory.create_linear_from_weights(
                weights.global_weights,
                W.qwen4_mtp_fc_hidden_w,
                hw_kernel_config=py_hw_kernel_config,
            ),
            hidden_size=model_config.hidden_size,
            norm_eps=model_config.layernorm_eps,
        )

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        input_hiddens: Optional[torch.Tensor] = getattr(inputs, "input_hiddens", None)
        if not isinstance(input_hiddens, torch.Tensor) or not input_hiddens.numel():
            raise RuntimeError(
                "qwen4_exp MTP requires hc_mult*hidden_size-wide target input_hiddens"
            )
        # The MTP draft consumes text ids only; do not enter Qwen35's multimodal
        # embedding injector from this override.
        inputs_embeds = self.embed_tokens(inputs.input_ids)
        return self.mtp_input_projection(inputs_embeds, input_hiddens)

    def _initial_hyper_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = int(self.config.hc_mult) * int(self.config.hidden_size)
        if int(hidden_states.shape[-1]) != expected:
            raise RuntimeError(
                "qwen4_exp MTP projected residual width mismatch: "
                f"expected {expected}, got {hidden_states.shape[-1]}"
            )
        return hidden_states
