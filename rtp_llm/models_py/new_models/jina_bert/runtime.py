"""Embedding engine adapter for the Jina noncausal encoder."""

import torch

from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.new_models.jina_bert.encoder import JinaEncoder


class JinaBertForEmbedding(JinaEncoder, GptModelBase):
    def __init__(self, config, load):
        GptModelBase.__init__(
            self,
            config,
            load.parallelism_config,
            None,
            max_generate_batch_size=load.max_generate_batch_size,
            fmha_config=load.fmha_config,
            py_hw_kernel_config=load.py_hw_kernel_config,
        )
        self.init_encoder(config, load)

    def runtime_weight_view(self):
        # These four names are consumed by the existing embedding engine ABI.
        # They are aliases to owned parameters, never checkpoint load targets.
        from rtp_llm.utils.model_weight import W

        return {
            W.embedding: self.embeddings.word_embeddings.weight,
            W.token_type_embedding: self.embeddings.token_type_embeddings.weight,
            W.pre_decoder_ln_gamma: self.embeddings.LayerNorm.weight,
            W.pre_decoder_ln_beta: self.embeddings.LayerNorm.bias,
            **self.custom_weight_view(),
        }

    def prepare_fmha_impl(self, *args, **kwargs):
        # PyTorch SDPA consumes the checkpoint's symmetric ALiBi bias. A
        # generic paged self-attention backend cannot supply it.
        return None

    def forward(self, inputs, fmha_impl=None):
        ids = inputs.input_ids
        metadata = get_primary_attention_inputs(inputs)
        types = inputs.bert_embedding_inputs.combo_tokens_type_ids
        if types is not None and types.numel() == 0:
            types = None
        # GPU tensor cumulative lengths remain live across graph replays; no
        # .tolist()/host-dependent iteration captured for packed requests.
        ends = metadata.cu_seqlens_device
        if ends is None or ends.numel() == 0:
            ends = metadata.cu_seqlens
        if ends is None or ends.numel() < 2:
            raise ValueError("Jina packed prefill requires cumulative query lengths")
        seq_ids = torch.bucketize(
            torch.arange(ids.numel(), device=ids.device),
            ends[1:].to(ids.device),
            right=True,
        )
        hidden = self.encode(
            ids.unsqueeze(0),
            None if types is None else types.unsqueeze(0),
            inputs.bert_embedding_inputs.input_embedding_scalar,
            sequence_ids=seq_ids,
        )
        from rtp_llm.ops.compute_ops import PyModelOutputs

        return PyModelOutputs(hidden.squeeze(0))
