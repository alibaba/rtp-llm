"""Embedding engine adapter for the Jina noncausal encoder."""

import torch

from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.new_models.jina_bert.encoder import JinaEncoder


from rtp_llm.models_py.new_models.resident_decoder import ResidentDecoderOwner


class JinaBertForEmbedding(ResidentDecoderOwner, JinaEncoder, GptModelBase):
    def prepare_resident_checkpoint_restore(self, state, values):
        super().prepare_resident_checkpoint_restore(state, values)
        for key, _, dtype in self._custom.values():
            value = values["custom_params." + key]
            if value.dtype != (dtype or self.embeddings.word_embeddings.weight.dtype):
                raise ValueError(f"Invalid resident Jina handler {key}")
            self.custom_params[key] = torch.nn.Parameter(value, requires_grad=False)

    def finalize_runtime_input_vocab(self, config, tokenizer):
        from rtp_llm.models_py.new_models.bert.checkpoint_metadata import publish_input_vocab

        publish_input_vocab(config, tokenizer, self.embeddings.word_embeddings.weight.shape[0])

    supports_ft_handler_heads = True

    def load_ft_checkpoint(self, checkpoint):
        from .ft_checkpoint import load_jina_ft_checkpoint

        load_jina_ft_checkpoint(self, checkpoint)

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
        # This owner supplies its own packed bidirectional ALiBi attention.
        return None

    def forward(self, inputs, fmha_impl=None):
        ids = inputs.input_ids
        metadata = get_primary_attention_inputs(inputs)
        types = inputs.bert_embedding_inputs.combo_tokens_type_ids
        if types is not None and types.numel() == 0:
            types = None
        host_ends = metadata.cu_seqlens
        ends = metadata.cu_seqlens_device
        if ends is None or ends.numel() == 0:
            ends = host_ends
        if ends is None or ends.numel() < 2:
            raise ValueError("Jina packed prefill requires cumulative query lengths")
        hidden = self.encode_cumulative(
            ids,
            types,
            ends,
            inputs.bert_embedding_inputs.input_embedding_scalar,
            host_ends=host_ends if host_ends is not None and host_ends.numel() else None,
            is_graph=metadata.is_cuda_graph,
        )
        from rtp_llm.ops.compute_ops import PyModelOutputs

        return PyModelOutputs(hidden)
