"""Megatron-BERT direct-weight model using main's packed attention runtime."""

from rtp_llm.models_py.new_models.resident_decoder import ResidentDecoderOwner
from torch import nn

from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.megatron_bert.backbone import MegatronParameters


class MegatronBertForEmbedding(ResidentDecoderOwner, MegatronParameters, GptModelBase, RtpModule):
    supports_ft_handler_heads = True

    def prepare_resident_checkpoint_restore(self, state, values):
        super().prepare_resident_checkpoint_restore(state, values)
        for name in self._custom:
            value = values["custom_params." + name]
            if value.dtype != self._custom_dtype[name]:
                raise ValueError(f"Invalid resident Megatron handler {name}")
            self.custom_params[name] = nn.Parameter(value, requires_grad=False)
        table = values.get("embeddings.token_type_embeddings.weight")
        if self.embeddings.token_type_embeddings is None and table is not None:
            anchor = self.embeddings.word_embeddings.weight
            if table.dtype != anchor.dtype or table.ndim != 2 or table.shape[1] != anchor.shape[1] or not table.shape[0]:
                raise ValueError("Invalid resident Megatron token type table")
            self.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(table)

    def finalize_runtime_input_vocab(self, config, tokenizer):
        from rtp_llm.models_py.new_models.bert.checkpoint_metadata import publish_input_vocab

        publish_input_vocab(config, tokenizer, self.embeddings.word_embeddings.weight.shape[0])

    def load_ft_checkpoint(self, checkpoint):
        from .ft_checkpoint import load_megatron_ft_checkpoint

        load_megatron_ft_checkpoint(self, checkpoint)

    def __init__(self, model_config, load_config):
        super().__init__(
            config=model_config,
            parallelism_config=load_config.parallelism_config,
            weight=None,
            max_generate_batch_size=load_config.max_generate_batch_size,
            fmha_config=load_config.fmha_config,
            py_hw_kernel_config=load_config.py_hw_kernel_config,
        )
        self.build_encoder(model_config, load_config)

    def forward(self, inputs, fmha_impl=None):
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        bert = inputs.bert_embedding_inputs
        hidden = self.encode_packed(
            inputs.input_ids,
            bert.combo_position_ids,
            bert.combo_tokens_type_ids,
            bert.input_embedding_scalar,
            lambda i: select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i),
            lambda i: self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
        )
        from rtp_llm.ops.compute_ops import PyModelOutputs

        return PyModelOutputs(hidden)

    def runtime_weight_view(self):
        # The native embedding engine consumes these non-owning references.
        result = self.custom_weight_view()
        result["embedding"] = self.embeddings.word_embeddings.weight
        result["position_encoding.weight"] = self.embeddings.position_embeddings.weight
        if self.embeddings.token_type_embeddings is not None:
            result["token_type_embedding.weight"] = (
                self.embeddings.token_type_embeddings.weight
            )
        return result
