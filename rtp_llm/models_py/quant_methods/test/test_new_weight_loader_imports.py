"""Smoke test for the public new_weight_loader Bazel target runfiles."""

import types
from unittest import mock

import torch


def _bert_config():
    return types.SimpleNamespace(
        model_type="bert",
        num_layers=1,
        hidden_size=4,
        inter_size=8,
        type_vocab_size=2,
        max_generate_batch_size=1,
        quant_config=None,
    )


def _bert_weights():
    h = 4
    inter = 8
    return {
        "bert.embeddings.word_embeddings.weight": torch.ones(8, h),
        "bert.embeddings.position_embeddings.weight": torch.ones(16, h),
        "bert.embeddings.token_type_embeddings.weight": torch.ones(2, h),
        "bert.embeddings.LayerNorm.weight": torch.ones(h),
        "bert.embeddings.LayerNorm.bias": torch.zeros(h),
        "bert.encoder.layer.0.attention.self.query.weight": torch.ones(h, h),
        "bert.encoder.layer.0.attention.self.query.bias": torch.zeros(h),
        "bert.encoder.layer.0.attention.self.key.weight": torch.ones(h, h),
        "bert.encoder.layer.0.attention.self.key.bias": torch.zeros(h),
        "bert.encoder.layer.0.attention.self.value.weight": torch.ones(h, h),
        "bert.encoder.layer.0.attention.self.value.bias": torch.zeros(h),
        "bert.encoder.layer.0.attention.output.dense.weight": torch.ones(h, h),
        "bert.encoder.layer.0.attention.output.dense.bias": torch.zeros(h),
        "bert.encoder.layer.0.attention.output.LayerNorm.weight": torch.ones(h),
        "bert.encoder.layer.0.attention.output.LayerNorm.bias": torch.zeros(h),
        "bert.encoder.layer.0.intermediate.dense.weight": torch.ones(inter, h),
        "bert.encoder.layer.0.intermediate.dense.bias": torch.zeros(inter),
        "bert.encoder.layer.0.output.dense.weight": torch.ones(h, inter),
        "bert.encoder.layer.0.output.dense.bias": torch.zeros(h),
        "bert.encoder.layer.0.output.LayerNorm.weight": torch.ones(h),
        "bert.encoder.layer.0.output.LayerNorm.bias": torch.zeros(h),
    }


def main():
    from rtp_llm.models_py.model_loader import LoadConfig, LoadMethod, NewModelLoader
    from rtp_llm.models_py.layers.linear import ColumnParallelLinear, RowParallelLinear
    from rtp_llm.models_py.new_models.bert import BertForEmbedding
    from rtp_llm.models_py.quant_methods import QuantizationConfig
    from rtp_llm.models_py.registry import list_models
    from rtp_llm.models_py.model_desc.bert import BertModel

    assert LoadMethod.AUTO == "auto"
    assert LoadConfig(device="cpu").device == "cpu"
    assert NewModelLoader is not None
    assert ColumnParallelLinear is not None
    assert RowParallelLinear is not None
    assert QuantizationConfig(quant_type="none").quant_type == "none"
    assert "bert" in list_models()

    model = BertForEmbedding(_bert_config(), LoadConfig(compute_dtype=torch.float32, device="cpu"))
    with mock.patch.object(BertModel, "__init__", return_value=None) as build_inner:
        model.load_weights(_bert_weights())
    assert build_inner.called
    assert isinstance(model.model, BertModel)


if __name__ == "__main__":
    main()
