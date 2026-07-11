"""Smoke test for the public new_weight_loader Bazel target runfiles."""

import tempfile
import types

import torch
from safetensors.torch import save_file


def _bert_config():
    from rtp_llm.config.model_config import ModelConfig

    config = ModelConfig()
    config.model_type = "bert"
    config.num_layers = 1
    config.vocab_size = 8
    config.hidden_size = 4
    config.inter_size = 8
    config.type_vocab_size = 2
    config.max_seq_len = 16
    config.layernorm_eps = 1e-5
    config.activation_type = "Gelu"
    config.quant_config = None
    config.attn_config.head_num = 1
    config.attn_config.kv_head_num = 1
    config.attn_config.size_per_head = 4
    return config


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
    from rtp_llm.ops import PyModelInitResources

    assert LoadMethod.AUTO == "auto"
    assert LoadConfig(device="cpu").device == "cpu"
    assert NewModelLoader is not None
    assert ColumnParallelLinear is not None
    assert RowParallelLinear is not None
    assert QuantizationConfig(quant_type="none").quant_type == "none"
    assert "bert" in list_models()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(_bert_weights(), f"{tmpdir}/model.safetensors")
        loader = NewModelLoader(
            _bert_config(),
            LoadConfig(compute_dtype=torch.float32, device="cpu", load_method=LoadMethod.SCRATCH),
            model_path=tmpdir,
        )
        model = loader.load()
    assert isinstance(model, BertForEmbedding)
    assert isinstance(model.model, BertModel)
    assert model.weights is not None
    assert model.embeddings.word_embeddings_weight.device.type == "cpu"
    assert model.initialize(PyModelInitResources())


if __name__ == "__main__":
    main()
