"""Smoke test for the public new_weight_loader Bazel target runfiles."""


def main():
    from rtp_llm.models_py.model_loader import LoadConfig, LoadMethod, NewModelLoader
    from rtp_llm.models_py.layers.linear import ColumnParallelLinear, RowParallelLinear
    from rtp_llm.models_py.quant_methods import QuantizationConfig
    from rtp_llm.models_py.registry import list_models

    assert LoadMethod.AUTO == "auto"
    assert LoadConfig(device="cpu").device == "cpu"
    assert NewModelLoader is not None
    assert ColumnParallelLinear is not None
    assert RowParallelLinear is not None
    assert QuantizationConfig(quant_type="none").quant_type == "none"
    assert "bert" in list_models()


if __name__ == "__main__":
    main()
