from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main

from rtp_llm.model_factory_register import ModelDict, register_hf_architecture
from rtp_llm.models_py.model_desc.multimodal_embedding import (
    embed_with_multimodal_features,
)

_RTP_LLM_ROOT = Path(__file__).resolve().parents[1]


class Qwen35VLMoeRoutingTest(TestCase):
    def test_qwen35_vl_routes_to_vl_model(self):
        config = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "vision_config": {},
        }
        self.assertEqual("qwen35_vl_moe", ModelDict.get_ft_model_type_by_config(config))

    def test_qwen35_text_still_routes_to_text_model(self):
        register_hf_architecture(
            "Qwen3_5MoeForConditionalGeneration",
            "qwen35_moe",
        )
        config = {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}
        self.assertEqual("qwen35_moe", ModelDict.get_ft_model_type_by_config(config))


class Qwen35VLMoeArchitectureTest(TestCase):
    def test_qwen3_next_text_backbone_does_not_own_multimodal_overlay(self):
        source = (_RTP_LLM_ROOT / "models_py/model_desc/qwen3_next.py").read_text()
        self.assertNotIn("multimodal_embedding", source)
        self.assertNotIn("embed_with_multimodal_features", source)

    def test_qwen35_vl_uses_specialized_python_model_wrapper(self):
        model_source = (_RTP_LLM_ROOT / "models/qwen3_5_vl/qwen3_5_vl.py").read_text()
        wrapper_source = (
            _RTP_LLM_ROOT / "models_py/model_desc/qwen35_vl_next.py"
        ).read_text()
        self.assertIn("def _create_python_model", model_source)
        self.assertIn("Qwen35VLNextModel", model_source)
        self.assertIn("class Qwen35VLNextModel", wrapper_source)
        self.assertIn("embed_with_multimodal_features", wrapper_source)


class _FakeTensor:
    def __init__(self, data, dtype="float32", device="cpu"):
        self.data = data
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        if isinstance(self.data, list):
            return (len(self.data),)
        return ()

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return _FakeTensor([item == other for item in self.data], dtype="bool")

    def __setitem__(self, key, value):
        if isinstance(key, _FakeTensor):
            for index, selected in enumerate(key.data):
                if selected:
                    self.data[index] = value
            return
        if isinstance(key, slice):
            self.data[key] = [row[:] for row in value.data]
            return
        self.data[key] = value

    def clone(self):
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            data = [row[:] for row in self.data]
        else:
            data = self.data[:]
        return _FakeTensor(data, dtype=self.dtype, device=self.device)

    def contiguous(self):
        return self

    def numel(self):
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return sum(len(row) for row in self.data)
        return len(self.data)

    def to(self, device=None, dtype=None):
        tensor = self.clone()
        if device is not None:
            tensor.device = device
        if dtype is not None:
            tensor.dtype = dtype
        return tensor

    def tolist(self):
        return self.clone().data


class _EmbeddingStub:
    def __call__(self, input_ids: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(
            [[float(item), float(item) + 100] for item in input_ids.tolist()],
            dtype="float32",
            device=input_ids.device,
        )


class Qwen35VLMoeEmbeddingOverlayTest(TestCase):
    def test_text_only_embedding_path_is_unchanged(self):
        input_ids = _FakeTensor([7, 8, 9], dtype="int64")
        inputs = SimpleNamespace(
            input_ids=input_ids,
            multimodal_features=[],
            text_tokens_mask=_FakeTensor([], dtype="int32"),
            mm_features_locs=_FakeTensor([], dtype="int32"),
        )
        hidden_states = embed_with_multimodal_features(_EmbeddingStub(), inputs)
        expected = _EmbeddingStub()(input_ids)
        self.assertEqual(expected.tolist(), hidden_states.tolist())

    def test_multimodal_features_overlay_embedding_spans(self):
        input_ids = _FakeTensor([7, 1000000, 1000001, 9], dtype="int64")
        inputs = SimpleNamespace(
            input_ids=input_ids,
            multimodal_features=[_FakeTensor([[1.5, 2.5], [3.5, 4.5]])],
            text_tokens_mask=_FakeTensor([1, 0, 0, 1], dtype="int32"),
            mm_features_locs=_FakeTensor([1], dtype="int32"),
        )
        hidden_states = embed_with_multimodal_features(_EmbeddingStub(), inputs)
        expected = [[7.0, 107.0], [1.5, 2.5], [3.5, 4.5], [9.0, 109.0]]
        self.assertEqual(expected, hidden_states.tolist())


if __name__ == "__main__":
    main()
