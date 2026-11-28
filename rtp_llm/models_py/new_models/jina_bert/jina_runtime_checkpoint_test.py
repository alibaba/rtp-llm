"""Acceptance at the actual main ABI. Missing native modules are failures, not skips."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.models.jina_bert.jina_bert import JinaBert
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.jina_bert.jina_checkpoint_test import (
    checkpoint,
    exact_forward,
)
from rtp_llm.models_py.new_models.jina_bert.runtime import JinaBertForEmbedding
from rtp_llm.ops import (
    FMHAConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
    TaskType,
)
from rtp_llm.utils.model_weight import W


class CpuJinaBert(JinaBert):
    """Exercise the unmodified service startup path on a deterministic CPU."""

    def _get_device_str(self):
        return "cpu"


class _JinaServiceFixture(unittest.TestCase):
    def check_service(self, *, native_output):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as cfg:
                json.dump(
                    dict(
                        model_type="bert",
                        num_attention_heads=3,
                        hidden_size=12,
                        num_hidden_layers=2,
                        vocab_size=29,
                        type_vocab_size=2,
                        layer_norm_eps=1e-5,
                        intermediate_size=17,
                    ),
                    cfg,
                )
            with open(os.path.join(directory, "vocab.txt"), "w") as vocab:
                vocab.write(
                    "\n".join(
                        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                        + [f"v{i}" for i in range(24)]
                    )
                    + "\n"
                )
            with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
                json.dump({"tokenizer_class": "BertTokenizer"}, f)
            tensors = checkpoint()
            save_file(tensors, os.path.join(directory, "model.safetensors"))
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                config = JinaBert.create_config(directory)
                config.model_type = "jina_bert_code"
                config.data_type = "fp32"
                config.task_type = TaskType.DENSE_EMBEDDING
                config.tokenizer_path = directory
                config.render_config = RenderConfig()
                config.generate_env_config = GenerateEnvConfig()
                service_args = dict(
                    model_config=config,
                    parallelism_config=ParallelismConfig(),
                    hw_kernel_config=HWKernelConfig(),
                    kv_cache_config=KVCacheConfig(),
                    fmha_config=FMHAConfig(),
                    moe_config=MoeConfig(),
                    max_generate_batch_size=2,
                    load_method=NewLoaderLoadMethod(method),
                    vit_config=None,
                    merge_lora=False,
                )
                service = CpuJinaBert(**service_args)
                service.load()
                model = service.py_model
                self.assertIsInstance(model, JinaBertForEmbedding)
                self.assertEqual(config.input_vocab_size, model.embeddings.word_embeddings.num_embeddings)
                self.assertIs(service.checkpoint_owner, model)
                self.assertIs(service.weight, model.weight)
                self.assertIs(
                    service.weight.get_global_weight(W.embedding),
                    model.embeddings.word_embeddings.weight,
                )
                self.assertIsNotNone(service.weight_manager)
                self.assertIsNotNone(service.rank_local_lora)
                self.assertEqual(
                    service.model_weights_loader.loaded_method.value, method
                )
                if not native_output:
                    continue
                from rtp_llm.ops.compute_ops import (
                    BertEmbeddingInputs,
                    PyAttentionInputs,
                    PyModelInputs,
                )

                ids = torch.tensor([1, 3, 9, 2, 14], dtype=torch.int32)
                types = torch.tensor([1, 1, 0, 1, 0], dtype=torch.int32)
                lengths = torch.tensor([2, 3], dtype=torch.int32)
                meta = PyAttentionInputs()
                meta.cu_seqlens_device = torch.tensor([0, 2, 5], dtype=torch.int32)
                inputs = PyModelInputs()
                inputs.input_ids = ids
                inputs.attention_inputs = meta
                b = BertEmbeddingInputs()
                b.combo_tokens_type_ids = types
                b.input_embedding_scalar = 1.0
                inputs.bert_embedding_inputs = b
                actual = model(inputs).hidden_states
                expected = torch.cat(
                    [
                        exact_forward(tensors, i, t)
                        for i, t in zip(
                            ids.split(lengths.tolist()), types.split(lengths.tolist())
                        )
                    ]
                )
                torch.testing.assert_close(actual, expected)
                pooled = service.custom_module.handler.forward(ids, actual, lengths)
                torch.testing.assert_close(
                    pooled, torch.nn.functional.normalize(expected[[0, 2]], dim=1)
                )
            # Even a single absent Jina-specific bias must fail service startup:
            # neither the owner nor the management interfaces may be published.
            del tensors["encoder.layer.1.attention.self.layer_norm_k.bias"]
            save_file(tensors, os.path.join(directory, "model.safetensors"))
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                service_args["load_method"] = NewLoaderLoadMethod(method)
                incomplete = CpuJinaBert(**service_args)
                with self.assertRaisesRegex(RuntimeError, "layer_norm_k.bias"):
                    incomplete.load()
                self.assertIsNone(incomplete.py_model)
                self.assertIsNone(incomplete.checkpoint_owner)
                self.assertIsNone(incomplete.weight)
                self.assertIsNone(incomplete.weight_manager)
                self.assertIsNone(incomplete.rank_local_lora)


class JinaServiceCheckpointTest(_JinaServiceFixture):
    def test_service_publishes_only_complete_checkpoint(self):
        self.check_service(native_output=False)


class JinaRuntimeCheckpointTest(_JinaServiceFixture):
    def test_native_forward_and_noncausal_pooling(self):
        self.check_service(native_output=True)


if __name__ == "__main__":
    unittest.main()
