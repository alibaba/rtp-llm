"""Main config/compute ABI acceptance. Requires the real candidate .so files."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    MegatronBertConfig,
    MegatronBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.model_factory_register import _model_factory, ensure_model_registered
from rtp_llm.models.megatron_bert import MegatronBert
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.megatron_bert.language import MegatronBertForEmbedding
from rtp_llm.models_py.new_models.megatron_bert.test_checkpoint import PackedSdpa
from rtp_llm.ops import ParallelismConfig, TaskType


class _MegatronServiceFixture(unittest.TestCase):
    def check_service(self, *, native_output):
        self.assertTrue(ensure_model_registered("megatron_bert"))
        service_class = _model_factory["megatron_bert"]
        self.assertIs(service_class, MegatronBert)

        class CpuService(service_class):
            # Real construction/tokenizer/handler/publication. The service test
            # selects CPU explicitly; there is no production CPU/GPU fallback.
            def _get_device_str(self):
                return "cpu"

        config = MegatronBertConfig(
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=16,
            max_position_embeddings=16,
            type_vocab_size=3,
            num_labels=3,
        )
        torch.manual_seed(31)
        ref = MegatronBertForSequenceClassification(config).eval()
        ids = torch.tensor([[3, 4, 5]])
        types = torch.tensor([[0, 1, 2]])
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            config.save_pretrained(tmp)
            # Old deployments may omit this optional JSON field even though
            # their checkpoint contains a trained segment table.
            config_file = Path(tmp) / "config.json"
            manifest = json.loads(config_file.read_text())
            manifest.pop("type_vocab_size")
            config_file.write_text(json.dumps(manifest))
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(
                    WordLevel(
                        {"[UNK]": 0, "[PAD]": 1, **{str(i): i for i in range(2, 16)}},
                        unk_token="[UNK]",
                    )
                ),
                unk_token="[UNK]",
                pad_token="[PAD]",
            )
            tokenizer.save_pretrained(tmp)
            save_file(ref.state_dict(), os.path.join(tmp, "model.safetensors"))
            cfg = service_class.create_config(tmp)
            self.assertEqual(cfg.type_vocab_size, 0)
            cfg.model_type = "megatron_bert"
            cfg.data_type = "fp32"
            cfg.tokenizer_path = tmp
            cfg.task_type = TaskType.SEQ_CLASSIFICATION
            cfg.render_config = RenderConfig()
            cfg.generate_env_config = GenerateEnvConfig()
            for method in ("scratch", "fastsafetensors"):
                service = CpuService.from_config(
                    model_config=cfg,
                    parallelism_config=ParallelismConfig(),
                    load_method=method,
                    kv_cache_config=SimpleNamespace(
                        multi_task_prompt="", multi_task_prompt_str=""
                    ),
                    fmha_config=None,
                    moe_config=None,
                    hw_kernel_config=SimpleNamespace(
                        enable_cuda_graph=False, use_swizzleA=False
                    ),
                    max_generate_batch_size=2,
                    merge_lora=False,
                    vit_config=None,
                )
                self.assertIsInstance(service.py_model, MegatronBertForEmbedding)
                self.assertEqual(cfg.input_vocab_size, ref.bert.embeddings.word_embeddings.num_embeddings)
                self.assertIs(service.checkpoint_owner, service.py_model)
                self.assertEqual(
                    service.model_weights_loader.loaded_method.value, method
                )
                self.assertIs(
                    service.weight.get_global_weight("embedding"),
                    service.py_model.embeddings.word_embeddings.weight,
                )
                torch.testing.assert_close(
                    service.weight.get_global_weight("token_type_embedding.weight"),
                    ref.bert.embeddings.token_type_embeddings.weight,
                )
                classifier = service.custom_module.handler
                self.assertEqual(
                    classifier.linear.weight.data_ptr(),
                    service.weight.get_global_weight(
                        "__custom__.classifier.weight"
                    ).data_ptr(),
                )
                if not native_output:
                    continue
                from rtp_llm.ops.compute_ops import BertEmbeddingInputs, PyModelInputs

                inputs = PyModelInputs()
                inputs.input_ids = ids[0]
                inputs.bert_embedding_inputs = BertEmbeddingInputs(
                    torch.arange(3),
                    torch.empty(0),
                    types[0],
                    torch.empty(0),
                    1.0,
                )
                hidden = service.py_model(inputs, PackedSdpa((3,))).hidden_states
                logits = classifier.forward(ids[0], hidden, torch.tensor([3]))
                torch.testing.assert_close(
                    logits,
                    ref(input_ids=ids, token_type_ids=types).logits.to(logits.dtype),
                )


class MegatronServiceCheckpointTest(_MegatronServiceFixture):
    def test_service_loads_and_publishes_classifier(self):
        self.check_service(native_output=False)


class NativeMegatronTest(_MegatronServiceFixture):
    def test_public_service_publishes_classifier_owner(self):
        self.check_service(native_output=True)

    def test_registered_class_and_native_inputs_outputs(self):
        from rtp_llm.ops.compute_ops import BertEmbeddingInputs, PyModelInputs

        config = MegatronBertConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=24,
            vocab_size=32,
            max_position_embeddings=16,
        )
        torch.manual_seed(17)
        ref = MegatronBertForSequenceClassification(config).eval()
        ids = torch.tensor([[1, 5, 2]])
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            config.save_pretrained(tmp)
            save_file(ref.state_dict(), os.path.join(tmp, "model.safetensors"))
            cfg = MegatronBert._create_config(tmp)
            cfg.model_type = "megatron_bert"
            for method in ("scratch", "fastsafetensors"):
                loader = NewModelLoader(
                    cfg,
                    NewLoaderConfig(
                        device="cpu",
                        compute_dtype=torch.float32,
                        load_method=method,
                        parallelism_config=ParallelismConfig(),
                    ),
                    model_path=tmp,
                )
                model = loader.load()
                inputs = PyModelInputs()
                inputs.input_ids = ids[0]
                inputs.bert_embedding_inputs = BertEmbeddingInputs(
                    torch.arange(3),
                    torch.empty(0),
                    torch.zeros(3, dtype=torch.long),
                    torch.empty(0),
                    1.0,
                )
                actual = model(inputs, PackedSdpa((3,))).hidden_states
                torch.testing.assert_close(
                    actual,
                    ref.bert(input_ids=ids).last_hidden_state[0],
                    atol=1e-5,
                    rtol=1e-5,
                )
                self.assertEqual(loader._selected_load_method.value, method)


if __name__ == "__main__":
    unittest.main()
