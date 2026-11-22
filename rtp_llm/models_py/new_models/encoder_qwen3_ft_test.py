"""Real offline checkpoints: production FT descriptors, shard writer and index."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    MegatronBertConfig,
    MegatronBertForSequenceClassification,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    RobertaConfig,
    RobertaForSequenceClassification,
)

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.config.quant_config import Int8PerTensorQuantConfig
from rtp_llm.models.bert import Bert, Roberta
from rtp_llm.models.jina_bert.jina_bert import JinaBert
from rtp_llm.models.megatron_bert import MegatronBert
from rtp_llm.models.qwen_3_dspark import Qwen3DSpark
from rtp_llm.models.qwen_v3 import QwenV3
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.jina_bert.jina_checkpoint_test import (
    checkpoint as jina_weights,
)
from rtp_llm.models_py.new_models.qwen3_dspark.checkpoint_test import (
    checkpoint as dspark_weights,
)
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.ops import (
    FMHAConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
    TaskType,
)
from rtp_llm.tools.convert.ft_export_job import create_ft_export_job
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index


class ProductionFtRoundTrip(unittest.TestCase):
    def round_trip(
        self, name, service_cls, hf_config, weights, task, *, aliases=(), static=False
    ):
        with tempfile.TemporaryDirectory() as tmp:
            hf, ft = Path(tmp) / "hf", Path(tmp) / "ft"
            hf.mkdir()
            ft.mkdir()
            hf_config.save_pretrained(hf)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(
                    WordLevel(
                        {
                            "[UNK]": 0,
                            "yes": 1,
                            "no": 2,
                            **{str(i): i for i in range(3, hf_config.vocab_size)},
                        },
                        unk_token="[UNK]",
                    )
                ),
                unk_token="[UNK]",
            )
            tokenizer.save_pretrained(hf)
            save_file(weights, hf / "model.safetensors")

            def config(path, tag):
                cfg = service_cls.create_config(str(path))
                cfg.model_type, cfg.ckpt_path, cfg.tokenizer_path = (
                    tag,
                    str(path),
                    str(path),
                )
                cfg.task_type, cfg.data_type = task, "fp32"
                if static:
                    cfg.quant_config = Int8PerTensorQuantConfig()
                if name == "qwen_3_dspark":
                    cfg.gen_num_per_cycle = 3
                    cfg.is_mtp = True
                cfg.render_config, cfg.generate_env_config = (
                    RenderConfig(),
                    GenerateEnvConfig(),
                )
                return cfg

            pc = ParallelismConfig()
            pc.ep_size = 1
            cfg = config(hf, name)
            service = service_cls(
                model_config=cfg,
                parallelism_config=pc,
                kv_cache_config=KVCacheConfig(),
                hw_kernel_config=HWKernelConfig(),
                fmha_config=FMHAConfig(),
                moe_config=MoeConfig(),
                max_generate_batch_size=1,
                load_method="scratch",
                vit_config=None,
                merge_lora=False,
            )
            handler = service._init_custom_module()
            mappings = (
                handler.checkpoint_weight_mappings() if handler is not None else ()
            )
            create_ft_export_job(service).dump_weight_as_ft_style("cpu", str(ft))
            shutil.copyfile(hf / "config.json", ft / "config.json")
            generate_ft_index(str(ft), {"TP_SIZE": "1", "EP_SIZE": "1", "DP_SIZE": "1"})

            def load(path, tag, method):
                options = NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.float32,
                    parallelism_config=pc,
                    load_method=method,
                    custom_weight_mappings=mappings,
                    quant_config=QuantizationConfig("pertensor_quant")
                    if static
                    else None,
                )
                return NewModelLoader(
                    config(path, tag), options, model_path=str(path)
                ).load()

            reference = load(hf, name, "scratch")
            for tag in (name, *aliases):
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(tag=tag, method=method):
                        result = load(ft, tag, method)
                        self.assertEqual(
                            reference.state_dict().keys(), result.state_dict().keys()
                        )
                        for key, value in reference.state_dict().items():
                            torch.testing.assert_close(
                                result.state_dict()[key], value, rtol=0, atol=0
                            )
                        for role, *_ in mappings:
                            torch.testing.assert_close(
                                result.runtime_weight_view()[role],
                                reference.runtime_weight_view()[role],
                            )
            if name == "megatron_bert":
                index_path = ft / "model.safetensors.index.json"
                index = json.loads(index_path.read_text())
                for key in ("gamma", "beta"):
                    del index["weight_map"][
                        "rank_00_00_00.global.final_layernorm." + key
                    ]
                index_path.write_text(json.dumps(index))
                with self.assertRaisesRegex(
                    ValueError, "Re-export from the original HF"
                ):
                    load(ft, name, "scratch")

    def test_encoder_classifiers(self):
        for name, cls, cfg_cls, ref_cls in (
            ("bert", Bert, BertConfig, BertForSequenceClassification),
            ("roberta", Roberta, RobertaConfig, RobertaForSequenceClassification),
            (
                "megatron_bert",
                MegatronBert,
                MegatronBertConfig,
                MegatronBertForSequenceClassification,
            ),
        ):
            with self.subTest(model=name):
                cfg = cfg_cls(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=24,
                    num_hidden_layers=2,
                    max_position_embeddings=16,
                    num_attention_heads=4,
                    type_vocab_size=2,
                    num_labels=3,
                )
                self.round_trip(
                    name,
                    cls,
                    cfg,
                    ref_cls(cfg).state_dict(),
                    TaskType.SEQ_CLASSIFICATION,
                )

    def test_jina_code(self):
        cfg = BertConfig(
            vocab_size=29,
            hidden_size=12,
            intermediate_size=17,
            num_hidden_layers=2,
            num_attention_heads=3,
            type_vocab_size=2,
            layer_norm_eps=1e-5,
        )
        cfg.num_labels = 3
        weights = {"bert." + k: v for k, v in jina_weights().items()}
        weights.update(
            {
                "bert.pooler.dense.weight": torch.randn(12, 12),
                "bert.pooler.dense.bias": torch.randn(12),
                "classifier.weight": torch.randn(3, 12),
                "classifier.bias": torch.randn(3),
            }
        )
        self.round_trip(
            "jina_bert_code", JinaBert, cfg, weights, TaskType.SEQ_CLASSIFICATION
        )

    def test_calibrated_bert(self):
        cfg = BertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            type_vocab_size=2,
            num_labels=3,
        )
        weights = BertForSequenceClassification(cfg).state_dict()
        for index in range(2):
            prefix = f"bert.encoder.layer.{index}."
            weights[prefix + "attention.self.qkv_input_scale"] = torch.tensor(
                [0.0625 * (index + 1)]
            )
            for suffix in (
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
                "intermediate.dense",
                "output.dense",
            ):
                stem = prefix + suffix
                weight = weights.pop(stem + ".weight")
                weights[stem + ".qweight"] = (
                    (weight.T * 1200)
                    .round()
                    .clamp(-128, 127)
                    .to(torch.int8)
                    .contiguous()
                )
                weights[stem + ".scales"] = (
                    torch.arange(1, weight.shape[0] + 1, dtype=torch.float32) / 1024
                )
                if not suffix.startswith("attention.self."):
                    weights[stem + "_input_scale"] = torch.tensor([0.125])
        self.round_trip(
            "bert", Bert, cfg, weights, TaskType.SEQ_CLASSIFICATION, static=True
        )

    def test_qwen3_and_tool(self):
        cfg = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            tie_word_embeddings=False,
        )
        weights = Qwen3ForCausalLM(cfg).state_dict()
        for task in (TaskType.LANGUAGE_MODEL, TaskType.RERANKER):
            self.round_trip(
                "qwen_3", QwenV3, cfg, weights, task, aliases=("qwen_3_tool",)
            )

    def test_dspark_draft(self):
        cfg = Qwen3Config(
            vocab_size=8,
            hidden_size=4,
            intermediate_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            tie_word_embeddings=False,
            target_layer_ids=[0, 3],
            mask_token_id=7,
            draft_vocab_size=4,
            markov_rank=2,
        )
        self.round_trip(
            "qwen_3_dspark", Qwen3DSpark, cfg, dspark_weights(), TaskType.LANGUAGE_MODEL
        )


if __name__ == "__main__":
    unittest.main()
