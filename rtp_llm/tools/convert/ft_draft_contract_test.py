import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rtp_llm.models_py.new_models.qwen3_moe_eagle3.body import Eagle3DraftBody
from rtp_llm.tools.convert.ft_export_job import create_ft_export_job
from rtp_llm.tools.convert.weights_convert import WeightConverter


class DraftExportContractTest(unittest.TestCase):
    def test_convert_drafts_and_ordinary_model(self):
        config = dict(
            hidden_size=8, vocab_size=24, head_dim=4, num_attention_heads=2,
            num_key_value_heads=1, intermediate_size=12, hidden_act="silu",
            rms_norm_eps=1e-5, num_hidden_layers=1, dtype="float16",
        )
        eagle = dict(config, architectures=["LlamaForCausalLMEagle3"], eagle_config=dict(
            eagle_aux_hidden_state_layer_ids=[2, 0, 1], use_aux_hidden_state=True,
            use_input_layernorm_in_first_layer=True, use_last_layernorm=True,
        ))
        body = Eagle3DraftBody(8, 12, 2, 1, 4, 1e-5, torch.float16)
        eagle_weights = {
            k: torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape)
            for k, p in body.named_parameters()
        }
        dflash = dict(config, architectures=["DFlashDraftModel"], model_type="qwen3", layer_types=["full_attention"],
            dflash_config=dict(mask_token_id=7, target_layer_ids=[1, 3], block_size=16))
        shapes = {
            "fc.weight": (8, 16), "hidden_norm.weight": (8,), "norm.weight": (8,),
            "layers.0.input_layernorm.weight": (8,), "layers.0.post_attention_layernorm.weight": (8,),
            "layers.0.self_attn.q_norm.weight": (4,), "layers.0.self_attn.k_norm.weight": (4,),
            "layers.0.self_attn.q_proj.weight": (8, 8), "layers.0.self_attn.k_proj.weight": (4, 8),
            "layers.0.self_attn.v_proj.weight": (4, 8), "layers.0.self_attn.o_proj.weight": (8, 8),
            "layers.0.mlp.gate_proj.weight": (12, 8), "layers.0.mlp.up_proj.weight": (12, 8),
            "layers.0.mlp.down_proj.weight": (8, 12),
        }
        dflash_weights = {k: torch.arange(torch.Size(s).numel(), dtype=torch.float16).reshape(s)
                          for k, s in shapes.items()}
        ordinary = dict(config, architectures=["Qwen3ForCausalLM"], model_type="qwen3")
        ordinary_weights = {"model." + k: v for k, v in dflash_weights.items()
                            if k.startswith("layers.") or k == "norm.weight"}
        ordinary_weights["model.embed_tokens.weight"] = torch.ones(24, 8, dtype=torch.float16)
        ordinary_weights["lm_head.weight"] = torch.full((24, 8), 2, dtype=torch.float16)
        for raw, kind, weights, suffix, expected in (
            (eagle, "qwen_3_moe_eagle3", eagle_weights, ".fc.weight", eagle_weights["fc.weight"]),
            (dflash, "qwen_3_dflash", dflash_weights, ".global.dspark_fc.weight", dflash_weights["fc.weight"].T),
            (ordinary, "qwen_3", ordinary_weights, ".global.embedding", ordinary_weights["model.embed_tokens.weight"]),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root, patch.dict(os.environ):
                source, out = Path(root, "draft"), Path(root, "ft")
                source.mkdir()
                (source / "config.json").write_text(json.dumps(raw))
                save_file(weights, str(source / "model.safetensors"))
                if kind == "qwen_3":
                    PreTrainedTokenizerFast(
                        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
                        unk_token="<unk>",
                    ).save_pretrained(source)
                job = WeightConverter(str(source), None, {
                    "TP_SIZE": "1", "DP_SIZE": "1", "ACT_TYPE": "fp16",
                    "LOADER_RECYCLE_HANDLES": "false", "MOE_PURE_TP_PRESHARD": "true",
                })
                self.assertEqual(job.model_type, kind)
                with patch.object(sys, "argv", [
                    "weights_convert", "--pretrained_model_dir", str(source),
                    "--output_dir_base", str(out),
                ]), patch("rtp_llm.tools.convert.weights_convert.create_ft_export_job",
                          wraps=create_ft_export_job) as build_job:
                    job.convert(str(out))
                descriptor, = build_job.call_args.args
                self.assertIs(descriptor.loader_recycle_handles, False)
                self.assertIs(descriptor.moe_pure_tp_preshard, True)
                self.assertTrue((out / "done").is_file())
                index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
                for role in (".global.embedding", ".global.lm_head"):
                    self.assertEqual(any(k.endswith(role) for k in index), kind == "qwen_3")
                key, = [k for k in index if k.endswith(suffix)]
                with safe_open(out / index[key], framework="pt") as shard:
                    torch.testing.assert_close(shard.get_tensor(key), expected, rtol=0, atol=0)
                if kind == "qwen_3_moe_eagle3":
                    (source / "config.json").write_text(json.dumps(dict(raw, draft_vocab_size=4)))
                    with self.assertRaisesRegex(ValueError, "reduced-vocabulary"):
                        WeightConverter(str(source), None, {})

    def test_contract_checked_before_database_open(self):
        class Dense:
            def load_ft_checkpoint(self, checkpoint):
                raise AssertionError("not a draft adapter")

        class MissingPrivateAdapter(Dense):
            pass

        class MissingVocabularyLoan:
            def load_ft_checkpoint(self, checkpoint):
                pass

        model = Mock()
        model.model_config.attn_config.head_num = 8
        model.parallelism_config.get_attn_tp_size.return_value = 1
        for kind, owner, reason in (
            ("qwen_2-mtp", MissingPrivateAdapter, "draft FT format"),
            ("qwen_3_dflash", MissingVocabularyLoan, "vocabulary FT loan contract"),
            ("qwen_3_moe_eagle3", MissingVocabularyLoan, "vocabulary FT loan contract"),
        ):
            model.model_config.model_type = kind
            with self.subTest(kind=kind), patch(
                "rtp_llm.models_py.registry.get_model_class", return_value=owner
            ), patch("rtp_llm.utils.database.CkptDatabase") as database:
                with self.assertRaisesRegex(ValueError, reason):
                    create_ft_export_job(model)
                database.assert_not_called()


if __name__ == "__main__":
    unittest.main()
