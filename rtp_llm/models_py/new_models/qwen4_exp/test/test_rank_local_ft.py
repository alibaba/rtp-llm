import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import Qwen4TargetWeights


def write_rank_pair(root):
    text = dict(
        num_hidden_layers=1,
        hidden_size=16,
        vocab_size=32,
        hc_count=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention"],
        ple_layer_ids=[1],
        ngram_size=3,
        heads_per_ngram=1,
        split_ngram_parts=2,
        ple_embed_dim=8,
        ple_conv_kernel_size=3,
        eos_token_id=2,
    )
    (root / "config.json").write_text(json.dumps({"text_config": text}))
    weight_map = {}
    for rank in range(2):
        shapes = {"global.embedding": (32, 8), "global.lm_head": (16, 16)}
        for prefix in (
            "global.qwen4.hc.mixer",
            "layers.0.qwen4.hc.attn",
            "layers.0.qwen4.hc.mlp",
        ):
            for field, shape in (
                ("norm", (32,)),
                ("mix_down", (4, 32)),
                ("mix_up", (32, 4)),
                ("inject", (2, 32)),
            ):
                if field != "inject" or not prefix.startswith("global."):
                    shapes[prefix + "_" + field] = shape
        shapes.update(
            {
                "layers.0." + name: shape
                for name, shape in (
                    ("self_attention_weights.query_weight.kernel", (16, 16)),
                    ("self_attention_weights.attention_output_weight.kernel", (8, 16)),
                    ("self_attention_weights.gate.weight", (16, 8)),
                    ("self_attention_weights.q_layernorm.gamma", (4,)),
                    ("self_attention_weights.k_layernorm.gamma", (4,)),
                    ("partial_moe_weights.gate.kernel", (16, 2)),
                    ("partial_moe_weights.intermediate_weight.kernel", (1, 32, 16)),
                    ("partial_moe_weights.intermediate_weight2.kernel", (1, 16, 16)),
                )
            }
        )
        shapes.update(
            {
                "layers.0.qwen4.ple." + key: shape
                for key, shape in (
                    ("conv.weight", (32, 1, 3)),
                    ("key_proj.weight", (32, 8)),
                    ("value_proj.weight", (16, 8)),
                    ("norm_key.gamma", (32,)),
                    ("norm_query.gamma", (32,)),
                    ("norm_conv.gamma", (32,)),
                    (f"ngram_shards.{rank}", (4, 4)),
                )
            }
        )
        name = f"worker-{rank}.safetensors"
        data = {
            f"rank_{rank}_0_{rank}."
            + key: (
                torch.arange(torch.Size(shape).numel()).reshape(shape) % 53 + rank * 60
            ).bfloat16()
            for key, shape in shapes.items()
        }
        for leaf, values in (
            ("multipliers", [1, 3, 7]),
            ("ngram_offsets", [0, 4]),
            ("ngram_vocab_sizes", [4, 4]),
        ):
            data[f"rank_{rank}_0_{rank}.layers.0.qwen4.ple." + leaf] = torch.tensor(
                values, dtype=torch.int64
            )
        save_file(data, str(root / name))
        weight_map.update(dict.fromkeys(data, name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "is_ft_style_weight": True,
                "ft_export_backend": "cuda",
                "__env__params__": {"TP_SIZE": 2, "EP_SIZE": 2},
                "weight_map": weight_map,
            }
        )
    )


class Qwen4FTTest(unittest.TestCase):
    def check_rank1(self, backend, pruned=False, restart=False):
        with tempfile.TemporaryDirectory() as directory:
            write_rank_pair(Path(directory))
            topology = CheckpointTopology(tp_size=2, tp_rank=1, ep_size=2, ep_rank=1)
            ft = FTCheckpoint.open(directory, topology)
            ft.read_backend = backend
            config = {"ckpt_path": directory, "model_type": "qwen4_exp", "moe_k": 1}
            if pruned:
                config.update(
                    enable_output_vocab_pruning=True, output_vocab_padded_size=16,
                    output_vocab_ids=[0, 1, 2, 3, 4, 5, 6, 7, 10, 29, 31],
                )
            load = NewLoaderConfig(
                tp_size=2,
                tp_rank=1,
                ep_size=2,
                ep_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                ffn_tp_size=2,
                ffn_tp_rank=1,
                lm_head_tp_size=2,
                lm_head_tp_rank=1,
                device="cpu",
                compute_dtype=torch.bfloat16,
            )
            from dataclasses import replace

            from rtp_llm.models_py.model_loader import NewModelLoader
            from rtp_llm.models_py.registry import register_model

            key = "_qwen4_rank_local_ft_cpu_owner"
            register_model(key)(Qwen4TargetWeights)
            config.update(model_type=key, num_experts=2)
            load = replace(load, load_method=backend)
            loader = NewModelLoader(config, load, model_path=directory)
            model = loader.load()
            self.assertEqual(loader._selected_load_method.value, backend)
            torch.testing.assert_close(
                model.embed_tokens.weight, ft.read("global.embedding")
            )
            if pruned:
                # Rank 1 needs ID 10 from rank 0 and IDs 29/31 from rank 1.
                full = torch.cat([
                    ft.read_rank_slice("global.lm_head", peer, (slice(None), slice(None)))
                    for peer in range(2)
                ])
                expected = full.new_zeros(8, 16)
                expected[:3] = full[[10, 29, 31]]
                torch.testing.assert_close(model.lm_head.weight, expected)
            else:
                torch.testing.assert_close(model.lm_head.weight, ft.read("global.lm_head"))
            torch.testing.assert_close(
                model.layers[0].self_attn.qkv_proj.weight,
                ft.read("layers.0.self_attention_weights.query_weight.kernel").t(),
            )
            torch.testing.assert_close(
                model.layers[0].mlp.experts.w13,
                ft.read("layers.0.partial_moe_weights.intermediate_weight.kernel"),
            )
            table = model.layers[0].ple.ple_embedding.ngram_embedding
            self.assertEqual(tuple(table._parameters), ("shard_1",))
            self.assertEqual(table.shard_1.device.type, "cpu")
            torch.testing.assert_close(
                table.shard_1, ft.read("layers.0.qwen4.ple.ngram_shards.1")
            )
            if restart:
                # This is a CPU construction/ownership test, not a VMM/device smoke.
                from rtp_llm.models_py.resident_checkpoint import ResidentFTHeader

                ref = ResidentFTHeader.recording(ft)
                Qwen4TargetWeights.from_ft_checkpoint_header(config, load, ref)
                state = json.loads(json.dumps(model.resident_constructor_state()))
                values = dict(model.named_parameters())
                for path in Path(directory).glob("*.safetensors"):
                    path.unlink()
                (Path(directory) / "model.safetensors.index.json").unlink()
                replay = ResidentFTHeader(directory, json.loads(json.dumps(ref.facts)))
                restored = Qwen4TargetWeights.from_ft_checkpoint_header(
                    config, replace(load, resident_construction=state), replay
                )
                restored.prepare_resident_checkpoint_restore(state, values)
                fresh = restored.layers[0].ple.ple_embedding.ngram_embedding
                self.assertEqual(fresh.shard_1.data_ptr(), table.shard_1.data_ptr())
                self.assertEqual(fresh.shard_rows, table.shard_rows)
                restored.layers[0].mlp.experts.validate_weights_loaded()
                restored.close()
            model.close()

    def test_rank1_attention_experts_and_global_parameters(self):
        self.check_rank1("scratch")

    def test_ft_restart_rebuilds_constructor_and_host_ple_without_shards(self):
        self.check_rank1("scratch", restart=True)

    def test_native_fst_rank1(self):
        self.check_rank1("fastsafetensors")

    def test_pruned_ft_reads_original_token_rows_from_peer_ranks(self):
        for backend in ("scratch", "fastsafetensors"):
            with self.subTest(backend=backend):
                self.check_rank1(backend, pruned=True)


if __name__ == "__main__":
    unittest.main()
