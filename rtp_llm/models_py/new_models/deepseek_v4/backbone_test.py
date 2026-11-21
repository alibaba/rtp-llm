"""CPU checkpoint tests; intentionally stop before CUDA-only preparation."""

import json
import os
import tempfile
import types
import unittest

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v4.backbone import (
    OwnedV4Args,
    OwnedV4Transformer,
)
from rtp_llm.models_py.weight_mapper import get_all_weights


class Dsv4BackboneCheckpointTest(unittest.TestCase):
    def args(self, rank=0):
        return OwnedV4Args(
            vocab_size=16,
            dim=256,
            n_heads=2,
            n_layers=3,
            n_mtp_layers=0,
            q_lora_rank=128,
            head_dim=512,
            rope_head_dim=64,
            o_lora_rank=128,
            o_groups=2,
            compress_ratios=[0, 0, 0],
            original_seq_len=0,
            max_batch_size=1,
            max_seq_len=16,
            index_n_heads=2,
            index_head_dim=128,
            index_topk=2,
            tp_size=2,
            tp_rank=rank,
            commit_only=True,
        )

    def weights(self):
        weights = {}
        for index in range(3):
            prefix = f"mtp.{index}.attn."
            weights[prefix + "wkv.weight"] = torch.full(
                (512, 256), index + 1, dtype=torch.float8_e4m3fn
            )
            weights[prefix + "wkv.scale"] = torch.full(
                (4, 2), 126 + index, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            weights[prefix + "kv_norm.weight"] = torch.full(
                (512,), index + 2, dtype=torch.bfloat16
            )
        return weights

    def load(self, weights, rank=0):
        model = OwnedV4Transformer(self.args(rank), layer_prefix="mtp")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pytorch_model.bin")
            torch.save(weights, path)
            loader = NewModelLoader(
                types.SimpleNamespace(model_type="deepseek_v4_dspark"),
                NewLoaderConfig(device="cpu", tp_size=2, tp_rank=rank),
                model_path=folder,
            )
            plan = {}
            resolver = loader._checkpoint_read_spec_resolver(
                model, model.checkpoint_weight_name_filter(), None, plan
            )
            model._rtp_checkpoint_load_plan = plan
            model.load_standalone_checkpoint(
                get_all_weights(
                    [path],
                    read_spec_resolver=resolver,
                    name_filter=model.checkpoint_weight_name_filter(),
                )
            )
        NewModelLoader._validate_loaded_weights(model)
        return model

    def test_real_three_layer_raw_checkpoint_commit_tree_tp2(self):
        weights = self.weights()
        # Full draft checkpoints contain proposal-only tensors; the pre-read
        # role filter must exclude them, not materialize and discard them.
        weights["mtp.0.attn.wq_a.weight"] = torch.ones(1)
        weights["mtp.2.markov_head.markov_w1.weight"] = torch.ones(1)
        weights["layers.0.attn.wkv.weight"] = torch.ones(1)
        for rank in (0, 1):
            model = self.load(weights, rank)
            self.assertEqual(len(list(model.named_parameters())), 9)
            self.assertIsNone(model.embed)
            self.assertIsNone(model.head_weight)
            for index, block in enumerate(model.layers):
                self.assertIsNone(block.ffn)
                self.assertIsNone(block.attn_hc)
                self.assertIsNone(block.attn.wq_a)
                self.assertIsNone(block.attn.wo_b)
                for name, value in block.attn.named_parameters():
                    source = "kv_norm.weight" if name == "kv_norm" else name
                    expected = weights[f"mtp.{index}.attn.{source}"]
                    torch.testing.assert_close(
                        value.view(torch.uint8),
                        expected.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                self.assertEqual(block.attn.wkv.scale.dtype, torch.float8_e8m0fnu)

    def test_missing_required_checkpoint_tensor_never_succeeds(self):
        for suffix in ("wkv.weight", "wkv.scale", "kv_norm.weight"):
            weights = self.weights()
            del weights["mtp.1.attn." + suffix]
            with self.subTest(missing=suffix), self.assertRaisesRegex(
                RuntimeError, "missing required"
            ):
                self.load(weights)

    def test_raw_scale_dtype_is_not_silently_converted(self):
        weights = self.weights()
        weights["mtp.1.attn.wkv.scale"] = torch.ones((4, 2), dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "Dtype mismatch"):
            self.load(weights)

    def test_commit_role_rejects_compressed_attention_and_invalid_partition(self):
        args = self.args()
        args.compress_ratios[1] = 4
        with self.assertRaisesRegex(ValueError, "requires SWA"):
            OwnedV4Transformer(args)
        with self.assertRaisesRegex(ValueError, "TP partition"):
            OwnedV4Transformer(self.args(rank=2))

    def test_commit_attention_cannot_accidentally_run_proposal_or_cpu_backend(self):
        model = self.load(self.weights())
        attention = model.layers[0].attn
        with self.assertRaisesRegex(RuntimeError, "proposal attention"):
            attention(torch.zeros((1, 1, 256), dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            attention.process_weights_after_loading()
        with self.assertRaisesRegex(RuntimeError, "Unexpected V4 commit-only"):
            attention.load_weights({"wq_a.weight": torch.ones(1)})


class Dsv4FullBackboneCheckpointTest(unittest.TestCase):
    def args(self, rank=0):
        args = Dsv4BackboneCheckpointTest().args(rank)
        args.dim = 128
        args.vocab_size = 17
        args.hc_mult = 2
        args.commit_only = False
        args.compress_ratios = [0, 4, 128]
        args.n_routed_experts = 4
        args.n_activated_experts = 2
        args.n_hash_layers = 1
        args.moe_inter_dim = 128
        args.ep_size = 2
        args.ep_rank = rank
        args.world_size = 2
        args.world_rank = rank
        return args

    @staticmethod
    def scales(shape, exponent=127):
        return torch.full(shape, exponent, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    def weights(self):
        weights = {
            "embed.weight": torch.arange(17 * 128).reshape(17, 128).bfloat16(),
            "head.weight": torch.arange(17 * 128).reshape(17, 128).bfloat16() / 16,
            "norm.weight": torch.full((128,), 0.5, dtype=torch.bfloat16),
            "hc_head_fn": torch.full((2, 256), 0.02),
            "hc_head_base": torch.tensor([0.1, 0.2]),
            "hc_head_scale": torch.tensor([0.3]),
        }
        for index, ratio in enumerate((0, 4, 128)):
            prefix = f"layers.{index}."
            for projection, shape in {
                "wq_a": (128, 128),
                "wq_b": (1024, 128),
                "wkv": (512, 128),
                "wo_a": (256, 512),
                "wo_b": (128, 256),
            }.items():
                weights[prefix + f"attn.{projection}.weight"] = torch.full(
                    shape, index + 1, dtype=torch.float8_e4m3fn
                )
                weights[prefix + f"attn.{projection}.scale"] = self.scales(
                    (shape[0] // 128, shape[1] // 128), 125 + index
                )
            weights[prefix + "attn.q_norm.weight"] = torch.ones(
                128, dtype=torch.bfloat16
            )
            weights[prefix + "attn.kv_norm.weight"] = torch.ones(
                512, dtype=torch.bfloat16
            )
            weights[prefix + "attn.attn_sink"] = torch.tensor([0.123456, 0.234567])

            def compressor(stem, head_dim):
                out = head_dim * (2 if ratio == 4 else 1)
                weights[prefix + stem + "ape"] = torch.ones(ratio, out)
                weights[prefix + stem + "wkv.weight"] = torch.ones(
                    out, 128, dtype=torch.bfloat16
                )
                weights[prefix + stem + "wgate.weight"] = torch.ones(
                    out, 128, dtype=torch.bfloat16
                )
                weights[prefix + stem + "norm.weight"] = torch.ones(
                    head_dim, dtype=torch.bfloat16
                )

            if ratio:
                compressor("attn.compressor.", 512)
            if ratio == 4:
                weights[prefix + "attn.indexer.wq_b.weight"] = torch.ones(
                    256, 128, dtype=torch.float8_e4m3fn
                )
                weights[prefix + "attn.indexer.wq_b.scale"] = self.scales((2, 1))
                weights[prefix + "attn.indexer.weights_proj.weight"] = torch.ones(
                    2, 128, dtype=torch.bfloat16
                )
                compressor("attn.indexer.compressor.", 128)
            for kind in ("attn", "ffn"):
                weights[prefix + kind + "_norm.weight"] = torch.ones(
                    128, dtype=torch.bfloat16
                )
                weights[prefix + f"hc_{kind}_fn"] = torch.full((8, 256), 0.01 + index)
                weights[prefix + f"hc_{kind}_base"] = torch.arange(8).float() / 32
                weights[prefix + f"hc_{kind}_scale"] = torch.tensor([0.1, 0.2, 0.3])
            for expert in range(4):
                for projection in ("w1", "w2", "w3"):
                    stem = prefix + f"ffn.experts.{expert}.{projection}."
                    weights[stem + "weight"] = torch.full(
                        (128, 64), expert + index + 1, dtype=torch.int8
                    )
                    weights[stem + "scale"] = self.scales((128, 4), 124 + expert)
            for projection in ("w1", "w2", "w3"):
                stem = prefix + f"ffn.shared_experts.{projection}."
                weights[stem + "weight"] = torch.ones(
                    128, 128, dtype=torch.float8_e4m3fn
                )
                weights[stem + "scale"] = self.scales((1, 1))
            weights[prefix + "ffn.gate.weight"] = torch.ones(
                4, 128, dtype=torch.bfloat16
            )
            if index == 0:
                weights[prefix + "ffn.gate.tid2eid"] = (
                    torch.arange(34).reshape(17, 2) % 4
                ).int()
            else:
                weights[prefix + "ffn.gate.bias"] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        return weights

    def load(self, weights, rank=0, *, cuda_norm=False, cuda_native_rms=False):
        from safetensors.torch import save_file

        args = self.args(rank)
        args.cuda_norm = cuda_norm
        args.cuda_native_rms = cuda_native_rms
        model = OwnedV4Transformer(args)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.safetensors")
            save_file(weights, path)
            loader = NewModelLoader(
                types.SimpleNamespace(model_type="deepseek_v4"),
                NewLoaderConfig(
                    device="cpu", tp_size=2, tp_rank=rank, ep_size=2, ep_rank=rank
                ),
                model_path=folder,
            )
            plan = {}
            resolver = loader._checkpoint_read_spec_resolver(
                model, model.checkpoint_weight_name_filter(), None, plan
            )
            model._rtp_checkpoint_load_plan = plan
            model.load_standalone_checkpoint(
                get_all_weights(
                    [path],
                    read_spec_resolver=resolver,
                    name_filter=model.checkpoint_weight_name_filter(),
                )
            )
        NewModelLoader._validate_loaded_weights(model)
        return model

    def test_cuda_norm_uses_original_prefill_output_workspace_and_loaded_parameter(
        self,
    ):
        from unittest.mock import patch

        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm as CudaRMSNorm
        from rtp_llm.models_py.modules.dsv4 import block

        weights = self.weights()
        model = self.load(weights, cuda_norm=True)
        for module, key in (
            (model.layers[0].attn_norm, "layers.0.attn_norm.weight"),
            (model.layers[0].ffn_norm, "layers.0.ffn_norm.weight"),
            (model.norm, "norm.weight"),
        ):
            self.assertIs(type(module), CudaRMSNorm)
            self.assertIs(module.weight, dict(module.named_parameters())["weight"])
            self.assertTrue(torch.equal(module.weight, weights[key]))
        # Test the runtime decision without loading a CUDA extension on CPU.
        x = torch.empty_like(model.layers[0].attn_norm.weight).unsqueeze(0)
        with patch.object(block, "RMSNorm", CudaRMSNorm), patch.object(
            CudaRMSNorm, "forward", autospec=True, return_value=x
        ) as forward:
            self.assertIs(block._prefill_fast_norm(model.layers[0].attn_norm, x), x)
        self.assertIs(forward.call_args.kwargs["output"], x)

    def test_arm_native_norm_bound_to_checkpointed_modules(self):
        from rtp_llm.models_py.new_models.deepseek_v4.cuda_rms import native_rmsnorm

        weights = self.weights()
        model = self.load(weights, cuda_norm=True, cuda_native_rms=True)
        for node, key in (
            (model.layers[0].attn_norm, "layers.0.attn_norm.weight"),
            (model.layers[0].ffn_norm, "layers.0.ffn_norm.weight"),
            (model.norm, "norm.weight"),
        ):
            self.assertIs(node._implementation, native_rmsnorm)
            self.assertTrue(torch.equal(node.weight, weights[key]))

    def test_real_native_checkpoint_full_swa_csa_hca_moe_hc_tree(self):
        weights = self.weights()
        for rank in (0, 1):
            model = self.load(weights, rank)
            torch.testing.assert_close(model.embed.weight, weights["embed.weight"])
            # LM-head TP=1, EP=2: the distributed loader still pads to 8.
            expected_head = torch.nn.functional.pad(
                weights["head.weight"], (0, 0, 0, 7)
            )
            torch.testing.assert_close(model.head_weight, expected_head)
            torch.testing.assert_close(model.head_hc.fn, weights["hc_head_fn"])
            self.assertIs(model.head_weight, model.head.weight)
            for index, block in enumerate(model.layers):
                torch.testing.assert_close(
                    block.attn_hc.fn, weights[f"layers.{index}.hc_attn_fn"]
                )
                torch.testing.assert_close(
                    block.attn.attn_sink,
                    weights[f"layers.{index}.attn.attn_sink"][rank : rank + 1],
                )
                self.assertEqual(block.ffn.routed_w13.shape, (2, 256, 64))
                self.assertIsNone(block.ffn._runtime)
                for name in block.ffn._expected:
                    actual = block.ffn._target(name)
                    expected = weights[f"layers.{index}.ffn.{name}"]
                    torch.testing.assert_close(
                        actual.contiguous().view(torch.uint8),
                        expected.contiguous().view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )

    def test_full_tree_required_attention_hc_router_expert_global_tensor_failures(self):
        for name in (
            "embed.weight",
            "head.weight",
            "hc_head_scale",
            "layers.0.hc_attn_fn",
            "layers.1.attn.indexer.wq_b.scale",
            "layers.2.attn.compressor.ape",
            "layers.0.ffn.gate.tid2eid",
            "layers.1.ffn.experts.1.w1.scale",
        ):
            weights = self.weights()
            del weights[name]
            with self.subTest(name=name), self.assertRaisesRegex(
                RuntimeError, "missing|required|Missing"
            ):
                self.load(weights)

    def test_typed_config_production_runtime_owns_checkpoint_without_modelweights(self):
        from safetensors.torch import save_file

        from rtp_llm.models.deepseek_v4 import DeepSeekV4
        from rtp_llm.models.direct_checkpoint import _runtime_view
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
        from rtp_llm.models_py.new_models.deepseek_v4.language import (
            DeepSeekV4ForCausalLM,
        )
        from rtp_llm.models_py.registry import get_model_class
        from rtp_llm.ops import KvCacheDataType, ParallelismConfig

        hf_config = dict(
            num_hidden_layers=3,
            hidden_size=128,
            vocab_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=128,
            o_groups=2,
            o_lora_rank=128,
            compress_ratios=[0, 4, 128],
            sliding_window=128,
            index_head_dim=128,
            index_n_heads=2,
            index_topk=2,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.0,
            num_experts_per_tok=2,
            n_routed_experts=4,
            moe_intermediate_size=128,
            n_shared_experts=1,
            hc_mult=2,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            num_hash_layers=1,
        )
        weights = self.weights()
        for name in ("embed.weight", "head.weight", "layers.0.ffn.gate.tid2eid"):
            weights[name] = weights[name][:16].contiguous()
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "config.json"), "w") as stream:
                json.dump(hf_config, stream)
            path = os.path.join(folder, "model.safetensors")
            save_file(weights, path)
            config = DeepSeekV4._create_config(folder)
            config.model_type = "deepseek_v4"
            config.ckpt_path = folder
            config.max_seq_len = 128
            config.gen_num_per_cycle = 1
            config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
            parallel = ParallelismConfig()
            parallel.tp_size = parallel.ep_size = parallel.world_size = 2
            parallel.ffn_tp_size = 2
            for rank in (0, 1):
                parallel.tp_rank = parallel.ep_rank = parallel.world_rank = rank
                parallel.ffn_tp_rank = rank
                load_config = NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.bfloat16,
                    tp_size=2,
                    tp_rank=rank,
                    ep_size=2,
                    ep_rank=rank,
                    parallelism_config=parallel,
                    max_generate_batch_size=3,
                )
                owner = get_model_class(config.model_type)
                self.assertIs(owner, DeepSeekV4ForCausalLM)
                model = owner(config, load_config)
                self.assertIsInstance(model, DeepSeekV4Model)
                self.assertIsNone(model.weight)
                self.assertEqual(model._max_generate_batch_size, 3)
                loader = NewModelLoader(config, load_config, model_path=folder)
                resolver = loader._checkpoint_read_spec_resolver(
                    model, model.checkpoint_weight_name_filter(), None
                )
                # Pre-materialization is a serving requirement, not just
                # skipping a nonlocal expert after the file has been read.
                for expert in range(4):
                    name = f"layers.0.ffn.experts.{expert}.w1.weight"
                    spec = resolver(name, tuple(weights[name].shape))
                    self.assertEqual(spec.load, expert // 2 == rank)
                proj = "layers.0.attn.wq_b.weight"
                spec = resolver(proj, tuple(weights[proj].shape))
                self.assertTrue(spec.is_local_shard)
                model.load_weights(get_all_weights([path], read_spec_resolver=resolver))
                NewModelLoader._validate_loaded_weights(model)
                exports = model.runtime_weight_view()
                self.assertIs(exports["embedding"], model.v4.embed.weight)
                self.assertIs(exports["lm_head"], model.v4.head.weight)
                self.assertEqual(exports["lm_head"].shape, (8, 128))
                torch.testing.assert_close(
                    exports["lm_head"],
                    weights["head.weight"].chunk(2)[rank].to(exports["lm_head"].dtype),
                    rtol=0,
                    atol=0,
                )
                self.assertFalse(model._materialized)
                self.assertTrue(
                    all(block.ffn._runtime is None for block in model.v4.layers)
                )
                parameter_ids = tuple(id(p) for p in model.parameters())
                published = _runtime_view(model, config, "cpu")
                model.weight = object()
                with self.assertRaisesRegex(RuntimeError, "legacy ModelWeights"):
                    model._release_runtime_weight_source()
                model.weight = published
                model._release_runtime_weight_source()
                self.assertIs(model.weight, published)
                self.assertIs(
                    published.get_global_weight("lm_head"), model.v4.head.weight
                )
                resource = types.SimpleNamespace(
                    kv_cache=None,
                    is_speculative=False,
                    is_decode_role=True,
                    max_context_batch_size=1,
                )
                with self.assertLogs(level="ERROR"), self.assertRaisesRegex(
                    RuntimeError, "accelerator post-load"
                ):
                    model.initialize(resource)
                self.assertTrue(
                    model.v4.layers[0].ffn._runtime_kwargs["is_decode_role"]
                )
                self.assertIsNone(model.v4.layers[0].ffn._runtime)
                self.assertEqual(
                    tuple(id(p) for p in model.parameters()), parameter_ids
                )

    def test_runtime_batch_configuration_is_not_guessed(self):
        from rtp_llm.models_py.new_models.deepseek_v4.language import (
            DeepSeekV4ForCausalLM,
        )

        with self.assertRaisesRegex(ValueError, "positive max_generate_batch_size"):
            DeepSeekV4ForCausalLM(None, NewLoaderConfig(device="cpu"))
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "max_generate_batch_size"
            ):
                NewLoaderConfig(max_generate_batch_size=value)


if __name__ == "__main__":
    unittest.main()
