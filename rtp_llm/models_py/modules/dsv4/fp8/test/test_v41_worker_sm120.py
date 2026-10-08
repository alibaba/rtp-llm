"""Native worker integration; small real weights, tokenizer, host table and kernels."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.engram import make_token_windows
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_transformer_sm120 import fixture
from rtp_llm.ops import KvCacheDataType, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs


def checkpoint(root, *, full_schedule=False):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"token{i}": i for i in range(128)}, unk_token="token0")
        ),
        unk_token="token0",
    )
    tokenizer.save_pretrained(root)
    tensors = {
        "embed.weight": (torch.randn(4000, 32) * 0.25).to(torch.float8_e4m3fn),
        "embed.scale": torch.ones(4000, 1).to(torch.float8_e8m0fnu),
        "wkv.weight": (torch.randn(2560, 384) * 0.025).to(torch.float8_e4m3fn),
        "wkv.scale": torch.ones(80, 12).to(torch.float8_e8m0fnu),
        "q_weight": torch.ones(4, 512, dtype=torch.bfloat16),
        "k_weight": torch.ones(4, 512, dtype=torch.bfloat16),
    }
    layers = [1, 14] if full_schedule else [1]
    tensors = {
        f"layers.{layer}.engram.{name}": value.clone()
        for layer in layers
        for name, value in tensors.items()
    }
    save_file(tensors, root / "engram.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "engram.safetensors" for name in tensors}})
    )
    return {
        "checkpoint_path": str(root),
        "num_hidden_layers": 40 if full_schedule else 6,
        "hidden_size": 512,
        "hc_mult": 4,
        "rms_norm_eps": 1e-20,
        "engram_layer_ids": layers,
        "engram_num_embeddings": [4000] * len(layers),
        "engram_max_ngram_size": 4,
        "engram_n_heads": 4,
        "engram_head_dim": 32,
        "engram_compressed_vocab_size": 128,
        "engram_pad_token_id": 2,
        "engram_vocab_size": 13,
    }


def model_config(args, engram, gen_num_per_cycle=0):
    config = ModelConfig()
    fields = {
        "vocab_size": args.vocab_size,
        "hidden_size": args.dim,
        "num_layers": args.n_layers,
        "inter_size": args.moe_inter_dim,
        "expert_num": args.n_routed_experts,
        "moe_k": args.n_activated_experts,
        "scoring_func": 2,
        "routed_scaling_factor": args.route_scale,
        "swiglu_limit": args.swiglu_limit,
        "num_hash_layers": 0,
        "hc_mult": args.hc_mult,
        "hc_sinkhorn_iters": args.hc_sinkhorn_iters,
        "hc_eps": args.hc_eps,
        "layernorm_eps": args.norm_eps,
        "max_seq_len": args.max_seq_len,
        "gen_num_per_cycle": gen_num_per_cycle,
        "ckpt_path": engram["checkpoint_path"],
        "deepseek_v41_config": {**args.v41_config, **engram},
    }
    for name, value in fields.items():
        setattr(config, name, value)
    for name, value in {
        "head_num": args.n_heads,
        "q_lora_rank": args.q_lora_rank,
        "size_per_head": args.head_dim,
        "rope_head_dim": args.rope_head_dim,
        "o_groups": args.o_groups,
        "o_lora_rank": args.o_lora_rank,
        "sliding_window": args.window_size,
        "layer_compress_ratios": args.compress_ratios,
        "compress_rope_theta": args.compress_rope_theta,
        "indexer_head_num": args.index_n_heads,
        "indexer_head_dim": args.index_head_dim,
        "indexer_topk": args.index_topk,
        "kv_cache_dtype": KvCacheDataType.FP8,
    }.items():
        setattr(config.attn_config, name, value)
    for name, value in {
        "base": args.rope_theta,
        "scale": args.rope_factor,
        "factor2": args.beta_fast,
        "factor1": args.beta_slow,
        "max_pos": args.original_seq_len,
    }.items():
        setattr(config.attn_config.rope_config, name, value)
    if args.n_layers == 40:
        from rtp_llm.models.dsv41_kv_cache import configure_dsv41_kv_cache

        config.attn_config.tokens_per_block = 256
        config.attn_config.kernel_tokens_per_block = 256
        configure_dsv41_kv_cache(config)
    return config


def resources(cache=None, **kwargs):
    # PyModelInitResources has no writable cache/role constructor. This resource
    # envelope supplies the same fields to the actual base initializer.
    return SimpleNamespace(
        **{
            "kv_cache": cache,
            "is_speculative": False,
            "is_decode_role": False,
            "max_context_batch_size": 2,
            **kwargs,
        }
    )


def tagged_cache(model):
    cache = NativeTaggedCache(
        [layer.attn for layer in model.v4.layers], gen=model._gen_num_per_cycle
    )
    cache.group_tags = list(cache.layouts)
    cache.layer_count = len(model.v4.layers)
    cache.get_layer_cache_groups = lambda layer: [
        group for (index, _), group in cache.groups.items() if index == layer
    ]
    return cache


def inputs(cache, ids=(3, 4, 5, 6, 7), lengths=(3, 2), prefixes=None, history=None):
    ids = torch.tensor(ids, device="cuda", dtype=torch.int32)
    lens = torch.tensor(lengths, dtype=torch.int32)
    pref = torch.tensor(prefixes or [0] * len(lengths), dtype=torch.int32)
    cu = torch.cat((torch.zeros(1, dtype=torch.int32), lens.cumsum(0).int()))
    tagged = {}
    for tag in cache.group_tags:
        attn = PyAttentionInputs()
        attn.is_prefill = True
        attn.input_lengths = lens
        attn.prefix_lengths = pref
        attn.cu_seqlens = cu
        attn.kv_cache_kernel_block_id_device = torch.tensor(
            [[2], [1]][: len(lengths)], device="cuda", dtype=torch.int32
        )
        tagged[tag] = attn
    lookback = torch.tensor(history or [[-1] * 3] * len(lengths), dtype=torch.int32)
    windows = make_token_windows(ids, cu, lookback).int().contiguous()
    return PyModelInputs(
        input_ids=ids, attention_inputs=tagged, engram_token_windows=windows
    )


class V41WorkerSM120Test(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41122)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.config = checkpoint(Path(temp.name))
        env = patch.dict(
            os.environ,
            {
                "DSV41_ENGRAM_HOST_BYTES": str(2**20),
                "DSV4_MOE_STRATEGY": "auto",
                "MOEDBG": "0",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        self.addCleanup(Dsv4SharedRuntimeBufferStore._reset_for_test)

    def worker(self, initialize=True, engram=None):
        args, weights = fixture()
        cfg = model_config(args, self.config if engram is None else engram)
        parallelism = ParallelismConfig()
        parallelism.role_type = RoleType.PREFILL
        model = DeepSeekV41Model(cfg, parallelism, weights, None, 2)
        self.addCleanup(model.close)
        if initialize:
            self.assertTrue(model.initialize(resources()))
            for layer in model.v4.layers:
                self.assertEqual(layer.attn._cache_layouts["csa_state"].entries, 2)
                self.assertEqual(layer.attn._cache_layouts["swa_kv"].entries, 128)
            self.assertTrue(model.initialize(resources(tagged_cache(model))))
        return model

    def assert_released(self, model):
        for layer in model.v4.layers:
            self.assertIsNone(layer.engram_hashes)
            self.assertIsNone(layer.engram_token_mask)
            self.assertIsNone(layer.attn_hc.pre_mix_out)
            self.assertIsNone(layer.ffn_hc.pre_mix_out)
            self.assertIsNone(layer.attn._prefill_meta_shared)
        shared = model.v4.layers[0].attn._shared_attention
        self.assertEqual(shared["global"], {})
        self.assertEqual(shared["topk"], {})
        self.assertIsNone(shared["candidates"])

    def test_native_worker_history_fast_normal_and_continuation(self):
        model = self.worker()
        self.assertEqual([layer.layer_id for layer, _ in model._engram_layers], [1])
        self.assertEqual(
            model.v4.layers[1].engram.embed_tokens.weight.device.type, "cpu"
        )
        self.assertFalse(hasattr(model, "weight"))
        actual = []
        caches = []
        for fast in ("0", "1"):
            model.kv_cache = tagged_cache(model)
            with patch.dict(os.environ, {"DSV4_PREFILL_FAST_PATH": fast}):
                first = model.forward(inputs(model.kv_cache)).hidden_states.clone()
                self.assert_released(model)
                second = model.forward(
                    inputs(
                        model.kv_cache,
                        (8, 9, 10),
                        (1, 2),
                        [3, 2],
                        [[5, 4, 3], [7, 6, -1]],
                    )
                ).hidden_states.clone()
                self.assert_released(model)
            actual.append((first, second))
            caches.append({k: v.clone() for k, v in model.kv_cache.storage.items()})
        for left, right in zip(*actual):
            self.assertTrue(torch.isfinite(left).all())
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for key in caches[0]:
            self.assertTrue(torch.equal(caches[0][key], caches[1][key]))
        model.kv_cache = tagged_cache(model)
        changed = inputs(model.kv_cache, history=[[11, 12, 13], [14, 15, 16]])
        self.assertFalse(
            torch.equal(actual[0][0], model.forward(changed).hidden_states)
        )
        self.assert_released(model)

    def test_invalid_history_and_native_modes_do_not_write_cache(self):
        model = self.worker()
        baseline = {k: v.clone() for k, v in model.kv_cache.storage.items()}
        for bad in (
            None,
            torch.zeros(5, 3, dtype=torch.int32),
            torch.zeros(5, 4, dtype=torch.int64),
            torch.full((5, 4), 128, dtype=torch.int32),
            torch.zeros(5, 4, dtype=torch.int32),
            torch.zeros(4, 5, dtype=torch.int32).t(),
        ):
            request = inputs(model.kv_cache)
            if bad is None:
                request = PyModelInputs(
                    input_ids=request.input_ids,
                    attention_inputs=request.attention_inputs,
                )
            else:
                request.engram_token_windows = bad
            with self.assertRaises(ValueError):
                model.forward(request)
            self.assert_released(model)
        for field in ("is_prefill", "is_target_verify", "is_cuda_graph"):
            request = inputs(model.kv_cache)
            for attn in request.attention_inputs.values():
                setattr(attn, field, field != "is_prefill")
            with self.assertRaises(NotImplementedError):
                model.forward(request)
        request = inputs(model.kv_cache)
        request.multimodal_inputs.multimodal_features = [torch.zeros(1, device="cuda")]
        with self.assertRaisesRegex(NotImplementedError, "image inputs"):
            model.forward(request)
        request = inputs(model.kv_cache)
        with self.assertRaises(NotImplementedError):
            model.prepare_fmha_impl(request, is_cuda_graph=True)
        with self.assertRaises(NotImplementedError):
            model.forward(request, object())
        for ids in (
            torch.empty(0, device="cuda", dtype=torch.int32),
            torch.tensor([-1], device="cuda", dtype=torch.int32),
        ):
            request.input_ids = ids
            with self.assertRaisesRegex(ValueError, "token IDs"):
                model.forward(request)
        for key, tensor in baseline.items():
            self.assertTrue(torch.equal(tensor, model.kv_cache.storage[key]))
        self.assertTrue(
            torch.isfinite(model.forward(inputs(model.kv_cache)).hidden_states).all()
        )
        self.assert_released(model)

    def test_failure_after_engram_clears_request_state_and_allows_retry(self):
        model = self.worker()
        layer = model.v4.layers[2]
        with patch.object(
            layer, "forward", side_effect=RuntimeError("injected block failure")
        ):
            with patch.dict(os.environ, {"DSV4_PREFILL_FAST_PATH": "0"}):
                with self.assertRaisesRegex(RuntimeError, "injected block failure"):
                    model.forward(inputs(model.kv_cache))
        self.assert_released(model)
        # Kernel failures do not roll back pages; a new allocation is required.
        model.kv_cache = tagged_cache(model)
        self.assertTrue(
            torch.isfinite(model.forward(inputs(model.kv_cache)).hidden_states).all()
        )

    def test_request_boundaries_and_stream_rejected_before_cache_writes(self):
        model = self.worker()
        baseline = {k: v.clone() for k, v in model.kv_cache.storage.items()}
        for field, value, error in (
            ("input_lengths", [3, 0], ValueError),
            ("input_lengths", [3, 3], ValueError),
            ("prefix_lengths", [-1, 0], ValueError),
            ("prefix_lengths", [254, 0], NotImplementedError),
            ("cu_seqlens", [0, 2, 5], ValueError),
            ("combo_position_ids", [0, 1, 2, 9, 10], ValueError),
        ):
            request = inputs(model.kv_cache)
            for attn in request.attention_inputs.values():
                setattr(attn, field, torch.tensor(value, dtype=torch.int32))
            with self.assertRaises(error):
                model.forward(request)
            self.assert_released(model)
        request = inputs(model.kv_cache)
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(
                NotImplementedError, "initialization CUDA stream"
            ):
                model.forward(request)
        for key, tensor in baseline.items():
            self.assertTrue(torch.equal(tensor, model.kv_cache.storage[key]))
        # Both host metadata fallback and the pinned native mirror are admitted.
        for cu in (
            torch.empty(0, dtype=torch.int32),
            torch.tensor([0, 3, 5], dtype=torch.int32).pin_memory(),
        ):
            request = inputs(model.kv_cache)
            for attn in request.attention_inputs.values():
                attn.cu_seqlens = cu
            self.assertTrue(torch.isfinite(model.forward(request).hidden_states).all())

    def test_initialization_failure_releases_registered_table(self):
        model = self.worker(initialize=False)
        owners = []

        def fail_projection(*args):
            owners.extend(model._engram_tables.values())
            raise RuntimeError("injected projection initialization failure")

        with patch(
            "rtp_llm.models_py.model_desc.deepseek_v41_model.V41MXFP8Linear",
            side_effect=fail_projection,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected projection"):
                model.initialize(resources())
        self.assertEqual(len(owners), 1)
        self.assertTrue(owners[0]._closed)
        self.assertIsNone(owners[0]._mapping)
        self.assertFalse(owners[0]._path.exists())
        self.assertEqual(model._engram_tables, {})
        with self.assertRaisesRegex(RuntimeError, "closed"):
            model.initialize(resources())

    def test_budget_and_capacity_reject_before_base_weight_consumption(self):
        model = self.worker(initialize=False)
        weights = model.weight
        keys = [set(layer) for layer in weights.weights]
        with patch.dict(os.environ, {"DSV41_ENGRAM_HOST_BYTES": "0"}):
            with self.assertRaisesRegex(ValueError, "host bytes"):
                model.initialize(resources())
        with patch(
            "rtp_llm.models_py.model_desc.deepseek_v41_model.shutil.disk_usage",
            return_value=SimpleNamespace(free=0),
        ):
            with self.assertRaisesRegex(RuntimeError, "combined tables"):
                model.initialize(resources())
        self.assertIs(model.weight, weights)
        self.assertEqual(keys, [set(layer) for layer in weights.weights])
        self.assertIsNone(model.v4)
        self.assertEqual(model._engram_tables, {})
        self.assertTrue(model.initialize(resources()))

    def test_shared_table_leases_close_and_failed_unregister_retry(self):
        first = self.worker()
        second = self.worker()
        table = first._engram_tables[1]
        other = second._engram_tables[1]
        path = table._path
        self.assertEqual(path, other._path)
        runtime = table._runtime
        table._runtime = SimpleNamespace(cudaHostUnregister=lambda _: (1,))
        try:
            with self.assertRaisesRegex(RuntimeError, "cudaHostUnregister"):
                first.close()
            self.assertIs(first._engram_tables[1], table)
            self.assertIsNotNone(table._mapping)
            self.assertTrue(path.exists())
            with self.assertRaisesRegex(RuntimeError, "closing"):
                first.forward(inputs(first.kv_cache))
        finally:
            table._runtime = runtime
        first.close()
        first.close()
        self.assertIsNone(table._mapping)
        self.assertTrue(path.exists())
        self.assertTrue(
            torch.isfinite(second.forward(inputs(second.kv_cache)).hidden_states).all()
        )
        second.close()
        self.assertFalse(path.exists())

    def test_no_owned_engram_does_not_open_checkpoint(self):
        cfg = {
            **self.config,
            "engram_layer_ids": [14],
            "checkpoint_path": "/missing/v41",
        }
        model = self.worker(engram=cfg)
        self.assertEqual(model._engram_tables, {})
        request = inputs(model.kv_cache)
        request.engram_token_windows = torch.empty(0, dtype=torch.int32)
        self.assertTrue(torch.isfinite(model.forward(request).hidden_states).all())

    def test_unsupported_initialization_preserves_weights(self):
        model = self.worker(initialize=False)
        for field in ("tp_size", "ep_size", "dp_size", "pp_size"):
            with patch.object(model._v4_args, field, 2):
                with self.assertRaisesRegex(NotImplementedError, "PP1/CP1"):
                    model.initialize(resources())
        for field in ("is_decode_role", "is_speculative"):
            with self.assertRaises(NotImplementedError):
                model.initialize(resources(**{field: True}))
        for gen in (-1, 2, 3):
            with patch.object(model, "_gen_num_per_cycle", gen):
                with self.assertRaises(NotImplementedError):
                    model.initialize(resources())
        self.assertIsNone(model.v4)
        self.assertTrue(hasattr(model, "weight"))

    def test_nas_header_only_combined_admission(self):
        root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        cfg = json.loads((root / "config.json").read_text())["text_config"]
        model = self.worker(initialize=False)
        model._v4_args.v41_config = {**cfg, "checkpoint_path": str(root)}
        model.layer_num = cfg["num_hidden_layers"]
        with patch.dict(os.environ, {"DSV41_ENGRAM_HOST_BYTES": str(256 * 2**30)}):
            model._engram_sources = model._plan_engram()
        self.assertEqual(set(model._engram_sources), {1, 14})
        total = sum(s.table.nbytes for s in model._engram_sources.values())
        self.assertAlmostEqual(total / 2**30, 188.833, places=3)
        # Deterministic admission failure without copying any NAS payload.
        with patch(
            "rtp_llm.models_py.model_desc.deepseek_v41_model.shutil.disk_usage",
            return_value=SimpleNamespace(free=total - 1),
        ):
            with self.assertRaisesRegex(RuntimeError, "combined tables"):
                model._check_engram_capacity()
        self.assertEqual(model._engram_tables, {})


if __name__ == "__main__":
    unittest.main()
