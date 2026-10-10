"""Named checkpoint loading into the upstream V4.1 worker with bounded weights."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.models.deepseek_v41 import DeepSeekV41, DeepSeekV41Weight
from rtp_llm.models.dsv41_kv_cache import configure_dsv41_kv_cache
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_transformer_sm120 import fixture
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    inputs,
    resources,
    tagged_cache,
)
from rtp_llm.ops import HWKernelConfig, KVCacheConfig, ParallelismConfig, RoleType
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


def save_checkpoint(root, args, weights):
    engram = checkpoint(root, full_schedule=True, hidden_size=args.dim)
    text = {
        **args.v41_config,
        **engram,
        "vocab_size": args.vocab_size,
        "moe_intermediate_size": args.moe_inter_dim,
        "num_attention_heads": args.n_heads,
        "num_key_value_heads": 1,
        "qk_rope_head_dim": args.rope_head_dim,
        "q_lora_rank": args.q_lora_rank,
        "o_lora_rank": args.o_lora_rank,
        "o_groups": args.o_groups,
        "max_position_embeddings": args.max_seq_len,
        "n_routed_experts": args.n_routed_experts,
        "n_shared_experts": args.n_shared_experts,
        "num_experts_per_tok": args.n_activated_experts,
        "scoring_func": args.score_func,
        "norm_topk_prob": True,
        "routed_scaling_factor": args.route_scale,
        "swiglu_limit": args.swiglu_limit,
        "hc_sinkhorn_iters": args.hc_sinkhorn_iters,
        "hc_eps": args.hc_eps,
        "index_n_heads": args.index_n_heads,
        "index_topk": args.index_topk,
        "rope_theta": args.rope_theta,
        "compress_rope_theta": args.compress_rope_theta,
        "rope_scaling": {
            "factor": args.rope_factor,
            "beta_fast": args.beta_fast,
            "beta_slow": args.beta_slow,
            "original_max_position_embeddings": args.original_seq_len,
        },
    }
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV41ForCausalLM"],
                "dtype": "bfloat16",
                "image_token_id": 127,
                "text_config": text,
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [32, 32],
                    "scale_fmt": "ue8m0",
                    "expert_dtype": "fp4",
                },
            }
        )
    )
    names = {
        W.v4_attn_norm: "attn_norm.weight",
        W.v4_attn_q_norm: "attn.q_norm.weight",
        W.v4_attn_kv_norm: "attn.kv_norm.weight",
        W.v4_attn_sink: "attn.attn_sink",
        W.v4_ffn_norm: "ffn_norm.weight",
        W.v4_router_w: "ffn.gate.weight",
        W.v4_router_bias: "ffn.gate.bias",
        W.v4_compressor_wkv: "attn.compressor.wkv.weight",
        W.v4_compressor_wgate: "attn.compressor.wgate.weight",
        W.v4_compressor_norm: "attn.compressor.norm.weight",
        W.v41_indexer_wk: "attn.indexer.wk.weight",
        W.v41_indexer_k_norm: "attn.indexer.k_norm.weight",
        W.v4_indexer_weights_proj_w: "attn.indexer.weights_proj.weight",
    }
    for part in ("attn", "ffn"):
        for kind in ("base", "fn", "scale"):
            names[getattr(W, f"v4_hc_{part}_{kind}")] = f"hc_{part}_{kind}"
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        for kind, suffix in (("w", "weight"), ("s", "scale")):
            names[getattr(W, f"v4_attn_{name}_{kind}")] = f"attn.{name}.{suffix}"
    for kind, suffix in (("w", "weight"), ("s", "scale")):
        names[getattr(W, f"v4_indexer_wq_b_{kind}")] = f"attn.indexer.wq_b.{suffix}"

    payload = {
        name: weights.global_weights[key].cpu()
        for key, name in (
            (W.embedding, "embed.weight"),
            (W.lm_head, "head.weight"),
            (W.final_ln_gamma, "norm.weight"),
        )
    }
    for layer_id, layer in enumerate(weights.weights):
        prefix = f"layers.{layer_id}."
        payload.update(
            {
                prefix + name: layer[key].cpu()
                for key, name in names.items()
                if key in layer
            }
        )
        for kind, suffix in (("w", "weight"), ("s", "scale")):
            w1, w3 = layer[getattr(W, f"v4_shared_w13_{kind}")].chunk(2, dim=0)
            w2 = layer[getattr(W, f"v4_shared_w2_{kind}")]
            for part, value in (("w1", w1), ("w2", w2), ("w3", w3)):
                payload[prefix + f"ffn.shared_experts.{part}.{suffix}"] = (
                    value.cpu().contiguous()
                )
            dtype = torch.int8 if kind == "w" else torch.float8_e8m0fnu
            for part in ("w1", "w2", "w3"):
                values = layer[getattr(W, f"v4_routed_{part}_{kind}")]
                for expert, value in enumerate(values):
                    payload[prefix + f"ffn.experts.{expert}.{part}.{suffix}"] = (
                        value.cpu().view(dtype)
                    )
    save_file(payload, root / "text.safetensors")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"].update({name: "text.safetensors" for name in payload})
    (root / "model.safetensors.index.json").write_text(json.dumps(index))


def load_configuration(root, *, max_seq_len=256):
    config = DeepSeekV41._create_config(str(root))
    config.model_type = "deepseek_v41"
    config.ckpt_path = str(root)
    config.phy2log_path = None
    config.max_seq_len = max_seq_len
    config.gen_num_per_cycle = 0
    config.enable_fp32_lm_head = False
    kv_config = KVCacheConfig()
    kv_config.fp8_kv_cache = True
    config.init_precision_config(kv_config, "bf16")
    config.attn_config.tokens_per_block = 256
    config.attn_config.kernel_tokens_per_block = 256
    configure_dsv41_kv_cache(config)
    return config, kv_config


def make_loader(config, kv_config, ep_size, rank):
    parallelism = ParallelismConfig()
    parallelism.role_type = RoleType.PREFILL
    for field in ("world_size", "dp_size", "ep_size", "local_world_size"):
        setattr(parallelism, field, ep_size)
    for field in ("world_rank", "dp_rank", "ep_rank", "local_rank"):
        setattr(parallelism, field, rank)
    descriptor = DeepSeekV41Weight(config, parallelism, HWKernelConfig(), kv_config)
    # Device setup parses server flags; unittest selectors are not server flags.
    with patch.object(sys, "argv", [sys.argv[0]]):
        loader = ModelLoader(
            config, descriptor, [], CkptDatabase(config.ckpt_path), LoadMethod.SCRATCH
        )
    return parallelism, loader


class V41LoaderWorkerSM120Test(unittest.TestCase):
    def test_named_loader_bytes_expert_ownership_and_worker_output(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41980)
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        self.addCleanup(Dsv4SharedRuntimeBufferStore._reset_for_test)
        args, expected = fixture(experts=16, full_schedule=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_checkpoint(root, args, expected)
            config, kv_config = load_configuration(root)
            loaded = None
            for ep_size, rank in ((8, 0), (8, 7), (1, 0)):
                parallelism, loader = make_loader(config, kv_config, ep_size, rank)
                names, _ = loader._generate_weight_info()
                self.assertFalse(any("engram" in name for name in names))
                loaded = loader.load_weights(f"cuda:{torch.cuda.current_device()}")
                for layer_id, actual in enumerate(loaded.weights):
                    self.assertTrue(actual)
                    for key, value in actual.items():
                        oracle = expected.weights[layer_id][key]
                        if key.startswith("v4.routed."):
                            count = args.n_routed_experts // ep_size
                            oracle = oracle[rank * count : (rank + 1) * count]
                        if key in (W.v4_hc_attn_scale, W.v4_hc_ffn_scale):
                            # AtomicWeight preserves named scales as matrices.
                            oracle = oracle.reshape(3, 1)
                        torch.testing.assert_close(
                            value.view(torch.uint8),
                            oracle.view(torch.uint8),
                            rtol=0,
                            atol=0,
                            msg=f"ep={ep_size} rank={rank} layer={layer_id} key={key}",
                        )
                for key, value in expected.global_weights.items():
                    torch.testing.assert_close(
                        loaded.global_weights[key], value, rtol=0, atol=0
                    )
                if ep_size != 1:
                    del loaded
            with patch.dict(
                os.environ,
                {
                    "DSV41_ENGRAM_HOST_BYTES": str(2**20),
                    "DSV41_ENGRAM_CACHE_DIR": str(root),
                    "DSV4_MOE_STRATEGY": "auto",
                    "MOEDBG": "0",
                },
            ):
                workers = []
                try:
                    for weights in (expected, loaded):
                        worker = DeepSeekV41Model(config, parallelism, weights, None, 2)
                        workers.append(worker)
                        worker.initialize(resources())
                        worker.initialize(resources(tagged_cache(worker)))
                    oracle = (
                        workers[0]
                        .forward(inputs(workers[0].kv_cache))
                        .hidden_states.clone()
                    )
                    actual = (
                        workers[1].forward(inputs(workers[1].kv_cache)).hidden_states
                    )
                    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
                finally:
                    for worker in workers:
                        worker.close()
            print(
                "V4.1 named loader EP ownership, byte parity and worker prefill PASS",
                flush=True,
            )

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_NAS_LOAD") == "1",
        "full NAS EP8 shard loading requires explicit device-memory admission",
    )
    def test_nas_ep8_shard_load_and_sampled_bytes(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        free, _ = torch.cuda.mem_get_info()
        self.assertGreater(free, 60 * 2**30)
        torch.cuda.reset_peak_memory_stats()
        root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        config, kv_config = load_configuration(root)
        _, loader = make_loader(config, kv_config, 8, 7)
        names, _ = loader._generate_weight_info()
        self.assertFalse(
            any(
                "engram" in name or "vision" in name or name.startswith("mtp.")
                for name in names
            )
        )
        expected_experts = set(range(config.expert_num * 7 // 8, config.expert_num))
        self.assertEqual(
            {int(name.split(".")[4]) for name in names if ".experts." in name},
            expected_experts,
        )
        started = time.monotonic()
        weights = loader.load_weights(f"cuda:{torch.cuda.current_device()}")
        self.assertEqual(len(weights.weights), 40)
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for layer in (0, 39):
            for local, expert in (
                (0, min(expected_experts)),
                (-1, max(expected_experts)),
            ):
                for part in ("w1", "w2", "w3"):
                    for kind, suffix in (("w", "weight"), ("s", "scale")):
                        name = f"layers.{layer}.ffn.experts.{expert}.{part}.{suffix}"
                        with safe_open(
                            root / index[name], framework="pt", device="cpu"
                        ) as shard:
                            expected = shard.get_tensor(name)
                        actual = weights.weights[layer][
                            getattr(W, f"v4_routed_{part}_{kind}")
                        ][local]
                        torch.testing.assert_close(
                            actual.cpu().view(torch.uint8),
                            expected.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
        stored = sum(
            value.nbytes
            for group in [weights.global_weights, *weights.weights]
            for value in group.values()
        )
        peak = torch.cuda.max_memory_allocated()
        self.assertGreater(stored, 40 * 2**30)
        self.assertLess(peak, 60 * 2**30)
        print(
            f"V4.1 NAS EP8 rank7 loader PASS bytes={stored} peak_allocated={peak} "
            f"seconds={time.monotonic() - started:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
