"""Distributed SP/zigzag contracts, including tiny and ragged requests."""

import os
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.sequence_parallel import (
    shard_tokens,
    token_shard_layout,
)
from rtp_llm.models_py.distributed.zigzag_token_layout import ZigzagTokenLayout

CASES = ([1], [2, 3, 17], [15, 16, 17, 127, 257], [8191, 8193], [131072] * 8)


def _expected_cp(x, lengths, size, rank):
    parts = []
    offset = 0
    for length in lengths:
        pair = (length + 2 * size - 1) // (2 * size)
        padded = x.new_zeros((2 * size * pair, *x.shape[1:]))
        padded[:length] = x[offset : offset + length]
        parts.extend(
            (
                padded[rank * pair : (rank + 1) * pair],
                padded[(2 * size - 1 - rank) * pair : (2 * size - rank) * pair],
            )
        )
        offset += length
    return torch.cat(parts)


class LayoutTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "GLM53_PREFILL_SHARED_EXPERT_LOCAL": "1",
            "GLM53_MOE_TP_TOKEN_SHARD": "1",
            "GLM5_MOE_GATE_CHUNK_ROWS": "0",
        },
    )
    def test_mtp_fused_shared_constructor_selects_token_sharding(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
        from rtp_llm.models_py.modules.glm5_mega_moe import mega_moe_fp8_se_wrapper
        from rtp_llm.ops import MoeConfig, ParallelismConfig, RoleType
        from rtp_llm.utils.model_weight import W

        config = ModelConfig()
        config.model_type = "glm_5_mtp"
        config.is_glm53_mtp = True
        config.hidden_size = 4
        config.inter_size = 8
        config.expert_num = 288
        config.moe_k = 8
        config.moe_style = 2
        backend = torch.nn.Identity()
        backend.expert_num = 36
        moe = MoeConfig()
        moe.moe_strategy = "mega_moe_fp8_se"
        for role in (RoleType.PREFILL, RoleType.DECODE):
            parallelism = ParallelismConfig()
            parallelism.role_type = role
            parallelism.world_size = parallelism.local_world_size = (
                parallelism.ep_size
            ) = 8
            parallelism.tp_size = 8 if role == RoleType.PREFILL else 1
            parallelism.ffn_tp_size = parallelism.tp_size
            parallelism.dp_size = 1 if role == RoleType.PREFILL else 8
            with patch.object(
                mega_moe_fp8_se_wrapper, "MegaMoeFp8SEWrapper", return_value=backend
            ):
                layer = GenericMoeLayer(
                    config, parallelism, {W.moe_gate: torch.zeros(4, 288)}, moe
                )
            self.assertEqual(layer.num_local_experts, 36)
            self.assertTrue(layer._use_mega_moe_fused_shared)
            self.assertEqual(layer.route_local_tokens, role == RoleType.PREFILL)
            self.assertEqual(layer.shared_expert_local, role == RoleType.PREFILL)
            self.assertEqual(layer.routed_tp_size, parallelism.tp_size)
            self.assertEqual(layer.ffn_tp_size, parallelism.tp_size)
            clone = layer.clone_for_cuda_graph()
            self.assertEqual(clone.route_local_tokens, layer.route_local_tokens)
            self.assertEqual(clone.shared_expert_local, layer.shared_expert_local)
            if role == RoleType.PREFILL:
                config.is_glm53_mtp = False
                with self.assertRaisesRegex(ValueError, "does not support EP"):
                    GenericMoeLayer(
                        config, parallelism, {W.moe_gate: torch.zeros(4, 288)}, moe
                    )
                config.is_glm53_mtp = True

    def test_glm53_mtp_parallel_selection_does_not_change_legacy_mtp(self):
        from rtp_llm.models.glm53_prefill_parallel import (
            glm53_moe_enabled,
            mla_cp_enabled,
            shared_expert_local_enabled,
        )
        from rtp_llm.ops import RoleType

        with patch.dict(
            os.environ,
            {
                "GLM53_PREFILL_SHARED_EXPERT_LOCAL": "1",
                "GLM53_PREFILL_MLA_CP": "1",
            },
        ):
            for is_glm53 in (False, True):
                config = SimpleNamespace(model_type="glm_5_mtp", is_glm53_mtp=is_glm53)
                self.assertEqual(glm53_moe_enabled(config), is_glm53)
                for role in (RoleType.PREFILL, RoleType.DECODE):
                    self.assertEqual(
                        shared_expert_local_enabled(
                            config.model_type, role, is_glm53_mtp=is_glm53
                        ),
                        is_glm53 and role == RoleType.PREFILL,
                    )
                    self.assertFalse(mla_cp_enabled(config.model_type, role))
                    self.assertEqual(
                        mla_cp_enabled(config.model_type, role, is_glm53_mtp=is_glm53),
                        is_glm53 and role == RoleType.PREFILL,
                    )

    @unittest.skipUnless(
        os.environ.get("GLM5_CKPT_PATH"), "real MTP checkpoint manifest"
    )
    @patch.dict(
        os.environ,
        {"GLM53_PREFILL_SHARED_EXPERT_LOCAL": "1", "GLM53_PREFILL_MLA_CP": "1"},
    )
    def test_glm53_mtp_quantized_shared_weights_and_ep_manifest(self):
        import json
        from pathlib import Path

        from safetensors import safe_open

        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight
        from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
        from rtp_llm.model_loader.weight_module import CompositeWeight
        from rtp_llm.models.deepseek_v2 import Glm5Mtp, Glm5MtpWeight
        from rtp_llm.ops import HWKernelConfig, ParallelismConfig, RoleType
        from rtp_llm.utils.model_weight import W

        checkpoint = Path(os.environ["GLM5_CKPT_PATH"])
        weight_map = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]
        for role in (RoleType.PREFILL, RoleType.DECODE):
            config = Glm5Mtp._create_config(str(checkpoint))
            config.model_type = "glm_5_mtp"
            config.ckpt_path = str(checkpoint)
            config.init_precision_config(KVCacheConfig(), "BF16")
            parallelism = ParallelismConfig()
            parallelism.role_type = role
            parallelism.world_size = parallelism.local_world_size = (
                parallelism.ep_size
            ) = 8
            parallelism.tp_size = 8 if role == RoleType.PREFILL else 1
            parallelism.ffn_tp_size = parallelism.tp_size
            parallelism.dp_size = 1 if role == RoleType.PREFILL else 8
            loader = Glm5MtpWeight(
                config, parallelism, HWKernelConfig(), KVCacheConfig()
            )
            loader._process_meta({}, set(weight_map))
            info = loader.get_weight_info()
            self.assertEqual(len(info.layer_weights), 1)
            found = {}

            def check(weight):
                if isinstance(weight, CompositeWeight):
                    for child in weight.sub_weights.values():
                        check(child)
                    return
                found[weight.name] = weight
                for ckpt in weight.weights:
                    name = ckpt.name.format(i=0, expert_id=0)
                    self.assertIn(name, weight_map)
                    self.assertIn(".layers.45.", name)
                if isinstance(weight, FfnAtomicWeight):
                    self.assertEqual(
                        weight.config.replicate_for_prefill_tokens,
                        role == RoleType.PREFILL,
                    )
                if (
                    isinstance(weight, MlaAttnAtomicWeight)
                    and weight.config is not None
                ):
                    self.assertEqual(
                        weight.config.replicate_for_prefill_cp,
                        role == RoleType.PREFILL,
                    )
                if isinstance(weight, MoeAtomicWeight):
                    self.assertFalse(
                        getattr(weight.config, "replicate_for_prefill_tokens", False)
                    )

            for weight in info.layer_weights[0]:
                check(weight)
            self.assertEqual(found[W.moe_gate].data_type, torch.float32)
            for name, ckpt_suffix in (
                (W.ffn_w2, "mlp.shared_experts.down_proj.weight"),
                (W.ffn_s2, "mlp.shared_experts.down_proj.weight_scale_inv"),
                (W.mla_q_b_w, "self_attn.q_b_proj.weight"),
                (W.mla_q_b_s, "self_attn.q_b_proj.weight_scale_inv"),
                (W.attn_o_w, "self_attn.o_proj.weight"),
                (W.attn_o_s, "self_attn.o_proj.weight_scale_inv"),
            ):
                ckpt_name = "model.language_model.layers.45." + ckpt_suffix
                with safe_open(
                    str(checkpoint / weight_map[ckpt_name]),
                    framework="pt",
                    device="cpu",
                ) as reader:
                    raw = reader.get_tensor(ckpt_name)
                if role == RoleType.PREFILL:
                    # The actual FP8 bytes and FP32 scales survive TP8 loading.
                    for rank in range(8):
                        loaded = found[name]._split(
                            raw, SimpleNamespace(tp_size=8, tp_rank=rank)
                        )[name]
                        self.assertEqual(loaded.shape, raw.shape)
                        torch.testing.assert_close(
                            loaded.view(torch.uint8),
                            raw.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )

    def test_shared_fp8_kernel_and_scales_replicate_only_when_selected(self):
        from rtp_llm.model_loader.ffn_weight import FfnConfig
        from rtp_llm.model_loader.per_block_fp8_quant_weight import (
            W8A8Fp8PerBlockFfnAtomicWeight,
        )
        from rtp_llm.models.glm53_prefill_parallel import shared_expert_local_enabled
        from rtp_llm.ops import RoleType
        from rtp_llm.utils.model_weight import W

        cfg = FfnConfig(replicate_for_prefill_tokens=True)
        for name in (W.ffn_w13, W.ffn_w2, W.ffn_s13, W.ffn_s2):
            weight = W8A8Fp8PerBlockFfnAtomicWeight(name, [], config=cfg)
            tensor = torch.arange(128).reshape(16, 8)
            result = weight._split({name: tensor}, None)[name]
            torch.testing.assert_close(result, tensor, rtol=0, atol=0)
            self.assertNotEqual(result.data_ptr(), tensor.data_ptr())
        with patch.dict(os.environ, {"GLM53_PREFILL_SHARED_EXPERT_LOCAL": "1"}):
            self.assertTrue(
                shared_expert_local_enabled("glm5_3_flash", RoleType.PREFILL)
            )
            self.assertFalse(
                shared_expert_local_enabled("glm5_3_flash", RoleType.DECODE)
            )
            self.assertFalse(
                shared_expert_local_enabled("kimi_linear", RoleType.PREFILL)
            )

    def test_mla_view_preserves_kda_tp_and_rejects_global_cp(self):
        from rtp_llm.models.glm53_prefill_parallel import MlaCPParallelismView

        cp = SimpleNamespace(kv_cache_sharded=True, is_enabled=lambda: False)
        parallelism = SimpleNamespace(
            tp_size=8,
            tp_rank=3,
            ep_size=8,
            prefill_cp_config=cp,
            get_attn_tp_size=lambda: 8,
            get_attn_tp_rank=lambda: 3,
        )
        view = MlaCPParallelismView(parallelism)
        self.assertEqual(view.get_attn_tp_size(), 1)
        self.assertEqual(view.get_attn_tp_rank(), 0)
        self.assertEqual(view.tp_rank, 3)
        self.assertEqual(parallelism.get_attn_tp_size(), 8)
        self.assertFalse(parallelism.prefill_cp_config.is_enabled())
        self.assertTrue(view.prefill_cp_config.is_enabled())
        cp.is_enabled = lambda: True
        with self.assertRaisesRegex(ValueError, "cp_rotate_method=DISABLED"):
            MlaCPParallelismView(parallelism)
        cp.is_enabled = lambda: False
        cp.kv_cache_sharded = False
        with self.assertRaisesRegex(ValueError, "KV_CACHE_SHARDED"):
            MlaCPParallelismView(parallelism)

    def test_model_cp_metadata_does_not_replace_kda_inputs(self):
        from rtp_llm.models_py.model_desc import kimi_linear
        from rtp_llm.models_py.modules import AttnImplFactory
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        model = object.__new__(kimi_linear.KimiLinearModel)
        torch.nn.Module.__init__(model)
        model.prefill_mla_cp = True
        model.parallelism_config = SimpleNamespace(tp_size=8, tp_rank=3)
        model.config = model.mla_parallelism = model.weight = model.fmha_config = None
        attn = PyAttentionInputs()
        attn.is_prefill = True
        attn.input_lengths = torch.tensor([257, 8193], dtype=torch.int32)
        attn.prefix_lengths = torch.tensor([0, 128], dtype=torch.int32)
        original_cp = attn.context_parallel_info
        inputs = SimpleNamespace(input_ids=torch.arange(8450), attention_inputs=attn)
        with patch.object(
            AttnImplFactory, "get_fmha_impl", return_value=SimpleNamespace()
        ) as factory:
            impl = model.prepare_fmha_impl(inputs)
        cp_attn = factory.call_args.args[3]
        self.assertIsNot(cp_attn, attn)
        self.assertIs(attn.context_parallel_info, original_cp)
        torch.testing.assert_close(cp_attn.input_lengths, attn.input_lengths)
        torch.testing.assert_close(cp_attn.prefix_lengths, attn.prefix_lengths)
        self.assertEqual(impl.glm53_cp_layout.q_lens, (257, 8193))
        torch.testing.assert_close(
            cp_attn.context_parallel_info.prefill_cp_chunk_lengths,
            torch.tensor([34, 1026], dtype=torch.int32),
        )

    def test_zigzag_and_inverse_metadata(self):
        for size in (2, 4, 8):
            for lengths in CASES:
                x = torch.arange(sum(lengths), dtype=torch.int64).reshape(-1, 1)
                gathered = torch.cat(
                    [_expected_cp(x, lengths, size, r) for r in range(size)]
                )
                for rank in range(size):
                    layout = token_shard_layout(len(x), size, rank)
                    cp = ZigzagTokenLayout(
                        lengths, layout, size, rank, torch.device("cpu")
                    )
                    info = cp.context_parallel_info()
                    restored = gathered[info.prefill_qkv_restore_indice.long()]
                    torch.testing.assert_close(
                        restored[info.prefill_qkv_padding_mask.bool()], x
                    )
                    self.assertEqual(sum(cp.send_counts), layout.local_valid_tokens)
                    self.assertEqual(cp.reverse_recv_counts, cp.send_counts)
                    self.assertEqual(
                        sum(cp.recv_counts),
                        sum(f.length for f in cp.fragments if f.cp_rank == rank),
                    )
                # The metadata is rank independent and causal work is balanced
                # by assigning one front and one back chunk to each rank.
                self.assertEqual(len(gathered), info.prefill_qkv_padding_mask.numel())

    def test_mla_fp8_weights_and_scales_replicate_only_when_selected(self):
        from rtp_llm.model_loader.attn_weight import MlaConfig
        from rtp_llm.model_loader.per_block_fp8_quant_weight import (
            W8A8Fp8PerBlockMlaAttnAtomicWeight,
        )
        from rtp_llm.models.glm53_prefill_parallel import mla_cp_enabled
        from rtp_llm.ops import RoleType
        from rtp_llm.utils.model_weight import W

        cfg = MlaConfig(replicate_for_prefill_cp=True)
        for name in (
            W.mla_q_b_w,
            W.mla_q_b_s,
            W.attn_o_w,
            W.attn_o_s,
            W.mla_kc,
            W.mla_vc,
        ):
            w = W8A8Fp8PerBlockMlaAttnAtomicWeight(name, [], config=cfg)
            tensor = torch.arange(128).reshape(16, 8)
            result = w._split({name: tensor}, None)[name]
            torch.testing.assert_close(result, tensor)
            self.assertNotEqual(result.data_ptr(), tensor.data_ptr())
        with patch.dict(os.environ, {"GLM53_PREFILL_MLA_CP": "1"}):
            self.assertTrue(mla_cp_enabled("glm5_3_flash", RoleType.PREFILL))
            self.assertFalse(mla_cp_enabled("glm5_3_flash", RoleType.DECODE))
            self.assertFalse(mla_cp_enabled("kimi_linear", RoleType.PREFILL))

    @unittest.skipUnless(os.environ.get("GLM5_CKPT_PATH"), "real checkpoint manifest")
    @patch.dict(
        os.environ,
        {"GLM53_PREFILL_MLA_CP": "1", "GLM53_PREFILL_SHARED_EXPERT_LOCAL": "1"},
    )
    def test_glm53_quantized_manifest_role_and_layer_scope(self):
        import json
        from pathlib import Path

        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight
        from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight
        from rtp_llm.model_loader.weight_module import CompositeWeight
        from rtp_llm.models.glm5_3_flash import Glm53Flash, Glm53FlashWeight
        from rtp_llm.ops import HWKernelConfig, ParallelismConfig, RoleType
        from rtp_llm.utils.model_weight import W

        checkpoint = Path(os.environ["GLM5_CKPT_PATH"])
        keys = set(
            json.loads((checkpoint / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
        )
        for role in (RoleType.PREFILL, RoleType.DECODE):
            config = Glm53Flash._create_config(str(checkpoint))
            config.init_precision_config(KVCacheConfig(), "BF16")
            parallelism = ParallelismConfig()
            parallelism.role_type = role
            parallelism.tp_size = parallelism.world_size = (
                parallelism.local_world_size
            ) = 8
            parallelism.ep_size = 8
            manifest = Glm53FlashWeight(
                config, parallelism, HWKernelConfig(), KVCacheConfig()
            )
            self.assertEqual(manifest.role_type, role)
            manifest._process_meta({}, keys)
            weight_info = manifest.get_weight_info()
            found = set()

            def check(weight, layer_id):
                if (
                    isinstance(weight, MlaAttnAtomicWeight)
                    and weight.config is not None
                ):
                    self.assertEqual(
                        weight.config.replicate_for_prefill_cp, role == RoleType.PREFILL
                    )
                    found.add(weight.name)
                elif isinstance(weight, FfnAtomicWeight):
                    self.assertEqual(
                        weight.config.replicate_for_prefill_tokens,
                        role == RoleType.PREFILL
                        and layer_id in manifest.moe_layer_index_,
                    )
                elif isinstance(weight, CompositeWeight):
                    for child in weight.sub_weights.values():
                        check(child, layer_id)
                else:
                    self.assertFalse(
                        getattr(
                            getattr(weight, "config", None),
                            "replicate_for_prefill_cp",
                            False,
                        )
                    )

            for layer_id, layer in enumerate(weight_info.layer_weights):
                for weight in layer:
                    check(weight, layer_id)
            for name in (
                W.mla_q_b_w,
                W.mla_q_b_s,
                W.attn_o_w,
                W.attn_o_s,
                W.mla_kc,
                W.mla_vc,
            ):
                self.assertIn(name, found)


def _distributed_worker(rank, size, port):
    from rtp_llm.models_py.distributed import collective_torch as coll

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size
    )
    coll._group_map[coll.Group.TP] = dist.group.WORLD
    coll._group_map[coll.Group.DP_AND_TP] = dist.group.WORLD
    coll._initialized = True
    coll._parallelism_config = SimpleNamespace(tp_size=size, dp_size=1, world_size=size)
    try:
        for lengths in CASES:
            logical = sum(lengths)
            # Keep the 1M-token test small in memory while checking exact row order.
            x = torch.arange(logical, device="cuda", dtype=torch.float32).reshape(-1, 1)
            layout = token_shard_layout(logical, size, rank)
            cp = ZigzagTokenLayout(lengths, layout, size, rank, x.device)
            local = shard_tokens(x, layout)
            zigzag = cp.sp_to_cp(local)
            torch.testing.assert_close(
                zigzag, _expected_cp(x, lengths, size, rank), rtol=0, atol=0
            )
            torch.testing.assert_close(cp.cp_to_sp(zigzag), local, rtol=0, atol=0)
            torch.testing.assert_close(
                coll.all_gather_trim(local, logical, coll.Group.TP), x, rtol=0, atol=0
            )
            # TP row projection -> RS -> token-local nonlinear operation -> AG
            partial = (x % 17) * (rank + 1)
            reduced = coll.reduce_scatter_padded(partial, coll.Group.TP)
            actual = coll.all_gather_trim(torch.square(reduced), logical, coll.Group.TP)
            expected = torch.square((x % 17) * (size * (size + 1) // 2))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Exercise real CP seed communication on all ranks, with ragged query
        # lengths, prefix hits and padding. Each request's final query has one
        # owner; every rank must receive the same frozen expanded selection.
        from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel

        lengths = [3, 17, 129]
        cp = ZigzagTokenLayout(
            lengths,
            token_shard_layout(sum(lengths), size, rank),
            size,
            rank,
            torch.device("cuda", rank),
        )
        rows = {}
        for fragment in cp.fragments:
            if fragment.cp_rank == rank:
                for offset in range(fragment.length):
                    rows[fragment.cp_offset + offset] = (
                        fragment.canonical_offset + offset
                    )
        ids = sorted(rows)
        pooled = torch.tensor(
            [[rows[row] % 31, 0, -1, -1] for row in ids],
            dtype=torch.int32,
            device="cuda",
        ).reshape(-1, 4)
        model = object.__new__(GenericMoeMTPModel)
        model.config = SimpleNamespace(
            is_glm53_mtp=True,
            attn_config=SimpleNamespace(indexer_topk=4, indexer_compress_ratio=4),
        )
        model._mtp_shared_topk_indices = torch.zeros(
            (3, 19), dtype=torch.int32, device="cuda"
        )
        fmha = SimpleNamespace(
            glm53_cp_layout=cp,
            cp_params=SimpleNamespace(
                total_local_ids=torch.tensor(ids, dtype=torch.int32, device="cuda")
            ),
        )
        prefix = torch.tensor([256, 256, 256], dtype=torch.int32, device="cuda")
        model._store_prefill_cp_seed(
            pooled, fmha, SimpleNamespace(prefix_lengths=prefix)
        )
        expected_pool = torch.tensor(
            [[2, 0, -1, -1], [19, 0, -1, -1], [148 % 31, 0, -1, -1]],
            dtype=torch.int32,
            device="cuda",
        )
        expected_seed = model._expand_mtp_seed(
            expected_pool, prefix + torch.tensor(lengths, device="cuda")
        )
        torch.testing.assert_close(
            model._mtp_shared_topk_indices, expected_seed, rtol=0, atol=0
        )
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


class DistributedTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("GLM53_TEST_DISTRIBUTED") == "1", "explicit eight-GPU test"
    )
    def test_eight_rank_collectives(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(_distributed_worker, args=(8, port), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
