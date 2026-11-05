"""PPU replay visibility, CP transport, native cache and decode continuation."""

import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv41.ppu.attention import build_sparse_workspace
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.ced import PPUCEDPlan
from rtp_llm.models_py.modules.dsv41.ppu.cp import (
    CPPrefillContext,
    ThreadedCPGroup,
    build_zigzag_plan,
)
from rtp_llm.models_py.modules.dsv41.ppu.decode import DecodeStepOrchestrator
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_model import NativeModelRunner
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import (
    native_cache_key_hash_seed,
)
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import (
    _build_reference_workspace,
    build_step_context,
    run_prefill,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache import NativePools
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)


class ReplayPools(NativePools):
    """Independent encoder/decoder native tables, with the real shared API."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        geo = V41Geometry.from_config(config)
        policy = PrefillPolicy.from_config(config)
        if not policy.bounded_replay:
            return
        decoder = copy.copy(self.inputs["swa_kv"])
        decoder.kv_cache_block_id_device = decoder.kv_cache_block_id_device.roll(1, 1)
        decoder.kv_cache_kernel_block_id_device = decoder.kv_cache_block_id_device
        decoder.cache_store_inputs = "decoder_swa_kv"
        self.inputs["decoder_swa_kv"] = decoder
        self.entries["decoder_swa_kv"] = self.entries["swa_kv"]
        self.group_tags.append("decoder_swa_kv")
        for layer in range(policy.decoder_start(geo), geo.num_layers):
            handle = self.handles.pop((layer, "swa_kv"))
            handle.tag = "decoder_swa_kv"
            self.handles[(layer, "decoder_swa_kv")] = handle


def make_ops(geo, rank=0, size=1, group=None):
    return PPUOps(
        ep_rank=rank,
        ep_size=size,
        num_routed_experts=geo.n_routed_experts,
        num_local_experts=geo.num_local_experts(size),
        hidden_size=geo.hidden_size,
        inter_dim=geo.moe_inter_dim,
        swiglu_limit=geo.swiglu_limit,
        mhc_config=geo.mhc_config(),
        sm_scale=geo.sm_scale(),
        window=geo.window,
        enpm_group=group,
    )


class CedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = PrefillPolicy(True, True).freeze(
            synthetic_config(
                num_layers=12,
                index_sources=(2, 6, 9),
                engram_layers=(),
                engram_num_embeddings=(),
            )
        )
        cls.geo = V41Geometry.from_config(cls.config)
        cls.root = tempfile.TemporaryDirectory(prefix="ppu-ced-")
        cls.checkpoint = write_synthetic_checkpoint(
            cls.config, Path(cls.root.name), seed=41
        )
        cls.ops = make_ops(cls.geo)
        cls.bundle = load_ppu_weights(cls.config, cls.checkpoint)
        cls.bundle.bind(cls.ops)

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def test_policy_is_frozen_and_semantic_cache_keys_are_separate(self):
        policy = PrefillPolicy.from_config(self.config)
        with patch.dict(
            "os.environ", {"DSV41_CED": "0", "DSV41_SWA_BOUNDED_REPLAY": "0"}
        ):
            self.assertEqual(PrefillPolicy.from_config(self.config), policy)
        self.assertNotEqual(
            native_cache_key_hash_seed(self.config),
            native_cache_key_hash_seed(PrefillPolicy().freeze(self.config)),
        )
        with self.assertRaisesRegex(ValueError, "requires DSV41_CED"):
            PrefillPolicy(False, True).validate()

    def test_mask_and_compact_have_same_tail_and_full_global_cache(self):
        ids = torch.arange(44) % self.geo.vocab_size
        cu = torch.tensor([0, 35, 44])
        results = []
        for compact in (False, True):
            cache = PPUKVCache(self.config)
            results.append(
                run_prefill(
                    self.bundle,
                    self.ops,
                    cache,
                    ids,
                    cu_seqlens=cu,
                    compact_ced=compact,
                    apply_lm_head=True,
                )
            )
        mask, compact = results
        self.assertTrue(compact.ced_info["compacted"])
        self.assertEqual(compact.ced_info["query_rows"], 16)
        for rid, end in enumerate((35, 44)):
            rows = slice(end - self.geo.window, end)
            torch.testing.assert_close(
                mask.hidden[rows], compact.hidden[rows], atol=0.02, rtol=0.02
            )
            for layer in range(self.geo.num_layers):
                positions = torch.arange((35, 9)[rid] - self.geo.window, (35, 9)[rid])
                self.assertTrue(
                    torch.equal(
                        mask.cache.read_swa(layer, rid, positions),
                        compact.cache.read_swa(layer, rid, positions),
                    )
                )
            for source in self.geo.kv_source_layer_ids:
                for left, right in zip(
                    mask.cache.read_source(source, rid),
                    compact.cache.read_source(source, rid),
                ):
                    self.assertTrue(torch.equal(left, right))
        self.assertEqual(int(torch.count_nonzero(compact.hidden[:27])), 0)

    def test_workspace_never_reads_before_replay_start(self):
        cache = PPUKVCache(self.config)
        cache.alloc(0, 35)
        # Deliberately poison the part of the SWA cache outside the replay.
        cache.insert_rows(
            7,
            "swa_kv",
            cache.slots_at_positions(7, "swa_kv", 0, torch.arange(27)),
            torch.full((27, 528), 127, dtype=torch.uint8),
        )
        tail = self.ops.encode_kv(
            torch.ones(8, 512, dtype=torch.bfloat16), pool_tag="swa_kv"
        )
        cache.insert_rows(
            7,
            "swa_kv",
            cache.slots_at_positions(7, "swa_kv", 0, torch.arange(27, 35)),
            tail,
        )
        context = build_step_context(
            torch.arange(27, 35), torch.zeros(8, dtype=torch.long), torch.tensor([0, 8])
        )
        selected = torch.full((8, 4), -1, dtype=torch.int32)
        expected = _build_reference_workspace(
            ops=self.ops,
            cache=cache,
            layer_id=7,
            geometry=self.geo,
            context=context,
            selected_blocks=selected,
            window=8,
            swa_starts={0: 27},
        )
        actual = build_sparse_workspace(
            selected,
            cache=cache.cache_view(7),
            layer_id=7,
            context=context,
            window=8,
            swa_starts={0: 27},
        )
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(actual[0]).all())
        self.assertEqual(actual[2].tolist(), list(range(1, 9)))

    def test_partial_prefix_requires_a_complete_decoder_replay_window(self):
        cache = PPUKVCache(self.config)
        cache.alloc(0, 32)
        with self.assertRaisesRegex(ValueError, "at least 8 replay tokens"):
            run_prefill(self.bundle, self.ops, cache, torch.tensor([1]))
        self.assertEqual(cache.request_length(0), 32)

    def test_cp8_selected_transport_roundtrips_mixed_lengths_and_absolute_positions(
        self,
    ):
        lengths, prefixes = [1, 9, 35], [17, 128, 301]
        ids = torch.arange(sum(lengths)) % self.geo.vocab_size
        cu = torch.tensor([0, 1, 10, 45])
        group = ThreadedCPGroup(8)

        def forward(rank):
            original = build_zigzag_plan(
                ids, cu, cp_size=8, cp_rank=rank, prefix_lengths=prefixes
            )
            ctx = CPPrefillContext(original, group.rank_view(rank))
            ced = PPUCEDPlan.create(
                self.geo,
                PrefillPolicy(True, True),
                original.local_input_ids,
                original.local_positions,
                original.local_request_ids,
                original.cu_local,
                ctx,
            )
            value = original.local_positions[:, None].float()
            compact = ced.compact(value)
            self.assertTrue(torch.equal(compact[:, 0], ced.positions.float()))
            selected = ced.context.all_gather_restore(compact)
            restored = ced.restore_global(selected)
            self.assertEqual(restored.shape[0], sum(lengths))
            self.assertEqual(selected.shape[0], 17)
            expected = original.full_positions.index_select(0, ced.selected)
            self.assertTrue(torch.equal(selected[:, 0], expected.float()))
            return ced.info

        with ThreadPoolExecutor(max_workers=8) as pool:
            reports = list(pool.map(forward, range(8)))
        self.assertTrue(all(report == reports[0] for report in reports))

    def test_native_bridge_and_graph_keep_decoder_swa_tables_independent(self):
        native = ReplayPools(self.config)
        bridge = NativeCacheBridge(self.config, native, native.inputs)
        graph = NativeGraphCache(self.config, native, native.inputs, device="cpu")
        self.assertEqual(bridge.pools[(7, "swa_kv")].framework_tag, "decoder_swa_kv")
        self.assertEqual(bridge.pools[(6, "swa_kv")].framework_tag, "swa_kv")
        positions = torch.tensor([3, 4])
        self.assertFalse(
            torch.equal(
                graph.slots(6, "swa_kv", positions), graph.slots(7, "swa_kv", positions)
            )
        )
        before = graph.slots(6, "swa_kv", positions).clone()
        native.inputs["decoder_swa_kv"].kv_cache_block_id_device.add_(1).remainder_(12)
        graph.prepare(native.inputs)
        self.assertTrue(torch.equal(before, graph.slots(6, "swa_kv", positions)))

    def test_cp8_forward_compacts_mhc_and_shared_topk_with_real_expert_shards(self):
        ids = torch.arange(44) % self.geo.vocab_size
        cu = torch.tensor([0, 35, 44])

        def cp8(compact):
            group = ThreadedCPGroup(8)
            bundles, ops_by_rank = [], {}
            for rank in range(8):
                view = group.rank_view(rank)
                ops = make_ops(self.geo, rank, 8, view)
                bundle = load_ppu_weights(
                    self.config,
                    self.checkpoint,
                    ep_rank=rank,
                    ep_size=8,
                    enp_rank=rank,
                    enp_size=8,
                )
                bundle.bind(ops, enpm_group=view)
                ops_by_rank[rank] = ops
                bundles.append(bundle)
            group.register_ops(ops_by_rank)

            def forward(rank):
                plan = build_zigzag_plan(ids, cu, cp_size=8, cp_rank=rank)
                return run_prefill(
                    bundles[rank],
                    ops_by_rank[rank],
                    PPUKVCache(self.config),
                    plan.local_input_ids,
                    positions=plan.local_positions,
                    request_ids=plan.local_request_ids,
                    cu_seqlens=plan.cu_local,
                    parallel_context=CPPrefillContext(
                        plan, group.rank_view(rank), True
                    ),
                    compact_ced=compact,
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                return list(pool.map(forward, range(8)))

        # Keep EP shard accumulation identical in the mask/compact comparison.
        # The CP1 mask/compact test above separately covers the single-rank math.
        reference = cp8(False)[0]
        results = cp8(True)
        for result in results:
            self.assertTrue(result.ced_info["compacted"])
            self.assertEqual(result.ced_info["query_rows"], 16)
            self.assertEqual(result.ced_info["cp_size"], 8)
            for end in (35, 44):
                torch.testing.assert_close(
                    result.hidden[end - 8 : end],
                    reference.hidden[end - 8 : end],
                    atol=0.02,
                    rtol=0.02,
                )
            torch.testing.assert_close(result.hidden, results[0].hidden, atol=0, rtol=0)

    def test_native_prefill_then_decode_uses_moved_state_without_prefill_on_d(self):
        native = ReplayPools(self.config, batch=1, blocks=16)
        core = PPUModelCore(self.config, self.bundle, self.ops)
        p = NativeModelRunner(core)
        p.initialize(NS(kv_cache=native, is_decode_role=False))
        tokens = (torch.arange(35) % self.geo.vocab_size).tolist()

        def inputs(prefill):
            prefix = 0 if prefill else len(tokens) - 1
            for item in native.inputs.values():
                item.is_prefill = prefill
                item.input_lengths = torch.tensor([len(tokens) - prefix])
                item.prefix_lengths = torch.tensor([prefix])
                item.sequence_lengths = torch.tensor([prefix])
            return NS(
                input_ids=torch.tensor(tokens[prefix:]),
                attention_inputs=native.inputs,
                engram_token_windows=None,
                is_fake_stream=False,
                warmup=False,
                need_all_logits=False,
                need_all_hidden_states=False,
            )

        actual = p.forward(inputs(True))
        reference_core = PPUModelCore(self.config, self.bundle, self.ops)
        reference = reference_core.prefill(torch.tensor(tokens), apply_lm_head=True)
        torch.testing.assert_close(actual[-8:], reference.hidden[-8:], atol=0, rtol=0)
        self.assertIn((7, "decoder_swa_kv"), native.calls)
        self.assertNotIn((7, "swa_kv"), native.calls)
        expected = DecodeStepOrchestrator(
            ops=self.ops, bundle=self.bundle, cache=reference.cache
        )
        expected.add_request(0, tokens)
        d = NativeModelRunner(PPUModelCore(self.config, self.bundle, self.ops))
        d.initialize(NS(kv_cache=native, is_decode_role=True))
        pending = int(reference.logits[-1].argmax())
        with patch.object(
            d.core, "prefill", side_effect=AssertionError("D executed prefill")
        ):
            for _ in range(4):
                tokens.append(pending)
                got = d.forward(inputs(False))
                want = expected.step({0: pending})
                torch.testing.assert_close(got, want.hidden, atol=0, rtol=0)
                pending = int(want.logits[0].argmax())

    def test_missing_encoder_state_rebuilds_without_writing_cached_global_prefix(self):
        config = PrefillPolicy(True, True, True).freeze(self.config)
        tokens = torch.arange(35) % self.geo.vocab_size
        reference = run_prefill(
            self.bundle,
            self.ops,
            PPUKVCache(config),
            tokens,
            apply_lm_head=True,
        )
        native = ReplayPools(config, batch=1, blocks=16)
        bridge = NativeCacheBridge(config, native, native.inputs)
        bridge.export_requests(reference.cache, [35])
        prefix = 17  # Includes an unfinished ratio-2 pair.
        full_tags = {"global_kv_1", "global_kv_2", "indexer_kv"}
        before = {
            key: bridge._copy_rows(key, 0, bridge._positions(key, prefix)).clone()
            for key in bridge.pools
            if key[1] in full_tags
        }
        for key, pool in bridge.pools.items():
            if key[1] not in full_tags:
                pool.tensor.fill_(
                    float("nan") if pool.tensor.is_floating_point() else 255
                )

        def inputs(prefill, history):
            cached = prefix if prefill else len(history) - 1
            for item in native.inputs.values():
                item.is_prefill = prefill
                item.input_lengths = torch.tensor([len(history) - cached])
                item.prefix_lengths = torch.tensor([cached])
                item.sequence_lengths = torch.tensor([cached])
            return NS(
                input_ids=torch.tensor(history[cached:]),
                attention_inputs=native.inputs,
                prefill_replay_token_ids=torch.tensor(history),
                engram_token_windows=None,
                is_fake_stream=False,
                warmup=False,
                need_all_logits=False,
                need_all_hidden_states=False,
            )

        core = PPUModelCore(config, self.bundle, self.ops)
        p = NativeModelRunner(core)
        p.initialize(NS(kv_cache=native, is_decode_role=False))
        original_prefill = core.prefill

        def protected_forward(*args, **kwargs):
            cache = kwargs["cache"]
            cached_slots = {
                key: cache._slots_for_blocks(
                    key, 0, prefix // cache.geometry.ratio(key[0])
                )
                for key in cache.pool_keys()
                if key[1] in full_tags
            }
            insert = cache.insert_rows

            def guarded(owner, tag, slots, rows):
                if tag in full_tags:
                    self.assertFalse(
                        bool(torch.isin(slots, cached_slots[(owner, tag)]).any())
                    )
                return insert(owner, tag, slots, rows)

            with patch.object(cache, "insert_rows", side_effect=guarded):
                result = original_prefill(*args, **kwargs)
            self.assertEqual(result.cache.request_length(0), 35)
            return result

        history = tokens.tolist()
        with patch.object(core, "prefill", side_effect=protected_forward):
            hidden = p.forward(inputs(True, history))
        self.assertEqual(hidden.shape[0], 35 - prefix)
        torch.testing.assert_close(hidden[-8:], reference.hidden[-8:], atol=0, rtol=0)
        for key, rows in before.items():
            self.assertTrue(
                torch.equal(
                    rows, bridge._copy_rows(key, 0, bridge._positions(key, prefix))
                )
            )

        d = NativeModelRunner(PPUModelCore(config, self.bundle, self.ops))
        d.initialize(NS(kv_cache=native, is_decode_role=True))
        expected = DecodeStepOrchestrator(
            ops=self.ops, bundle=self.bundle, cache=reference.cache
        )
        expected.add_request(0, history)
        pending = int(reference.logits[-1].argmax())
        with patch.object(
            d.core, "prefill", side_effect=AssertionError("D executed prefill")
        ):
            for _ in range(3):
                history.append(pending)
                got = d.forward(inputs(False, history))
                want = expected.step({0: pending})
                torch.testing.assert_close(got, want.hidden, atol=0, rtol=0)
                pending = int(want.logits[0].argmax())


if __name__ == "__main__":
    unittest.main()
