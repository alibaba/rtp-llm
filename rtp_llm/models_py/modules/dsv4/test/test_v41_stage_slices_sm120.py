"""Synthetic CP1 controls for the maintained 21/19 owner loop.

These tests project already constructed PP1 controls into slices. Production
constructor/worker admission and native allocation/transport are not exercised.
"""

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_transformer_sm120 import fixture
from rtp_llm.models_py.modules.dsv4.hc.pp_state import imported_delayed_pre_mix
from rtp_llm.models_py.modules.dsv4.native_l20_prefill import (
    IMPORT_KEY,
    imported_native_l20_prefill,
    prepare_native_l20_prefill_source,
)
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
    native_l20_cache_views,
)
from rtp_llm.models_py.modules.dsv4.prefill.forward import forward_layers
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_prefill import setup
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import (
    CONTRACT,
    TEXT,
    frame_for,
    native_fixture,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer
from rtp_llm.models_py.modules.dsv4.v41_cache import (
    logical_to_physical_slots,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_pp import (
    V41PPFrame,
    import_v41_pp_frame,
    pack_v41_pp_frame,
)
from rtp_llm.utils.model_weight import W


def repeated(value, *sizes):
    if value.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return value.view(torch.uint8).repeat(*sizes).view(value.dtype)
    return value.repeat(*sizes)


def wide_fixture():
    """Widen synthetic hidden dimensions; keep 512/128 KV/index latents."""
    args, weights = fixture(full_schedule=True)
    for layer in weights.weights:
        for prefix in ("v4_attn_wq_a", "v4_attn_wkv", "v4_shared_w13"):
            for suffix in ("w", "s"):
                key = getattr(W, f"{prefix}_{suffix}")
                layer[key] = repeated(layer[key], 1, 10)
        for prefix in ("v4_attn_wo_b", "v4_shared_w2"):
            for suffix in ("w", "s"):
                key = getattr(W, f"{prefix}_{suffix}")
                layer[key] = repeated(layer[key], 10, 1)
        for name in ("w1", "w3"):
            for suffix in ("w", "s"):
                key = getattr(W, f"v4_routed_{name}_{suffix}")
                layer[key] = repeated(layer[key], 1, 1, 10)
        for suffix in ("w", "s"):
            key = getattr(W, f"v4_routed_w2_{suffix}")
            layer[key] = repeated(layer[key], 1, 10, 1)
        layer[W.v4_router_w] = repeated(layer[W.v4_router_w], 1, 10)
        for part in ("attn", "ffn"):
            norm = getattr(W, f"v4_{part}_norm")
            layer[norm] = repeated(layer[norm], 10)
            fn = getattr(W, f"v4_hc_{part}_fn")
            layer[fn] = layer[fn].reshape(24, 4, 512).repeat(1, 1, 10).flatten(1) * 0.1
        for key in (W.v4_compressor_wkv, W.v4_compressor_wgate):
            if key in layer:
                layer[key] = repeated(layer[key], 1, 10)
        if W.v4_indexer_weights_proj_w in layer:
            layer[W.v4_indexer_weights_proj_w] = repeated(
                layer[W.v4_indexer_weights_proj_w], 1, 10
            )
    for key in (W.embedding, W.lm_head):
        weights.global_weights[key] = repeated(weights.global_weights[key], 1, 10)
    weights.global_weights[W.final_ln_gamma] = repeated(
        weights.global_weights[W.final_ln_gamma], 10
    )
    return replace(args, dim=5120, gen_num_per_cycle=0), weights


def model_for(args, weights):
    # Maintained providers consume loader entries after packing their weights.
    # Each comparison owner needs an independent loader dictionary and tensors.
    owned = SimpleNamespace(
        global_weights={k: v.clone() for k, v in weights.global_weights.items()},
        weights=[{k: v.clone() for k, v in layer.items()} for layer in weights.weights],
    )
    model = V4Transformer(args, owned)
    model._prefill_ws_q_rows = 256
    model._prefill_ws_q_dim = args.n_heads * args.head_dim
    for layer in model.layers:
        layer.attn.reset_rope_cache(torch.device("cuda"))
    return model


def project_control(model, rank):
    """Test-only projection of an initialized owner, preserving its layer loop."""
    ids = list(range(21)) if rank == 0 else list(range(21, 40))
    model.layers = nn.ModuleList([model.layers[i] for i in ids])
    model.pp_global_layer_ids = ids
    model.args = replace(
        model.args,
        pp_size=2,
        pp_rank=rank,
        pp_layer_ids=ids,
        pp_has_embedding=rank == 0,
        pp_has_lm_head=rank == 1,
    )
    model.pp_has_embedding, model.pp_has_lm_head = rank == 0, rank == 1
    model._owns_embedding, model._owns_lm_head = rank == 0, rank == 1
    shared = {
        "layers": {layer.layer_id: layer.attn for layer in model.layers},
        "cache_bindings": {
            (b.source_layer_id, b.tag): b for b in dsv41_cache_bindings(TEXT, 2, rank)
        },
    }
    for local, layer in enumerate(model.layers):
        layer.attn.cache_layer_id = local
        layer.attn._shared_attention = shared
    if rank == 0:
        model.norm = model.head_hc = model.head_weight = None
    else:
        model.embed = None
        model.layers[0].attn_hc.set_previous(None)
    return model, shared


def tables(cache, hca=(1, 2), index=(2, 1)):
    result = {
        tag: torch.tensor([[1], [2]], device="cuda", dtype=torch.int32)
        for tag in cache.layouts
    }
    result["hca_kv"] = torch.tensor(hca, device="cuda", dtype=torch.int32)[:, None]
    result["indexer_kv"] = torch.tensor(index, device="cuda", dtype=torch.int32)[
        :, None
    ]
    return result


def capture_frame(block, rows, hidden):
    template = frame_for(rows)
    return V41PPFrame(
        hidden.clone().contiguous(),
        block.ffn_hc.pre_mix_out.clone().contiguous(),
        rows.clone(),
        block.attn._shared_attention["topk"][20].clone(),
        template.candidate_blocks.cuda(),
        template.candidate_counts.cuda(),
    )


def assert_cleared(test, stage, shared):
    test.assertIsNone(stage.layers[0].attn_hc._previous_ref)
    test.assertNotIn(IMPORT_KEY, shared)
    test.assertEqual(shared["global"], {})
    test.assertEqual(shared["topk"], {})
    test.assertTrue(all(l.attn._prefill_meta_shared is None for l in stage.layers))
    test.assertTrue(all(l.ffn.gate._v41_cp_ctx is None for l in stage.layers))


class NativeL20StageAdmissionTest(unittest.TestCase):
    def owners(self):
        entry, common, receiver, frame, rows, spans, _, _ = setup()
        source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
        shared = entry._shared_attention
        for layer_id in range(22, 40):
            index_source = max(i for i in (20, 24, 28, 32, 36) if i <= layer_id)
            shared["layers"][layer_id] = SimpleNamespace(
                **{
                    **vars(entry),
                    "layer_id": layer_id,
                    "index_source_layer_id": index_source,
                    "is_index_source": layer_id == index_source,
                }
            )
        return entry, common, source, shared

    def test_later_consumers_require_local_index_owners_and_cannot_republish(self):
        entry, common, source, shared = self.owners()
        with imported_native_l20_prefill(entry, source):
            for owner in shared["layers"].values():
                source.require_owner(owner)
            with self.assertRaises(NotImplementedError):
                source.publish(shared["layers"][24], common)
            self.assertEqual(shared["global"], {})
            self.assertEqual(shared["topk"], {})
            source.publish(entry, common)
            self.assertEqual(list(shared["global"]), [20])

    def test_missing_or_wrong_index_provenance_fails_before_publication(self):
        entry, _, source, shared = self.owners()
        with imported_native_l20_prefill(entry, source):
            consumer = shared["layers"][25]
            source.require_owner(consumer)
            producer = shared["layers"].pop(24)
            with self.assertRaises(NotImplementedError):
                source.require_owner(consumer)
            shared["layers"][24] = producer
            producer.is_index_source = False
            with self.assertRaises(NotImplementedError):
                source.require_owner(consumer)
            producer.is_index_source = True
            stale = SimpleNamespace(**vars(consumer))
            with self.assertRaises(NotImplementedError):
                source.require_owner(stale)
            consumer.index_source_layer_id = 20
            with self.assertRaises(NotImplementedError):
                source.require_owner(consumer)
            consumer.index_source_layer_id = 24
            consumer.is_index_source = True
            with self.assertRaises(NotImplementedError):
                source.require_owner(consumer)
            self.assertEqual(shared["global"], {})
            self.assertEqual(shared["topk"], {})


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "Maintained 21/19 controls require a leased SM120 GPU",
)
class V41StageSlicesSM120Test(unittest.TestCase):
    def setUp(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        env = patch.dict(os.environ, {"DSV4_MOE_STRATEGY": "auto", "MOEDBG": "0"})
        env.start()
        self.addCleanup(env.stop)

    @torch.no_grad()
    def test_21_19_hidden_logits_and_cache_bytes_match_full_owner(self):
        torch.manual_seed(412119)
        args, weights = wide_fixture()
        full = model_for(args, weights)
        stage0, _ = project_control(model_for(args, weights), 0)
        stage1, shared1 = project_control(model_for(args, weights), 1)
        self.assertEqual([l.layer_id for l in stage0.layers], list(range(21)))
        self.assertEqual([l.layer_id for l in stage1.layers], list(range(21, 40)))
        self.assertNotIn(20, shared1["layers"])
        self.assertTrue(all(not l.attn.is_kv_source for l in stage1.layers))
        self.assertIsNone(stage0.norm)
        self.assertIsNone(stage1.embed)
        full_cache = NativeTaggedCache([l.attn for l in full.layers], gen=0, pages=6)
        first_cache = NativeTaggedCache([l.attn for l in stage0.layers], gen=0, pages=6)
        first_cache.layer_count = 21
        replica, backing = native_fixture(1, "cuda")
        second_swa = NativeTaggedCache([l.attn for l in stage1.layers], gen=0, pages=6)
        replica.layers.update(second_swa.groups)
        replica.layouts = second_swa.layouts
        full_tables, first_tables = tables(full_cache), tables(first_cache)
        second_tables = tables(replica, (3, 4), (4, 3))
        identities = (V41CacheIdentity(1201, 1, 0), V41CacheIdentity(1202, 2, 1))
        producer_pages = {
            identities[0]: {"hca_kv": (1,), "indexer_kv": (2,)},
            identities[1]: {"hca_kv": (2,), "indexer_kv": (1,)},
        }
        replica_pages = {
            identities[0]: {"hca_kv": (3,), "indexer_kv": (4,)},
            identities[1]: {"hca_kv": (4,), "indexer_kv": (3,)},
        }
        receiver = NativeL20ReplicaReceiver(TEXT, replica, CONTRACT, 0)
        for identity in identities:
            receiver.begin_request(identity, replica_pages[identity])
        prefixes = [0, 0]
        with patch.dict(os.environ, {"DSV4_PREFILL_FAST_PATH": "0"}):
            for chunk, counts in enumerate(([3, 2], [4, 3], [122, 124])):
                positions = torch.cat(
                    [
                        torch.arange(p, p + n, device="cuda")
                        for p, n in zip(prefixes, counts)
                    ]
                )
                cu = torch.tensor([0, counts[0], sum(counts)], device="cuda")
                attn_inputs = SimpleNamespace(
                    input_lengths=torch.tensor(
                        counts, device="cuda", dtype=torch.int32
                    ),
                    prefix_lengths=torch.tensor(
                        prefixes, device="cuda", dtype=torch.int32
                    ),
                    is_prefill=True,
                    cache_store_inputs=None,
                )
                ids = (torch.arange(sum(counts), device="cuda") + chunk * 17).remainder(
                    args.vocab_size
                )
                rows = torch.tensor(
                    [
                        [r, p, 1]
                        for r, (s, n) in enumerate(zip(prefixes, counts))
                        for p in range(s, s + n)
                    ],
                    dtype=torch.int64,
                )
                captured = []

                def capture_l20(block, inputs, hidden):
                    captured.append(capture_frame(block, rows, hidden))

                hook = full.layers[20].register_forward_hook(capture_l20)
                try:
                    expected = forward_layers(
                        full, full_cache, ids, positions, cu, full_tables, attn_inputs
                    )
                finally:
                    hook.remove()
                self.assertEqual(len(captured), 1)
                full_frame = captured.pop()
                hook = stage0.layers[-1].register_forward_hook(capture_l20)
                try:
                    hidden = forward_layers(
                        stage0,
                        first_cache,
                        ids,
                        positions,
                        cu,
                        first_tables,
                        attn_inputs,
                    )
                finally:
                    hook.remove()
                self.assertEqual(len(captured), 1)
                frame = captured[0]
                torch.testing.assert_close(hidden, frame.hidden_states, rtol=0, atol=0)
                for name, expected_value in full_frame.__dict__.items():
                    torch.testing.assert_close(
                        frame.__dict__[name], expected_value, rtol=0, atol=0
                    )
                self.assertEqual(stage0.layers[0].attn._shared_attention["topk"], {})
                spans = [
                    V41L20Span(identity, chunk, start, count)
                    for identity, start, count in zip(identities, prefixes, counts)
                ]
                delta = export_native_l20_delta(
                    TEXT, first_cache, CONTRACT, 0, chunk, spans, producer_pages
                )
                frame = import_v41_pp_frame(
                    pack_v41_pp_frame(delta, frame, spans, rows), receiver, spans, rows
                )
                source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
                frame.topk.zero_()
                frame.row_map.fill_(99)
                before = backing.clone()
                if chunk == 0:
                    # Failure after imported publication and three real local
                    # blocks must release metadata, source and predecessor.
                    with self.assertRaisesRegex(
                        RuntimeError, "stage selection failure"
                    ):
                        with imported_native_l20_prefill(
                            stage1.layers[0].attn, source
                        ), imported_delayed_pre_mix(
                            stage1.layers[0].attn_hc,
                            frame.hidden_states,
                            frame.prev_pre_mix,
                        ), patch.object(
                            stage1.layers[3].attn,
                            "_prefill_compute_qkv",
                            side_effect=RuntimeError("stage selection failure"),
                        ):
                            forward_layers(
                                stage1,
                                replica,
                                ids,
                                positions,
                                cu,
                                second_tables,
                                attn_inputs,
                                prepare_hidden_fn=lambda **kwargs: frame.hidden_states.clone(),
                            )
                    assert_cleared(self, stage1, shared1)
                    torch.testing.assert_close(backing, before, rtol=0, atol=0)
                selected_owners = {}
                selection_hooks = []
                for layer in stage1.layers:
                    if layer.attn.is_index_source:

                        def capture_selection(block, inputs, outputs):
                            selected_owners[block.layer_id] = (
                                block.attn._shared_attention["topk"][
                                    block.layer_id
                                ].clone()
                            )

                        selection_hooks.append(
                            layer.register_forward_hook(capture_selection)
                        )
                try:
                    with imported_native_l20_prefill(
                        stage1.layers[0].attn, source
                    ), imported_delayed_pre_mix(
                        stage1.layers[0].attn_hc,
                        frame.hidden_states,
                        frame.prev_pre_mix,
                    ):
                        frame.prev_pre_mix.zero_()
                        actual = forward_layers(
                            stage1,
                            replica,
                            ids,
                            positions,
                            cu,
                            second_tables,
                            attn_inputs,
                            prepare_hidden_fn=lambda **kwargs: frame.hidden_states,
                        )
                finally:
                    for handle in selection_hooks:
                        handle.remove()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                torch.testing.assert_close(
                    F.linear(actual, stage1.head_weight),
                    F.linear(expected, full.head_weight),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(backing, before, rtol=0, atol=0)
                self.assertEqual(set(selected_owners), {24, 28, 32, 36})
                for selections in selected_owners.values():
                    torch.testing.assert_close(selections, source.topk, rtol=0, atol=0)
                assert_cleared(self, stage1, shared1)
                for (local, tag), group in first_cache.groups.items():
                    torch.testing.assert_close(
                        group.kv_cache_base,
                        full_cache.groups[local, tag].kv_cache_base,
                        rtol=0,
                        atol=0,
                    )
                for (local, tag), group in second_swa.groups.items():
                    torch.testing.assert_close(
                        group.kv_cache_base,
                        full_cache.groups[local + 21, tag].kv_cache_base,
                        rtol=0,
                        atol=0,
                    )
                source_views = native_l20_cache_views(TEXT, first_cache, 0)
                replica_views = native_l20_cache_views(TEXT, replica, 1)
                for request, span in enumerate(spans):
                    logical = torch.arange(span.start + span.count, device="cuda")
                    for tag in ("hca_kv", "indexer_kv"):
                        planned = []
                        for views, pages in (
                            (source_views, producer_pages),
                            (replica_views, replica_pages),
                        ):
                            slots = logical_to_physical_slots(
                                torch.tensor(pages[span.identity][tag], device="cuda"),
                                logical,
                                tag,
                                1,
                            )
                            planned.append(views[tag].gather(slots))
                        torch.testing.assert_close(
                            planned[0], planned[1], rtol=0, atol=0
                        )
                prefixes = [p + n for p, n in zip(prefixes, counts)]


if __name__ == "__main__":
    unittest.main()
