"""Wrapper equivalence and live Graph contracts for the Decode fusion batch."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import decode_fusions as fusion


class DecodeFusionContracts(unittest.TestCase):
    def test_switches_and_unsupported_inputs(self):
        with patch.dict(os.environ, {}, clear=True):
            for component in ("ENGRAM", "STATS", "SHARED", "ROUTES", "MASK", "JOURNAL", "METADATA"):
                self.assertTrue(fusion.enabled(component), component)
            os.environ["RTP_V41_PPU_DECODE_FUSIONS"] = "0"
            self.assertFalse(fusion.enabled("SHARED"))
            self.assertFalse(fusion.enabled("ENGRAM"))
            os.environ["RTP_V41_PPU_DECODE_FUSE_SHARED"] = "1"
            self.assertTrue(fusion.enabled("SHARED"))
            os.environ["RTP_V41_PPU_DECODE_FUSIONS"] = "1"
            os.environ["RTP_V41_PPU_DECODE_FUSE_SHARED"] = "0"
            self.assertFalse(fusion.enabled("SHARED"))
            self.assertTrue(fusion.enabled("ENGRAM"))
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(
                fusion.try_shared_activation_quant(torch.zeros(8, 4608), 10.0)
            )
            self.assertFalse(
                fusion.compact_routes_enabled(
                    NS(backend="deepep", _fake_balance_config=(True,))
                )
            )
            self.assertTrue(
                fusion.compact_routes_enabled(
                    NS(backend="megamoe", _fake_balance_config=(True,))
                )
            )

    def test_compact_gate_defers_count_and_avoids_padding(self):
        from rtp_llm.models_py.modules.dsv41.ppu import mega_gate

        hidden = torch.zeros(8, 5120)
        valid = torch.arange(8) % 3 != 0
        count = torch.tensor(-1)

        def native(*args, **kwargs):
            ids, weights = kwargs["out"]
            ids.copy_(
                torch.where(valid[:, None], torch.arange(ids.shape[1])[None, :], -1)
            )
            weights.fill_(0.25)

        for top_k in (3, 6):
            with patch.object(
                mega_gate, "_pack_dispatch", side_effect=AssertionError("padded routes")
            ):
                ids, weights = mega_gate._execute_gate(
                    hidden,
                    torch.zeros(128 if top_k == 3 else 384, 5120),
                    None,
                    top_k=top_k,
                    routed_scaling_factor=1.5,
                    image_rows=None,
                    router_bias_vl=None,
                    graph_routes=(valid, count),
                    ep_rank=0,
                    operation=native,
                    compact_graph_routes=True,
                )
            self.assertEqual(ids.shape, (8, top_k))
            self.assertEqual(weights.shape, ids.shape)
            self.assertEqual(count.item(), -1)
            self.assertTrue((ids[~valid] == -1).all())


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_GRAPH_DEVICE") == "1" and torch.cuda.is_available(),
    "explicit PPU qualification required",
)
class DecodeFusionDevice(unittest.TestCase):
    def setUp(self):
        self.flags = patch.dict(os.environ, {"RTP_V41_PPU_DECODE_FUSIONS": "1"})
        self.flags.start()
        self.addCleanup(self.flags.stop)

    @torch.inference_mode()
    def test_direct_journal_replay_restores_reserved_pages_and_publishes(self):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import VerifyGraphJournal
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import (
            BatchedVerifyPublication,
        )
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import (
            GraphPackedAttention,
        )

        rows, entries = 32, 128
        keys = [(0, "swa_kv"), (1, "swa_kv"), (1, "global_kv_2")]
        width = [528, 528, 288]
        pool = {k: torch.zeros(4, entries * w, dtype=torch.uint8, device="cuda")
                for k, w in zip(keys, width)}
        table = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int32).repeat(8, 1)
        row_table = table.repeat_interleave(4, dim=0)
        native = NativeGraphCache.__new__(NativeGraphCache)
        native.device = table.device
        native.geometry = NS(ratio=lambda _: 2)
        native.pool = lambda owner, tag: pool[(owner, tag)]
        native._table_tag = lambda _, tag: tag
        native.spans = {"swa_kv": entries, "global_kv_2": entries * 2}
        native.entries = dict.fromkeys(native.spans, entries)
        j = VerifyGraphJournal.__new__(VerifyGraphJournal)
        j.rows, j.native_cache = rows, native
        j.row_tables = dict.fromkeys(native.spans, row_table)
        j.requests = torch.arange(rows, device="cuda") // 4
        j.starts = torch.arange(8, device="cuda") * 4
        j.prefixes = torch.empty(8, dtype=torch.int64, device="cuda")
        j.valid = torch.ones(rows, dtype=torch.bool, device="cuda")
        pos = torch.empty(rows, dtype=torch.int64, device="cuda")
        raw = torch.empty_like(pos)
        global_raw = torch.full_like(pos, -1)
        j.generations = {k: torch.empty_like(pos) for k in keys}
        private = {k: torch.empty(rows, w, dtype=torch.uint8, device="cuda")
                   for k, w in zip(keys, width)}
        j.payloads = private
        publication = BatchedVerifyPublication.try_create(
            [(j.generations[k], private[k], pool[k], table, native.spans[k[1]], entries,
              2 if k[1] == "global_kv_2" else 1, k[1] == "global_kv_2", False, False)
             for k in keys], rows, table.device,
        )
        self.assertIsNotNone(publication)
        adapter = GraphPackedAttention.__new__(GraphPackedAttention)
        adapter.cache, adapter.journal = native, j
        adapter.offsets = torch.arange(entries, device=table.device)
        adapter.swa_domains = {0: ("swa_kv", 4), 1: ("swa_kv", 4)}
        adapter._tail_transaction_active = False
        pages = [pool[k].view(4, entries, w)[0, :rows] for k, w in zip(keys, width)]
        backups = [torch.empty_like(p) for p in pages]
        def pointers(tensors):
            return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device="cuda")
        adapter._tail_transaction = NS(
            pages=pages, backups=backups, page_ptrs=pointers(pages), backup_ptrs=pointers(backups),
            sizes=torch.tensor([p.numel() for p in pages], dtype=torch.int32, device="cuda"),
            max_size=max(p.numel() for p in pages),
        )
        adapter._direct_journal = NS(
            private=private, mapped=dict(zip(keys,pages)), payload_ptrs=pointers(list(private.values())),
        )
        incoming = {k: torch.empty_like(v) for k,v in private.items()}

        def forward(fail=False):
            with adapter.verify_tail_transaction():
                self.assertIs(j.payloads, adapter._direct_journal.mapped)
                for k in keys:
                    j.write_rows(*k, raw if k[1] == "swa_kv" else global_raw, incoming[k])
                with adapter._stage_verify_tail(keys[1:]):
                    if fail:
                        raise RuntimeError("direct journal consumer")
                shared = {}
                one = adapter._swa_indices(0, pos, j.valid, shared, verify_slots=raw)
                two = adapter._swa_indices(1, pos, j.valid, shared, verify_slots=raw)
                self.assertIs(one[0], two[0])
            self.assertIs(j.payloads, private)
            return one

        def inputs(step):
            # Every request has a distinct old sequence, including a page edge.
            j.prefixes.copy_(torch.arange(8, device="cuda") * 4 + 126 + step)
            pos.copy_(j.prefixes[j.requests] + torch.arange(rows, device="cuda") % 4)
            j.valid.fill_(True)
            j.valid[-4 * (step + 1):] = False
            raw.copy_(torch.where(j.valid, pos, -1))
            global_raw.copy_(torch.where(j.valid & (pos % 2 == 1), pos-1, -1))
            for k in keys:
                incoming[k].random_(0, 256)
                pool[k].random_(0, 256)

        inputs(0)
        before = [p.clone() for p in pages]
        with self.assertRaisesRegex(RuntimeError, "direct journal consumer"):
            forward(fail=True)
        for p, old in zip(pages, before):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
        self.assertIs(j.payloads, private)
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_slots = forward()
        for i in (0, 1):
            inputs(i)
            # Mapping changes as well as payload and token positions.
            row_table[:, 1] = 2 if not i else 3
            table[:, 1] = 2 if not i else 3
            before = {k:v.clone() for k,v in pool.items()}
            graph.replay()
            ends = j.prefixes + (1 if i == 0 else 2)
            for k in keys:
                torch.testing.assert_close(private[k], incoming[k], rtol=0, atol=0)
                torch.testing.assert_close(pool[k], before[k], rtol=0, atol=0)
            expected = adapter._native_indices(0, "swa_kv", pos, j.valid, entries)
            for actual, wanted in zip(replay_slots, expected):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            count = publication.publish(j.requests, j.prefixes, ends, j.valid)
            self.assertIsNotNone(count)
            for k,w in zip(keys,width):
                want = before[k].view(4, entries, w)
                ratio = 2 if k[1] == "global_kv_2" else 1
                span = native.spans[k[1]]
                for token in range(rows):
                    request = token // 4
                    generation = int(j.generations[k][token])
                    if int(j.prefixes[request])//ratio <= generation//ratio < int(ends[request])//ratio:
                        block = int(table[request, generation//span])
                        want[block, (generation % span)//ratio].copy_(incoming[k][token])
                torch.testing.assert_close(pool[k].view_as(want), want, rtol=0, atol=0)

    @torch.inference_mode()
    def test_shared_silu_quant_matches_two_kernel_boundary_and_live_graph(self):
        from rtp_llm.models_py.modules.dsv41.moe._silu_mul_bf16_triton import (
            silu_mul_split_bf16,
        )
        from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import (
            quantize_ptpc_activation,
        )

        for rows in (8, 24, 32):
            values = torch.randn(rows, 4608, device="cuda", dtype=torch.bfloat16) * 7
            fusion.try_shared_activation_quant(values, 10.0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = fusion.try_shared_activation_quant(values, 10.0)
            for i in range(3):
                values.normal_(0, 7)
                values[:, :6] = torch.tensor(
                    [-12.0, -10.0, -0.0, 0.0, 10.0, 12.0], device="cuda"
                )
                if i == 1:
                    values[::2].zero_()
                graph.replay()
                expected = quantize_ptpc_activation(
                    silu_mul_split_bf16(values, clamp_limit=10.0)
                )
                self.assertTrue(
                    torch.equal(got[0].view(torch.uint8), expected[0].view(torch.uint8))
                )
                torch.testing.assert_close(got[1], expected[1], rtol=0, atol=0)

    @torch.inference_mode()
    def test_mask_quant_projection_preserves_live_values_idle_nan_and_ownership(self):
        from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear

        values = torch.randn(32, 8, 1024, device="cuda", dtype=torch.bfloat16)
        valid = torch.arange(32, device="cuda") % 3 != 0
        weight = (torch.randn(128, 8192, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        linear = PPUPTPCLinear(weight, torch.ones(128, 1, device="cuda"))
        fusion.masked_output_projection(linear, values, valid)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = fusion.masked_output_projection(linear, values, valid)
        for i in range(3):
            values.normal_()
            valid.fill_(i != 2)
            if i == 1:
                valid[::3] = False
            values[~valid] = float("nan")
            graph.replay()
            expected = linear(torch.where(valid[:, None], values.flatten(1), 0))
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            owned = got.clone()
            saved = owned.clone()
            values.zero_()
            graph.replay()
            torch.testing.assert_close(owned, saved, rtol=0, atol=0)

    @torch.inference_mode()
    def test_compact_balance_matches_padded_reference_for_live_slot_masks(self):
        from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE

        for top_k in (3, 6):
            engine = PPUMoE.__new__(PPUMoE)
            engine.backend, engine.ep_size = "megamoe", 8
            engine.num_routed_experts = 128 if top_k == 3 else 384
            for policy in ("rtp", "sail_round_robin"):
                engine._fake_balance_config = (True, top_k, 3, 8)
                engine._fake_balance_policy = policy
                ids = torch.arange(32 * top_k, device="cuda").reshape(32, top_k).long()
                weights = torch.ones_like(ids, dtype=torch.float32)
                valid = torch.ones(32, device="cuda", dtype=torch.bool)
                count = torch.empty((), device="cuda", dtype=torch.int64)

                def forward():
                    return engine._apply_fake_balance(
                        ids,
                        weights,
                        layer_id=14,
                        routed_count=count,
                        route_valid_mask=valid,
                    )

                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    got = forward()
                for i in range(3):
                    ids.random_(0, engine.num_routed_experts)
                    ids[::4, 1] = -7
                    valid.fill_(i != 2)
                    valid[::3] = False
                    graph.replay()
                    padded = torch.nn.functional.pad(
                        torch.where(valid[:, None], ids, -1), (0, 8 - top_k), value=-1
                    )
                    reference = engine._apply_fake_balance(
                        padded,
                        torch.ones_like(padded, dtype=torch.float32),
                        layer_id=14,
                    )
                    torch.testing.assert_close(got[0], reference[0], rtol=0, atol=0)
                    torch.testing.assert_close(got[1], reference[1], rtol=0, atol=0)
                    torch.testing.assert_close(count, valid.sum(), rtol=0, atol=0)

    @torch.inference_mode()
    def test_expert_statistics_are_exact_including_int32_wrap(self):
        from rtp_llm.models_py.modules.dsv41.ppu.moe import _get_triton_kernels

        kernels = _get_triton_kernels()
        current = torch.arange(48, device="cuda", dtype=torch.int32) + (2**31 - 64)
        previous = current.clone()
        counts = torch.arange(48, device="cuda", dtype=torch.int32)

        def forward():
            kernels.accumulate_mega_expert_load[(1,)](
                counts, current, previous, EXPERTS=48, BLOCK=128
            )

        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        for _ in range(3):
            current.add_(100)
            expected = counts + (current - previous)
            graph.replay()
            torch.testing.assert_close(counts, expected, rtol=0, atol=0)
            previous.copy_(current)

    @torch.inference_mode()
    def test_engram_gate_matches_bf16_oracle_and_live_mask(self):
        from rtp_llm.models_py.modules.dsv41.ppu.engram import engram_gate_inject

        torch.manual_seed(321)
        hidden = torch.randn(32, 4, 5120, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(32, 5 * 5120, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        mask = torch.ones(32, device="cuda", dtype=torch.bool)
        engram_gate_inject(hidden, qw, kw, kv, token_mask=mask, fused_mask=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = engram_gate_inject(
                hidden, qw, kw, kv, token_mask=mask, fused_mask=True
            )
        differences = []
        for i in range(3):
            hidden.normal_()
            kv.normal_()
            mask.fill_(i != 2)
            if i == 1:
                mask[::3] = False
            graph.replay()
            expected = engram_gate_inject(
                hidden, qw, kw, kv, token_mask=mask, fused_mask=False
            )

            # FP32 reduction trees can cross a BF16 rounding boundary. ULPs
            # alone are unsuitable after cancellation near zero: also bound
            # FP32 error relative to the two terms of the final addition.
            def ordered(value):
                bits = value.view(torch.int16).int()
                return torch.where(bits < 0, -32768 - bits, bits)

            delta = (ordered(got) - ordered(expected)).abs()
            absolute = (got.float() - expected.float()).abs()
            record = dict(
                iteration=i,
                different=int((delta != 0).sum()),
                max_bf16_ulp=int(delta.max()),
                max_absolute=float(absolute.max()),
                over_one_ulp=int((delta > 1).sum()),
                over_one_ulp_max_absolute=float(
                    torch.where(delta > 1, absolute, 0).max()
                ),
            )
            print("ENGRAM_ROUNDING", json.dumps(record), flush=True)
            shared = kv[:, 4 * 5120 :].reshape(32, 1, 5120).float()
            fp32_budget = (hidden.float().abs() + shared.abs()) * (8 * 2**-23)
            self.assertTrue(bool(((delta <= 1) | (absolute <= fp32_budget)).all()))
            torch.testing.assert_close(got[~mask], hidden[~mask], rtol=0, atol=0)
            differences.append(record)
        (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "engram-rounding.json"
        ).write_text(json.dumps(differences))

    @torch.inference_mode()
    def test_joint_journal_slots_match_independent_domains_and_live_tables(self):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import VerifyGraphJournal

        for ratio in (1, 2):
            raw = torch.arange(32, device="cuda", dtype=torch.int64) * 127
            tables = {
                tag: torch.arange(1, 35, device="cuda", dtype=torch.int32).repeat(32, 1)
                for tag in ("global_kv_1", "indexer_kv")
            }
            pool = torch.empty(36, 1, device="cuda")
            native = NS(
                _table_tag=lambda owner, tag: tag,
                geometry=NS(ratio=lambda owner: ratio),
                spans={tag: 128 for tag in tables},
                entries={tag: 128 // ratio for tag in tables},
                pool=lambda *_: pool,
            )
            journal = VerifyGraphJournal.__new__(VerifyGraphJournal)
            journal.native_cache, journal.row_tables = native, tables
            fusion.try_joint_journal_slots(journal, 0, "global_kv_1", raw)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = fusion.try_joint_journal_slots(journal, 0, "global_kv_1", raw)
            for i in range(3):
                raw.add_(1)
                raw[::4] = -1
                tables["indexer_kv"][:, 2] = 0 if i == 1 else 3
                graph.replay()
                for actual, tag in zip(got, tables):
                    torch.testing.assert_close(
                        actual, journal.slots(0, tag, raw), rtol=0, atol=0
                    )

    @torch.inference_mode()
    def test_staging_fuses_into_metadata_and_restores_on_failure(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import (
            GraphPackedAttention,
        )

        rows = 32
        positions = torch.arange(rows, device="cuda", dtype=torch.int64) % 4 + 2048
        valid = torch.ones(rows, device="cuda", dtype=torch.bool)
        keys, widths = [(0, "swa_kv"), (0, "global_kv_1")], [528, 288]
        views = {
            key: torch.randint(
                256, (36, 128, 1, width), device="cuda", dtype=torch.uint8
            )
            for key, width in zip(keys, widths)
        }
        payloads = {
            key: torch.randint(256, (rows, width), device="cuda", dtype=torch.uint8)
            for key, width in zip(keys, widths)
        }
        pages = [views[key][0, :rows, 0] for key in keys]
        backups = [torch.empty_like(p) for p in pages]
        native = NativeGraphCache.__new__(NativeGraphCache)
        native._table_tag = lambda *_: "swa"
        native.pool = lambda *_: views[keys[0]].flatten(1)
        native.spans, native.entries = {"swa": 128}, {"swa": 128}
        native.geometry = NS(ratio=lambda *_: 1)
        table = torch.arange(1, 35, device="cuda", dtype=torch.int32).repeat(rows, 1)
        adapter = GraphPackedAttention.__new__(GraphPackedAttention)
        adapter.cache, adapter.views = native, views
        adapter.journal = NS(
            rows=rows,
            payloads=payloads,
            row_tables={"swa": table},
            requests=torch.arange(rows, device="cuda") // 4,
            starts=torch.arange(0, rows, 4, device="cuda"),
            prefixes=torch.full((8,), 2048, device="cuda", dtype=torch.int64),
            generations={keys[0]: positions.clone()},
        )
        adapter._tail_transaction_active = False
        adapter._tail_transaction = NS(
            pages=pages,
            backups=backups,
            page_ptrs=torch.tensor(
                [p.data_ptr() for p in pages], device="cuda", dtype=torch.int64
            ),
            backup_ptrs=torch.tensor(
                [p.data_ptr() for p in backups], device="cuda", dtype=torch.int64
            ),
            sizes=torch.tensor(
                [p.numel() for p in pages], device="cuda", dtype=torch.int32
            ),
            max_size=max(p.numel() for p in pages),
        )
        observed = [torch.empty_like(p) for p in pages]

        def forward(observe=True, fail=False):
            with adapter.verify_tail_transaction():
                indices = adapter._native_indices(
                    0,
                    "swa_kv",
                    positions,
                    valid,
                    128,
                    shared={},
                    stage=(pages, list(payloads.values())),
                )
                with adapter._stage_verify_tail(keys, already_staged=True):
                    if observe:
                        for output, page in zip(observed, pages):
                            output.copy_(page)
                    if fail:
                        raise RuntimeError("consumer failure")
            return indices

        forward()
        before = [p.clone() for p in pages]
        with self.assertRaisesRegex(RuntimeError, "consumer failure"):
            forward(fail=True)
        for p, old in zip(pages, before):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = forward()
        for i in range(3):
            for p in pages:
                p.random_(0, 256)
            for payload in payloads.values():
                payload.random_(0, 256)
            before = [p.clone() for p in pages]
            valid[::3] = i != 1
            graph.replay()
            for output, payload in zip(observed, payloads.values()):
                torch.testing.assert_close(output, payload, rtol=0, atol=0)
            for p, old in zip(pages, before):
                torch.testing.assert_close(p, old, rtol=0, atol=0)
            expected = adapter._native_indices(
                0, "swa_kv", positions, valid, 128, shared={}
            )
            for actual, reference in zip(got, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            forward(observe=False)
            torch.cuda.synchronize()
        path = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
            / "journal-metadata.trace.json"
        )
        profile.export_chrome_trace(str(path))
        kernels = [
            e["name"]
            for e in json.loads(path.read_text())["traceEvents"]
            if e.get("cat") == "kernel"
        ]
        self.assertEqual(len(kernels), 3, kernels)
        self.assertEqual(sum("copy_reserved_pages" in name for name in kernels), 2)
