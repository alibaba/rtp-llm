"""CPU contracts for bulk integers and unchanged segment float orchestration."""

import ast
import importlib.util
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "rtp_llm.models_py.modules.dsv4.fp8"
SPEC = importlib.util.spec_from_file_location(
    "producer_metadata_cpu", ROOT / "_v41_producer_metadata.py"
)
metadata = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metadata
SPEC.loader.exec_module(metadata)


def functions():
    path = ROOT / "attention_v41.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AttentionV41FP8"
    )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_prefill_x_tile_plan"
    ]
    nodes += [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_produce_global", "_write_states", "_begin_forward")
    ]
    scope = dict(
        os=os,
        torch=torch,
        F=F,
        _PRODUCE_GLOBAL_TILE_ROWS=32768,
        INDEXER_KV="index",
        CSA_STATE="state",
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def inputs(prefixes, lengths):
    cp = SimpleNamespace(
        cp_size=4,
        prefix_lengths_host=tuple(prefixes),
        input_lengths_global_host=tuple(lengths),
        chunk_lengths_per_req=tuple(2 * ((n + 7) // 8) for n in lengths),
    )
    pos = torch.tensor([p for s, n in zip(prefixes, lengths) for p in range(s, s + n)])
    req = torch.tensor([b for b, n in enumerate(lengths) for _ in range(n)])
    return cp, pos, req, torch.tensor(prefixes), torch.tensor(lengths)


def slot_reference(region, positions, requests, state_end=None, rank=0):
    # Independent scalar model: virtual TPB512, state ring EB2 per CP rank.
    result = []
    for p, b in zip(positions.tolist(), requests.tolist()):
        if region == "state":
            block = b * 16 + (p // 512) % 8 + 1
            valid = (p % 8) // 2 == rank
            if state_end is not None:
                valid &= p + 8 >= min((p // 512 + 1) * 512, int(state_end[b]))
            slot = block * 2 + p % 2
        else:
            block = b * 16 + p // 512 + 1
            valid = (p // 128) % 4 == rank
            slot = block * 512 + p % 512
        result.append(slot if valid else -1)
    return torch.tensor(result, dtype=torch.int64)


class ProducerMetadataCPU(unittest.TestCase):
    def test_product_default_gate_excludes_unsharded_cp_and_single_request(self):
        tree = ast.parse((ROOT / "attention_v41.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AttentionV41FP8"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_produce_global"
        )
        gate = next(
            n.test
            for n in fn.body
            if isinstance(n, ast.If) and "kv_cache_sharded" in ast.unparse(n.test)
        )
        code = compile(ast.Expression(gate), "producer_gate", "eval")
        scope = dict(
            prefill=True,
            main_pool=object(),
            index_pool=object(),
            cp=SimpleNamespace(cp_size=4, kv_cache_sharded=True),
            lengths_host=[1, 1],
            raw_tiles=None,
            projected_tiles=None,
        )
        self.assertTrue(eval(code, scope))
        self.assertNotIn("environ", ast.unparse(gate))
        for change in (
            dict(cp=SimpleNamespace(cp_size=4, kv_cache_sharded=False)),
            dict(cp=SimpleNamespace(cp_size=1, kv_cache_sharded=True)),
            dict(lengths_host=[1]),
            dict(raw_tiles=object()),
            dict(projected_tiles=object()),
        ):
            self.assertFalse(eval(code, {**scope, **change}))

    def test_existing_forward_boundary_discards_counts_and_bounds(self):
        shared = {
            "layers": {0: object(), 8: object()},
            "prefill_score_bounds": {("producer_key_counts", 2): object()},
            "prefill_sparse_plans": object(),
        }
        scope = functions()
        scope["_begin_forward"](SimpleNamespace(layer_id=8, _shared_attention=shared))
        self.assertIn("prefill_score_bounds", shared)
        scope["_begin_forward"](SimpleNamespace(layer_id=0, _shared_attention=shared))
        self.assertNotIn("prefill_score_bounds", shared)
        self.assertNotIn("prefill_sparse_plans", shared)

    def test_structured_layouts_keep_physical_capacities_independent(self):
        table = torch.arange(32 * 10, dtype=torch.int32).reshape(32, 10)[:, :8]
        layouts = tuple(
            metadata.SlotLayout(table, eb, tpb, 128, 4, 1)
            for eb, tpb in ((64, 128), (128, 128), (3, 512))
        )
        self.assertTrue(metadata._slot_layouts_supported(layouts, 2, 32, table.device))
        from dataclasses import replace

        for bad in (
            layouts[:2],
            (replace(layouts[0], entries_per_block=63), *layouts[1:]),
            (replace(layouts[0], owner_tokens_per_block=64), *layouts[1:]),
            (layouts[0], replace(layouts[1], cp_rank=2), layouts[2]),
            (layouts[0], replace(layouts[1], table=table.t()), layouts[2]),
            (layouts[0], replace(layouts[1], table=table.long()), layouts[2]),
        ):
            self.assertFalse(metadata._slot_layouts_supported(bad, 2, 32, table.device))
        self.assertTrue(
            metadata._slot_layouts_supported(
                (replace(layouts[0], entries_per_block=128), layouts[1]),
                1,
                32,
                table.device,
            )
        )

    def test_batched_selection_reuses_only_matching_producer_counts(self):
        spec = importlib.util.spec_from_file_location(
            PACKAGE + ".select_counts_test", ROOT / "_v41_batched_prefill_select.py"
        )
        select = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(select)
        prefix, length = torch.tensor([128, 256]), torch.tensor([3, 5])
        cp = SimpleNamespace(prefix_lengths=prefix, input_lengths_global=length)
        counts = (prefix + length).div(2, rounding_mode="floor").int()
        score = Mock(return_value=None)
        package = ModuleType(PACKAGE)
        package._v41_prefill_topk = SimpleNamespace()
        with patch.dict(
            sys.modules,
            {
                PACKAGE: package,
                PACKAGE
                + "._v41_grouped_prefill_score": SimpleNamespace(
                    _mask_tail_kernel=None, try_grouped_scores=score
                ),
                PACKAGE
                + ".attention_v41": SimpleNamespace(_apply_prefill_candidates=None),
            },
        ):
            for same in (True, False):
                shared = {
                    "prefill_score_bounds": {
                        ("producer_key_counts", 2): (
                            prefix if same else prefix.clone(),
                            length,
                            counts,
                        )
                    }
                }
                attn = SimpleNamespace(
                    _shared_attention=shared, _cp_ctx=cp, compress_ratio=2
                )
                self.assertFalse(
                    select.try_select_batched(
                        attn,
                        SimpleNamespace(is_cuda=True),
                        None,
                        None,
                        [None, None],
                        (slice(0, 1), slice(1, 2)),
                        None,
                        None,
                        candidate_source=-1,
                        publish_candidates=False,
                        candidate_size=8,
                        candidate_blocks=2048,
                        req_ids=torch.tensor([0, 1]),
                    )
                )
                actual = score.call_args.kwargs["key_counts"]
                torch.testing.assert_close(actual, counts, rtol=0, atol=0)
                self.assertEqual(actual is counts, same)

            shared = {
                "prefill_score_bounds": {
                    ("producer_key_counts", 2): (prefix, length, counts)
                }
            }
            attn._shared_attention = shared

            def consume():
                select.try_select_batched(
                    attn,
                    SimpleNamespace(is_cuda=True),
                    None,
                    None,
                    [None, None],
                    (slice(0, 1), slice(1, 2)),
                    None,
                    None,
                    candidate_source=-1,
                    publish_candidates=False,
                    candidate_size=8,
                    candidate_blocks=2048,
                    req_ids=torch.tensor([0, 1]),
                )
                return score.call_args.kwargs["key_counts"]

            first = consume()
            cache = shared["prefill_score_bounds"]
            cache[("producer_key_counts", 2)] = (prefix, length, counts.clone())
            self.assertIs(consume(), first)
            cache[("grouped", "stale")] = object()
            shared["prefill_sparse_plans"] = object()
            cp.prefix_lengths = prefix + 10
            replaced = consume()
            torch.testing.assert_close(
                replaced, ((cp.prefix_lengths + length) // 2).int(), rtol=0, atol=0
            )
            self.assertIsNot(replaced, first)
            self.assertNotIn(("grouped", "stale"), cache)
            self.assertNotIn("prefill_sparse_plans", shared)

    def test_raw_tile_rows_default_and_finite_gate(self):
        cp, pos, req, starts, lengths = inputs([1, 0], [65535, 130])
        self.assertEqual(metadata._RAW_TILE_ROWS, 32768)
        for ratio in (1, 2):
            self.assertEqual(
                metadata._raw_plan(cp, ratio), metadata._raw_plan(cp, ratio, 32768)
            )
            full = torch.where((pos + 1) % ratio == 0)[0]
            segments, rows, count = metadata._raw_plan(cp, ratio, 65536)
            self.assertEqual(list(segments), [0, 65536])
            self.assertEqual(count, len(full))
            for start, (end, first, stop, _) in segments.items():
                self.assertEqual(end, min(start + 65536, rows))
                expected = torch.where((pos[start:end] + 1) % ratio == 0)[0]
                torch.testing.assert_close(
                    full[first:stop] % 65536, expected, rtol=0, atol=0
                )
        for tile_rows in (None, True, 0, -32768, 16384, 65535, 131072, 65536.0):
            with self.subTest(tile_rows=tile_rows), patch.object(
                torch, "empty", side_effect=AssertionError("invalid tile allocates")
            ):
                self.assertIsNone(metadata._raw_plan(cp, 2, tile_rows))
                self.assertIsNone(
                    metadata.prepare_raw(
                        cp,
                        pos,
                        req,
                        starts,
                        lengths,
                        2,
                        Mock(),
                        (1, 2, 3),
                        tile_rows=tile_rows,
                    )
                )

    def test_raw_host_plan_preserves_32768_cross_request_boundaries(self):
        for ratio in (1, 2):
            for prefixes, lengths in (
                ([0, 1, 512, 513], [32767, 1, 1, 32770]),
                ([0] * 32, [1] * 32),
                ([i % 2 for i in range(32)], [1 + i * 137 for i in range(32)]),
                ([1, 0], [32768, 32769]),
            ):
                cp, pos, _, _, _ = inputs(prefixes, lengths)
                segments, rows, count = metadata._raw_plan(cp, ratio)
                full = torch.where((pos + 1) % ratio == 0)[0]
                self.assertEqual(count, len(full))
                self.assertEqual(rows, sum(lengths))
                for start, (end, first, stop, _) in segments.items():
                    self.assertEqual(start % 32768, 0)
                    self.assertEqual(end, min(start + 32768, rows))
                    expected = torch.where((pos[start:end] + 1) % ratio == 0)[0]
                    torch.testing.assert_close(
                        full[first:stop] % 32768, expected, rtol=0, atol=0
                    )
                if ratio == 2 and prefixes == [0] * 32:
                    self.assertEqual(count, 0)

    def test_raw_gates_cpu_metadata_and_budget_before_allocation(self):
        cp, pos, req, starts, lengths = inputs([0, 1], [129, 257])
        for changes in (
            {"cp_size": 1},
            {"prefix_lengths_host": None},
            {"prefix_lengths_host": (0, -1)},
            {"input_lengths_global_host": (0, 386)},
        ):
            self.assertIsNone(
                metadata._raw_plan(SimpleNamespace(**{**vars(cp), **changes}), 2)
            )
        with patch.object(
            torch, "empty", side_effect=AssertionError("CPU fallback allocates nothing")
        ):
            self.assertIsNone(
                metadata.prepare_raw(
                    cp, pos, req, starts, lengths, 2, Mock(), (1, 2, 3)
                )
            )
        with patch.object(metadata, "_MAX_METADATA_BYTES", 1):
            self.assertIsNone(metadata._raw_plan(cp, 2))

    def test_warmup_covers_odd_slot_views_for_all_three_writers(self):
        spec = importlib.util.spec_from_file_location(
            PACKAGE + ".producer_warmup_cpu", ROOT / "_v41_attention_jit_warmup.py"
        )
        warmup = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = warmup
        spec.loader.exec_module(warmup)
        seen = {"main": set(), "index": set(), "state": set()}

        def main(*args):
            seen["main"].add((args[4].data_ptr() % 16, args[11].data_ptr() % 16))
            return torch.empty((len(args[8]), 512), dtype=torch.bfloat16)

        def index(*args):
            seen["index"].add((args[3].data_ptr() % 16, args[6].data_ptr() % 16))
            return True

        def state(*args):
            seen["state"].add(args[2].data_ptr() % 16)
            return True

        fp4 = SimpleNamespace(
            **{
                name: Mock(return_value=torch.empty(0))
                for name in (
                    "quantize_indexer_k_fp4",
                    "gather_indexer_k_fp4",
                    "dequantize_indexer_k_fp4",
                    "quantize_and_insert_k_cache_fp4",
                    "dequantize_k_cache_slots_fp4",
                    "gather_k_cache_bytes_fp4",
                    "dequantize_k_cache_bytes_fp4",
                )
            }
        )
        package = SimpleNamespace(
            _v41_grouped_gemm=SimpleNamespace(
                _enabled=lambda: False, _global_enabled=lambda: False
            ),
            _v41_batched_producer=SimpleNamespace(is_supported=lambda t: False),
            _v41_fp4_triton=fp4,
            _v41_prefill_global=SimpleNamespace(
                store_index=index,
                store_states=state,
                compress_main=main,
                _enabled=lambda t: True,
            ),
            _v41_prefill_metadata=SimpleNamespace(
                try_slot_mapping=lambda *a, **kw: torch.empty(0),
                try_chunk_metadata=lambda *a, **kw: (torch.empty(0),) * 3,
            ),
            _v41_swa_triton=SimpleNamespace(),
        )
        names = SimpleNamespace(CSA_KV=1, HCA_KV=2, INDEXER_KV=3, CSA_STATE=5, SWA_KV=7)
        attn = SimpleNamespace(
            window_size=128,
            global_norm=torch.ones(512),
            index_k_norm=torch.ones(128),
            eps=1e-6,
            freqs_cis=torch.ones(64, 32, dtype=torch.complex64),
        )
        with patch.dict(
            sys.modules,
            {
                PACKAGE: package,
                "rtp_llm.models_py.modules.dsv4.attn_type": names,
                PACKAGE
                + "._v41_prefill_pools": SimpleNamespace(
                    _group_row_metadata=lambda *a: None
                ),
            },
        ), patch.object(warmup, "_warm_raw_producer_metadata") as raw_warmup:
            for region, width in ((1, 288), (2, 288), (3, 68), (5, 4096)):
                warmup._warm_pool(
                    attn,
                    warmup.PoolLayout(
                        region, 16, 16 * width, 128, 128, 2 if region == 1 else 1
                    ),
                    4,
                    0,
                    32,
                    "cpu",
                )
        self.assertEqual(raw_warmup.call_args_list, [call(2, "cpu")])
        self.assertEqual(seen["main"], {(0, 0), (0, 8), (8, 0), (8, 8)})
        self.assertEqual(seen["index"], seen["main"])
        self.assertEqual(seen["state"], {0, 8})

    def test_all_ranks_odd_prefix_right_endpoints_ring_and_views(self):
        scope = functions()
        for ratio in (1, 2):
            for prefixes, lengths in (
                ([0, 1, 512, 513], [1, 7, 129, 2049]),
                ([4095, 4096, 4097], [33, 513, 1025]),
            ):
                cp, pos, req, starts, sizes = inputs(prefixes, lengths)
                tiles = tuple(scope["_prefill_x_tile_plan"](cp))
                for rank in range(4):
                    mapper = Mock(
                        side_effect=lambda region, p, b, **kw: slot_reference(
                            region, p, b, rank=rank, **kw
                        )
                    )
                    plan = metadata.prepare(
                        cp,
                        pos,
                        req,
                        starts,
                        sizes,
                        ratio,
                        tiles,
                        mapper,
                        ("main", "index", "state"),
                    )
                    self.assertIsNotNone(plan)
                    self.assertEqual(mapper.call_count, 3 if ratio == 2 else 2)
                    begin = 0
                    for _, _, count in tiles:
                        end = begin + count
                        idx = torch.tensor(
                            [
                                i
                                for i in range(count)
                                if (int(pos[begin + i]) + 1) % ratio == 0
                            ],
                            dtype=torch.long,
                        )
                        actual = plan.tile(begin, end)
                        wanted = (idx, pos[begin:end][idx], req[begin:end][idx])
                        for got, expected in zip(actual[:3], wanted):
                            torch.testing.assert_close(got, expected, rtol=0, atol=0)
                            self.assertTrue(got.is_contiguous())
                        for region, got in zip(("main", "index"), actual[3:5]):
                            torch.testing.assert_close(
                                got,
                                slot_reference(region, wanted[1], wanted[2], rank=rank),
                                rtol=0,
                                atol=0,
                            )
                        if ratio == 2:
                            torch.testing.assert_close(
                                actual[5],
                                slot_reference(
                                    "state",
                                    pos[begin:end],
                                    req[begin:end],
                                    starts + sizes,
                                    rank,
                                ),
                                rtol=0,
                                atol=0,
                            )
                        self.assertEqual(
                            actual[3].untyped_storage().data_ptr(),
                            plan.main_slots.untyped_storage().data_ptr(),
                        )
                        begin = end
                    self.assertEqual(begin, len(pos))

    def test_gates_budget_and_wrong_segments(self):
        scope = functions()
        cp, pos, req, starts, sizes = inputs([0, 1], [129, 257])
        tiles = tuple(scope["_prefill_x_tile_plan"](cp))
        mapper = Mock(side_effect=slot_reference)
        for changes in ({"cp_size": 1}, {"prefix_lengths_host": None}):
            bad = SimpleNamespace(**{**vars(cp), **changes})
            self.assertIsNone(
                metadata.prepare(
                    bad,
                    pos,
                    req,
                    starts,
                    sizes,
                    2,
                    tiles,
                    mapper,
                    ("main", "index", "state"),
                )
            )
        with patch.object(metadata, "_MAX_METADATA_BYTES", 1), patch.object(
            torch, "arange", side_effect=AssertionError("must gate before allocation")
        ):
            self.assertIsNone(
                metadata.prepare(
                    cp,
                    pos,
                    req,
                    starts,
                    sizes,
                    2,
                    tiles,
                    mapper,
                    ("main", "index", "state"),
                )
            )
        mapper.assert_not_called()
        self.assertIsNone(
            metadata.prepare(
                cp,
                pos,
                req,
                starts,
                sizes,
                2,
                [(0, 0, len(pos))],
                mapper,
                ("main", "index", "state"),
            )
        )
        plan = metadata.prepare(
            cp, pos, req, starts, sizes, 2, tiles, mapper, ("main", "index", "state")
        )
        with self.assertRaises(ValueError):
            plan.tile(0, 1)

    def _producer(self, ratio, use_metadata, state_fast=True):
        scope = functions()
        cp, positions, requests, starts, lengths = inputs(
            [0, 1, 512, 513], [5, 16, 129, 257]
        )
        tiles = tuple(scope["_prefill_x_tile_plan"](cp))
        total, head = len(positions), 4
        x = torch.arange(total * head).reshape(total, head).remainder(61).bfloat16()
        owner = SimpleNamespace(
            global_wkv=torch.eye(head).bfloat16(),
            global_wgate=torch.eye(head).bfloat16() / 8,
            global_norm=torch.ones(head),
            index_wk=torch.eye(head).bfloat16(),
            index_k_norm=torch.ones(head),
        )
        pools = {
            name: torch.zeros(40000, head if name != "state" else 2 * head)
            for name in ("main", "index", "state")
        }
        events, mappings = [], []

        def slots(region, pos, req, **kwargs):
            mappings.append(region)
            return slot_reference(region, pos, req, **kwargs)

        def store(data, pool, mapped):
            valid = mapped >= 0
            pool[mapped[valid]] = data[valid].float()

        def state_store(v, s, mapped, pool):
            events.append(("state", v.clone(), s.clone(), mapped.clone()))
            if state_fast:
                store(torch.cat((v, s), 1), pool, mapped)
            return state_fast

        def compress(
            v,
            s,
            norm,
            eps,
            pos,
            req,
            starts,
            previous,
            idx,
            freqs,
            pool,
            mapped,
            actual_ratio,
            carry,
        ):
            events.append(
                (
                    "main",
                    v.clone(),
                    idx.clone(),
                    pos.clone(),
                    req.clone(),
                    mapped.clone(),
                )
            )
            if s is not None:
                events.append(("scores", s.clone()))
            if carry is not None:
                events.append(("carry", carry[0].clone(), carry[1].clone()))
            latent = v[idx].bfloat16()
            store(latent, pool, mapped)
            return latent

        def index_store(projected, norm, eps, pos, freqs, pool, mapped, actual_ratio):
            events.append(("index", projected.clone(), pos.clone(), mapped.clone()))
            store(projected, pool, mapped)
            return True

        float_stages = SimpleNamespace(
            compress_main=compress, store_states=state_store, store_index=index_store
        )
        scope.update(
            prefill_global=float_stages,
            prefill_indexer=SimpleNamespace(is_supported=lambda *a: False),
            fp8_roundtrip=lambda x: x,
        )
        source = SimpleNamespace(
            compress_ratio=ratio,
            head_dim=head,
            eps=1e-6,
            layer_id=20,
            freqs_cis=None,
            _cp_ctx=cp,
            _owner=lambda: owner,
            _global_region=lambda: "main",
            _source_pool=lambda region: pools[region],
            _slots=slots,
            _read_state=lambda *a: torch.zeros(4, 8),
            _gather_shards=lambda x: x,
            _shared_attention={},
        )
        source._write_states = lambda *a, **kw: scope["_write_states"](source, *a, **kw)
        package = ModuleType(PACKAGE)
        package._v41_producer_metadata = metadata
        compressor = ModuleType(PACKAGE + ".compressor")

        def linear(tensor, weight):
            events.append(("gemm", tensor.clone()))
            return F.linear(tensor.float(), weight.float())

        compressor._linear_bf16_bf16_fp32 = linear
        codec = ModuleType(PACKAGE + "._v41_fp4_triton")
        codec.quantize_and_insert_k_cache_fp4 = Mock(
            side_effect=AssertionError("unexpected float fallback")
        )
        codec.quantize_indexer_k_fp4 = Mock(
            side_effect=AssertionError("unexpected float fallback")
        )
        codec.dequantize_indexer_k_fp4 = lambda pool, mapped: pool[mapped.clamp_min(0)]
        codec.gather_k_cache_bytes_fp4 = lambda pool, mapped: pool[mapped.clamp_min(0)]
        codec.dequantize_k_cache_bytes_fp4 = lambda raw: raw

        def raw_tiles():
            begin = 0
            for _, _, count in tiles:
                yield begin, x[begin : begin + count]
                begin += count

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        PACKAGE: package,
                        compressor.__name__: compressor,
                        codec.__name__: codec,
                    },
                )
            )
            if not use_metadata:
                stack.enter_context(
                    patch.object(metadata, "prepare", return_value=None)
                )
            scope["_produce_global"](
                source,
                x,
                positions,
                requests,
                starts,
                lengths,
                prefill=True,
                raw_tiles=raw_tiles(),
            )
        return events, pools, mappings, len(tiles), source

    def test_actual_producer_preserves_all_float_stage_inputs_order_and_pool_bytes(
        self,
    ):
        for ratio in (1, 2):
            for state_fast in (True, False):
                old = self._producer(ratio, False, state_fast)
                new = self._producer(ratio, True, state_fast)
                self.assertEqual(len(old[0]), len(new[0]))
                for left, right in zip(old[0], new[0]):
                    self.assertEqual(left[0], right[0])
                    for x, y in zip(left[1:], right[1:]):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
                for region in old[1]:
                    torch.testing.assert_close(
                        new[1][region], old[1][region], rtol=0, atol=0
                    )
                # Post-producer pool materialization adds the same calls to both.
                saved = old[3] - 1
                self.assertEqual(
                    len(old[2]) - len(new[2]), saved * (3 if ratio == 2 else 2)
                )
                self.assertNotIn(
                    "global_pairs",
                    new[4]._shared_attention.get("prefill_meta_common", {}),
                )


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ProducerMetadataRawCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fused = _load_file("raw_slot_fused", ROOT / "_v41_prefill_metadata.py")
        cls.old = _load_file("raw_slot_old", ROOT / "_cp_slot_mapping.py")

    def _case(self, prefixes, lengths, ratio, rank=0, stride=1, tile_rows=32768):
        cp, *cpu = inputs(prefixes, lengths)

        def transfer(t):
            result = torch.empty(t.numel() * stride + 1, device="cuda", dtype=t.dtype)[
                1::stride
            ]
            result.copy_(t)
            return result

        pos, req, starts, sizes = [transfer(t) for t in cpu]
        # Real physical/kernel TPB128, STATE TPB512/EB3; padded block-table stride.
        table = torch.arange(
            1, len(lengths) * 1056 + 1, device="cuda", dtype=torch.int32
        ).view(len(lengths), 1056)[:, :1024]
        table[:, 3::17] = 0
        state_table = table[:, :8]

        def slots(region, p, r, state_end=None):
            return self.fused.try_slot_mapping(
                p,
                r,
                state_table if region == "state" else table,
                3 if region == "state" else 128 // ratio,
                512 if region == "state" else 128,
                ratio,
                4,
                rank,
                owner_tokens_per_block=128,
                state=region == "state",
                seq_ends=state_end,
            )

        def old(region, p, r, state_end=None):
            if region == "state":
                return self.old.cp_state_slot_mapping(
                    p, state_table, r, 3, 512, 4, rank, state_end
                )
            return self.old.cp_kv_slot_mapping(
                p, table, r, 128, 128 // ratio, ratio, 4, rank, 128
            )

        def build():
            return metadata.prepare_raw(
                cp,
                pos,
                req,
                starts,
                sizes,
                ratio,
                slots,
                ("main", "index", "state"),
                tile_rows=tile_rows,
            )

        return SimpleNamespace(
            cp=cp,
            pos=pos,
            req=req,
            starts=starts,
            sizes=sizes,
            ratio=ratio,
            tile_rows=tile_rows,
            build=build,
            old=old,
            slots=slots,
        )

    def _check(self, c, plan):
        self.assertIsNotNone(plan)
        torch.testing.assert_close(plan.seq_ends, c.starts + c.sizes, rtol=0, atol=0)
        for start in range(0, len(c.pos), c.tile_rows):
            end = min(start + c.tile_rows, len(c.pos))
            idx = torch.where((c.pos[start:end] + 1) % c.ratio == 0)[0]
            bp, br = c.pos[start:end][idx], c.req[start:end][idx]
            wanted = [
                idx,
                bp,
                br,
                c.old("main", bp, br),
                c.old("index", bp, br),
                (
                    c.old(
                        "state", c.pos[start:end], c.req[start:end], c.starts + c.sizes
                    )
                    if c.ratio == 2
                    else None
                ),
            ]
            for got, expected in zip(plan.tile(start, end), wanted):
                if expected is None:
                    self.assertIsNone(got)
                else:
                    self.assertTrue(got.is_contiguous())
                    self.assertTrue(
                        torch.equal(
                            got.view(torch.uint8),
                            expected.contiguous().view(torch.uint8),
                        )
                    )
        with self.assertRaises(ValueError):
            plan.tile(0, 1)
        if c.tile_rows == 65536 and len(c.pos) >= 65536:
            self.assertTrue(bool((plan.tile(0, 65536)[0] >= 32768).any()))
            with self.assertRaises(ValueError):
                plan.tile(0, 32768)

    def test_raw_bytes_crossings_odd_empty_pairs_all_ranks_strides(self):
        fixtures = [
            ([0, 1, 512, 513], [32767, 1, 1, 32770]),
            ([i % 2 + 512 for i in range(32)], [1 + i * 137 for i in range(32)]),
            ([0] * 32, [1] * 32),
            ([0, 0], [32768, 1]),
            ([1, 0], [32768, 32769]),
        ]
        for ratio in (1, 2):
            for rank in range(4):
                for i, (prefixes, lengths) in enumerate(fixtures):
                    with self.subTest(ratio=ratio, rank=rank, fixture=i):
                        c = self._case(
                            prefixes, lengths, ratio, rank, stride=2 if rank % 2 else 1
                        )
                        self._check(c, c.build())

    def test_raw_graph_replay_updates_all_integer_outputs(self):
        for ratio, tile_rows in ((1, 32768), (2, 32768), (1, 65536), (2, 65536)):
            c = self._case(
                [0, 513, 1024],
                [32767, 4, 32768],
                ratio,
                stride=2,
                tile_rows=tile_rows,
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    c.build()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan = c.build()
            graph.replay()
            self._check(c, plan)
            c.pos.add_(512)
            c.starts.add_(512)
            c.cp.prefix_lengths_host = tuple(s + 512 for s in c.cp.prefix_lengths_host)
            graph.replay()
            self._check(c, plan)

    def test_raw_65536_crossings_all_ranks_strides_and_empty_tail(self):
        for ratio in (1, 2):
            for rank in range(4):
                for prefixes, lengths in (
                    ([1, 3, 5, 7], [32767, 2, 32768, 3]),
                    ([0, 0], [65536, 1]),
                    ([1, 0], [64511, 3]),
                    ([1, 0], [65535, 130]),
                    ([0] * 128, [1] * 128),
                ):
                    with self.subTest(ratio=ratio, rank=rank, lengths=lengths):
                        c = self._case(
                            prefixes,
                            lengths,
                            ratio,
                            rank,
                            stride=2 if rank % 2 else 1,
                            tile_rows=65536,
                        )
                        self._check(c, c.build())

    def test_raw_side_stream_and_unsupported_strides(self):
        c = self._case([0, 1], [32769, 129], 2, stride=2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan = c.build()
        torch.cuda.current_stream().wait_stream(stream)
        self._check(c, plan)
        for tensor in (c.pos.int(), c.pos[:1].expand(len(c.pos)), c.pos[:-1]):
            with patch.object(
                torch, "empty", side_effect=AssertionError("unsupported allocates")
            ):
                self.assertIsNone(
                    metadata.prepare_raw(
                        c.cp,
                        tensor,
                        c.req,
                        c.starts,
                        c.sizes,
                        2,
                        c.slots,
                        ("main", "index", "state"),
                    )
                )

    def test_raw_launch_budget_no_h2d_or_per_request_kernels(self):
        import json
        import tempfile

        c = self._case([i % 2 for i in range(32)], [2049] * 32, 2)
        c.build()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            c.build()
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        self.assertEqual(len(kernels), 4, [e["name"] for e in kernels])
        self.assertFalse(any(e.get("cat") == "gpu_memcpy" for e in events))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedProducerMetadataCUDA(unittest.TestCase):
    def test_actual_startup_warmup(self):
        spec = importlib.util.spec_from_file_location(
            PACKAGE + ".fused_metadata_warm_test", ROOT / "_v41_attention_jit_warmup.py"
        )
        warm = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = warm
        spec.loader.exec_module(warm)
        package = ModuleType(PACKAGE)
        package._v41_producer_metadata = metadata
        with patch.dict(sys.modules, {PACKAGE: package}):
            for rank in range(4):
                for ratio in (1, 2):
                    layouts = (
                        warm.PoolLayout(
                            1, 128 // ratio, (128 // ratio) * 288, 128, 128, ratio
                        ),
                        warm.PoolLayout(2, 128, 128 * 68, 128, 128, ratio),
                    )
                    if ratio == 2:
                        layouts += (warm.PoolLayout(3, 3, 3 * 4096, 512, 128, 2),)
                    warm._warm_fused_producer_metadata(
                        layouts, rank, torch.device("cuda")
                    )
        torch.cuda.synchronize()

    def _check(
        self,
        ratio,
        prefixes,
        sizes,
        rank,
        *,
        strided=False,
        dtype=torch.int32,
        tile_rows=65536,
        table_dtype=torch.int32,
        offset=0
    ):
        cp, pos, req, starts, lengths = inputs(prefixes, sizes)

        def device(t, kind):
            t = t.to(device="cuda", dtype=kind)
            if strided or offset:
                stride = 2 if strided else 1
                out = torch.empty(
                    t.numel() * stride + offset, device="cuda", dtype=kind
                )
                out[offset::stride] = t
                t = out[offset::stride]
            return t

        pos, req = device(pos, torch.int64), device(req, torch.int64)
        starts, lengths = device(starts, dtype), device(lengths, dtype)
        batch = len(sizes)
        columns = (max(p + n for p, n in zip(prefixes, sizes)) + 511) // 512 + 3
        table = torch.arange(
            1, batch * (columns + 2) + 1, dtype=table_dtype, device="cuda"
        ).reshape(batch, columns + 2)[:, offset : columns + offset]
        table[0, 0] = 0
        table[-1, :] = -1
        layouts = tuple(
            metadata.SlotLayout(table, eb, tpb, 128, 4, rank)
            for eb, tpb in ((128 // ratio, 128), (128, 128), (3, 512))[
                : (3 if ratio == 2 else 2)
            ]
        )
        spec = importlib.util.spec_from_file_location(
            "fused_slot_reference", ROOT / "_v41_prefill_metadata.py"
        )
        mapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mapper)

        def slots(region, p, r, state_end=None):
            layout = layouts[region]
            return mapper.try_slot_mapping(
                p,
                r,
                layout.table,
                layout.entries_per_block,
                layout.tokens_per_block,
                ratio,
                4,
                rank,
                owner_tokens_per_block=128,
                state=region == 2,
                seq_ends=state_end,
            )

        def build():
            return metadata.prepare_raw(
                cp,
                pos,
                req,
                starts,
                lengths,
                ratio,
                slots,
                (0, 1, 2),
                slot_layouts=layouts,
                tile_rows=tile_rows,
            )

        got = build()
        self.assertIsNotNone(got)
        keep = torch.where((pos + 1) % ratio == 0)[0]
        for actual, expected in (
            (got.raw_indices, keep % tile_rows),
            (got.positions, pos[keep]),
            (got.requests, req[keep]),
            (got.seq_ends, starts.long() + lengths.long()),
            (got.starts, starts.long()),
            (got.lengths, lengths.long()),
            (got.key_counts, ((starts.long() + lengths.long()) // ratio).int()),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for i, actual in enumerate((got.main_slots, got.index_slots)):
            if keep.numel():
                torch.testing.assert_close(
                    actual, slots(i, pos[keep], req[keep]), rtol=0, atol=0
                )
        if ratio == 2:
            torch.testing.assert_close(
                got.state_slots,
                slots(2, pos, req, starts.long() + lengths.long()),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                got.previous_slots,
                slots(
                    2,
                    (starts.long() - 1).clamp_min(0),
                    torch.arange(batch, device="cuda"),
                ),
                rtol=0,
                atol=0,
            )
        else:
            self.assertIs(got.positions, pos)
            self.assertIs(got.requests, req)
        self.assertEqual(got.key_counts.untyped_storage().nbytes(), 4 * batch)
        return (
            build,
            got,
            layouts,
            (cp, pos, req, starts, lengths, ratio, slots, (0, 1, 2)),
        )

    def test_actual_layout_all_ranks_strides_crossings_and_empty_pairs(self):
        for ratio in (1, 2):
            for rank in range(4):
                self._check(
                    ratio,
                    [0, 127, 129, 513],
                    [32767, 3, 32766, 1],
                    rank,
                    strided=True,
                    tile_rows=32768,
                )
                self._check(ratio, [1, 0], [65536, 3], rank, dtype=torch.int64)
        self._check(2, [0] * 32, [1] * 32, 0)

    def test_side_stream_graph_and_prelaunch_rejection(self):
        from dataclasses import replace

        for ratio in (1, 2):
            build, expected, layouts, args = self._check(
                ratio, [127, 512], [2049, 129], 1
            )
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                actual = build()
            side.synchronize()
            torch.testing.assert_close(
                actual.main_slots, expected.main_slots, rtol=0, atol=0
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = build()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                captured.main_slots, expected.main_slots, rtol=0, atol=0
            )
            bad = (replace(layouts[0], owner_tokens_per_block=64), *layouts[1:])
            with patch.object(
                torch, "empty", side_effect=AssertionError("gate allocated")
            ):
                self.assertIsNone(metadata.prepare_raw(*args, slot_layouts=bad))

    def test_one_launch_no_row_h2d(self):
        import json
        import tempfile

        for ratio in (1, 2):
            build, _, _, _ = self._check(
                ratio, [i % 2 for i in range(32)], [2049] * 32, 0
            )
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                build()
                torch.cuda.synchronize()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "trace.json"
                p.export_chrome_trace(str(path))
                events = json.loads(path.read_text())["traceEvents"]
            gpu = [
                e
                for e in events
                if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
            ]
            self.assertEqual(len(gpu), 1, [e["name"] for e in gpu])
            self.assertIn("_fused_raw_metadata_kernel", gpu[0]["name"])

    def test_bounded_dtype_and_alignment_compilation(self):
        for ratio in (1, 2):
            for dtype in (torch.int32, torch.int64):
                for table_dtype in (torch.int32, torch.int64):
                    self._check(
                        ratio, [0, 1], [8, 5], 0, dtype=dtype, table_dtype=table_dtype
                    )
                    kernel = metadata._fused_raw_metadata_kernel
                    with patch.object(
                        kernel,
                        "_do_compile",
                        side_effect=AssertionError(
                            "alignment/count compiled another variant"
                        ),
                    ):
                        for n in (1, 2, 3, 4):
                            self._check(
                                ratio,
                                [0, 1],
                                [n, 4],
                                0,
                                dtype=dtype,
                                table_dtype=table_dtype,
                                offset=1,
                                strided=True,
                            )
        _, _, layouts, args = self._check(2, [0, 1], [1, 3], 0)
        for position_kind, request_kind, start_kind, length_kind in (
            (torch.int32, torch.int64, torch.int32, torch.int32),
            (torch.int64, torch.int32, torch.int32, torch.int32),
            (torch.int64, torch.int64, torch.int32, torch.int64),
        ):
            cp, pos, req, starts, lengths, ratio, slots, regions = args
            vectors = [
                t.to(kind)
                for t, kind in zip(
                    (pos, req, starts, lengths),
                    (position_kind, request_kind, start_kind, length_kind),
                )
            ]
            with patch.object(
                torch, "empty", side_effect=AssertionError("mixed dtype allocated")
            ):
                self.assertIsNone(
                    metadata.prepare_raw(
                        cp, *vectors, ratio, slots, regions, slot_layouts=layouts
                    )
                )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
