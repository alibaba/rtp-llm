"""CPU contracts for bounded CP4 raw gather, restore and producer lifetime.

Compile complete, unchanged production methods to avoid native/CUDA imports.
Only collectives, projection and cache/kernel boundaries use CPU substitutes.
"""

import ast
import os
import sys
import types
import unittest
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


def _load_attention():
    source = Path(__file__).resolve().parents[1] / "attention_v41.py"
    tree = ast.parse(source.read_text())
    functions = {
        "_use_small_cp_x_gather",
        "_prefill_x_group_plan",
        "_prefill_raw_x_groups",
        "_prefill_projected_x_groups",
        "_prefill_x_tile_plan",
        "_start_prefill_x_gather_async",
        "_wait_prefill_x_gather",
        "_prefill_x_tiles",
        "rms_norm",
        "compress_pairs",
    }
    constants = {
        "_PRODUCE_GLOBAL_TILE_ROWS",
        "_SMALL_CP_X_GATHER_MAX_ROWS",
        "_CP_X_GROUP_MAX_ROWS",
        "_CP_X_GROUP_MAX_BYTES",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        )
    ]
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AttentionV41FP8"
    )
    nodes.extend(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_produce_global", "_prefill_produce"}
    )
    module = types.ModuleType("attention_cp_raw_cpu")
    module.__dict__.update(
        torch=torch,
        F=F,
        os=os,
        INDEXER_KV="index",
        record_function_range=lambda *args: nullcontext(),
        prefill_indexer=types.SimpleNamespace(is_supported=lambda *args: False),
        rope_only=lambda x, *args: x,
        fp8_roundtrip=lambda x: x,
    )
    # The wait annotation is unrelated to the CPU collective substitute.
    module._V41AsyncXGather = lambda work, event, gathered: types.SimpleNamespace(
        work=work, completion_event=event, gathered=gathered
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
        module.__dict__,
    )
    return module


ATTENTION = _load_attention()
COMPRESSOR_NAME = "rtp_llm.models_py.modules.dsv4.fp8.compressor"


@contextmanager
def _cpu_compressor(module):
    # Restore only our injected entry; preserve PyTorch's lazy registrations.
    previous = sys.modules.get(COMPRESSOR_NAME)
    sys.modules[COMPRESSOR_NAME] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(COMPRESSOR_NAME, None)
        else:
            sys.modules[COMPRESSOR_NAME] = previous


def _case(lengths):
    """Independent zigzag oracle: label every real token, invert rank packing."""
    chunks = tuple(2 * ((length + 7) // 8) for length in lengths)
    parts = [[] for _ in range(4)]
    base = 0
    for length, chunk in zip(lengths, chunks):
        padded = torch.full((chunk * 4,), -1, dtype=torch.long)
        padded[:length] = torch.arange(base, base + length)
        halves = padded.reshape(8, chunk // 2)
        for rank in range(4):
            parts[rank].extend((halves[rank], halves[7 - rank]))
        base += length
    packed = [torch.cat(part) for part in parts]
    gathered_ids = torch.cat(packed)
    restore = torch.empty(base, dtype=torch.long)
    valid = gathered_ids >= 0
    restore[gathered_ids[valid]] = torch.arange(len(gathered_ids))[valid]
    rows = torch.arange(base)
    expected = torch.stack([(rows // (128**i)) % 128 for i in range(4)], 1).bfloat16()
    local = [
        expected[p.clamp_min(0)].masked_fill((p < 0)[:, None], -999) for p in packed
    ]
    cp = types.SimpleNamespace(
        cp_size=4,
        cp_rank=0,
        chunk_length=sum(chunks),
        padded_seq_len=sum(chunks) * 4,
        seq_len_full=base,
        input_lengths_global_host=tuple(lengths),
        chunk_lengths_per_req=chunks,
        input_lengths_global=torch.tensor(lengths),
        unpad_restore=restore,
        gather_restore_positions=None,
        swa_replay_start=None,
        prefix_lengths_host=tuple(i % 4 for i in range(len(lengths))),
    )
    return cp, local, expected


def _mixed(delta=0):
    values = (128, 257, 512, 1025, 1537, 2048, 3073, 4096)
    return [values[(5 * i + i // 7) % 8] + delta for i in range(32)]


class CPRawGroupsCPU(unittest.TestCase):
    def setUp(self):
        ceiling = patch.object(ATTENTION, "_SMALL_CP_X_GATHER_MAX_ROWS", 65536)
        ceiling.start()
        self.addCleanup(ceiling.stop)

    def test_large_batch_gather_keeps_single_request_boundary(self):
        with patch.object(ATTENTION, "_SMALL_CP_X_GATHER_MAX_ROWS", 131072):
            for lengths, expected in (
                ([16384], True),
                ([65536], True),
                ([65537], False),
                ([131072], False),
                ([32768] * 4, True),
                ([32768] * 3 + [32769], False),
            ):
                cp, _, _ = _case(lengths)
                self.assertEqual(ATTENTION._use_small_cp_x_gather(cp), expected)

    def test_actual_mixed_boundary_plans_and_cp4_consensus(self):
        for delta, padded, sizes in (
            (0, 47344, None),
            (568, 65520, None),
            (576, 65776, [61104, 4672]),
            (2048, 112880, [63928, 48952]),
        ):
            cp, local, _ = _case(_mixed(delta))
            self.assertEqual(cp.padded_seq_len, padded)
            plans = []
            for rank in range(4):
                cp.cp_rank = rank
                plan = ATTENTION._prefill_x_group_plan(local[rank], cp)
                plans.append(plan)
                self.assertEqual(
                    None if plan is None else [4 * (e - s) for s, e, _, _ in plan],
                    sizes,
                )
            self.assertTrue(all(p == plans[0] for p in plans))

    def test_padded_boundary_and_single_request_limits(self):
        for lengths, expected in (
            ([2048] * 31 + [2040], None),
            ([2048] * 32, None),
            ([2048] * 31 + [2049], 2),
            ([65536, 1], 2),
            ([65537, 1], None),
            ([131072], None),
            ([0, 65536, 1], None),
        ):
            cp, local, _ = _case(lengths)
            plan = ATTENTION._prefill_x_group_plan(local[0], cp)
            self.assertEqual(None if plan is None else len(plan), expected)

    def test_overridden_threshold_preserves_single_shot_and_explicit_disable(self):
        cp, local, _ = _case(_mixed(576))
        for ceiling in (0, -1, 131072):
            with patch.object(ATTENTION, "_SMALL_CP_X_GATHER_MAX_ROWS", ceiling):
                self.assertIsNone(ATTENTION._prefill_x_group_plan(local[0], cp))
        with patch.object(ATTENTION, "_SMALL_CP_X_GATHER_MAX_ROWS", 32768):
            plan = ATTENTION._prefill_x_group_plan(local[0], cp)
            self.assertTrue(
                all(4 * (end - start) <= 32768 for start, end, _, _ in plan)
            )

    def test_missing_or_special_metadata_and_layout_fall_back_without_device_reads(
        self,
    ):
        cp, local, _ = _case(_mixed(576))
        bad = (
            ("cp_size", 2),
            ("input_lengths_global_host", None),
            ("chunk_lengths_per_req", None),
            ("chunk_lengths_per_req", (1,)),
            ("chunk_length", cp.chunk_length + 1),
            ("seq_len_full", cp.seq_len_full + 1),
            ("padded_seq_len", cp.padded_seq_len + 8),
            ("gather_restore_positions", torch.tensor([0])),
            ("swa_replay_start", 0),
            ("unpad_restore", None),
            ("unpad_restore", cp.unpad_restore.int()),
            ("unpad_restore", cp.unpad_restore[:-1]),
        )
        with patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("D2H")
        ), patch.object(torch.Tensor, "item", side_effect=AssertionError("D2H")):
            for name, value in bad:
                with self.subTest(field=name), patch.object(cp, name, value):
                    self.assertIsNone(ATTENTION._prefill_x_group_plan(local[0], cp))
            for x in (local[0].float(), local[0][:, ::2], local[0][:-1]):
                self.assertIsNone(ATTENTION._prefill_x_group_plan(x, cp))

    def test_actual_bytes_bound_can_split_wide_hidden_below_row_limit(self):
        cp, _, _ = _case([20000] * 4)
        # Meta tensors exercise the real shape/dtype budget without allocating GBs.
        cp.unpad_restore = cp.unpad_restore.to("meta")
        for width, groups in ((5120, 2), (12000, 4), (40000, None)):
            x = torch.empty(
                (cp.chunk_length, width), dtype=torch.bfloat16, device="meta"
            )
            plan = ATTENTION._prefill_x_group_plan(x, cp)
            self.assertEqual(None if plan is None else len(plan), groups)
            for l0, l1, r0, r1 in plan or ():
                live_bytes = (4 * (l1 - l0) + r1 - r0) * width * 2 + 3 * (r1 - r0) * 8
                self.assertLessEqual(live_bytes, ATTENTION._CP_X_GROUP_MAX_BYTES)

    def _gather(self, cp, local, plan, references=None):
        calls = []

        def start(x, actual_cp, tile, buffer, weights=()):
            if references is not None:
                self.assertTrue(
                    all(ref() is None for ref in references),
                    "Previous group still alive at next allocation",
                )
            l0, l1, _, _ = plan[len(calls)]
            self.assertIs(actual_cp, cp)
            self.assertEqual(tile, (None, 0, 4 * (l1 - l0)))
            self.assertEqual(weights, ())
            self.assertEqual(buffer.shape, (tile[2], x.shape[1]))
            torch.testing.assert_close(x, local[cp.cp_rank][l0:l1], rtol=0, atol=0)
            buffer.copy_(torch.cat([rank[l0:l1] for rank in local]))
            calls.append(tile[2])
            if references is not None:
                references.append(weakref.ref(buffer))
            return types.SimpleNamespace(gathered=buffer)

        return (
            calls,
            patch.object(ATTENTION, "_start_prefill_x_gather_async", new=start),
            patch.object(
                ATTENTION,
                "_wait_prefill_x_gather",
                new=lambda pending: pending.gathered,
            ),
        )

    def test_ragged_restore_exact_order_all_ranks_and_multiple_groups(self):
        for lengths in (_mixed(576), _mixed(2048), [32769, 1, 65535, 3, 65536]):
            cp, local, expected = _case(lengths)
            plan = ATTENTION._prefill_x_group_plan(local[0], cp)
            original_restore = cp.unpad_restore.clone()
            for rank in range(4):
                cp.cp_rank = rank
                calls, start, wait = self._gather(cp, local, plan)
                with start, wait:
                    tiles = ATTENTION._prefill_raw_x_groups(local[rank], cp, plan)
                    owner_segments = iter(ATTENTION._prefill_x_tile_plan(cp))
                    end = 0
                    for offset, tile in tiles:
                        owner, start_row, rows = next(owner_segments)
                        self.assertEqual(offset, end)
                        self.assertEqual(len(tile), rows)
                        torch.testing.assert_close(
                            tile,
                            local[owner][start_row : start_row + rows],
                            rtol=0,
                            atol=0,
                        )
                        end += len(tile)
                        torch.testing.assert_close(
                            tile, expected[offset:end], rtol=0, atol=0
                        )
                        del tile
                    self.assertEqual(end, cp.seq_len_full)
                    self.assertIsNone(next(owner_segments, None))
                self.assertEqual(len(calls), len(plan))
            torch.testing.assert_close(
                cp.unpad_restore, original_restore, rtol=0, atol=0
            )

    def _owner_projected(self, cp, local, ratio):
        offset = 0
        for owner, start, rows in ATTENTION._prefill_x_tile_plan(cp):
            values = local[owner][start : start + rows].float()
            # The CPU fixture uses identity value weights and identity/8 gates.
            yield offset, values if ratio == 1 else torch.cat((values, values / 8), 1)
            offset += rows

    def _produce(self, cp, x, ratio, tiles=None, references=None, projected=None):
        starts = torch.tensor(cp.prefix_lengths_host)
        lengths = cp.input_lengths_global
        ids = torch.repeat_interleave(torch.arange(len(starts)), lengths)
        positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lengths)])
        state = torch.arange(len(starts) * 8).view(-1, 8).float() / 16
        state_read = Mock(side_effect=lambda *args: state.clone())
        writes, projection_rows, index_rows = [], [], []

        def write(values, scores, pos, req, ends):
            writes.append((values.clone(), scores.clone(), pos.clone(), req.clone()))
            state.fill_(999)

        def linear(raw, weight):
            projection_rows.append(len(raw))
            if references is not None and tiles is not None:
                references.append(
                    weakref.ref(raw._base if raw._base is not None else raw)
                )
            return F.linear(raw.float(), weight.float())

        compressor = types.ModuleType(COMPRESSOR_NAME)
        compressor._linear_bf16_bf16_fp32 = linear
        owner = types.SimpleNamespace(
            global_wkv=torch.eye(4).bfloat16(),
            global_wgate=torch.eye(4).bfloat16() / 8,
            global_norm=torch.ones(4).bfloat16(),
            index_wk=torch.eye(4).bfloat16(),
            index_k_norm=torch.ones(4).bfloat16(),
        )
        shared = {"prefill_chunk_meta": object()}
        source = types.SimpleNamespace(
            compress_ratio=ratio,
            head_dim=4,
            rope_head_dim=2,
            eps=1e-6,
            layer_id=2,
            freqs_cis=torch.ones(int((starts + lengths).max()) + 1, 1),
            _cp_ctx=cp,
            _owner=lambda: owner,
            _global_region=lambda: "main",
            _source_pool=lambda *args: None,
            _read_state=state_read,
            _write_states=write,
            _shared_attention=shared,
        )
        original_linear = F.linear

        def observe_index(tensor, weight, *args, **kwargs):
            if weight is owner.index_wk:
                index_rows.append(len(tensor))
            return original_linear(tensor, weight, *args, **kwargs)

        with _cpu_compressor(compressor), patch.object(F, "linear", new=observe_index):
            ATTENTION._produce_global(
                source,
                x,
                positions,
                ids,
                starts,
                lengths,
                prefill=True,
                raw_tiles=tiles,
                projected_tiles=projected,
            )
        self.assertEqual(state_read.call_count, int(ratio == 2))
        self.assertNotIn("prefill_chunk_meta", shared)
        return shared["global"][2], writes, projection_rows, index_rows

    def test_actual_producer_carry_snapshot_and_raw_group_lifetime(self):
        # Tiny chunks force pairs across tiles and a group ending on an odd row.
        cp, local, expected = _case([11, 2, 9, 7])
        with patch.object(ATTENTION, "_SMALL_CP_X_GATHER_MAX_ROWS", 32), patch.object(
            ATTENTION, "_PRODUCE_GLOBAL_TILE_ROWS", 3
        ):
            plan = ATTENTION._prefill_x_group_plan(local[0], cp)
            for ratio in (1, 2):
                reference = self._produce(
                    cp,
                    local[0],
                    ratio,
                    projected=self._owner_projected(cp, local, ratio),
                )
                references = []
                calls, start, wait = self._gather(cp, local, plan, references)
                with start, wait:
                    raw = ATTENTION._prefill_raw_x_groups(local[0], cp, plan)
                    actual = self._produce(cp, local[0], ratio, raw, references)
                self.assertEqual(len(calls), len(plan))
                self.assertTrue(all(ref() is None for ref in references))
                torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
                self.assertEqual(actual[3], reference[3])
                for column in range(4):
                    if ratio == 2:
                        torch.testing.assert_close(
                            torch.cat([w[column] for w in actual[1]]),
                            torch.cat([w[column] for w in reference[1]]),
                            rtol=0,
                            atol=0,
                        )

    def test_full_mixed_producer_and_all_gemm_shapes_match_original_owner(self):
        for delta in (576, 2048):
            cp, local, _ = _case(_mixed(delta))
            plan = ATTENTION._prefill_x_group_plan(local[0], cp)
            for ratio in (1, 2):
                baseline = self._produce(
                    cp,
                    local[0],
                    ratio,
                    projected=self._owner_projected(cp, local, ratio),
                )
                _, start, wait = self._gather(cp, local, plan)
                with start, wait:
                    actual = self._produce(
                        cp,
                        local[0],
                        ratio,
                        ATTENTION._prefill_raw_x_groups(local[0], cp, plan),
                    )
                torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], baseline[1], rtol=0, atol=0)
                self.assertEqual(actual[3], baseline[3])
                self.assertEqual(
                    actual[2],
                    [
                        rows
                        for _, _, rows in ATTENTION._prefill_x_tile_plan(cp)
                        for _ in range(ratio)
                    ],
                )

    def test_raw_and_projected_interface_rejects_ambiguous_input(self):
        with self.assertRaises(ValueError):
            ATTENTION._produce_global(
                None,
                None,
                None,
                None,
                None,
                None,
                prefill=True,
                raw_tiles=iter(()),
                projected_tiles=iter(()),
            )

    def test_whole_projected_groups_preserve_order_and_release_storage(self):
        for lengths in ([11, 2, 9, 7], [7, 9, 2, 11], [1, 1, 25, 17]):
            cp, local, expected = _case(lengths)
            with patch.object(ATTENTION, "_SMALL_CP_X_GATHER_MAX_ROWS", 32):
                plan = ATTENTION._prefill_x_group_plan(local[0], cp)
            self.assertIsNotNone(plan)
            calls, refs = [], []

            def gather(send, context, tile, buffer, weights=()):
                self.assertTrue(all(ref() is None for ref in refs))
                begin, end, _, _ = plan[len(calls)]
                values = torch.cat([part[begin:end].float() for part in local])
                buffer.copy_(torch.cat((values, values / 8), 1))
                calls.append(tile)
                return types.SimpleNamespace(gathered=buffer)

            compressor = types.ModuleType(COMPRESSOR_NAME)
            compressor._linear_bf16_bf16_fp32 = lambda x, w: F.linear(
                x.float(), w.float()
            )
            with _cpu_compressor(compressor), patch.object(
                ATTENTION, "_start_prefill_x_gather_async", side_effect=gather
            ), patch.object(
                ATTENTION, "_wait_prefill_x_gather", side_effect=lambda h: h.gathered
            ):
                tiles = ATTENTION._prefill_projected_x_groups(
                    local[0],
                    cp,
                    plan,
                    (torch.eye(4).bfloat16(), torch.eye(4).bfloat16() / 8),
                    whole_groups=True,
                )
                for start, tensor in tiles:
                    _, _, begin, end = plan[len(refs)]
                    self.assertEqual(start, begin)
                    ref = expected[begin:end].float()
                    torch.testing.assert_close(
                        tensor, torch.cat((ref, ref / 8), 1), rtol=0, atol=0
                    )
                    refs.append(weakref.ref(tensor))
                    del tensor
                self.assertEqual(len(calls), len(plan))
                self.assertTrue(all(ref() is None for ref in refs))

    def test_batched_producer_requires_complete_projected_groups(self):
        with self.assertRaisesRegex(ValueError, "complete projected groups"):
            ATTENTION._produce_global(
                None,
                None,
                None,
                None,
                None,
                None,
                prefill=True,
                raw_tiles=iter(()),
                batched_groups=True,
            )

    def test_prefill_produce_routes_once_and_closes_group_on_consumer_failure(self):
        cp, local, _ = _case(_mixed(576))
        common = types.SimpleNamespace(
            cp_on=True,
            cp_ctx=cp,
            prefix_lengths=torch.tensor(cp.prefix_lengths_host),
            batch_size=32,
        )
        qkv = types.SimpleNamespace(kv_full=object())
        events = []
        raw = Mock()
        consumer = Mock(side_effect=RuntimeError("consumer failed"))
        source = types.SimpleNamespace(
            swa_bounded_replay=False,
            is_kv_source=True,
            compress_ratio=2,
            _owner=lambda: types.SimpleNamespace(
                global_wkv=torch.eye(4), global_wgate=torch.eye(4)
            ),
            _begin_forward=lambda: None,
            _prefill_common_setup=lambda *a: common,
            _prefill_compute_qkv=lambda *a, **kw: qkv,
            _can_fuse_swa_fresh=lambda *a: False,
            _swa_prefill_workspace=lambda *a, **kw: (None, None),
            _prefill_write_swa_fp8_paged=lambda *a: events.append("swa"),
            _produce_global=consumer,
        )
        with patch.object(
            ATTENTION, "_prefill_projected_x_groups", return_value=raw
        ), patch.object(
            ATTENTION,
            "_start_prefill_x_gather_async",
            side_effect=AssertionError("eager gather"),
        ):
            with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                ATTENTION._prefill_produce(source, local[0], None)
        self.assertEqual(events, ["swa"])
        consumer.assert_called_once()
        self.assertIsNone(consumer.call_args.kwargs["raw_tiles"])
        self.assertIs(consumer.call_args.kwargs["projected_tiles"], raw)
        self.assertTrue(consumer.call_args.kwargs["grouped_projection"])
        raw.close.assert_called_once()

    def test_owner_groups_preserve_segments_packing_and_consumer_lifetime(self):
        for lengths in ([11, 2, 9, 7], [7, 9, 2, 11], [1, 1, 25, 17]):
            cp, local, _ = _case(lengths)
            with patch.object(ATTENTION, "_SMALL_CP_X_GATHER_MAX_ROWS", 32):
                plan = ATTENTION._prefill_x_group_plan(local[0], cp)
            self.assertIsNotNone(plan)
            for rank in range(4):
                cp.cp_rank = rank
                for ratio in (1, 2):
                    weights = (torch.eye(4).bfloat16(),) * ratio
                    if ratio == 2:
                        weights = (weights[0], weights[1] / 8)
                    projected_shapes, gather_calls, packs, refs = [], [], [], []

                    def project(raw, weight):
                        self.assertTrue(raw.is_contiguous())
                        projected_shapes.append((len(raw), raw.stride(), weight.shape))
                        return F.linear(raw.float(), weight.float())

                    def start(send, ctx, tile, buffer, weights=()):
                        self.assertTrue(all(ref() is None for ref in refs))
                        l0, l1, _, _ = plan[len(gather_calls)]
                        expected = torch.cat(
                            [local[r][l0:l1].float() for r in range(4)]
                        )
                        if ratio == 2:
                            expected = torch.cat((expected, expected / 8), 1)
                        # Padding is zeroed rather than projected. All selected
                        # owner rows must agree with independent CP rank packing.
                        for owner, row, size in ATTENTION._prefill_x_tile_plan(cp):
                            if owner == rank and l0 <= row < l1:
                                offset = row - l0
                                target = expected[
                                    rank * (l1 - l0)
                                    + offset : rank * (l1 - l0)
                                    + offset
                                    + size
                                ]
                                torch.testing.assert_close(
                                    send[offset : offset + size], target, rtol=0, atol=0
                                )
                        buffer.copy_(expected)
                        gather_calls.append(tile)
                        return types.SimpleNamespace(gathered=buffer)

                    original_copy = torch._foreach_copy_
                    original_select = torch.Tensor.index_select

                    def pack(destinations, sources):
                        packs.append(len(sources))
                        self.assertTrue(all(t.is_contiguous() for t in sources))
                        return original_copy(destinations, sources)

                    def select(tensor, *args):
                        result = original_select(tensor, *args)
                        refs.append(weakref.ref(result))
                        return result

                    compressor = types.ModuleType(COMPRESSOR_NAME)
                    compressor._linear_bf16_bf16_fp32 = project
                    with _cpu_compressor(compressor), patch.object(
                        ATTENTION, "_start_prefill_x_gather_async", side_effect=start
                    ), patch.object(
                        ATTENTION,
                        "_wait_prefill_x_gather",
                        side_effect=lambda h: h.gathered,
                    ), patch.object(
                        torch, "_foreach_copy_", side_effect=pack
                    ), patch.object(
                        torch.Tensor, "index_select", new=select
                    ):
                        actual = []
                        tiles = ATTENTION._prefill_projected_x_groups(
                            local[rank], cp, plan, weights
                        )
                        try:
                            for offset, tile in tiles:
                                actual.append((offset, tile.clone()))
                                del tile
                        finally:
                            tiles.close()
                    expected = list(self._owner_projected(cp, local, ratio))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(
                        [shape[0] for shape in projected_shapes],
                        [
                            n
                            for owner, _, n in ATTENTION._prefill_x_tile_plan(cp)
                            if owner == rank
                            for _ in weights
                        ],
                    )
                    self.assertEqual(sum(packs), len(projected_shapes))
                    self.assertEqual(len(gather_calls), len(plan))
                    self.assertTrue(all(ref() is None for ref in refs))
                    reference = self._produce(
                        cp,
                        local[rank],
                        ratio,
                        projected=self._owner_projected(cp, local, ratio),
                    )
                    gather_calls.clear()
                    refs.clear()
                    with patch.object(
                        ATTENTION, "_start_prefill_x_gather_async", side_effect=start
                    ), patch.object(
                        ATTENTION,
                        "_wait_prefill_x_gather",
                        side_effect=lambda h: h.gathered,
                    ), patch.object(
                        torch.Tensor, "index_select", new=select
                    ):
                        tiles = ATTENTION._prefill_projected_x_groups(
                            local[rank], cp, plan, weights
                        )
                        try:
                            produced = self._produce(
                                cp, local[rank], ratio, projected=tiles
                            )
                        finally:
                            tiles.close()
                    torch.testing.assert_close(
                        produced[:2], reference[:2], rtol=0, atol=0
                    )
                    self.assertEqual(produced[3], reference[3])
                    self.assertTrue(all(ref() is None for ref in refs))

    def test_prefill_dispatch_keeps_small_b1_oversized_and_noncp_routes(self):
        for lengths, cp_on, route in (
            (_mixed(568), True, "small"),
            ([65536], True, "small"),
            ([65537], True, "projected"),
            ([65537, 7], True, "projected"),
            ([7, 13], False, "plain"),
        ):
            cp, local, expected = _case(lengths)
            x = local[0] if cp_on else expected
            common = types.SimpleNamespace(
                cp_on=cp_on,
                cp_ctx=cp if cp_on else None,
                prefix_lengths=torch.tensor(cp.prefix_lengths_host),
                input_lengths=cp.input_lengths_global,
                batch_size=len(lengths),
            )
            qkv = types.SimpleNamespace(kv_full=object())
            owner = types.SimpleNamespace(
                global_wkv=torch.eye(4), global_wgate=torch.eye(4)
            )
            consumer = Mock()
            source = types.SimpleNamespace(
                swa_bounded_replay=False,
                is_kv_source=True,
                compress_ratio=2,
                head_dim=4,
                _owner=lambda: owner,
                _begin_forward=lambda: None,
                _prefill_common_setup=lambda *a: common,
                _prefill_compute_qkv=lambda *a, **kw: qkv,
                _can_fuse_swa_fresh=lambda *a: False,
                _swa_prefill_workspace=lambda *a, **kw: (None, None),
                _prefill_write_swa_fp8_paged=lambda *a: None,
                _produce_global=consumer,
            )
            with self.subTest(route=route, lengths=lengths), patch.object(
                ATTENTION, "_start_prefill_x_gather_async"
            ) as start, patch.object(ATTENTION, "_wait_prefill_x_gather"), patch.object(
                ATTENTION,
                "_cp_restore_gathered_full_2d",
                create=True,
                return_value=expected,
            ) as restore, patch.object(
                ATTENTION, "_prefill_x_tiles"
            ) as projected, patch.object(
                ATTENTION,
                "_prefill_raw_x_groups",
                side_effect=AssertionError("unexpected raw groups"),
            ):
                ATTENTION._prefill_produce(source, x, None)
            consumer.assert_called_once()
            self.assertIsNone(consumer.call_args.kwargs["raw_tiles"])
            self.assertEqual(start.call_count, int(cp_on))
            self.assertEqual(restore.call_count, int(route == "small"))
            self.assertEqual(projected.call_count, int(route == "projected"))
            if route == "small":
                self.assertEqual(start.call_args.args[2], (None, 0, cp.padded_seq_len))
                self.assertIs(consumer.call_args.args[0], expected)
            elif route == "projected":
                self.assertIs(
                    consumer.call_args.kwargs["projected_tiles"], projected.return_value
                )
            else:
                self.assertIs(consumer.call_args.args[0], x)

    def test_existing_async_event_chain_is_used_for_every_group(self):
        # Execute the real transport helper; only CUDA/NCCL APIs are mocked.
        log = []
        current = types.SimpleNamespace(
            wait_event=lambda event: log.append("consume.wait_event")
        )
        comm = types.SimpleNamespace(
            wait_stream=lambda stream: log.append("comm.wait_current")
        )
        work = types.SimpleNamespace(wait=lambda: log.append("work.wait"))
        event = types.SimpleNamespace(record=lambda stream: log.append("event.record"))
        group = object()
        collective = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")
        collective.Group = types.SimpleNamespace(TP="TP")
        collective._get_group = lambda kind: group
        parent = types.ModuleType("rtp_llm.models_py.distributed")
        parent.collective_torch = collective

        def gather(output, local, **kwargs):
            log.append("all_gather")
            self.assertIs(kwargs["group"], group)
            self.assertTrue(kwargs["async_op"])
            output.copy_(local.repeat(4, 1))
            return work

        with patch.dict(
            sys.modules, {parent.__name__: parent, collective.__name__: collective}
        ), patch.object(
            ATTENTION, "_get_cp_comm_stream", create=True, return_value=comm
        ), patch.object(
            torch.cuda, "current_stream", return_value=current
        ), patch.object(
            torch.cuda, "stream", side_effect=lambda stream: nullcontext()
        ), patch.object(
            torch.cuda, "Event", return_value=event
        ), patch.object(
            torch.distributed, "all_gather_into_tensor", side_effect=gather
        ):
            for rows in (3, 7):
                x = torch.ones(rows, 4).bfloat16()
                buffer = x.new_empty(rows * 4, 4)
                handle = ATTENTION._start_prefill_x_gather_async(
                    x, types.SimpleNamespace(cp_rank=0), (None, 0, rows * 4), buffer
                )
                result = ATTENTION._wait_prefill_x_gather(handle)
                self.assertTrue(torch.equal(result, x.repeat(4, 1)))
        self.assertEqual(
            log,
            [
                "comm.wait_current",
                "all_gather",
                "work.wait",
                "event.record",
                "consume.wait_event",
                "work.wait",
            ]
            * 2,
        )


if __name__ == "__main__":
    unittest.main()
