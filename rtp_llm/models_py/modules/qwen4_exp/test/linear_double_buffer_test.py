"""Single/double SSM ownership, acceptance, indexed records and graph checks."""

import unittest
from dataclasses import replace

import torch

from rtp_llm.models_py.modules.qwen4_exp.linear_double_buffer import (
    DUMMY_REQUEST_STATE_BLOCK_ID,
    LinearDoubleBufferStage,
    LinearDoubleBufferTransaction,
    _resolve_pool_indices,
    _validate_acceptance,
    replay_partial_acceptance,
    resolve_pool_indices,
    split_pool_views,
)


def reference_prefix(initial, record, count):
    result = initial.clone()
    for token in range(count):
        key = record[token, :, 0, :]
        update = record[token, :, 1, :]
        decay = record[token, :, 2, 0]
        result = result * decay[:, None, None] + update[:, :, None] * key[:, None, :]
    return result


def cpu_partial_commit(stage, accepted):
    """Independent CPU backend for transaction tests, not a production fallback."""
    for state, records in zip(stage.states, stage.records):
        physical = state.transpose(-1, -2)
        for row, count in enumerate(accepted.tolist()):
            if count > 0 and (stage.in_place or count < stage.query):
                destination = (
                    stage.reads[row] if stage.in_place else stage.writes[row, 0]
                )
                physical[destination] = reference_prefix(
                    physical[stage.reads[row]], records[stage.record_ids[row]], count
                )


def fixture(device="cpu", count=4, *, in_place=False, query=3):
    torch.manual_seed(1711)
    heads, dim = 2, 128
    slots = 1 if in_place else 2
    physical = torch.randn(1 + slots * count, heads, dim, dim, device=device) * 0.1
    records = torch.randn(count + 4, query, heads, 3, dim, device=device) * 0.1
    records[:, :, :, 2, 0] = 0.9
    table = torch.arange(1, 1 + slots * count, dtype=torch.int32, device=device).view(
        count, slots
    )
    record_ids = torch.arange(count, 0, -1, dtype=torch.int32, device=device)
    # Runtime outputs are separate small buffers, including the one-slot case.
    reads = table[:, 0].clone()
    writes = table[:, -1:].expand(count, query).contiguous()
    state = physical.transpose(-1, -2)
    stage = LinearDoubleBufferStage(
        (state,),
        (records,),
        torch.tensor([state.data_ptr()], device=device, dtype=torch.int64),
        torch.tensor([records.data_ptr()], device=device, dtype=torch.int64),
        reads,
        writes,
        record_ids,
        in_place=in_place,
    )
    return stage, table


def simulate_full_candidate(stage):
    """Populate scratch as the verify kernel does, retaining committed."""
    for state, records in zip(stage.states, stage.records):
        physical = state.transpose(-1, -2)
        for row in range(stage.reads.numel()):
            physical[stage.writes[row, 0]] = reference_prefix(
                physical[stage.reads[row]], records[stage.record_ids[row]], stage.query
            )


class LinearDoubleBufferContractTest(unittest.TestCase):
    def test_single_ssm_commits_every_nonempty_prefix_without_slot_changes(self):
        stage, table = fixture(in_place=True)
        physical = stage.states[0].transpose(-1, -2)
        self.assertEqual(physical.shape[0], 1 + table.shape[0])
        tx = LinearDoubleBufferTransaction(cpu_partial_commit, in_place=True)
        original_table = table.clone()
        for counts in ([0, 1, 2, 3], [3, 2, 1, 0], [0] * 4, [3] * 4):
            before = physical.clone()
            tx.bind(stage)
            tx.prepare(torch.tensor(counts, dtype=torch.int32))
            torch.testing.assert_close(physical, before, rtol=0, atol=0)
            tx.finish(True)
            tx.finalize()  # CPU backend is synchronous; native gates GPU completion.
            expected = before.clone()
            for row, count in enumerate(counts):
                expected[stage.reads[row]] = reference_prefix(
                    before[stage.reads[row]],
                    stage.records[0][stage.record_ids[row]],
                    count,
                )
            torch.testing.assert_close(physical, expected, rtol=0, atol=0)
            torch.testing.assert_close(table, original_table, rtol=0, atol=0)

    def test_single_ssm_abort_is_allowed_only_before_write_submission(self):
        stage, _ = fixture(in_place=True)
        tx = LinearDoubleBufferTransaction(cpu_partial_commit, in_place=True)
        physical = stage.states[0].transpose(-1, -2)
        before = physical.clone()
        tx.bind(stage)
        tx.prepare(torch.tensor([0, 1, 2, 3], dtype=torch.int32))
        tx.finish(False)
        torch.testing.assert_close(physical, before, rtol=0, atol=0)
        tx.bind(stage)
        tx.prepare(torch.tensor([0, 1, 2, 3], dtype=torch.int32))
        tx.finish(True)
        with self.assertRaisesRegex(RuntimeError, "fail-stop"):
            tx.finish(False)
        for action in (tx.finalize, tx.clear, lambda: tx.save(12)):
            with self.assertRaises(RuntimeError):
                action()

    def test_single_ssm_marks_irreversible_before_failed_launch(self):
        stage, _ = fixture(in_place=True)

        def failed_commit(active, accept):
            self.assertTrue(tx.submitted)
            cpu_partial_commit(active, accept)
            raise RuntimeError("injected launch failure after a write")

        tx = LinearDoubleBufferTransaction(failed_commit, in_place=True)
        tx.bind(stage)
        tx.prepare(torch.tensor([1, 0, 2, 3], dtype=torch.int32))
        with self.assertRaisesRegex(RuntimeError, "injected launch"):
            tx.finish(True)
        self.assertTrue(tx.poisoned)
        with self.assertRaisesRegex(RuntimeError, "fail-stop"):
            tx.finish(False)
        for action in (lambda: tx.bind(stage), tx.finalize, lambda: tx.finish(True)):
            with self.assertRaises(RuntimeError):
                action()

    def test_transaction_rejects_state_mode_mismatch(self):
        single, _ = fixture(in_place=True)
        double, _ = fixture()
        for stage, mode in ((single, False), (double, True)):
            with self.assertRaisesRegex(ValueError, "layouts disagree"):
                LinearDoubleBufferTransaction(cpu_partial_commit, in_place=mode).bind(
                    stage
                )
        with self.assertRaises(TypeError):
            LinearDoubleBufferTransaction(in_place=1)

    def test_views_share_manager_storage_and_preserve_fp32(self):
        conv_base = torch.zeros(13, 3 * 256, dtype=torch.bfloat16)
        state_base = torch.zeros(9, 2 * 128 * 128, dtype=torch.float32)
        replay_base = torch.zeros(7, 3 * 2 * 3 * 128, dtype=torch.float32)
        conv, state, replay = split_pool_views(
            conv_base,
            state_base,
            replay_base,
            heads=2,
            key_dim=128,
            value_dim=128,
            conv_dim=256,
            conv_length=3,
            query=3,
        )
        self.assertEqual(conv.data_ptr(), conv_base.data_ptr())
        self.assertEqual(state.data_ptr(), state_base.data_ptr())
        self.assertEqual(replay.data_ptr(), replay_base.data_ptr())
        self.assertEqual(state.stride()[1:], (128 * 128, 1, 128))
        self.assertEqual(state.dtype, torch.float32)
        self.assertEqual(replay.dtype, torch.float32)
        state[3, 1, 7, 9] = 17
        self.assertEqual(state_base.view(9, 2, 128, 128)[3, 1, 9, 7].item(), 17)
        with self.assertRaisesRegex(ValueError, "float32"):
            split_pool_views(
                conv_base,
                state_base.bfloat16(),
                replay_base,
                heads=2,
                key_dim=128,
                value_dim=128,
                conv_dim=256,
                conv_length=3,
                query=3,
            )

    def test_mixed_acceptance_and_multiple_bank_swaps(self):
        stage, table = fixture()
        tx = LinearDoubleBufferTransaction(cpu_partial_commit)
        physical = stage.states[0].transpose(-1, -2)
        for accepted in (
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [3, 0, 1, 2],
            [2, 3, 0, 1],
            [3, 3, 3, 3],
        ):
            stage.reads.copy_(table[:, 0])
            stage.writes.copy_(table[:, 1:].expand_as(stage.writes))
            before = physical.clone()
            simulate_full_candidate(stage)
            tx.bind(stage)
            accept = torch.tensor(accepted, dtype=torch.int32)
            tx.prepare(accept)
            tx.finish(True)
            # Even a successful finish has not published a bank swap yet.
            torch.testing.assert_close(
                physical[table[:, 0]], before[table[:, 0]], rtol=0, atol=0
            )
            for row, count in enumerate(accepted):
                expected = reference_prefix(
                    before[table[row, 0]],
                    stage.records[0][stage.record_ids[row]],
                    count,
                )
                if count:
                    torch.testing.assert_close(physical[table[row, 1]], expected)
                    # This mirrors the cache manager's publication, deliberately
                    # outside the Python transaction implementation.
                    table[row] = table[row].flip(0)
                torch.testing.assert_close(physical[table[row, 0]], expected)
            tx.finalize()

    def test_abort_after_scratch_write_leaves_committed_recoverable(self):
        stage, table = fixture()
        tx = LinearDoubleBufferTransaction(cpu_partial_commit)
        physical = stage.states[0].transpose(-1, -2)
        before = physical[table[:, 0]].clone()
        simulate_full_candidate(stage)
        tx.bind(stage)
        tx.prepare(torch.tensor([1, 0, 2, 3], dtype=torch.int32))
        tx.finish(True)
        tx.finish(False)  # A peer failed before C++ publication.
        torch.testing.assert_close(physical[table[:, 0]], before, rtol=0, atol=0)
        self.assertIsNone(tx.pending)
        self.assertIsNone(tx.active)
        simulate_full_candidate(stage)
        tx.bind(stage)
        tx.prepare(torch.tensor([3, 3, 3, 3], dtype=torch.int32))
        tx.finish(True)
        tx.finalize()

    def test_graph_registry_uses_current_indices_and_rejects_pending_rebind(self):
        stage, _ = fixture()
        smaller, _ = fixture(count=2)
        tx = LinearDoubleBufferTransaction(cpu_partial_commit)
        tx.bind(stage)
        tx.save(12)
        tx.bind(smaller)
        tx.save(6)
        tx.activate(12)
        self.assertIs(tx.active, stage)
        stage.reads.copy_(torch.tensor([7, 5, 3, 1], dtype=torch.int32))
        self.assertEqual(tx.active.reads.tolist(), [7, 5, 3, 1])
        tx.prepare(torch.zeros(4, dtype=torch.int32))
        for action in (lambda: tx.activate(6), tx.clear, lambda: tx.save(12)):
            with self.assertRaises(RuntimeError):
                action()
        tx.finish(False)
        tx.activate(6)
        self.assertIs(tx.active, smaller)
        with self.assertRaises(RuntimeError):
            tx.finalize()

    def test_acceptance_validation_does_not_start_transaction(self):
        stage, _ = fixture()
        tx = LinearDoubleBufferTransaction(cpu_partial_commit)
        tx.bind(stage)
        for accept in (
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
        ):
            with self.assertRaises(RuntimeError):
                tx.prepare(accept)
            self.assertIsNone(tx.pending)
        stage.reads[0] = 0
        with self.assertRaises(RuntimeError):
            tx.prepare(torch.tensor([1], dtype=torch.int32))
        tx.prepare(torch.zeros(1, dtype=torch.int32))
        tx.finish(True)
        tx.finalize()


@unittest.skipUnless(
    torch.cuda.is_available(), "GPU required for actual Triton/graph validation"
)
class LinearDoubleBufferKernelTest(unittest.TestCase):
    def test_fused_acceptance_validation_matches_eager_for_strided_inputs(self):
        for values, indices in [([0, 1, 2, 4], [0, 2, 3, 4]),
                                ([-1, 1, 2, 4], [0, 2, 3, 4]),
                                ([0, 1, 2, 5], [0, 2, 3, 4]),
                                ([0, 1, 2, 4], [0, 0, 3, 4])]:
            for stride in (1, 2):
                backing = torch.empty(4 * stride, dtype=torch.int32, device="cuda")
                reads_backing = torch.empty_like(backing)
                accept, reads = backing[::stride], reads_backing[::stride]
                accept.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
                reads.copy_(torch.tensor(indices, device="cuda", dtype=torch.int32))
                result = torch.empty((), dtype=torch.bool, device="cuda")
                _validate_acceptance[(1,)](accept, reads, result, 4, 4, 4,
                                           accept.stride(0), reads.stride(0), num_warps=4)
                expected = torch.all((accept >= 0) & (accept <= 4)
                                     & ((accept == 0) | (reads > 0)))
                torch.testing.assert_close(result, expected)

    def test_single_resolver_strided_tables_and_dummy_validation(self):
        state_rows, record_rows, query = 9, 8, 3
        cases = [
            (1, 3, 13, "live"),
            (-2, -2, 13, "dummy"),
            (0, 0, 0, "padding"),
            (0, 0, 13, "invalid"),
            (-1, -1, 13, "invalid"),
            (-2, 1, 13, "invalid"),
            (1, -2, 13, "invalid"),
            (-2, -1, 13, "invalid"),
            (-1, -2, 13, "invalid"),
            (state_rows, 1, 13, "invalid"),
            (1, record_rows, 13, "invalid"),
            (7, 2, 13, "live"),
            (-999, -999, 0, "padding"),
        ] * 12
        count = len(cases)
        wide_state = torch.full((count, 9), -999, dtype=torch.int32, device="cuda")
        wide_record = torch.full((count, 7), -999, dtype=torch.int32, device="cuda")
        table, record_table = wide_state[:, :1], wide_record[:, :1]
        table[:, 0].copy_(torch.tensor([c[0] for c in cases], device="cuda"))
        record_table[:, 0].copy_(torch.tensor([c[1] for c in cases], device="cuda"))
        prefix = torch.tensor([c[2] for c in cases], dtype=torch.int32, device="cuda")
        reads = torch.empty(count, dtype=torch.int32, device="cuda")
        writes = torch.empty(count, query, dtype=torch.int32, device="cuda")
        record_ids = torch.empty_like(reads)
        invalid = torch.empty((count + 127) // 128, dtype=torch.int32, device="cuda")
        _resolve_pool_indices[(invalid.numel(),)](
            table,
            record_table,
            prefix,
            reads,
            writes,
            record_ids,
            invalid,
            count,
            table.stride(0),
            record_table.stride(0),
            table.stride(1),
            state_rows,
            record_rows,
            query,
            0,
            DUMMY_REQUEST_STATE_BLOCK_ID,
            128,
            STATE_SLOTS=1,
        )
        expected_reads = [c[0] if c[3] == "live" else 0 for c in cases]
        self.assertEqual(reads.tolist(), expected_reads)
        self.assertEqual(writes.tolist(), [[row] * query for row in expected_reads])
        self.assertEqual(
            record_ids.tolist(), [c[1] if c[3] == "live" else 0 for c in cases]
        )
        self.assertEqual(
            invalid.tolist(),
            [
                sum(c[3] == "invalid" for c in cases[start : start + 128])
                for start in range(0, count, 128)
            ],
        )
        # Q1 uses the same single-slot resolver, but sequence-length-plus-one
        # makes length 1 an inactive pad rather than a live first token.
        stage, table = fixture("cuda", in_place=True)
        record_table = stage.record_ids[:, None].clone()
        lengths = torch.tensor([9, 13, 1, 1], device="cuda", dtype=torch.int32)
        resolve_pool_indices(
            table,
            record_table,
            lengths,
            (stage.reads, stage.writes, stage.record_ids, invalid[:1]),
            stage.states[0].shape[0],
            stage.records[0].shape[0],
            minimum_length=1,
            state_slots=1,
        )
        self.assertEqual(stage.reads.tolist(), [1, 2, 0, 0])
        self.assertEqual(stage.record_ids.tolist(), [4, 3, 0, 0])

    def test_single_inplace_replay_acceptance_matches_fp32_reference(self):
        for query in (2, 3, 4):
            with self.subTest(query=query):
                stage, table = fixture(
                    "cuda", count=query + 1, in_place=True, query=query
                )
                # One launch must update every layer independently, without
                # allocating another full state bank or touching reserved row 0.
                second = stage.states[0].clone(memory_format=torch.preserve_format)
                second_records = stage.records[0].clone()
                stage = replace(
                    stage,
                    states=(*stage.states, second),
                    records=(*stage.records, second_records),
                    state_ptrs=torch.tensor(
                        [stage.states[0].data_ptr(), second.data_ptr()],
                        dtype=torch.int64,
                        device="cuda",
                    ),
                    record_ptrs=torch.tensor(
                        [stage.records[0].data_ptr(), second_records.data_ptr()],
                        dtype=torch.int64,
                        device="cuda",
                    ),
                )
                expected = [state.transpose(-1, -2).clone() for state in stage.states]
                original_table = table.clone()
                records_before = [records.clone() for records in stage.records]
                generator = torch.Generator().manual_seed(1717 + query)
                for iteration in range(64):
                    counts = (
                        list(range(query + 1))
                        if iteration == 0
                        else torch.randint(
                            query + 1, (query + 1,), generator=generator
                        ).tolist()
                    )
                    # Reorder live requests while preserving their owned slots.
                    order = torch.randperm(query + 1, generator=generator).to("cuda")
                    stage.reads.copy_(table[order, 0])
                    stage.record_ids.copy_(order.to(torch.int32) + 1)
                    accepted = torch.tensor(counts, dtype=torch.int32, device="cuda")
                    if iteration == 0:
                        # Padding must never be interpreted as a request's accept length.
                        backing = accepted.new_full((2 * accepted.numel(),), -1)
                        backing[::2].copy_(accepted)
                        accepted = backing[::2]
                    before = [state.transpose(-1, -2).clone() for state in stage.states]
                    replay_partial_acceptance(stage, accepted)
                    for layer, physical in enumerate(expected):
                        for row, count in enumerate(counts):
                            state_id = stage.reads[row]
                            physical[state_id] = reference_prefix(
                                physical[state_id],
                                stage.records[layer][stage.record_ids[row]],
                                count,
                            )
                        torch.testing.assert_close(
                            stage.states[layer].transpose(-1, -2),
                            physical,
                            rtol=3e-5,
                            atol=2e-6,
                        )
                        for row, count in enumerate(counts):
                            if count == 0:
                                state_id = stage.reads[row]
                                torch.testing.assert_close(
                                    stage.states[layer].transpose(-1, -2)[state_id],
                                    before[layer][state_id],
                                    rtol=0,
                                    atol=0,
                                )
                    torch.testing.assert_close(table, original_table, rtol=0, atol=0)
                for records, before in zip(stage.records, records_before):
                    torch.testing.assert_close(records, before, rtol=0, atol=0)

    @unittest.skipUnless(
        torch.version.hip, "Record-only FlyDSL integration requires ROCm"
    )
    def test_flydsl_record_only_verify_matches_hybrid_and_replays_full_acceptance(self):
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_mtp_hybrid

        torch.manual_seed(1718)
        for query in (2, 3, 4):
            batch, key_heads, heads, dim = query + 1, 2, 6, 128
            q, k = [
                torch.randn(
                    batch, query, key_heads, dim, device="cuda", dtype=torch.bfloat16
                )
                * 0.1
                for _ in range(2)
            ]
            v = (
                torch.randn(
                    batch, query, heads, dim, device="cuda", dtype=torch.bfloat16
                )
                * 0.1
            )
            a, b = [
                torch.randn(batch, query, heads, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            bias = torch.randn(heads, device="cuda", dtype=torch.bfloat16)
            alog = torch.randn(heads, device="cuda", dtype=torch.float32)
            reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
            # Deliberately distinct write IDs prove NO_STATE_WRITE suppresses
            # checkpoints, independently of compact_record's src == dst guard.
            writes = (reads[:, None] + batch).expand(batch, query).contiguous()
            record_ids = reads.flip(0).contiguous()
            initial = torch.randn(1 + 2 * batch, heads, dim, dim, device="cuda") * 0.1
            legacy, single = initial.clone(), initial.clone()
            reference_record = torch.zeros(batch, query, heads, 3, dim, device="cuda")
            record = torch.zeros(batch + 2, query, heads, 3, dim, device="cuda")
            expected, output = torch.empty_like(v), torch.empty_like(v)
            flydsl_gdr_mtp_hybrid(
                q,
                k,
                v,
                a,
                b,
                bias,
                alog,
                legacy,
                expected,
                reads,
                writes,
                reference_record,
            )
            flydsl_gdr_mtp_hybrid(
                q,
                k,
                v,
                a,
                b,
                bias,
                alog,
                single,
                output,
                reads,
                writes,
                record,
                record_indices=record_ids,
                record_only=True,
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(single, initial, rtol=0, atol=0)
            torch.testing.assert_close(
                record[record_ids], reference_record, rtol=0, atol=0
            )
            self.assertEqual(torch.count_nonzero(record[[0, batch + 1]]).item(), 0)
            stage = LinearDoubleBufferStage(
                (single.transpose(-1, -2),),
                (record,),
                torch.tensor([single.data_ptr()], dtype=torch.int64, device="cuda"),
                torch.tensor([record.data_ptr()], dtype=torch.int64, device="cuda"),
                reads,
                writes,
                record_ids,
                in_place=True,
            )
            accepted = torch.arange(batch, dtype=torch.int32, device="cuda")
            replay_partial_acceptance(stage, accepted)
            reference = initial.clone()
            for row, count in enumerate(accepted.tolist()):
                reference[reads[row]] = reference_prefix(
                    initial[reads[row]], reference_record[row], count
                )
            torch.testing.assert_close(single, reference, rtol=3e-5, atol=2e-6)
            torch.testing.assert_close(
                single[reads[-1]], legacy[writes[-1, 0]], rtol=3e-5, atol=2e-6
            )

    @unittest.skipUnless(torch.version.hip, "Record-only FlyDSL graph requires ROCm")
    def test_single_graph_readonly_verify_and_commit_rebatch(self):
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_mtp_hybrid

        torch.manual_seed(1719)
        stage, table = fixture("cuda", in_place=True)
        batch, query, heads, dim = 4, 3, 2, 128
        physical = stage.states[0].transpose(-1, -2)
        original = physical.clone()
        # All captured SSM/record rows start as graph dummies. No private bank.
        table.fill_(DUMMY_REQUEST_STATE_BLOCK_ID)
        record_table = torch.full_like(table, DUMMY_REQUEST_STATE_BLOCK_ID)
        prefix = torch.full((batch,), 13, dtype=torch.int32, device="cuda")
        invalid = torch.empty(1, dtype=torch.int32, device="cuda")
        stage.records[0].zero_()
        q, k, v = [
            torch.randn(batch, query, heads, dim, device="cuda", dtype=torch.bfloat16)
            * 0.1
            for _ in range(3)
        ]
        a, b = [
            torch.randn(batch, query, heads, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        bias = torch.randn(heads, device="cuda", dtype=torch.bfloat16)
        alog = torch.randn(heads, device="cuda", dtype=torch.float32)
        output = torch.empty_like(v)

        def verify():
            resolve_pool_indices(
                table,
                record_table,
                prefix,
                (stage.reads, stage.writes, stage.record_ids, invalid),
                physical.shape[0],
                stage.records[0].shape[0],
                state_slots=1,
            )
            flydsl_gdr_mtp_hybrid(
                q,
                k,
                v,
                a,
                b,
                bias,
                alog,
                physical,
                output,
                stage.reads,
                stage.writes,
                stage.records[0],
                record_indices=stage.record_ids,
                record_only=True,
            )

        verify()
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verify()
        graph.replay()
        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(physical, original, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        self.assertEqual(torch.count_nonzero(stage.records[0]).item(), 0)
        tx = LinearDoubleBufferTransaction(in_place=True)
        tx.bind(stage)
        tx.save(batch * query)
        generator = torch.Generator().manual_seed(1720)
        for iteration in range(32):
            live = batch - (iteration % 3 == 1)
            state_ids = torch.randperm(batch, generator=generator).add(1).tolist()
            record_ids = (
                torch.randperm(stage.records[0].shape[0] - 1, generator=generator)
                .add(1)
                .tolist()[:batch]
            )
            table[:, 0].copy_(torch.tensor(state_ids, device="cuda"))
            record_table[:, 0].copy_(torch.tensor(record_ids, device="cuda"))
            prefix.fill_(13)
            prefix[live:] = 0
            table[live:] = 0
            record_table[live:] = 0
            before = physical.clone()
            table_before = table.clone()
            record_before = stage.records[0].clone()
            graph.replay()
            torch.testing.assert_close(physical, before, rtol=0, atol=0)
            if live != batch:
                self.assertEqual(torch.count_nonzero(output[live:]).item(), 0)
            untouched_records = [
                row
                for row in range(record_before.shape[0])
                if row not in record_ids[:live]
            ]
            torch.testing.assert_close(
                stage.records[0][untouched_records],
                record_before[untouched_records],
                rtol=0,
                atol=0,
            )
            counts = (
                list(range(batch))
                if iteration == 0
                else torch.randint(query + 1, (batch,), generator=generator).tolist()
            )
            counts[live:] = [0] * (batch - live)
            accepted = torch.tensor(counts, dtype=torch.int32, device="cuda")
            tx.activate(batch * query)
            tx.prepare(accepted)
            torch.testing.assert_close(physical, before, rtol=0, atol=0)
            tx.finish(True)
            # Production performs this current-stream completion gate in C++
            # before TP consensus. A standalone helper test has no executor.
            torch.cuda.current_stream().synchronize()
            tx.finalize()
            expected = before.clone()
            for row in range(live):
                expected[state_ids[row]] = reference_prefix(
                    before[state_ids[row]],
                    stage.records[0][record_ids[row]],
                    counts[row],
                )
            torch.testing.assert_close(physical, expected, rtol=3e-5, atol=2e-6)
            for row in range(live):
                if counts[row] == 0:
                    torch.testing.assert_close(
                        physical[state_ids[row]], before[state_ids[row]], rtol=0, atol=0
                    )
            torch.testing.assert_close(table, table_before, rtol=0, atol=0)

    def test_resolver_mixed_malformed_dummy_rows_preserve_validation(self):
        state_rows, record_rows, query = 9, 8, 3
        # Explicit outcomes include mixed negative markers, null/zero IDs,
        # aliasing banks, out-of-range IDs, and positive-length graph dummies.
        cases = [
            (1, 2, 1, 13, "live"),
            (-2, -2, -2, 13, "dummy"),
            (0, 0, 0, 0, "padding"),
            (0, 0, 0, 13, "invalid"),
            (-1, -1, -1, 13, "invalid"),
            (-2, -2, 1, 13, "invalid"),
            (-2, 2, -2, 13, "invalid"),
            (1, -2, -2, 13, "invalid"),
            (-2, -2, -1, 13, "invalid"),
            (-2, -1, -2, 13, "invalid"),
            (-1, -2, -2, 13, "invalid"),
            (-2, -2, 0, 13, "invalid"),
            (-2, 0, -2, 13, "invalid"),
            (0, -2, -2, 13, "invalid"),
            (1, 1, 1, 13, "invalid"),
            (state_rows, 2, 1, 13, "invalid"),
            (1, 2, record_rows, 13, "invalid"),
            (-2, -2, -3, 13, "invalid"),
            (3, 4, 2, 13, "live"),
        ] * 8
        count = len(cases)
        # Also exercise padded table widths and a non-unit SSM column stride.
        wide_ssm = torch.full((count, 4), -999, device="cuda", dtype=torch.int32)
        wide_records = torch.full((count, 3), -999, device="cuda", dtype=torch.int32)
        table, records_table = wide_ssm[:, ::2], wide_records[:, :1]
        table.copy_(
            torch.tensor([row[:2] for row in cases], device="cuda", dtype=torch.int32)
        )
        records_table[:, 0].copy_(
            torch.tensor([row[2] for row in cases], device="cuda", dtype=torch.int32)
        )
        prefix = torch.tensor(
            [row[3] for row in cases], device="cuda", dtype=torch.int32
        )
        reads = torch.empty(count, device="cuda", dtype=torch.int32)
        writes = torch.empty((count, query), device="cuda", dtype=torch.int32)
        record_ids = torch.empty_like(reads)
        invalid = torch.empty((count + 127) // 128, device="cuda", dtype=torch.int32)
        # Inspect the real kernel's validation flags without deliberately
        # triggering a device assert and poisoning the remaining GPU tests.
        _resolve_pool_indices[(invalid.numel(),)](
            table,
            records_table,
            prefix,
            reads,
            writes,
            record_ids,
            invalid,
            count,
            table.stride(0),
            records_table.stride(0),
            table.stride(1),
            state_rows,
            record_rows,
            query,
            0,
            DUMMY_REQUEST_STATE_BLOCK_ID,
            128,
        )
        self.assertEqual(
            reads.tolist(), [row[0] if row[4] == "live" else 0 for row in cases]
        )
        self.assertEqual(
            record_ids.tolist(), [row[2] if row[4] == "live" else 0 for row in cases]
        )
        self.assertEqual(
            writes.tolist(),
            [[row[1] if row[4] == "live" else 0] * query for row in cases],
        )
        self.assertEqual(
            invalid.tolist(),
            [
                sum(row[4] == "invalid" for row in cases[start : start + 128])
                for start in range(0, count, 128)
            ],
        )

    def test_graph_dummy_rows_transition_to_live_pool_indices(self):
        stage, table = fixture("cuda")
        table.fill_(DUMMY_REQUEST_STATE_BLOCK_ID)
        records_table = torch.full(
            (4, 1), DUMMY_REQUEST_STATE_BLOCK_ID, device="cuda", dtype=torch.int32
        )
        prefix = torch.full((4,), 13, device="cuda", dtype=torch.int32)
        invalid = torch.empty(1, device="cuda", dtype=torch.int32)
        buffers = stage.reads, stage.writes, stage.record_ids, invalid
        accepted = torch.zeros(4, dtype=torch.int32, device="cuda")
        physical = stage.states[0].transpose(-1, -2)
        before = physical.clone()
        records_before = stage.records[0].clone()

        def body():
            resolve_pool_indices(
                table,
                records_table,
                prefix,
                buffers,
                stage.states[0].shape[0],
                stage.records[0].shape[0],
            )
            replay_partial_acceptance(stage, accepted)

        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(stage.reads.tolist(), [0] * 4)
        self.assertEqual(stage.record_ids.tolist(), [0] * 4)
        self.assertEqual(invalid.item(), 0)
        torch.testing.assert_close(physical, before, rtol=0, atol=0)

        # The same captured pointers now describe two live requests, one
        # positive-length graph dummy, and one ordinary zero-prefix pad row.
        table.copy_(
            torch.tensor(
                [[5, 6], [-2, -2], [1, 2], [0, 0]], device="cuda", dtype=torch.int32
            )
        )
        records_table.copy_(
            torch.tensor([[6], [-2], [2], [0]], device="cuda", dtype=torch.int32)
        )
        prefix[-1] = 0
        accepted.copy_(torch.tensor([1, 0, 2, 0], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(stage.reads.tolist(), [5, 0, 1, 0])
        self.assertEqual(stage.writes[:, 0].tolist(), [6, 0, 2, 0])
        self.assertEqual(stage.record_ids.tolist(), [6, 0, 2, 0])
        self.assertEqual(invalid.item(), 0)
        expected = before.clone()
        expected[6] = reference_prefix(before[5], records_before[6], 1)
        expected[2] = reference_prefix(before[1], records_before[2], 2)
        torch.testing.assert_close(physical, expected, rtol=3e-5, atol=1e-6)
        torch.testing.assert_close(stage.records[0], records_before, rtol=0, atol=0)

    def test_single_decode_padding_uses_sequence_lengths_plus_one(self):
        stage, table = fixture("cuda")
        record_table = stage.record_ids[:, None].clone()
        wide_ssm = torch.full((4, 11), -999, device="cuda", dtype=torch.int32)
        wide_record = torch.full((4, 11), -999, device="cuda", dtype=torch.int32)
        wide_ssm[:, :2].copy_(table)
        wide_record[:, :1].copy_(record_table)
        lengths = torch.tensor([9, 13, 1, 1], device="cuda", dtype=torch.int32)
        invalid = torch.empty(1, device="cuda", dtype=torch.int32)
        resolve_pool_indices(
            wide_ssm[:, :2],
            wide_record[:, :1],
            lengths,
            (stage.reads, stage.writes, stage.record_ids, invalid),
            stage.states[0].shape[0],
            stage.records[0].shape[0],
            minimum_length=1,
        )
        self.assertEqual(stage.reads.tolist(), [1, 3, 0, 0])
        self.assertEqual(stage.record_ids.tolist(), [4, 3, 0, 0])

    def test_partial_replay_matches_fp32_reference(self):
        stage, table = fixture("cuda")
        physical = stage.states[0].transpose(-1, -2)
        before = physical.clone()
        simulate_full_candidate(stage)
        accepted = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
        replay_partial_acceptance(stage, accepted)
        torch.testing.assert_close(
            physical[table[:, 0]], before[table[:, 0]], rtol=0, atol=0
        )
        for row, count in enumerate(accepted.tolist()):
            if count:
                expected = reference_prefix(
                    before[table[row, 0]],
                    stage.records[0][stage.record_ids[row]],
                    count,
                )
                torch.testing.assert_close(
                    physical[table[row, 1]], expected, rtol=3e-5, atol=1e-6
                )

    def test_graph_replay_refreshes_request_indices_after_swaps_and_rebatch(self):
        stage, table = fixture("cuda")
        records_table = stage.record_ids[:, None].clone()
        prefix = torch.ones(4, device="cuda", dtype=torch.int32)
        invalid = torch.empty(1, device="cuda", dtype=torch.int32)
        buffers = stage.reads, stage.writes, stage.record_ids, invalid
        accepted = torch.tensor([1, 2, 0, 1], dtype=torch.int32, device="cuda")

        def body():
            resolve_pool_indices(
                table,
                records_table,
                prefix,
                buffers,
                stage.states[0].shape[0],
                stage.records[0].shape[0],
            )
            replay_partial_acceptance(stage, accepted)

        body()  # Compile and warm outside capture.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        physical = stage.states[0].transpose(-1, -2)
        for order in ([3, 0, 2, 1], [1, 3, 0, 2]):
            # Swap banks and reorder requests, including non-contiguous record
            # rows. Capture must not retain the original request-to-slot map.
            table.copy_(table.flip(1)[order])
            records_table.copy_(records_table[order])
            prefix[-1] = 0  # Captured B=4 bucket now has only three live rows.
            accepted[-1] = 0
            before = physical.clone()
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(stage.reads[-1].item(), 0)
            self.assertEqual(stage.record_ids[-1].item(), 0)
            for row, count in enumerate(accepted.tolist()):
                if count:
                    expected = reference_prefix(
                        before[table[row, 0]],
                        stage.records[0][records_table[row, 0]],
                        count,
                    )
                    torch.testing.assert_close(
                        physical[table[row, 1]], expected, rtol=3e-5, atol=1e-6
                    )
            torch.testing.assert_close(
                physical[table[:, 0]], before[table[:, 0]], rtol=0, atol=0
            )

    @unittest.skipUnless(torch.version.hip, "Indexed FlyDSL integration requires ROCm")
    def test_flydsl_indexed_pool_matches_existing_fp32_hybrid(self):
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_mtp_hybrid

        torch.manual_seed(1712)
        batch, query, heads, dim = 3, 3, 2, 128
        q, k, v = [
            torch.randn(batch, query, heads, dim, dtype=torch.bfloat16, device="cuda")
            * 0.1
            for _ in range(3)
        ]
        a, b = [
            torch.randn(batch, query, heads, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        bias = torch.randn(heads, dtype=torch.bfloat16, device="cuda")
        alog = torch.randn(heads, dtype=torch.float32, device="cuda")
        reads = torch.tensor([5, 1, 9], device="cuda", dtype=torch.int32)
        writes = (
            torch.tensor([6, 2, 10], device="cuda", dtype=torch.int32)[:, None]
            .expand(batch, query)
            .contiguous()
        )
        record_ids = torch.tensor([7, 2, 5], device="cuda", dtype=torch.int32)
        initial = torch.randn(13, heads, dim, dim, device="cuda") * 0.1
        legacy, indexed = initial.clone(), initial.clone()
        legacy_record = torch.empty(batch, query, heads, 3, dim, device="cuda")
        pool_record = torch.zeros(11, query, heads, 3, dim, device="cuda")
        expected, actual = torch.empty_like(v), torch.empty_like(v)
        flydsl_gdr_mtp_hybrid(
            q, k, v, a, b, bias, alog, legacy, expected, reads, writes, legacy_record
        )
        flydsl_gdr_mtp_hybrid(
            q,
            k,
            v,
            a,
            b,
            bias,
            alog,
            indexed,
            actual,
            reads,
            writes,
            pool_record,
            record_indices=record_ids,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(indexed, legacy, rtol=0, atol=0)
        for row, record_id in enumerate(record_ids.tolist()):
            torch.testing.assert_close(
                pool_record[record_id, :, :, :2],
                legacy_record[row, :, :, :2],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                pool_record[record_id, :, :, 2, 0],
                legacy_record[row, :, :, 2, 0],
                rtol=0,
                atol=0,
            )
        # No records may leak into batch-numbered or reserved pool rows.
        untouched = [
            row for row in range(pool_record.shape[0]) if row not in record_ids.tolist()
        ]
        self.assertEqual(torch.count_nonzero(pool_record[untouched]).item(), 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flydsl_gdr_mtp_hybrid(
                q,
                k,
                v,
                a,
                b,
                bias,
                alog,
                indexed,
                actual,
                reads,
                writes,
                pool_record,
                record_indices=record_ids,
            )
        # Same captured pointers, different request-owned record rows. The last
        # request is padding and must zero output without touching pool row 0.
        reads.copy_(torch.tensor([1, 9, 0], device="cuda", dtype=torch.int32))
        writes.copy_(
            torch.tensor([2, 10, 0], device="cuda", dtype=torch.int32)[
                :, None
            ].expand_as(writes)
        )
        record_ids.copy_(torch.tensor([3, 8, 0], device="cuda", dtype=torch.int32))
        indexed.copy_(initial)
        legacy.copy_(initial)
        pool_record.zero_()
        actual.fill_(float("nan"))
        graph.replay()
        flydsl_gdr_mtp_hybrid(
            q, k, v, a, b, bias, alog, legacy, expected, reads, writes, legacy_record
        )
        torch.testing.assert_close(actual[:2], expected[:2], rtol=0, atol=0)
        torch.testing.assert_close(indexed, legacy, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(actual[2]).item(), 0)
        self.assertEqual(torch.count_nonzero(pool_record[0]).item(), 0)
        for row, record_id in enumerate([3, 8]):
            torch.testing.assert_close(
                pool_record[record_id, :, :, :2],
                legacy_record[row, :, :, :2],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                pool_record[record_id, :, :, 2, 0],
                legacy_record[row, :, :, 2, 0],
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
