"""CPU tests; explicit CUDA harness: torchrun --nproc-per-node={4,8} FILE --gpu.

No CUDA discovery or initialization occurs in the ordinary unittest path.
"""

from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "models_py/distributed/zigzag_row_exchange.py"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DirectedExchangeCPU(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), "directed row transport is missing")
        self.module = load_file("k31_directed_cpu", PATH)
        self.plan = load_file("k31_plan_cpu", ROOT / "models_py/modules/hybrid/natural_zigzag_plan.py")

    def test_chunk_intersections_copy_exact_tagged_rows(self):
        # Catches wrong incoming direction, peer offset or chunk-local offset.
        sources = torch.arange(24 * 3).reshape(4, 6, 3).to(torch.bfloat16)
        expected = [[0, 1, 14, 15, 16, 23], [2, 3, 12, 13, 17, 22],
                    [4, 5, 10, 11, 18, 21], [6, 7, 8, 9, 19, 20]]
        zigzag = []
        for rank in range(4):
            plan = self.plan.NaturalZigzagPlan([11, 5], 4, rank)
            output = torch.full((6, 3), -1, dtype=torch.bfloat16)
            for start in range(0, 6, 3):
                for peer, source, destination, rows in self.module.chunk_intersections(plan.to_zigzag, start, min(start + 3, 6)):
                    output[start + destination:start + destination + rows].copy_(sources[peer, source:source + rows])
            self.assertTrue(torch.equal(output, sources.reshape(24, 3)[expected[rank]]))
            zigzag.append(output)
        for rank in range(4):
            plan = self.plan.NaturalZigzagPlan([11, 5], 4, rank)
            output = torch.empty((6, 3), dtype=torch.bfloat16)
            for start in range(0, 6, 5):
                for peer, source, destination, rows in self.module.chunk_intersections(plan.to_natural, start, min(start + 5, 6)):
                    output[start + destination:start + destination + rows].copy_(zigzag[peer][source:source + rows])
            self.assertTrue(torch.equal(output, sources[rank]))

    def test_capacity_budget_counts_one_source_and_two_bounded_slots(self):
        self.assertEqual(self.module.payload_bytes(8192, 7168, 1024), 146800640)
        self.assertEqual(self.module.payload_bytes(6, 3, 1024), 108)
        for args in ((0, 3, 2), (6, 0, 2), (6, 3, 0), (-1, 3, 2)):
            with self.assertRaises(ValueError):
                self.module.payload_bytes(*args)

    def test_cp8_literal_fixtures_and_production_block_mapping(self):
        # Both directions are checked against independent literal maps; no
        # forward/reverse roundtrip can cancel a shared routing mistake.
        for lengths, expected in known_fixtures(8):
            for rank in range(8):
                plan = self.plan.NaturalZigzagPlan(lengths, 8, rank)
                natural = torch.arange(plan.physical_rows).reshape(8, plan.local_rows)
                zigzag = torch.tensor(expected, dtype=torch.int64)
                for direction, source, wanted in (("to_zigzag", natural, zigzag[rank]),
                                                   ("to_natural", zigzag, natural[rank])):
                    output = torch.empty_like(wanted)
                    for start in range(0, plan.local_rows, 3):
                        for peer, src, dst, rows in self.module.chunk_intersections(
                                getattr(plan, direction), start, min(start + 3, plan.local_rows)):
                            output[start + dst:start + dst + rows].copy_(source[peer, src:src + rows])
                    self.assertTrue(torch.equal(output, wanted))
        natural = torch.arange(65536).reshape(8, 8192)
        block_pairs = [(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8)]
        zigzag = torch.stack([torch.cat([torch.arange(block * 4096, (block + 1) * 4096)
                                        for block in pair]) for pair in block_pairs])
        for rank in range(8):
            plan = self.plan.NaturalZigzagPlan([65536], 8, rank)
            for direction, source, wanted in (("to_zigzag", natural, zigzag[rank]),
                                               ("to_natural", zigzag, natural[rank])):
                output = torch.empty_like(wanted)
                for start in range(0, 8192, 1023):
                    for peer, src, dst, rows in self.module.chunk_intersections(
                            getattr(plan, direction), start, min(start + 1023, 8192)):
                        output[start + dst:start + dst + rows].copy_(source[peer, src:src + rows])
                self.assertTrue(torch.equal(output, wanted))

    def test_plan_contract_rejects_gaps_wrong_rank_or_capacity(self):
        plan = self.plan.NaturalZigzagPlan([11, 5], 4, 2)
        self.module.validate_plan(plan, "to_zigzag", 4, 2, 6)
        for direction, rank, capacity in (("wrong", 2, 6), ("to_zigzag", 1, 6), ("to_natural", 2, 5)):
            with self.assertRaises(ValueError):
                self.module.validate_plan(plan, direction, 4, rank, capacity)
        plan.to_zigzag = plan.to_zigzag[1:]
        with self.assertRaises(ValueError):
            self.module.validate_plan(plan, "to_zigzag", 4, 2, 6)

    def test_tensor_contract_rejects_private_aliases_and_wrong_precision(self):
        # Exercise real Torch storage/strides without constructing CUDA state.
        transport = self.module.DirectedRowExchange.__new__(self.module.DirectedRowExchange)
        transport.device, transport.hidden_size = torch.device("cpu"), 3
        transport._source = torch.empty((6, 3), dtype=torch.bfloat16)
        transport._slots = torch.empty((2, 3, 3), dtype=torch.bfloat16)
        values = torch.empty((6, 3), dtype=torch.bfloat16)
        transport._validate_tensor(values, 6, "values")
        bad = [transport._source, transport._slots.reshape(6, 3), values.float(),
               torch.empty((3, 6), dtype=torch.bfloat16).t(), values[:5]]
        for tensor in bad:
            with self.assertRaises(ValueError):
                transport._validate_tensor(tensor, 6, "out")

    def test_host_consensus_is_unanimous_and_releases_completed_vote_keys(self):
        # A real CPU Store proves local setup failure reaches every participant.
        store = torch.distributed.HashStore()
        participants = []
        for rank in range(4):
            participant = self.module.DirectedRowExchange.__new__(self.module.DirectedRowExchange)
            participant._store, participant.rank, participant.size = store, rank, 4
            participant.enabled = True
            participants.append(participant)
        results = [None] * 4

        def vote(rank, phase, fail=False):
            try:
                results[rank] = participants[rank]._agree(phase, rank, "allocation failed" if fail else "")
            except RuntimeError as exc:
                results[rank] = str(exc)

        for iteration in range(4):
            threads = [threading.Thread(target=vote, args=(rank, str(iteration), iteration == 3 and rank == 1))
                       for rank in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            if iteration == 3:
                self.assertTrue(all("allocation failed" in result for result in results))
                self.assertTrue(all(not participant.enabled for participant in participants))
            else:
                self.assertEqual(results, [[0, 1, 2, 3]] * 4)
            self.assertEqual(store.num_keys(), 0, "per-call validation keys must not accumulate")

    def test_prepared_plan_is_cached_and_rejects_changed_geometry(self):
        self.assertTrue(hasattr(self.module.DirectedRowExchange, "prepare_plan"),
                        "transport lacks a once-per-geometry collective preparation API")
        store = torch.distributed.HashStore()
        participants, plans = [], []
        for rank in range(4):
            participant = self.module.DirectedRowExchange.__new__(self.module.DirectedRowExchange)
            participant._store, participant.rank, participant.size = store, rank, 4
            participant.enabled = True
            participant.max_rows, participant.chunk_capacity = 6, 3
            participant._prepared, participant._prepare_epoch = {}, 0
            participant._lock = threading.Lock()
            participants.append(participant)
            plans.append(self.plan.NaturalZigzagPlan([11, 5], 4, rank))
        failures = []

        def prepare(rank):
            try:
                participants[rank].prepare_plan(plans[rank], chunk_rows=2)
            except Exception as exc:
                failures.append(exc)

        threads = [threading.Thread(target=prepare, args=(rank,)) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        # A cached local recheck needs no peers or Store rendezvous.
        for rank in range(4):
            participants[rank].prepare_plan(plans[rank], chunk_rows=2)
            participants[rank]._require_prepared(plans[rank], 2)
            participants[rank]._require_prepared(plans[rank], 3)
            self.assertEqual(participants[rank]._prepare_epoch, 1)
            with self.assertRaises(ValueError):
                participants[rank]._require_prepared(plans[rank], 1)
        self.assertEqual(store.num_keys(), 0)
        original_plans = plans
        for length in range(1, 17):
            plans = [self.plan.NaturalZigzagPlan([length], 4, rank) for rank in range(4)]
            threads = [threading.Thread(target=prepare, args=(rank,)) for rank in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            for participant in participants:
                self.assertLessEqual(len(participant._prepared), 16, "geometry cache must stay bounded")
        for rank in range(4):
            with self.assertRaises(ValueError):
                participants[rank]._require_prepared(original_plans[rank], 2)
        plans = original_plans
        threads = [threading.Thread(target=prepare, args=(rank,)) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        for rank in range(4):
            participants[rank]._require_prepared(plans[rank], 2)
            self.assertEqual(participants[rank]._prepare_epoch, 18)
        plans[0].to_zigzag = plans[0].to_zigzag[1:]
        with self.assertRaises(ValueError):
            participants[0]._require_prepared(plans[0], 2)


def known_fixtures(size):
    """Hand-specified natural row IDs for each Zigzag rank, including padding."""
    return {
        4: [([16], [[0, 1, 14, 15], [2, 3, 12, 13], [4, 5, 10, 11], [6, 7, 8, 9]]),
            ([11, 5], [[0, 1, 14, 15, 16, 23], [2, 3, 12, 13, 17, 22],
                       [4, 5, 10, 11, 18, 21], [6, 7, 8, 9, 19, 20]]),
            ([1], [[0, 7], [1, 6], [2, 5], [3, 4]]),
            ([0], [[], [], [], []])],
        8: [([16], [[0, 15], [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8]]),
            ([11, 5], [[0, 15, 16, 31], [1, 14, 17, 30], [2, 13, 18, 29], [3, 12, 19, 28],
                       [4, 11, 20, 27], [5, 10, 21, 26], [6, 9, 22, 25], [7, 8, 23, 24]]),
            ([1], [[0, 15], [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8]]),
            ([0], [[], [], [], [], [], [], [], []])],
    }[size]


def gpu_main():
    """All ranks print proof JSON only after exact BF16 checks synchronize."""
    import torch.distributed as dist

    def assert_bits(actual, expected):
        torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)

    module = load_file("k31_directed_gpu", PATH)
    plans = load_file("k31_plan_gpu", ROOT / "models_py/modules/hybrid/natural_zigzag_plan.py")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    group = dist.group.WORLD
    rank, size = dist.get_rank(), dist.get_world_size()
    if size not in (4, 8):
        raise ValueError("the independent CUDA fixtures require CP4 or CP8")
    allocator_handoff_regression(module, plans, group, device, assert_bits)
    transport = module.DirectedRowExchange(group, device, 12, 3, chunk_capacity=3)
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    fixtures = known_fixtures(size)
    checks = 0
    held = []
    for iteration in range(8):
        for lengths, expected in fixtures:
            plan = plans.NaturalZigzagPlan(lengths, size, rank)
            transport.prepare_plan(plan, chunk_rows=2)
            rows = plan.local_rows
            stream = streams[iteration % 2]
            with torch.cuda.stream(stream):
                tags = (torch.arange(plan.physical_rows * 3, device=device).reshape(-1, 3)
                        + 0x3F00 + iteration * 128).to(torch.int16).view(torch.bfloat16)
                values = tags[rank * rows:(rank + 1) * rows].clone()
                wanted = tags[expected[rank]]
                zigzag = transport.exchange(values, plan, "to_zigzag")
                assert_bits(zigzag, wanted)
                held.append((zigzag, wanted.clone()))
                restored = torch.empty_like(values)
                self_out = transport.exchange(zigzag, plan, "to_natural", out=restored)
                assert self_out is restored
                assert_bits(restored, values)
                # Alias original input is safe only after source publication.
                transport.exchange(values, plan, "to_zigzag", out=values)
                assert_bits(values, wanted)
                consumed = torch.empty_like(wanted)

                def consume(start, chunk):
                    # Deliberately keep a slot reader busy while the next pulls run.
                    if rank == 1:
                        torch.cuda._sleep(2_000_000)
                    consumed[start:start + len(chunk)].copy_(chunk)

                # Delay source publication on one rank to exercise ready barrier.
                if rank == 2:
                    torch.cuda._sleep(2_000_000)
                transport.consume_chunks(restored, plan, "to_zigzag", 3, consume)
                assert_bits(consumed, wanted)
                # Delay a peer pull itself; republishing must await all readers.
                with torch.cuda.stream(transport._stream):
                    if rank == 3:
                        torch.cuda._sleep(2_000_000)
                transport.consume_chunks(consumed, plan, "to_natural", 2, consume)
                assert_bits(consumed, restored)
                checks += 5
    # Queue a whole chain without host tensor assertions between calls. This
    # exposes early source reuse and cross-caller-stream lifetime races that a
    # synchronizing assertion after each exchange could accidentally conceal.
    plan = plans.NaturalZigzagPlan([11, 5], size, rank)
    transport.prepare_plan(plan, chunk_rows=2)
    expected_rows = fixtures[1][1][rank]
    for iteration in range(12):
        with torch.cuda.stream(streams[iteration % 2]):
            tags = (torch.arange(plan.physical_rows * 3, device=device).reshape(-1, 3)
                    + 0x4300 + iteration * 128).to(torch.int16).view(torch.bfloat16)
            values = tags[rank * plan.local_rows:(rank + 1) * plan.local_rows].clone()
            with torch.cuda.stream(transport._stream):
                if rank == 3:
                    torch.cuda._sleep(2_000_000)
            result = transport.exchange(values, plan, "to_zigzag")
            held.append((result, tags[expected_rows].clone()))
            restored = transport.exchange(result, plan, "to_natural")
            held.append((restored, values.clone()))
            checks += 2
    torch.cuda.synchronize(device)
    for actual, expected in held:
        assert_bits(actual, expected)
    assert transport.resident_bytes == module.payload_bytes(12, 3, 3)
    print(json.dumps({"proof": "exact_bf16_directed_rows", "rank": rank,
                      "cp_size": size,
                      "checks": checks, "owned_outputs": len(held),
                      "resident_bytes": transport.resident_bytes}), flush=True)
    production_shape_regression(module, plans, group, device, assert_bits)
    dist.destroy_process_group()


def allocator_handoff_regression(module, plans, group, device, assert_bits):
    """Inject real dirty allocator reuse immediately before slot allocation.

    Instrumenting the exact torch.empty call avoids unrelated raw symmetric
    initialization draining the pending write before the ownership edge under
    test. No tensor, CUDA stream/event, allocator or transport method is mocked.
    """
    from unittest.mock import patch

    constructor_stream = torch.cuda.Stream(device=device)
    caller_stream = torch.cuda.Stream(device=device)
    old_write_done = torch.cuda.Event()
    real_empty = torch.empty
    evidence = {}
    shape = (2, 3, 3)

    def dirty_recycled_empty(*args, **kwargs):
        if args and args[0] == shape and kwargs.get("dtype") == torch.bfloat16:
            assert not evidence, "exact receive-slot allocation must occur once"
            assert torch.cuda.current_stream(device) == constructor_stream
            dirty = real_empty(*args, **kwargs)
            pointer = dirty.data_ptr()
            # Prime this dtype/shape's fill kernel, the sleep kernel and event
            # creation before creating the hazard. Lazy CUDA setup may drain
            # earlier stream work and would make a first-use sleep inconclusive.
            dirty.fill_(-123)
            torch.cuda._sleep(1)
            old_write_done.record(constructor_stream)
            constructor_stream.synchronize()
            assert old_write_done.query(), "allocator regression warmup did not complete"
            # This work owns the allocation stream but the tensor is released
            # before it finishes. Same-stream allocator reuse is intentional.
            torch.cuda._sleep(2_000_000_000)
            dirty.fill_(-123)
            old_write_done.record(constructor_stream)
            del dirty
            recycled = real_empty(*args, **kwargs)
            assert recycled.data_ptr() == pointer, "regression did not force recycled slot storage"
            assert not old_write_done.query(), "old write completed before the tested handoff"
            evidence.update(reused_pointer=pointer, old_write_pending=True)
            return recycled
        return real_empty(*args, **kwargs)

    with torch.cuda.stream(constructor_stream):
        with patch.object(torch, "empty", dirty_recycled_empty):
            transport = module.DirectedRowExchange(group, device, 12, 3, chunk_capacity=3)
    assert evidence.get("old_write_pending"), "slot allocation interception did not execute"
    assert old_write_done.query(), "constructor did not join the slot allocation stream"
    assert transport._slots.data_ptr() == evidence["reused_pointer"]
    plan = plans.NaturalZigzagPlan([16], group.size(), group.rank())
    transport.prepare_plan(plan, chunk_rows=2)
    known_rows = {
        4: [[0, 1, 14, 15], [2, 3, 12, 13], [4, 5, 10, 11], [6, 7, 8, 9]],
        8: [[0, 15], [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8]],
    }[group.size()][group.rank()]
    # The wrapper is restored before the first call, on a different stream.
    with torch.cuda.stream(caller_stream):
        tags = (torch.arange(48, device=device).reshape(16, 3) + 0x3F00).to(torch.int16).view(torch.bfloat16)
        values = tags[group.rank() * plan.local_rows:(group.rank() + 1) * plan.local_rows].clone()
        received = torch.empty_like(values)

        def consume(start, chunk):
            received[start:start + len(chunk)].copy_(chunk)

        transport.consume_chunks(values, plan, "to_zigzag", 2, consume)
        assert_bits(received, tags[known_rows])
        result = transport.exchange(received, plan, "to_natural")
        assert_bits(result, values)
    caller_stream.synchronize()
    print(json.dumps({"proof": "dirty_slot_allocation_handoff", "rank": group.rank(),
                      "cp_size": group.size(), "reused_storage": True,
                      "prior_write_pending_at_reuse": True, "different_first_caller": True}), flush=True)


def production_shape_regression(module, plans, group, device, assert_bits):
    """Exact 64K x 7168 routing without constructing a global full-H tensor."""
    import torch.distributed as dist

    size, rank = group.size(), group.rank()
    rows, hidden, chunk = 65536 // size, 7168, 1024
    # Explicit paired block IDs, independent of the receive plan's segments.
    paired_blocks = {
        4: [(0, 7), (1, 6), (2, 5), (3, 4)],
        8: [(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8)],
    }[size][rank]
    half = rows // 2
    natural_rows = torch.arange(rank * rows, (rank + 1) * rows, dtype=torch.int32, device=device)
    zigzag_rows = torch.cat([torch.arange(block * half, (block + 1) * half,
                                        dtype=torch.int32, device=device) for block in paired_blocks])
    columns = torch.arange(hidden, dtype=torch.int32, device=device)

    def tagged(row_ids, iteration):
        # All 16 bits are compared, including NaN payloads; transport performs
        # no arithmetic on BF16 values. Row tags cover all 65536 physical rows.
        bits = row_ids[:, None].bitwise_xor(columns[None, :] * 17 + iteration * 131)
        return bits.to(torch.int16).view(torch.bfloat16)

    dist.barrier(group=group)
    torch.cuda.synchronize(device)
    free_before = torch.cuda.mem_get_info(device)[0]
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    transport = module.DirectedRowExchange(group, device, rows, hidden, chunk_capacity=chunk)
    torch.cuda.synchronize(device)
    measured = dict(
        cuda_free_delta=free_before - torch.cuda.mem_get_info(device)[0],
        caching_allocated_delta=torch.cuda.memory_allocated(device) - allocated_before,
        caching_reserved_delta=torch.cuda.memory_reserved(device) - reserved_before,
    )
    expected_bytes = 2 * rows * hidden + 4 * chunk * hidden
    assert transport.resident_bytes == expected_bytes
    assert transport.source.shape == (rows, hidden)
    assert transport._slots.shape == (2, chunk, hidden)
    assert transport._handle.offset == 0
    assert all(peer.shape == (rows, hidden) for peer in transport._peers)
    plan = plans.NaturalZigzagPlan([65536], size, rank)
    transport.prepare_plan(plan, chunk_rows=1023)
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    # Row/column metadata was produced on the entry stream before use below.
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
    checks = 0
    for iteration in range(3):
        with torch.cuda.stream(streams[iteration % 2]):
            values = tagged(natural_rows, iteration)
            expected = tagged(zigzag_rows, iteration)
            result = transport.exchange(values, plan, "to_zigzag")
            assert_bits(result, expected)
            # Independently tagged Zigzag input proves reverse mapping without
            # relying on a forward/reverse pair cancelling the same mistake.
            reverse = transport.exchange(expected, plan, "to_natural")
            assert_bits(reverse, values)
            consumed = torch.empty_like(values)

            def consume(start, received):
                if rank == size - 1:
                    torch.cuda._sleep(100_000)
                consumed[start:start + len(received)].copy_(received)

            transport.consume_chunks(values, plan, "to_zigzag", 1023, consume)
            assert_bits(consumed, expected)
            checks += 3
        # Bound live caller-owned outputs and their allocator lifetime in this
        # test; transport capacity stays fixed through all repeated calls.
        streams[iteration % 2].synchronize()
        del values, expected, result, reverse, consumed
    torch.cuda.synchronize(device)
    print(json.dumps({"proof": "production_64k_bf16_directed_rows", "rank": rank,
                      "cp_size": size, "hidden_size": hidden, "physical_rows": 65536,
                      "local_rows": rows, "checks": checks,
                      "resident_bytes": transport.resident_bytes,
                      "signal_pad_bytes": transport.signal_pad_bytes,
                      "raw_offset": transport._handle.offset,
                      "allocation_measurements": measured}), flush=True)


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        gpu_main()
    else:
        unittest.main()
