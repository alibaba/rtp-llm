"""CPU transport simulation; actual NCCL/IPC is covered by distributed tests."""

import threading
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import torch

import rtp_llm.eplb.shuffle as shuffle


class _Work:
    def __init__(self):
        self.done = threading.Event()

    def wait(self):
        if not self.done.wait(10):
            raise AssertionError("Unmatched simulated P2P send/receive")


class _Group:
    def __init__(self, rank, size, barrier):
        self._rank = rank
        self._size = size
        self._barrier = barrier
        self.barrier_count = 0

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def barrier(self):
        self.barrier_count += 1
        return SimpleNamespace(wait=lambda: self._barrier.wait(timeout=10))


class _CpuTransport:
    """Match each pair's ordered byte messages without knowing expert maps."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sends = defaultdict(deque)
        self.recvs = defaultdict(deque)
        self.batches = []

    def batch(self, ops):
        requests = []
        with self.lock:
            self.batches.append(ops)
            for op in ops:
                assert op.tensor.is_contiguous()
                assert op.tensor.dtype == torch.uint8
                work = _Work()
                requests.append(work)
                if op.op is shuffle.dist.isend:
                    key = (op.group.rank(), op.peer)
                    self.sends[key].append((op.tensor, work))
                else:
                    key = (op.peer, op.group.rank())
                    self.recvs[key].append((op.tensor, work))
                while self.sends[key] and self.recvs[key]:
                    source, sent = self.sends[key].popleft()
                    destination, received = self.recvs[key].popleft()
                    destination.copy_(source)
                    sent.done.set()
                    received.done.set()
        return requests


def _p2p_op(op, tensor, peer, group):
    return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)


def _bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


class ShuffleTest(unittest.TestCase):
    def test_sender_allocation_matches_reference_known_vectors(self):
        for senders, receivers, expected in (
            (1, 4, [0, 0, 0, 0]),
            (2, 5, [0, 0, 1, 1, 0]),
            (3, 8, [0, 0, 1, 1, 2, 2, 0, 1]),
            (4, 2, [0, 1]),
            (2, 0, []),
        ):
            with self.subTest(senders=senders, receivers=receivers):
                self.assertEqual(
                    [
                        shuffle._paired_sender(i, senders, receivers)
                        for i in range(receivers)
                    ],
                    expected,
                )

    def test_routes_use_first_local_owner_and_deduplicate_receivers(self):
        origin = [0, 0, 1, 2, 0, 2, 3, 3, 1, 4, 4, 2]
        current = [0, 2, 2, 1, 1, 2, 0, 0, 3, 0, 4, 4]
        self.assertEqual(
            shuffle._remote_routes(origin, current, 3, 0),
            ([(0, 0), (1, 1)], [2, 3]),
        )
        self.assertEqual(
            shuffle._remote_routes(origin, current, 3, 1),
            ([(0, 2), (2, 2)], [1]),
        )

    def test_intra_plan_covers_cycles_duplicate_and_noop_slots(self):
        origin = [4, 5, 6, 4]
        current = [5, 6, 4, 4]
        received, copies = shuffle._plan_intra_copies(origin, current)
        self.assertEqual(received, [True] * 4)
        values = origin.copy()
        for source, destination in copies:
            values[destination] = origin[source]
        self.assertEqual(values, current)
        self.assertEqual(copies, [(0, 2), (1, 0), (2, 1)])
        self.assertEqual(shuffle._plan_intra_copies(origin, origin), ([True] * 4, []))

    def _weights(self, mapping, size):
        num_layers, physical = mapping.shape
        local = physical // size
        logical = int(mapping.max()) + 1
        dtypes = (
            torch.int8,
            torch.int32,
            torch.bfloat16,
            torch.float8_e4m3fn,
        )
        if hasattr(torch, "float8_e8m0fnu"):
            dtypes += (torch.float8_e8m0fnu,)
        catalog = []
        for layer in range(num_layers):
            tensors = []
            for index, dtype in enumerate(dtypes):
                tensor = torch.empty(logical, 2, 4, dtype=dtype)
                for expert in range(logical):
                    raw = torch.arange(tensor[expert].numel() * tensor.element_size())
                    tensor[expert].view(torch.uint8).reshape(-1).copy_(
                        (raw + layer * 67 + expert * 19 + index * 7).to(torch.uint8)
                    )
                tensors.append(tensor)
            catalog.append(tensors)
        weights = []
        for rank in range(size):
            layers = []
            for layer in range(num_layers):
                tensors = []
                for index, dtype in enumerate(dtypes):
                    if index == 1:
                        tensor = torch.empty(local, 4, 2, dtype=dtype).mT
                    elif index == 2:
                        tensor = torch.empty(2, local, 4, dtype=dtype).permute(1, 0, 2)
                    else:
                        tensor = torch.empty(local, 2, 4, dtype=dtype)
                    for slot in range(local):
                        expert = int(mapping[layer, rank * local + slot])
                        tensor[slot].copy_(catalog[layer][index][expert])
                    tensors.append(tensor)
                layers.append(tensors)
            weights.append(layers)
        return catalog, weights

    def _check_weights(self, mapping, weights, catalog, identities):
        local = weights[0][0][0].size(0)
        for rank, layers in enumerate(weights):
            for layer, tensors in enumerate(layers):
                for index, tensor in enumerate(tensors):
                    self.assertEqual(
                        (tensor.data_ptr(), tuple(tensor.shape), tensor.stride()),
                        identities[rank][layer][index],
                    )
                    for slot in range(local):
                        expert = int(mapping[layer, rank * local + slot])
                        self.assertTrue(
                            torch.equal(
                                _bytes(tensor[slot]),
                                _bytes(catalog[layer][index][expert]),
                            ),
                            (rank, layer, index, slot, expert),
                        )

    def _exercise(self, intranode, p2p_groups=1, noop=False):
        size = 4
        origin = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5],
                [5, 4, 3, 2, 1, 0, 5, 3, 1, 4, 2, 0],
            ]
        )
        next_mapping = torch.tensor(
            [
                [3, 3, 0, 0, 5, 2, 1, 1, 4, 4, 2, 5],
                [2, 2, 5, 5, 0, 3, 4, 4, 1, 1, 3, 0],
            ]
        )
        if noop:
            next_mapping = origin.clone()
        catalog, weights = self._weights(origin, size)
        identities = [
            [
                [(w.data_ptr(), tuple(w.shape), w.stride()) for w in layer]
                for layer in layers
            ]
            for layers in weights
        ]
        barrier = threading.Barrier(size)
        groups = [_Group(rank, size, barrier) for rank in range(size)]
        mirrors = [
            [torch.empty(w.shape, dtype=w.dtype) for w in layers[0]]
            for layers in weights
        ]
        transport = _CpuTransport()
        with patch.object(shuffle.dist, "P2POp", _p2p_op), patch.object(
            shuffle.dist, "batch_isend_irecv", transport.batch
        ), patch.object(shuffle.dist, "get_global_rank", lambda group, rank: rank):
            # Reuse the same storage/mirrors over two layers and two updates.
            for old, new in ((origin, next_mapping), (next_mapping, origin)):

                def run(rank):
                    if intranode:
                        shuffle.expert_model_parallel_shuffle_inplace_intranode(
                            mirrors, old, weights[rank], new, groups[rank]
                        )
                    else:
                        shuffle.expert_model_parallel_shuffle_inplace(
                            old, weights[rank], new, groups[rank], p2p_groups
                        )

                with ThreadPoolExecutor(size) as executor:
                    list(executor.map(run, range(size)))
                self._check_weights(new, weights, catalog, identities)
        self.assertTrue(all(not queue for queue in transport.sends.values()))
        self.assertTrue(all(not queue for queue in transport.recvs.values()))
        self.assertEqual(
            [group.barrier_count for group in groups],
            [8 if intranode else 0] * size,
        )
        if noop or intranode:
            self.assertEqual(transport.batches, [])
        else:
            self.assertGreater(len(transport.batches), 0)
            for ops in transport.batches:
                buckets = {
                    (
                        op.group.rank() - op.peer
                        if op.op is shuffle.dist.isend
                        else op.peer - op.group.rank()
                    )
                    % p2p_groups
                    for op in ops
                }
                self.assertEqual(len(buckets), 1)

    def test_p2p_moves_bytes_and_preserves_executor_addresses(self):
        for groups in (1, 2, 3, 8):
            with self.subTest(p2p_groups=groups):
                self._exercise(False, groups)

    def test_intranode_mirror_reuse_preserves_all_weight_representations(self):
        self._exercise(True)

    def test_unchanged_maps_submit_no_p2p(self):
        self._exercise(False, noop=True)

    def test_intranode_unchanged_maps_keep_weights(self):
        self._exercise(True, noop=True)

    def test_p2p_rejects_zero_bucket_count(self):
        with self.assertRaisesRegex(ValueError, "p2p_groups must be positive"):
            shuffle.expert_model_parallel_shuffle_inplace(None, None, None, None, 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
