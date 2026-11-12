"""CPU simulation of symmetric-memory expert migration."""

import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

import rtp_llm.eplb.shuffle as shuffle


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

    def _exercise(self, noop=False):
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
        # Reuse the same storage and mirrors over two updates.
        for old, new in ((origin, next_mapping), (next_mapping, origin)):
            def run(rank):
                shuffle.expert_model_parallel_shuffle_inplace_intranode(
                    mirrors, old, weights[rank], new, groups[rank]
                )

            with ThreadPoolExecutor(size) as executor:
                list(executor.map(run, range(size)))
            self._check_weights(new, weights, catalog, identities)
        self.assertEqual([group.barrier_count for group in groups], [8] * size)

    def test_symmetric_memory_mirror_reuse_preserves_all_weight_representations(self):
        self._exercise()

    def test_intranode_unchanged_maps_keep_weights(self):
        self._exercise(noop=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
