"""Fixed-capacity MTP draft metadata and validation before graph replay."""

from collections.abc import Mapping

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)


def draft_graph_lanes(inputs, token_capacity):
    batch = inputs.input_lengths_device.numel()
    if not batch or token_capacity % batch or not 1 <= token_capacity // batch <= 4:
        raise RuntimeError(
            "QSA draft graph requires one-to-four-token request capacity"
        )
    return token_capacity // batch


def draft_graph_rows(inputs, token_capacity):
    """Map packed live tokens to fixed request lanes using device metadata."""
    lengths = inputs.input_lengths_device
    prefixes = inputs.prefix_lengths_device
    cu = inputs.cu_seqlens_device
    batch = lengths.numel()
    lanes = draft_graph_lanes(inputs, token_capacity)
    offsets = torch.arange(lanes, device=cu.device, dtype=torch.int64)
    active = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    sources = cu[:-1].long().unsqueeze(1) + offsets.unsqueeze(0)
    sources = torch.where(active, sources, 0).reshape(-1)
    rows = torch.arange(token_capacity, device=cu.device, dtype=torch.int64)
    request = (rows.unsqueeze(1) >= cu[1:].unsqueeze(0)).sum(1).clamp(max=batch - 1)
    local = rows - cu.long().index_select(0, request)
    packed_valid = rows < cu[-1]
    inverse = torch.where(packed_valid, request * lanes + local, 0)
    positions = prefixes.long().index_select(0, request) + local
    positions = torch.where(packed_valid, positions, 0)
    visible = prefixes.unsqueeze(1) + offsets.unsqueeze(0) + 1
    visible = torch.where(active, visible, 0).to(torch.int32)
    return sources, inverse, packed_valid, positions, visible


class DraftPrefillGraphReplay(Mapping):
    """Validate all tagged pools before the graph runner launches any writes.

    This intentionally is not a dict: the existing graph runner calls the
    model-wide callback with all cache groups, rather than each FMHA separately.
    """

    def __init__(self, impls, bounds, position_factor, token_capacity):
        self.impls = impls
        self.bounds = bounds
        self.position_factor = position_factor
        self.token_capacity = token_capacity

    def __getitem__(self, key):
        return self.impls[key]

    def __iter__(self):
        return iter(self.impls)

    def __len__(self):
        return len(self.impls)

    def prepare_cuda_graph(self, groups):
        anchor = groups[next(iter(self.impls))]
        lengths = anchor.input_lengths
        prefixes = anchor.prefix_lengths
        cu = anchor.cu_seqlens
        if any(value.device.type != "cpu" for value in (lengths, prefixes, cu)):
            raise RuntimeError("QSA draft graph replay requires host length mirrors")
        lengths = lengths.tolist()
        prefixes = prefixes.tolist()
        starts = cu.tolist()
        if (
            not lengths
            or len(prefixes) != len(lengths)
            or len(starts) != len(lengths) + 1
        ):
            raise RuntimeError("QSA draft graph replay has invalid length geometry")
        expected = [0]
        lanes = draft_graph_lanes(anchor, self.token_capacity)
        padding_started = False
        positions = []
        for length, prefix in zip(lengths, prefixes):
            if length == 0:
                padding_started = True
            elif padding_started or not 1 <= length <= lanes or prefix <= 0:
                raise RuntimeError(
                    "QSA draft graph requires positive live prefixes and lengths within request capacity followed by padding"
                )
            expected.append(expected[-1] + length)
            positions.extend(range(prefix, prefix + length))
        live_batch = next(
            (i for i, length in enumerate(lengths) if length == 0), len(lengths)
        )
        # Only the live prefix of the host cu_seqlens mirror is copied by the
        # runner. Its device padding tail is filled separately for every replay.
        if (
            starts[: live_batch + 1] != expected[: live_batch + 1]
            or expected[-1] > self.token_capacity
        ):
            raise RuntimeError(
                "QSA draft graph cu_seqlens do not partition the live token capacity"
            )
        for tag, pool_size, page_size, compressed in self.bounds:
            inputs = groups[tag]
            if not torch.equal(
                inputs.input_lengths, anchor.input_lengths
            ) or not torch.equal(inputs.prefix_lengths, anchor.prefix_lengths):
                raise RuntimeError("QSA draft graph cache groups disagree on lengths")
            if tag == INDEXER_STATE_TAG:
                # The graph runner refreshes the raw state table on device;
                # unlike kernel tables, it has no refreshed host mirror.
                table = inputs.kv_cache_block_id_device
                length_device = anchor.input_lengths_device
                prefix_device = anchor.prefix_lengths_device
                first = (prefix_device - prefix_device.remainder(4)) // page_size
                last = (prefix_device + length_device - 1) // page_size
                active = length_device > 0
                invalid = active & ((first < 0) | (last >= table.shape[1]))
                columns = (
                    torch.stack((first, last), dim=1)
                    .clamp(0, table.shape[1] - 1)
                    .long()
                )
                physical = table.gather(1, columns)
                invalid_state = (
                    invalid.any()
                    | (
                        active.unsqueeze(1)
                        & ((physical <= 0) | (physical >= pool_size))
                    ).any()
                )
                continue
            table = (
                inputs.kv_cache_block_id
                if tag == INDEXER_STATE_TAG
                else inputs.kv_cache_kernel_block_id
            )
            if (
                table.device.type != "cpu"
                or table.dtype != torch.int32
                or table.ndim != 2
                or table.shape[0] != len(lengths)
            ):
                raise RuntimeError(
                    "QSA draft graph replay requires host int32 block tables"
                )
            values = table.numpy()
            for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
                if not length:
                    continue
                if tag == INDEXER_STATE_TAG:
                    first = (prefix - prefix % 4) // page_size
                    end = (prefix + length - 1) // page_size + 1
                elif compressed:
                    first = 0
                    end = ((prefix + length) // 4 + page_size - 1) // page_size
                else:
                    first = 0
                    end = (prefix + length + page_size - 1) // page_size
                if end > table.shape[1]:
                    raise RuntimeError(
                        "QSA draft graph block table does not cover the visible tail"
                    )
                if any(
                    not 0 < int(block) < pool_size for block in values[row, first:end]
                ):
                    raise RuntimeError(
                        "QSA draft graph live row resolves to an unallocated or out-of-range physical block"
                    )
        # The raw-key ring has no historical position axes. Prove the text
        # position contract before generating canonical positions in the graph.
        transported = anchor.combo_position_ids
        if (
            transported is None
            or transported.numel() != self.token_capacity * self.position_factor
        ):
            raise RuntimeError("QSA draft graph transported position capacity changed")
        expected_positions = torch.tensor(
            positions, device=transported.device, dtype=transported.dtype
        )
        actual = transported.reshape(self.token_capacity, self.position_factor)[
            : len(positions)
        ]
        invalid_positions = (actual != expected_positions.unsqueeze(1)).any()
        if bool((invalid_state | invalid_positions).item()):
            raise RuntimeError(
                "QSA draft graph has invalid raw-state pages or positions disagree with logical cache positions"
            )
        for tag, impl in self.impls.items():
            impl.prepare_cuda_graph(groups[tag])
