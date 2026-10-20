"""Validate UQI bounds and build CUDA mask layouts or AITER token permutations."""

from array import array


def build_uqi_graph_layout(lengths, starts, b_lengths, token_capacity):
    """Return (length, B start, B length) for fixed slots plus padding sequence."""
    if not (len(lengths) == len(starts) == len(b_lengths)):
        raise ValueError("UQI graph metadata sizes differ")
    total_tokens = sum(lengths)
    if any(n < 0 for n in lengths) or total_tokens > token_capacity:
        raise ValueError("UQI graph token token_capacity exceeded")
    result = []
    for n, start, count in zip(lengths, starts, b_lengths):
        if count < 0 or (count and (start < 0 or start + count > n)):
            raise ValueError("invalid UQI segment bounds")
        result.append((n, start if count else n, count))
    padding = token_capacity - total_tokens
    result.append((padding, padding, 0))
    return result


def build_uqi_aiter_layout(lengths, starts, b_lengths, token_capacity, max_seq_len):
    """Fixed-slot AITER indptrs, stable permutations and output masks."""
    layout = build_uqi_graph_layout(lengths, starts, b_lengths, token_capacity)
    if max_seq_len <= 0 or any(n > max_seq_len for n in lengths):
        raise ValueError("UQI sequence exceeds AITER profile")
    perm, split_lengths, full_lengths, b_mask = [], [], [], []
    offset = 0
    for n, start, count in layout[:-1]:
        perm.extend(range(offset, offset + start))
        perm.extend(range(offset + start + count, offset + n))
        perm.extend(range(offset + start, offset + start + count))
        split_lengths.extend((n - count, count))
        full_lengths.append(n)
        b_mask.extend([False] * (n - count) + [True] * count)
        offset += n
    perm.extend(range(offset, token_capacity))
    b_mask.extend([False] * (token_capacity - offset))
    # Bound synthetic padding sequences individually: no bucket-sized quadratic
    # attention. Slot count stays constant when real token b_lengths change.
    padding_slots = (token_capacity + max_seq_len - 1) // max_seq_len
    padding = [
        min(max_seq_len, max(0, token_capacity - offset - i * max_seq_len))
        for i in range(padding_slots)
    ]

    def indptr(lens):
        ptr = [0]
        for n in lens:
            ptr.append(ptr[-1] + n)
        return ptr

    inv_perm = [0] * token_capacity
    for new, old in enumerate(perm):
        inv_perm[old] = new
    return (
        perm,
        inv_perm,
        indptr(split_lengths + padding),
        indptr(full_lengths + padding),
        b_mask,
        [True] * offset + [False] * (token_capacity - offset),
    )


class BertUqiAiterGraphLayout:
    """Reusable packed metadata for one graph bucket, without per-token loops.

    All six arrays share one byte buffer, allowing a single H2D copy. Native
    int64/int32 views match the PyTorch views of the device buffer. Only the
    request slots are visited in Python; token ranges are copied in bulk.
    """

    def __init__(self, token_capacity: int, request_capacity: int, max_seq_len: int):
        if (
            token_capacity <= 0
            or token_capacity > 2**31 - 1
            or request_capacity <= 0
            or max_seq_len <= 0
        ):
            raise ValueError("invalid UQI graph token_capacity")
        self.token_capacity = token_capacity
        self.request_capacity = request_capacity
        self.max_seq_len = max_seq_len
        self.padding_slots = (token_capacity + max_seq_len - 1) // max_seq_len
        self.split_indptr_size = 2 * request_capacity + self.padding_slots + 1
        self.full_indptr_size = request_capacity + self.padding_slots + 1
        self.inv_perm_offset = token_capacity * 8
        self.split_indptr_offset = self.inv_perm_offset + token_capacity * 8
        self.full_indptr_offset = self.split_indptr_offset + self.split_indptr_size * 4
        self.b_mask_offset = self.full_indptr_offset + self.full_indptr_size * 4
        self.valid_mask_offset = self.b_mask_offset + token_capacity
        self.buffer = bytearray(self.valid_mask_offset + token_capacity)
        view = memoryview(self.buffer)
        self.perm = view[: self.inv_perm_offset].cast("q")
        self.inv_perm = view[self.inv_perm_offset : self.split_indptr_offset].cast("q")
        self.split_indptr = view[
            self.split_indptr_offset : self.full_indptr_offset
        ].cast("i")
        self.full_indptr = view[self.full_indptr_offset : self.b_mask_offset].cast("i")
        self.b_mask = view[self.b_mask_offset : self.valid_mask_offset]
        self.valid_mask = view[self.valid_mask_offset :]
        self._identity = memoryview(array("q", range(token_capacity)))
        self._zeros = bytes(token_capacity)
        self._ones = b"\x01" * token_capacity

    def update(self, lengths, starts, b_lengths) -> None:
        if len(lengths) != self.request_capacity:
            raise ValueError("UQI graph request token_capacity changed")
        layout = build_uqi_graph_layout(lengths, starts, b_lengths, self.token_capacity)
        if any(n > self.max_seq_len for n in lengths):
            raise ValueError("UQI sequence exceeds AITER profile")

        self.perm[:] = self._identity
        self.inv_perm[:] = self._identity
        self.b_mask[:] = self._zeros
        self.split_indptr[0] = self.full_indptr[0] = 0
        offset = 0
        for i, (n, start, count) in enumerate(layout[:-1]):
            a_end = offset + n - count
            end = offset + n
            if count:
                b_start, b_end = offset + start, offset + start + count
                self.perm[b_start:a_end] = self._identity[b_end:end]
                self.perm[a_end:end] = self._identity[b_start:b_end]
                self.inv_perm[b_start:b_end] = self._identity[a_end:end]
                self.inv_perm[b_end:end] = self._identity[b_start:a_end]
                self.b_mask[a_end:end] = self._ones[:count]
            self.split_indptr[2 * i + 1] = a_end
            self.split_indptr[2 * i + 2] = end
            self.full_indptr[i + 1] = end
            offset = end
        self.valid_mask[:offset] = self._ones[:offset]
        self.valid_mask[offset:] = self._zeros[: self.token_capacity - offset]
        for i in range(self.padding_slots):
            end = min(self.token_capacity, offset + (i + 1) * self.max_seq_len)
            self.split_indptr[2 * self.request_capacity + i + 1] = end
            self.full_indptr[self.request_capacity + i + 1] = end
