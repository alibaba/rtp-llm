"""Validate host UQI bounds; preserve original token order for masked attention."""

from array import array


def build_uqi_graph_layout(lengths, starts, b_lengths, capacity):
    """Return (length, B start, B length) for fixed slots plus padding sequence."""
    if not (len(lengths) == len(starts) == len(b_lengths)):
        raise ValueError("UQI graph metadata sizes differ")
    total_tokens = sum(lengths)
    if any(n < 0 for n in lengths) or total_tokens > capacity:
        raise ValueError("UQI graph token capacity exceeded")
    result = []
    for n, start, count in zip(lengths, starts, b_lengths):
        if count < 0 or (count and (start < 0 or start + count > n)):
            raise ValueError("invalid UQI segment bounds")
        result.append((n, start if count else n, count))
    padding = capacity - total_tokens
    result.append((padding, padding, 0))
    return result


def build_uqi_aiter_layout(lengths, starts, counts, capacity, max_seq_len):
    """Fixed-slot AITER indptrs, stable permutations and output masks."""
    layout = build_uqi_graph_layout(lengths, starts, counts, capacity)
    if max_seq_len <= 0 or any(n > max_seq_len for n in lengths):
        raise ValueError("UQI sequence exceeds AITER profile")
    perm, first_lengths, full_lengths, b_mask = [], [], [], []
    offset = 0
    for n, start, count in layout[:-1]:
        perm.extend(range(offset, offset + start))
        perm.extend(range(offset + start + count, offset + n))
        perm.extend(range(offset + start, offset + start + count))
        first_lengths.extend((n - count, count))
        full_lengths.append(n)
        b_mask.extend([False] * (n - count) + [True] * count)
        offset += n
    perm.extend(range(offset, capacity))
    b_mask.extend([False] * (capacity - offset))
    # Bound synthetic padding sequences individually: no bucket-sized quadratic
    # attention. Slot count stays constant when real token counts change.
    padding_slots = (capacity + max_seq_len - 1) // max_seq_len
    padding = [min(max_seq_len, max(0, capacity - offset - i * max_seq_len))
               for i in range(padding_slots)]
    def indptr(lens):
        ptr = [0]
        for n in lens:
            ptr.append(ptr[-1] + n)
        return ptr
    inv = [0] * capacity
    for new, old in enumerate(perm):
        inv[old] = new
    return (perm, inv, indptr(first_lengths + padding),
            indptr(full_lengths + padding), b_mask,
            [True] * offset + [False] * (capacity - offset))


class BertUqiAiterGraphLayout:
    """Reusable packed metadata for one graph bucket, without per-token loops.

    All six arrays share one byte buffer, allowing a single H2D copy. Native
    int64/int32 views match the PyTorch views of the device buffer. Only the
    request slots are visited in Python; token ranges are copied in bulk.
    """

    def __init__(self, capacity: int, request_capacity: int, max_seq_len: int):
        if capacity <= 0 or capacity > 2**31 - 1 or request_capacity <= 0 or max_seq_len <= 0:
            raise ValueError("invalid UQI graph capacity")
        self.capacity = capacity
        self.request_capacity = request_capacity
        self.max_seq_len = max_seq_len
        self.padding_slots = (capacity + max_seq_len - 1) // max_seq_len
        self.first_count = 2 * request_capacity + self.padding_slots + 1
        self.full_count = request_capacity + self.padding_slots + 1
        self.inv_offset = capacity * 8
        self.first_offset = self.inv_offset + capacity * 8
        self.full_offset = self.first_offset + self.first_count * 4
        self.b_mask_offset = self.full_offset + self.full_count * 4
        self.valid_mask_offset = self.b_mask_offset + capacity
        self.buffer = bytearray(self.valid_mask_offset + capacity)
        view = memoryview(self.buffer)
        self.perm = view[: self.inv_offset].cast("q")
        self.inv = view[self.inv_offset : self.first_offset].cast("q")
        self.first = view[self.first_offset : self.full_offset].cast("i")
        self.full = view[self.full_offset : self.b_mask_offset].cast("i")
        self.b_mask = view[self.b_mask_offset : self.valid_mask_offset]
        self.valid = view[self.valid_mask_offset :]
        self._identity = memoryview(array("q", range(capacity)))
        self._zeros = bytes(capacity)
        self._ones = b"\x01" * capacity

    def update(self, lengths, starts, counts) -> None:
        if len(lengths) != self.request_capacity:
            raise ValueError("UQI graph request capacity changed")
        layout = build_uqi_graph_layout(lengths, starts, counts, self.capacity)
        if any(n > self.max_seq_len for n in lengths):
            raise ValueError("UQI sequence exceeds AITER profile")

        self.perm[:] = self._identity
        self.inv[:] = self._identity
        self.b_mask[:] = self._zeros
        self.first[0] = self.full[0] = 0
        offset = 0
        for i, (n, start, count) in enumerate(layout[:-1]):
            a_end = offset + n - count
            end = offset + n
            if count:
                b_start, b_end = offset + start, offset + start + count
                self.perm[b_start:a_end] = self._identity[b_end:end]
                self.perm[a_end:end] = self._identity[b_start:b_end]
                self.inv[b_start:b_end] = self._identity[a_end:end]
                self.inv[b_end:end] = self._identity[b_start:a_end]
                self.b_mask[a_end:end] = self._ones[:count]
            self.first[2 * i + 1] = a_end
            self.first[2 * i + 2] = end
            self.full[i + 1] = end
            offset = end
        self.valid[:offset] = self._ones[:offset]
        self.valid[offset:] = self._zeros[: self.capacity - offset]
        for i in range(self.padding_slots):
            end = min(self.capacity, offset + (i + 1) * self.max_seq_len)
            self.first[2 * self.request_capacity + i + 1] = end
            self.full[self.request_capacity + i + 1] = end
