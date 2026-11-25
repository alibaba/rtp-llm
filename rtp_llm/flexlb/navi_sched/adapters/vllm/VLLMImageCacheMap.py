from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from .VLLMBlockHash import calculate_vllm_block_hashes
from .VLLMCachePolicy import VLLMCacheDelta


def media_expansion_spans(prediction, actual):
    """Return (native start, native length, scheduling length) per expansion."""
    predicted = sorted(prediction.mm_features or [], key=lambda f: f.mm_position.offset)
    native = sorted(actual.mm_features or [], key=lambda f: f.mm_position.offset)
    if Counter(f.modality for f in predicted) != Counter(f.modality for f in native):
        raise ValueError("native media items differ from scheduling prediction")
    if not predicted:
        return ()
    source_tokens, target_tokens = prediction.prompt_token_ids, actual.prompt_token_ids
    for features, tokens, scheduling in (
        (predicted, source_tokens, True),
        (native, target_tokens, False),
    ):
        for feature in features:
            position = feature.mm_position
            if (
                feature.modality not in ("image", "video", "audio")
                or position.offset < 0
                or position.length < 1
                or position.offset + position.length > len(tokens)
                or (scheduling and position.length != 1)
            ):
                raise ValueError("invalid scheduling-to-native media positions")
    if len({f.mm_position.offset for f in predicted}) != len(predicted):
        raise ValueError("overlapping scheduling media placeholders")
    spans, shift = [], 0
    for source, target in zip(predicted, native):
        if source.modality != target.modality:
            break
        source, target = source.mm_position, target.mm_position
        if target.offset != source.offset + shift:
            break
        spans.append((target.offset, target.length, 1))
        shift += target.length - 1
    else:
        if len(target_tokens) == len(source_tokens) + shift:
            return tuple(spans)

    # Some native processors insert frame timestamps, delimiters or interleave
    # audio/video. Treat that region as one expansion; never claim its partial
    # cache coverage from a single modality's embedding mask.
    first_source = predicted[0].mm_position.offset
    first_native = min(f.mm_position.offset for f in native)
    start = 0
    while (
        start < min(first_source, first_native)
        and source_tokens[start] == target_tokens[start]
    ):
        start += 1
    source_end = max(f.mm_position.offset + f.mm_position.length for f in predicted)
    native_end = max(f.mm_position.offset + f.mm_position.length for f in native)
    suffix = 0
    limit = min(len(source_tokens) - source_end, len(target_tokens) - native_end)
    while suffix < limit and source_tokens[-suffix - 1] == target_tokens[-suffix - 1]:
        suffix += 1
    return (
        (
            start,
            len(target_tokens) - suffix - start,
            len(source_tokens) - suffix - start,
        ),
    )


def project_media_prefix(count, spans):
    removed = 0
    for offset, native_length, scheduling_length in spans:
        if count <= offset:
            break
        if count < offset + native_length:
            return offset - removed
        removed += native_length - scheduling_length
    return count - removed


@dataclass
class _Mapping:
    real_hashes: tuple
    logical_hashes: tuple
    required_blocks: tuple
    definition: tuple
    prefix: int = 0
    states: dict = field(default_factory=dict)
    owners: set = field(default_factory=set)


class VLLMMediaCacheMap:
    """Project native cache readiness into the scheduling token namespace.

    Sparse media expansions produce block boundaries; records retain neither
    media tensors nor per-token maps. Calls use the Observer lifecycle lock.
    """

    def __init__(self, cache_state, hybrid_tail_required, max_records=4096,
                 max_block_references=262144):
        self._cache = cache_state
        self._hybrid = hybrid_tail_required
        self._full_cache = cache_state._full_ready if self._hybrid else cache_state._flat_ready
        self._max_records = max_records
        self._max_references = max_block_references
        self._references = 0
        self._records = OrderedDict()
        self._definitions = {}
        self._next_id = 0
        self._by_hash = {}
        self._owners = {}
        self._counts = (Counter(), Counter(), Counter())
        self._changes = {}

    def _delta(self):
        added = ([], [], [])
        removed = ([], [], [])
        for block_hash, before in self._changes.items():
            for index, counts in enumerate(self._counts):
                after = block_hash in counts
                if after != before[index]:
                    (added if after else removed)[index].append(block_hash)
        return VLLMCacheDelta(
            added=tuple(added[0]), removed=tuple(removed[0]),
            added_full_ready=tuple(added[1]), removed_full_ready=tuple(removed[1]),
            added_tail_ready=tuple(added[2]), removed_tail_ready=tuple(removed[2]),
        )

    def _set_state(self, record, index, state):
        old = record.states.get(index, (False, False, False))
        block_hash = record.logical_hashes[index]
        self._changes.setdefault(block_hash, tuple(block_hash in c for c in self._counts))
        for counts, previous, current in zip(self._counts, old, state):
            if previous != current:
                counts[block_hash] += 1 if current else -1
                if counts[block_hash] == 0:
                    del counts[block_hash]
        if any(state):
            record.states[index] = state
        else:
            record.states.pop(index, None)

    def _refresh(self, record, changed_positions):
        previous = bisect_right(record.required_blocks, record.prefix)
        lost = [position for position in changed_positions
                if position < record.prefix and record.real_hashes[position] not in self._full_cache]
        if lost:
            record.prefix = min(lost)
        while (record.prefix < len(record.real_hashes)
               and record.real_hashes[record.prefix] in self._full_cache):
            record.prefix += 1
        current = bisect_right(record.required_blocks, record.prefix)
        affected = set(range(min(previous, current), max(previous, current)))
        for position in changed_positions:
            index = bisect_left(record.required_blocks, position + 1)
            end = bisect_right(record.required_blocks, position + 1)
            affected.update(range(index, end))
        for index in affected:
            full = index < current
            tail = full and record.real_hashes[
                record.required_blocks[index] - 1
            ] in self._cache._tail_ready
            self._set_state(record, index, (full and not self._hybrid,
                                           full and self._hybrid, tail))

    def _drop(self, key):
        record = self._records.pop(key)
        self._definitions.pop(record.definition)
        for index in tuple(record.states):
            self._set_state(record, index, (False, False, False))
        for block_hash in set(record.real_hashes):
            refs = self._by_hash[block_hash]
            del refs[key]
            if not refs:
                del self._by_hash[block_hash]
        for owner in record.owners:
            self._owners.pop(owner, None)
        self._references -= len(record.real_hashes) + len(record.logical_hashes)

    def register(
        self, owner, prediction, actual, block_size, logical_hashes=None, spans=None
    ):
        self._changes = {}
        spans = media_expansion_spans(prediction, actual) if spans is None else spans
        if not spans:
            return VLLMCacheDelta()
        positions = []
        shift = 0
        for offset, native_length, scheduling_length in spans:
            source_start = offset - shift
            shift += native_length - scheduling_length
            positions.append(
                (
                    source_start,
                    source_start + scheduling_length,
                    offset + native_length,
                    shift,
                )
            )
        first_block = positions[0][0] // block_size
        logical = (logical_hashes if logical_hashes is not None else calculate_vllm_block_hashes(
            prediction.prompt_token_ids, block_size, request=prediction,
        ))[first_block:]
        if not logical:
            return VLLMCacheDelta()
        ends = []
        span_index, shift = 0, 0
        for block in range(first_block, first_block + len(logical)):
            end = (block + 1) * block_size
            while span_index < len(positions) and positions[span_index][1] <= end:
                shift = positions[span_index][3]
                span_index += 1
            native_end = end + shift
            if span_index < len(positions) and positions[span_index][0] < end:
                native_end = positions[span_index][2]
            ends.append((native_end + block_size - 1) // block_size)
        real = tuple(bytes(value) for value in actual.block_hashes[:ends[-1]])
        count = bisect_right(ends, len(real))
        logical, ends = logical[:count], ends[:count]
        if not logical:
            return VLLMCacheDelta()
        real = real[:ends[-1]]
        logical = tuple(value.to_bytes(32, "big") for value in logical)
        definition = (logical, real, tuple(ends))
        if owner in self._owners:
            raise ValueError("duplicate media cache mapping owner")
        key = self._definitions.get(definition)
        record = self._records.get(key)
        if record is None:
            references = len(real) + len(logical)
            if references > self._max_references or self._max_records <= 0:
                return VLLMCacheDelta()
            while self._records and (
                len(self._records) >= self._max_records
                or self._references + references > self._max_references
            ):
                self._drop(next(iter(self._records)))
            self._next_id += 1
            key = self._next_id
            record = _Mapping(real, logical, tuple(ends), definition)
            self._records[key] = record
            self._definitions[definition] = key
            self._references += references
            for index, block_hash in enumerate(real):
                self._by_hash.setdefault(block_hash, {}).setdefault(key, index)
            self._refresh(record, ())
        self._records.move_to_end(key)
        record.owners.add(owner)
        self._owners[owner] = key
        return self._delta()

    def update(self, changed_hashes, cleared=False):
        self._changes = {}
        if cleared:
            for key in tuple(self._records):
                self._drop(key)
        else:
            affected = {}
            for block_hash in changed_hashes:
                for key, position in self._by_hash.get(block_hash, {}).items():
                    affected.setdefault(key, set()).add(position)
            for key, positions in affected.items():
                record = self._records[key]
                self._refresh(record, positions)
                if not record.owners and not record.states:
                    self._drop(key)
        return self._delta()

    def finish(self, owner):
        key = self._owners.pop(owner, None)
        if key is None:
            return
        record = self._records[key]
        record.owners.discard(owner)
        if not record.owners and not record.states:
            self._drop(key)


# Preserve imports used by existing image-only integrations.
VLLMImageCacheMap = VLLMMediaCacheMap
