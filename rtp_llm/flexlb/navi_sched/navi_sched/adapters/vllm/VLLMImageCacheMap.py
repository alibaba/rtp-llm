from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from .VLLMBlockHash import calculate_vllm_block_hashes
from .VLLMCachePolicy import VLLMCacheDelta


@dataclass
class _Mapping:
    real_hashes: tuple
    logical_hashes: tuple
    required_blocks: tuple
    definition: tuple
    prefix: int = 0
    states: dict = field(default_factory=dict)
    owners: set = field(default_factory=set)


class VLLMImageCacheMap:
    """Project native cache readiness into the scheduling token namespace.

    Sparse image positions produce block boundaries; records retain neither
    image tensors nor per-token maps. Calls use the Observer lifecycle lock.
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
            if (index < len(record.required_blocks)
                    and record.required_blocks[index] == position + 1):
                affected.add(index)
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

    def register(self, owner, prediction, actual, block_size, logical_hashes=None):
        self._changes = {}
        predicted_images = prediction.mm_features or []
        actual_images = actual.mm_features or []
        if len(predicted_images) != len(actual_images):
            raise ValueError("native image count differs from scheduling prediction")
        positions = []
        shift = 0
        for predicted, native in zip(predicted_images, actual_images):
            source, target = predicted.mm_position, native.mm_position
            if (predicted.modality != "image" or source.length != 1
                    or native.modality != "image" or target.offset != source.offset + shift
                    or target.length < 1
                    or (positions and source.offset <= positions[-1][0])):
                raise ValueError("invalid scheduling-to-native image positions")
            shift += target.length - 1
            positions.append((source.offset, shift))
        if not positions:
            return VLLMCacheDelta()
        if len(actual.prompt_token_ids) != len(prediction.prompt_token_ids) + shift:
            raise ValueError("native image expansion does not match token lengths")
        first_block = positions[0][0] // block_size
        logical = (logical_hashes if logical_hashes is not None else calculate_vllm_block_hashes(
            prediction.prompt_token_ids, block_size, request=prediction,
        ))[first_block:]
        if not logical:
            return VLLMCacheDelta()
        ends = []
        image_index, shift = 0, 0
        for block in range(first_block, first_block + len(logical)):
            end = (block + 1) * block_size
            while image_index < len(positions) and positions[image_index][0] < end:
                shift = positions[image_index][1]
                image_index += 1
            ends.append((end + shift + block_size - 1) // block_size)
        real = tuple(bytes(value) for value in actual.block_hashes[:ends[-1]])
        # A native final partial block cannot confirm a complete logical block.
        count = bisect_right(ends, len(real))
        logical, ends = logical[:count], ends[:count]
        if not logical:
            return VLLMCacheDelta()
        real = real[:ends[-1]]
        logical = tuple(value.to_bytes(32, "big") for value in logical)
        definition = (logical, real, tuple(ends))
        if owner in self._owners:
            raise ValueError("duplicate image cache mapping owner")
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
