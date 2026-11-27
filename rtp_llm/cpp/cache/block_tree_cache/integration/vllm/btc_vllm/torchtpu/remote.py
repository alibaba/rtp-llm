# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map locally shared Host slots for the BTC PACE storage backend."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack

from btc_kvcm import PaceSpec, PaceStorageBackend


class BtcLocalPaceStorage:
    """A single Host mapping shared with the local TPU workers."""

    def __init__(
        self, geometry, config, last_checkpoint, *, memory: memoryview, memory_fd: int,
        resolve_slot: Callable[[int, int], memoryview],
    ):
        self.capacity = geometry.num_slots
        self._resolve_slot = resolve_slot
        specs = config.get("specs")
        if specs is None:
            rank_bytes = (geometry.rank_bytes,) * geometry.num_ranks
            specs = {
                group: (
                    PaceSpec(f"{'L' if checkpoint else 'F'}{group}", 0, rank_bytes),
                )
                for group, checkpoint in enumerate(last_checkpoint)
            }
        if set(specs) != set(range(len(last_checkpoint))):
            raise ValueError("BTC PACE requires specs for every cache group")
        self.storage = PaceStorageBackend(
            dict(config, specs=specs), geometry.rank_bytes, geometry.num_ranks,
            memory=memory, memory_fd=memory_fd,
        )

    def describe(self, keys):
        return self.storage.describe(keys)

    def read_described(self, items):
        with ExitStack() as views:
            return self.storage.read_described([
                (group, key, views.enter_context(self._resolve_slot(group, slot)), proof)
                for group, key, slot, proof in items
            ])

    def write(self, group, key, slot):
        with self._resolve_slot(group, slot) as source:
            return self.storage.write(group, key, source)

    def close(self):
        self.storage.close()
