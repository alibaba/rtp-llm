"""Private prepare outputs owned by one captured PLE target transaction."""

import torch

from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit_prepare import CommitValueBuffers


class PLECommitWorkspace:
    def __init__(self, transaction):
        self.capacity = transaction.batch_size
        self.busy = False
        payload = torch.empty(
            10 + 3 * self.capacity,
            device=transaction.prefixes.device,
            dtype=torch.int64,
        )
        self._payloads = {self.capacity: payload}
        self._values = {}
        for stage in transaction.stages.values():
            slots = (transaction.query_len + stage.page_size - 1) // stage.page_size
            # Even a boundary parked on the final destination needs its own
            # immutable undo and selected values until commit/rollback finishes.
            for slot in ("final", *range(slots)):
                self._values[(stage.layer_idx, slot)] = CommitValueBuffers(
                    stage.initial_state, stage.initial_context
                )

    def metadata_payload(self, batch):
        if not 0 < batch <= self.capacity:
            return None
        payload = self._payloads.get(batch)
        if payload is None:
            # The fused metadata kernel packs all three vectors using live B,
            # not captured capacity. Keep an exact flat prefix for its views.
            payload = self._payloads[self.capacity][: 10 + 3 * batch]
            self._payloads[batch] = payload
        return payload

    def values(self, stage, slot):
        return self._values.get((stage.layer_idx, slot))
