"""Ownership of one collected batch during optional asynchronous input staging.

The scheduler owns admission, cancellation and batching. The model owns tensor
transfer/lifetimes. This envelope only maps prepared inputs back to surviving
work items; it never mutates the preprocessing cache or completes a request.
"""


class PrefetchedMMBatch:
    def __init__(self, chunks):
        self.chunks = chunks
        self.prepared = None
        self.data_by_item = {}
        self.failure = None

    def prepare(self, model):
        items = [item for chunk in self.chunks for item in chunk.work_items]
        prepare = getattr(model, "prepare_embedding_batch", None)
        if prepare is None:
            return
        self.prepared = prepare(
            [item.preprocess_result for item in items],
            [item.mm_type for item in items],
        )
        if self.prepared is None:
            return
        if len(self.prepared.data_list) != len(items):
            raise ValueError("prepared batch input count differs from work items")
        self.data_by_item = dict(zip(map(id, items), self.prepared.data_list))

    def inputs_for(self, items):
        if self.prepared is None:
            return None
        self.prepared.wait()
        return [self.data_by_item[id(item)] for item in items]

    def close(self):
        try:
            if self.prepared is not None:
                self.prepared.close()
        finally:
            self.prepared = None
            self.data_by_item.clear()
            self.chunks = []
