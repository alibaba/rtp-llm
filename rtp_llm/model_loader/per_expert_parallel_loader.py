import logging
from typing import Dict, Generator, Tuple

import fastsafetensors
import torch
from fastsafetensors import ParallelLoader
from fastsafetensors.parallel_loader import TimingContext

_REQUIRED_FST_VERSION = "0.1.19"


class PerExpertParallelLoader(ParallelLoader):
    """ParallelLoader subclass that splits stacked MoE tensors before NCCL broadcast.

    Instead of broadcasting the full stacked tensor [num_experts, ...] to every
    rank, this loader splits on the source rank and broadcasts individual expert
    tensors.  Peak GPU memory during broadcast drops from the full stacked tensor
    size to a single expert slice (stacked_size / num_experts).

    Args:
        stacked_key_config: Mapping from stacked checkpoint key to a per-expert
            name template containing ``{expert_id}``.  Keys not in this dict
            are handled by the normal broadcast path.
    """

    def __init__(self, stacked_key_config: Dict[str, str], *args, **kwargs):
        fst_ver = getattr(fastsafetensors, "__version__", "unknown")
        if not fst_ver.startswith(_REQUIRED_FST_VERSION):
            raise RuntimeError(
                f"PerExpertParallelLoader is tested with fastsafetensors "
                f"{_REQUIRED_FST_VERSION}*, current version: {fst_ver}. "
                f"Internal API changes may cause breakage."
            )
        super().__init__(*args, **kwargs)
        self.stacked_key_config = stacked_key_config or {}

    def _consume_single_batch(self):
        # Mirrors ParallelLoader._consume_single_batch; only the key iteration
        # loop below (marked CUSTOM) differs — stacked keys are split per-expert.
        with TimingContext("wait_queue", self._log_message) as timer:
            batch_item = self.batch_queue.get()

            if self.queue_size == 0 and self.consumer_processed is not None:
                self.consumer_processed.set()
            if batch_item is None:
                self._log_error("get batch_item is None, will break")
                return
            if isinstance(batch_item, Exception):
                self._log_error("get batch_item is Exception, will raise")
                raise batch_item

            batch = batch_item
            timer.batch_id = batch.batch_id
        queue_wait_time = timer.elapsed_ms

        if queue_wait_time / 1000 > 10:
            self._log_message(
                f"Warning: Batch {batch.batch_id}: Queue wait took "
                f"{queue_wait_time:.3f} ms",
                is_error=True,
            )

        try:
            self._log_message(
                f"Batch {batch.batch_id}: tensor key len: {len(batch.keys)}"
            )
            with TimingContext(
                "get_tensor", self._log_message, batch.batch_id
            ) as timer:
                # --- BEGIN CUSTOM LOGIC (differs from ParallelLoader) ---
                for key in batch.keys:
                    if key in self.stacked_key_config:
                        yield from self._broadcast_per_expert(batch, key)
                    else:
                        tensor = batch.fb.get_tensor(key)
                        yield key, tensor
                # --- END CUSTOM LOGIC ---
            get_tensor_time = timer.elapsed_ms

        finally:
            with TimingContext("fb.close", self._log_message, batch.batch_id) as timer:
                batch.fb.close()
            close_time = timer.elapsed_ms

        self._log_message(
            f"Batch {batch.batch_id} summary: "
            f"add_filenames={batch.add_filenames_time:.3f}ms, "
            f"copy_files={batch.copy_files_time:.3f}ms, "
            f"get_tensor={get_tensor_time:.3f}ms, "
            f"close={close_time:.3f}ms"
        )

        if self.queue_size < 0 and self.consumer_processed is not None:
            self.consumer_processed.set()

    def _broadcast_per_expert(
        self, batch, key: str
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Split a stacked tensor on source rank and broadcast per-expert slices.

        Uses fastsafetensors internal APIs (version-sensitive):
          fb._get_rank_lidx, fb.rank_loaders, fb.instantiated, fb.auto_mem_delete,
          factory.metadata.tensors, factory.tensors, factory.framework,
          factory.device, factory.free_dev_ptrs
        """
        template = self.stacked_key_config[key]
        fb = batch.fb
        (rank, lidx) = fb._get_rank_lidx(key)
        factory = fb.rank_loaders[rank][lidx]
        frame = factory.metadata.tensors[key]
        pg = fb.pg

        num_experts = frame.shape[0]
        expert_shape = list(frame.shape)[1:]

        logging.debug(f"per-expert broadcast: {key} [{num_experts} x {expert_shape}]")

        if pg.size() == 1:
            src_tensor = factory.tensors[key]
            for eid in range(num_experts):
                yield (
                    template.format(expert_id=eid),
                    src_tensor[eid].clone().detach().get_raw(),
                )
        else:
            for eid in range(num_experts):
                if pg.rank() == rank:
                    expert_t = factory.tensors[key][eid].clone().detach()
                else:
                    expert_t = factory.framework.get_empty_tensor(
                        expert_shape, frame.dtype, factory.device
                    )
                pg.broadcast(expert_t, rank)
                yield template.format(expert_id=eid), expert_t.get_raw()

        if fb.auto_mem_delete:
            fb.instantiated[rank][lidx][key] = True
            if len(fb.instantiated[rank][lidx]) == len(factory.metadata.tensors):
                factory.free_dev_ptrs()
