"""Layer-local DSA CP/DCP metadata over the shared PageRR cache geometry."""

import torch

from .mla_page_rr_cache_gather import MlaPageRRCacheGather
from .page_rr_mla_metadata import PageRRMlaMetadata


class DsaPageRRParams:
    def __init__(
        self,
        attention,
        parallelism,
        *,
        max_seq_len=None,
        workspace_owner=None,
        is_cuda_graph=False
    ):
        self.workspace_owner = workspace_owner
        self.is_cuda_graph = is_cuda_graph
        self.indexer_execution = None
        self.sfa_execution = None
        self.page_size = int(attention.tokens_per_block)
        self.kernel_page_size = int(attention.kernel_tokens_per_block)
        self.shard_size = int(parallelism.tp_size)
        self.shard_rank = int(parallelism.tp_rank)
        self.max_owner_columns = None
        if max_seq_len is not None:
            if type(max_seq_len) is not int or max_seq_len <= 0:
                raise ValueError(
                    "DSA maximum sequence length must be a positive integer"
                )
            pages = (max_seq_len + self.page_size - 1) // self.page_size
            owner_pages = (pages + self.shard_size - 1) // self.shard_size
            self.max_owner_columns = owner_pages * (
                self.page_size // self.kernel_page_size
            )
        if self.shard_size > 1 and not parallelism.prefill_cp_config.kv_cache_sharded:
            raise ValueError(
                "DSA CP/DCP requires the existing PageRR cache-sharding switch"
            )
        self.query_cp = bool(workspace_owner and workspace_owner.prefill_query_cp)
        self.query_layout = None
        self.producer_layout = None
        self.ingress_plans = {}
        if parallelism.prefill_cp_config.is_enabled() and not self.query_cp:
            raise ValueError(
                "K3.1 selects query CP inside DSA; outer Prefill CP must remain disabled"
            )
        self.cache_adapter = MlaPageRRCacheGather(
            self.page_size, self.shard_size, self.shard_rank, self.kernel_page_size
        )
        self.decode = None

    def prepare(self, inputs, forbid_realloc=False):
        self.forbid_realloc = forbid_realloc
        self.is_warmup = bool(getattr(inputs, "is_warmup", False))
        self.is_prefill = bool(
            inputs.is_prefill
            and not inputs.is_target_verify
            and not inputs.is_mtp_draft_update
        )
        self.block_table = inputs.kv_cache_kernel_block_id_device
        if self.is_prefill:
            if forbid_realloc:
                raise ValueError("DSA Prefill uses eager metadata")
            if self.query_cp:
                self._prepare_query_cp(inputs)
                return
            full = PageRRMlaMetadata(self.page_size, self.kernel_page_size, self.shard_size, self.shard_rank)
            full.prepare_prefill(inputs, sparse_query_positions=True)
            self.positions_d = full.positions_d
            self.batch_indice_d = full.batch_indice_d
            logical = int(inputs.logical_token_count)
            physical = int(inputs.physical_token_count)
            # SP reserves physical tail rows without allocating their cache.
            # Keep their Q rows for A2A, but never gather them as real history.
            remaining = logical if physical > logical else sum(full.q_lens_host)
            self.kv_lens_host = []
            for q_len, prefix in zip(full.q_lens_host, full.prefix_lens_host):
                valid_q = min(q_len, max(0, remaining))
                self.kv_lens_host.append(prefix + valid_q if valid_q else 0)
                remaining -= q_len
            self.cache_adapter.validate_block_table_capacity(
                self.block_table, self.kv_lens_host
            )
            self.slot_mapping = self.cache_adapter.slot_mapping(
                self.positions_d,
                self.batch_indice_d,
                self.block_table,
                valid_token_count=logical if physical > logical else None,
                allow_reserved_block_zero=self.is_warmup,
            )
            tokens = self.positions_d.numel()
            if tokens % self.shard_size:
                raise ValueError("DSA query CP requires TP-aligned physical token rows")
            start = self.shard_rank * (tokens // self.shard_size)
            end = start + tokens // self.shard_size
            self.query_indices = torch.arange(
                start, end, dtype=torch.int64, device=self.positions_d.device
            )
            self.query_positions = self.positions_d[start:end]
            if physical > logical:
                self.query_positions = torch.where(
                    self.query_indices < logical, self.query_positions, -1
                )
            self.query_requests = self.batch_indice_d[start:end]
            self.prefill_requests = []
            query_start = 0
            kv_start = 0
            kv_offsets = []
            for q_len, kv_len in zip(full.q_lens_host, self.kv_lens_host):
                left = min(end, max(start, query_start))
                right = min(end, max(start, query_start + q_len))
                rows = torch.arange(
                    left - start, right - start, device=self.positions_d.device
                )
                self.prefill_requests.append((rows, kv_start, kv_len))
                kv_offsets.append(kv_start)
                query_start += q_len
                kv_start += kv_len
            offsets = torch.tensor(
                kv_offsets, dtype=torch.int32, device=self.positions_d.device
            )
            self.query_kv_offsets = offsets.index_select(
                0, self.query_requests.to(torch.int64)
            )
        else:
            if self.decode is None:
                self.decode = PageRRMlaMetadata(
                    self.page_size,
                    self.kernel_page_size,
                    self.shard_size,
                    self.shard_rank,
                )
            self.decode.prepare_decode(inputs, forbid_realloc=forbid_realloc)
            self.positions_d = self.decode.positions_d
            self.slot_mapping = self.decode.slot_mapping
            valid_queries = getattr(inputs, "valid_token_mask", None)
            if valid_queries is not None:
                self.positions_d.copy_(torch.where(valid_queries, self.positions_d, -1))
                self.slot_mapping.copy_(
                    torch.where(valid_queries, self.slot_mapping, -1)
                )
            batch, queries = self.decode.local_causal_lens.shape
            if (
                not hasattr(self, "batch_indice_d")
                or self.batch_indice_d.numel() != batch * queries
            ):
                if forbid_realloc:
                    raise ValueError(
                        "DSA Decode batch metadata cannot grow during graph replay"
                    )
                self.batch_indice_d = torch.arange(
                    batch, dtype=torch.int32, device=self.positions_d.device
                ).repeat_interleave(queries)
            self.block_table = self.decode.query_block_tables.view(batch, queries, -1)[
                :, 0, :
            ]
            if self.max_owner_columns is not None:
                # Graph inputs reserve a global-length padded page table, but
                # its live prefix contains only this owner's compact pages.
                # Do not read or score the global padding as owner history.
                self.block_table = self.block_table[:, : self.max_owner_columns]
                torch._assert_async(
                    (
                        self.decode.local_causal_lens
                        <= self.block_table.shape[1] * self.kernel_page_size
                    ).all(),
                    "DSA owner page table is too short for the live causal lengths",
                )

    def _prepare_query_cp(self, inputs):
        layout = self.workspace_owner.query_layout
        if layout is None:
            raise ValueError("DSA query CP requires original zigzag positions")
        self.query_layout = layout
        producer = getattr(self.workspace_owner, "producer_layout", None)
        self.producer_layout = producer = layout if producer is None else producer
        self.positions_d = self.query_positions = layout.local_positions
        self.batch_indice_d = self.query_requests = layout.local_requests
        self.query_indices = torch.arange(layout.local_rows, dtype=torch.int64,
                                          device=self.positions_d.device)
        self.kv_lens_host = [n + p for n, p in zip(layout.lengths, layout.prefixes)]
        self.cache_adapter.validate_block_table_capacity(self.block_table, self.kv_lens_host)
        self.slot_mapping = self.cache_adapter.slot_mapping(
            self.positions_d, self.batch_indice_d, self.block_table,
            valid_token_mask=layout.local_valid,
            allow_reserved_block_zero=self.is_warmup,
        )
        self.producer_positions = producer.local_positions
        self.producer_requests = producer.local_requests
        self.producer_valid = producer.local_valid
        self.producer_slot_mapping = self.slot_mapping if producer is layout else self.cache_adapter.slot_mapping(
            self.producer_positions, self.producer_requests, self.block_table,
            valid_token_mask=self.producer_valid,
            allow_reserved_block_zero=self.is_warmup,
        )
        offsets = [0]
        for length in self.kv_lens_host:
            offsets.append(offsets[-1] + length)
        offset_tensor = torch.tensor(offsets[:-1], device=self.positions_d.device, dtype=torch.int32)
        self.query_kv_offsets = offset_tensor.index_select(0, self.query_requests.long())
        self.prefill_requests = [
            (torch.nonzero((self.query_requests == r) & (self.query_positions >= 0)).flatten(),
             offsets[r], length)
            for r, length in enumerate(self.kv_lens_host)
        ]
        from .dsa_kv_ingress import DsaKVIngress

        self.ingress_plans = {
            "indexer": DsaKVIngress(self, "indexer", self.workspace_owner.index_dim),
            "sfa": DsaKVIngress(self, "sfa", self.workspace_owner.latent_dim + self.workspace_owner.suffix_dim),
        }

    def prepare_execution(self, phase):
        from .dsa_kv_chunk_execution import DsaKVExecution

        name = phase + "_execution"
        execution = getattr(self, name)
        if execution is None or (
            not self.forbid_realloc
            and execution._planning_key != DsaKVExecution.planning_key(self, phase)
        ):
            execution = DsaKVExecution(self, phase)
            setattr(self, name, execution)
        elif self.forbid_realloc and (
            execution.rows != self.positions_d.numel()
            or execution.batch != self.decode.local_causal_lens.shape[0]
        ):
            raise ValueError("DSA captured KV plan cannot change its query shape")
        if execution.block_ids is not None:
            execution.block_ids.copy_(self.block_table)
        execution.update_indexer_metadata(self)
        return execution

    def local_sparse_slots(self, indices):
        """Mask other owners before translating global request positions to slots."""
        page = indices.to(torch.int64) // self.page_size
        local_token = (page // self.shard_size) * self.page_size + indices.to(
            torch.int64
        ) % self.page_size
        column = local_token // self.kernel_page_size
        valid = (indices >= 0) & (page % self.shard_size == self.shard_rank)
        valid &= (column >= 0) & (column < self.block_table.shape[1])
        safe_column = column.clamp(0, max(0, self.block_table.shape[1] - 1))
        if self.block_table.shape[1]:
            blocks = self.block_table[
                self.batch_indice_d.to(torch.int64)[:, None], safe_column
            ]
            valid &= blocks > 0
        else:
            blocks = torch.zeros_like(column)
            valid &= False
        slots = (
            blocks.to(torch.int64) * self.kernel_page_size
            + local_token % self.kernel_page_size
        )
        return torch.where(valid, slots, -1).to(torch.int32)
