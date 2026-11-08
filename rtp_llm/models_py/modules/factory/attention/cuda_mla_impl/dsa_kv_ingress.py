"""Budgeted new-KV transport from query CP producers to PageRR owners."""

import math

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather_into
from rtp_llm.models_py.utils.module_profiler import profile_scope

from ..mla_prefill_plan import prefill_kv_workspace_budget_bytes


class DsaKVIngress:
    def __init__(self, params, phase, features):
        self.params, self.phase, self.features = params, phase, features
        owner, layout = params.workspace_owner, params.producer_layout
        self.workspace = owner.phase(phase + "_ingress")
        rows, shards = layout.local_rows, params.shard_size
        # Charge routing metadata and pack/gather scratch. The projected KV
        # rows already produced by the model follow the scheduler's token budget.
        fixed = layout.physical_rows * 24 + 64 * 1024 * 1024
        available = prefill_kv_workspace_budget_bytes() - fixed
        per_row = features * 2 * (1 + shards) + shards * 24
        maximum = available // per_row
        page = params.kernel_page_size
        required = math.ceil(rows / page) * page
        capacity = min(required, maximum // page * page)
        if capacity < page:
            raise ValueError(
                "RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB cannot hold one aligned DSA KV ingress block"
            )
        self.capacity = capacity
        self.shapes = {
            "send": ((capacity, features), torch.bfloat16),
            "gather": ((shards * capacity, features), torch.bfloat16),
        }
        self.blocks = []
        offsets = torch.arange(capacity, device=owner.device, dtype=torch.int64)
        ranks = torch.arange(shards, device=owner.device, dtype=torch.int64)[:, None]
        for start in range(0, rows, capacity):
            count = min(rows - start, capacity)
            logical = (ranks * rows + start + offsets).reshape(-1)
            safe = logical.clamp(0, layout.physical_rows - 1)
            positions = layout.rankmajor_positions.index_select(0, safe)
            requests = layout.rankmajor_requests.index_select(0, safe)
            valid = (offsets < count).expand(shards, -1).reshape(-1)
            positions = torch.where(valid, positions, -1)
            slots = params.cache_adapter.slot_mapping(
                positions, requests, params.block_table,
                valid_token_mask=positions >= 0,
                allow_reserved_block_zero=params.is_warmup,
            )
            self.blocks.append((start, count, slots))
        self.required_bytes = fixed + capacity * per_row
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(owner.device))

    def run(self, values, writer):
        if values.shape != (self.params.producer_layout.local_rows, self.features):
            raise ValueError("DSA ingress requires complete local producer KV values")
        stream = torch.cuda.current_stream(values.device)
        stream.wait_event(self.ready)
        self.workspace.reserve(self.shapes)
        send = self.workspace("send", self.shapes["send"][0], torch.bfloat16)
        gathered = self.workspace("gather", self.shapes["gather"][0], torch.bfloat16)
        for start, count, slots in self.blocks:
            slots.record_stream(stream)
            with profile_scope(f"RTP::attention.dsa.{self.phase}.ingress_pack"):
                source = values[start : start + count]
                if count == self.capacity and source.is_contiguous():
                    # The producer stays live until the collective and owner
                    # write have completed on this stream. No extra wire copy.
                    wire = source
                else:
                    send[:count].copy_(source)
                    if count < self.capacity:
                        send[count:].zero_()
                    wire = send
            with profile_scope(f"RTP::attention.dsa.{self.phase}.ingress_all_gather"):
                all_gather_into(wire, gathered, Group.TP)
            with profile_scope(f"RTP::attention.dsa.{self.phase}.ingress_owner_write"):
                writer(gathered, slots)
        self.workspace.release()
