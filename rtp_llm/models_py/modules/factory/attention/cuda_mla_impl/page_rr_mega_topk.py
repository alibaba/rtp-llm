"""Exact global top-k of DSA Indexer logits across the decode DCP group (Page-RR KV).

Every rank holds the Indexer logits of its Page-RR share of each row. PageRRMegaTopK selects each
row's global top-k with rtp_llm_ops.mega_topk, which exchanges histogram segments and boundary
values over NVLink inside one kernel, and returns the global positions this rank owns (then -1).
Ties at the k-th value go to the lowest (rank, local position), so equal inputs give equal outputs.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from rtp_llm.models_py.distributed.collective_torch import Group, get_process_group
from rtp_llm.ops.compute_ops import rtp_llm_ops


class PageRRMegaTopK:
    """MegaTopk state shared by the serial decode layers of one DCP (TP) group.

    It owns the symmetric exchange workspace, the per-slot exchange epochs and the co-resident
    cluster table the ranks agree on. Calls must run in the same order on every rank, one at a
    time (the layers of one decode step on one stream).

    max_local_kvlen bounds every rank's local row length and, with the row count, picks the
    kernel's launch shape: under CUDA graphs the logits are as wide as the owner capacity of the
    configured max length, so pass that bound. kv_page_tokens is the kernel page of the block
    tables passed in; page_tokens is the Page-RR page that maps local to global positions.
    """

    def __init__(self, topk, max_local_kvlen, kv_page_tokens, page_tokens, device):
        self.topk, self.max_local_kvlen = topk, max_local_kvlen
        self.kv_page_tokens, self.page_tokens = kv_page_tokens, page_tokens
        self.group = get_process_group(Group.TP)
        self.cp_size = dist.get_world_size(self.group)
        self.cp_rank = dist.get_rank(self.group)
        # The launch shape must match on every rank: agree on the co-resident cluster counts.
        resident = torch.tensor(rtp_llm_ops.mega_topk_resident_clusters(), dtype=torch.int64, device=device)
        dist.all_reduce(resident, op=dist.ReduceOp.MIN, group=self.group)
        self.resident = resident.tolist()
        slots = max(self.resident)  # one exchange slot per row owner of any launch shape
        self.workspace = symm.empty(
            (rtp_llm_ops.mega_topk_workspace_bytes(self.cp_size, slots),), dtype=torch.uint8, device=device
        )
        handle = symm.rendezvous(self.workspace, group=self.group)
        self.peers = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
        self.multicast_ptr = int(handle.multicast_ptr)  # 0 without NVLS: per-peer stores
        self.epochs = torch.zeros(slots, dtype=torch.int32, device=device)
        self.workspace.zero_()
        # Every rank's workspace must be zeroed before any peer's first exchange.
        torch.cuda.synchronize(device)
        dist.barrier(group=self.group)

    def __call__(self, logits, local_lens, positions, block_table, rows_per_request, out, count):
        """Fill out [rows, topk] with this rank's selected global positions (then -1) and count [rows].

        logits [rows, >= local len] float32 over this rank's local positions; local_lens [rows]
        int32; positions [rows] int32, each row's global query position (-1 marks padding);
        block_table [requests, pages] int32 kernel pages, row r of request r // rows_per_request.
        """
        rtp_llm_ops.mega_topk(
            logits, local_lens, positions, block_table, rows_per_request, self.kv_page_tokens,
            out, count, self.peers, self.multicast_ptr, self.epochs, self.cp_rank, self.cp_size,
            self.topk, self.max_local_kvlen, self.resident, True, self.page_tokens,
        )
        return out, count


_instances = {}


def get_page_rr_mega_topk(topk, max_local_kvlen, kv_page_tokens, page_tokens, device):
    """The group's shared instance: one workspace serves every layer of a decode step."""
    key = (get_process_group(Group.TP), device, topk, max_local_kvlen, kv_page_tokens, page_tokens)
    if key not in _instances:
        _instances[key] = PageRRMegaTopK(topk, max_local_kvlen, kv_page_tokens, page_tokens, device)
    return _instances[key]
