#pragma once

#include <torch/extension.h>

#include <vector>

namespace rtp_llm {

// Exact global top-k of each row across a DCP group with Page-RR KV sharding (K3.1 DSA decode).
// Every rank passes its own share of each row; the kernel exchanges histogram segments and
// boundary values in-kernel over NVLink and writes the selected elements this rank owns, as KV
// slots or, with global_page_tokens, as global positions.
//
//   logits        : [rows, width] float32, 16-byte aligned rows; this rank's local positions
//   local_lens    : [rows] int32, valid local positions of each row (<= min(width, max_local_kvlen))
//   positions     : [rows] int32, the row's global query position, so its length over the whole group
//                   is position + 1 (identical on every rank; -1 for padding rows)
//   block_table   : [requests, pages] int32, kernel KV pages of kv_page_tokens tokens (a power of two >= 16)
//                   holding the local positions; row r belongs to request r / rows_per_request
//   output        : [rows, >= k] int32, this rank's selected slots (or global positions), then -1 up
//                   to k: first those above the boundary coarse bin, then the boundary ones, each in
//                   local position order
//   count         : [rows] int32, number of selected slots
//   peers         : [world] int64, every rank's address of the symmetric exchange workspace
//   multicast_ptr : its NVLS multicast address, or 0 to store to each peer instead
//   epochs        : [slots] int32, zeroed once; rows run on min(rows, slots, resident) owners;
//                   slots = the largest resident entry gives every launch shape its owners
//   max_local_kvlen : bound on any rank's local KV length per row, identical on every rank
//                     (e.g. the Page-RR share of the configured max length); with the row count
//                     it picks the launch shape, so it must hold for every row
//   resident      : co-resident owner counts per launch shape, mega_topk_resident_clusters()
//                   reduced by MIN over the group once
//   global_page_tokens : 0 to write KV slots through block_table; else the Page-RR page size
//                   (local page j of rank r is global page j * world + r) to write global positions
//
// The workspace (mega_topk_workspace_bytes(world, slots) bytes per rank) is zeroed once
// before first use and reused by later calls; calls sharing it must be serialized in the same
// order on every rank. Ties at the k-th value go to lower ranks first. With deterministic, a rank
// also fills its tie quota from its lowest positions and writes each group in position order, so
// equal inputs give equal outputs; without it warps claim output ranges with atomics: the values
// selected are the same exact top-k, but the order and which equal-valued elements fill the quota
// may vary between runs (fewer block-wide scans).
void mega_topk(const torch::Tensor&        logits,
                   const torch::Tensor&        local_lens,
                   const torch::Tensor&        positions,
                   const torch::Tensor&        block_table,
                   int64_t                     rows_per_request,
                   int64_t                     kv_page_tokens,
                   torch::Tensor&              output,
                   torch::Tensor&              count,
                   const torch::Tensor&        peers,
                   int64_t                     multicast_ptr,
                   torch::Tensor&              epochs,
                   int64_t                     rank,
                   int64_t                     world,
                   int64_t                     k,
                   int64_t                     max_local_kvlen,
                   const std::vector<int64_t>& resident,
                   bool                        deterministic      = true,
                   int64_t                     global_page_tokens = 0);

// This device's co-resident cluster counts for every launch shape (9 entries). Also enables
// clusters of 16 for this device, so call it before the first mega_topk.
std::vector<int64_t> mega_topk_resident_clusters();

int64_t mega_topk_workspace_bytes(int64_t world, int64_t slots);

}  // namespace rtp_llm
