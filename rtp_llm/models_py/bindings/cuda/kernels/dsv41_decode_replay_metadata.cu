#include "rtp_llm/models_py/bindings/cuda/kernels/dsv41_decode_replay_metadata.h"

#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace rtp_llm {

namespace {

constexpr int kReplayThreads = 256;

// Role tags kept in sync with Dsv41ReplayPoolRole in the header.
enum PoolRole : int32_t { kSwaKv = 0, kDecoderSwaKv = 1, kCsaKv = 2, kHcaKv = 3, kIndexerKv = 4, kStateRing = 5 };

// One block per request row. Thread q (== threadIdx.x < q_len) owns query q;
// the remaining threads only join the block-table mirror refresh below. The
// whole capacity tail of valid_requests is cleared so dummy rows from a
// larger earlier replay cannot survive a batch shrink.
__global__ void dsv41DecodeReplayMetadataKernel(Dsv41DecodeReplayMetadataPlan plan,
                                                 const int32_t* __restrict__ start_pos,
                                                 int32_t start_offset,
                                                 int32_t live_request_count) {
    const int r = blockIdx.x;
    if (threadIdx.x == 0) {
        plan.valid_requests[r] = (r < live_request_count);
    }

    // ---- Block-table mirror refresh (all threads of the block cooperate) ----
    // Live rows snapshot the captured per-tag mirror (refreshed by the C++
    // runner's fused strided D2D copy); rows past the live batch are zeroed
    // so a request that left a larger earlier replay cannot survive in the
    // metadata's pool_block_tables. Matches the Python chain's mirror
    // contract (dst_bt[bs:].zero_() plus full-row copy).
    for (int p = 0; p < plan.pool_count; ++p) {
        int32_t* mirror = plan.pool_block_table_mirror[p];
        if (mirror == nullptr) {
            continue;
        }
        const Dsv41ReplayPoolDesc& pool = plan.pools[p];
        const int64_t              cols = min(pool.block_stride, plan.pool_mirror_stride[p]);
        const int64_t              src_row = static_cast<int64_t>(r) * pool.block_stride;
        const int64_t              dst_row = static_cast<int64_t>(r) * plan.pool_mirror_stride[p];
        if (r < live_request_count) {
            for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
                mirror[dst_row + c] = pool.block_table[src_row + c];
            }
        } else if (threadIdx.x == 0) {
            for (int64_t c = 0; c < cols; ++c) {
                mirror[dst_row + c] = 0;
            }
        }
    }
    if (r >= live_request_count) {
        return;
    }

    const int Q = plan.q_len;
    const int W = plan.window;

    const int q = static_cast<int>(threadIdx.x);
    if (q < Q) {
        int           sp      = start_pos[r] - start_offset;
        // Mirror _build_start_pos_from_attention_inputs' clamp so the whole
        // [sp, sp + q_len) window stays within KV/pool capacity.
        if (sp < 0) {
            sp = 0;
        }
        if (sp > plan.max_start) {
            sp = plan.max_start;
        }
        const int     abs_pos = sp + q;
        const int64_t out_rq  = static_cast<int64_t>(r) * Q + q;

        // ---- Per-request scalars ----
        if (q == 0) {
            plan.start_pos_out[r] = sp;
            if (plan.decode_seq_start_per_req != nullptr) {
                plan.decode_seq_start_per_req[r] = static_cast<int64_t>(sp);
            }
            int swa_len                     = sp + Q;
            if (swa_len < 0) {
                swa_len = 0;
            }
            if (swa_len > W) {
                swa_len = W;
            }
            if (plan.swa_topk_length != nullptr) {
                plan.swa_topk_length[r] = swa_len;
            }
        }

        // ---- Positions ----
        plan.position_ids[out_rq] = abs_pos;
        if (plan.position_ids_long != nullptr) {
            plan.position_ids_long[out_rq] = static_cast<int64_t>(abs_pos);
        }

        // ---- SWA ring slot: r * W + pos % W ----
        if (plan.slot_mapping_swa != nullptr) {
            plan.slot_mapping_swa[out_rq] = r * W + (abs_pos % W);
        }

        // ---- Window indices and absolute-position window ----
        // topk_window_idxs: full ring (in_ring + 1 + k) % W once the request
        // is past one full window, else partial k <= abs_pos / -1 pad.
        // swa_abs_idx: left-aligned abs positions [win_start .. abs_pos].
        const int  in_ring   = abs_pos % W;
        const bool is_full   = abs_pos >= (W - 1);
        const int  win_start = static_cast<int>(max(abs_pos - W + 1, 0));
        if (plan.topk_window_idxs != nullptr && plan.swa_abs_idx != nullptr) {
            int32_t* topk_win = plan.topk_window_idxs + (static_cast<int64_t>(r) * Q + q) * W;
            int32_t* swa_abs  = plan.swa_abs_idx + (static_cast<int64_t>(r) * Q + q) * W;
            for (int k = 0; k < W; ++k) {
                topk_win[k] = is_full ? (in_ring + 1 + k) % W : (k <= abs_pos ? k : -1);
                swa_abs[k]  = (win_start + k <= abs_pos) ? (win_start + k) : -1;
            }
        }

        // ---- CSA indexer output buffer reset ----
        if (plan.topk_buffer_compressed != nullptr) {
            int32_t* buf = plan.topk_buffer_compressed + (static_cast<int64_t>(r) * Q + q) * plan.index_topk;
            for (int k = 0; k < plan.index_topk; ++k) {
                buf[k] = -1;
            }
        }

        // ---- Per-pool write-slot mappings ----
        for (int p = 0; p < plan.pool_count; ++p) {
            const Dsv41ReplayPoolDesc& pool  = plan.pools[p];
            int64_t*                   slots = plan.pool_write_slots[p];
            if (slots == nullptr) {
                continue;
            }
            const int64_t row = static_cast<int64_t>(r) * pool.block_stride;
            if (pool.role == kSwaKv || pool.role == kDecoderSwaKv || pool.role == kHcaKv) {
                // Ratio-1: every token writes.
                const int64_t bis     = abs_pos / pool.tokens_per_block;
                const int64_t clamped = min(max(bis, int64_t(0)), static_cast<int64_t>(pool.max_blocks - 1));
                const int64_t bid     = static_cast<int64_t>(pool.block_table[row + clamped]);
                const int     in_blk  = abs_pos % pool.entries_per_block;
                slots[out_rq]         = (bid > 0) ? bid * pool.entries_per_block + in_blk : -1;
            } else if (pool.role == kCsaKv || pool.role == kIndexerKv) {
                // Ratio-2 boundary: (abs+1) % 2 == 0.
                const bool    on_b    = ((abs_pos + 1) % 2) == 0;
                const int64_t cmp_idx = (abs_pos + 1) / 2 - 1;
                const bool    skip    = !on_b || cmp_idx < 0;
                const int64_t safe    = skip ? 0 : cmp_idx;
                const int64_t bis     = safe / pool.tokens_per_block;
                const int64_t clamped = min(max(bis, int64_t(0)), static_cast<int64_t>(pool.max_blocks - 1));
                const int64_t bid     = static_cast<int64_t>(pool.block_table[row + clamped]);
                const int     in_blk  = static_cast<int>(safe % pool.entries_per_block);
                slots[out_rq]         = (!skip && bid > 0) ? bid * pool.entries_per_block + in_blk : -1;
            } else {  // kStateRing: cyclic (pos // tpb) % max_blocks
                const int64_t block_in_seq =
                    (static_cast<int64_t>(abs_pos) / pool.tokens_per_block) % pool.max_blocks;
                const int64_t bid    = static_cast<int64_t>(pool.block_table[row + block_in_seq]);
                const int     in_blk = abs_pos % pool.entries_per_block;
                slots[out_rq]        = (bid > 0) ? bid * pool.entries_per_block + in_blk : -1;
            }
        }

        // ---- SWA global read slots: translate swa_abs_idx through the SWA table ----
        // Two consumers exist: the parent metadata (SWA_KV table) and, under
        // bounded replay, the decoder-SWA child metadata (DECODER_SWA table).
        for (int s = 0; s < kMaxDsv41ReplaySwaSlotSets; ++s) {
            int32_t* dst = plan.swa_slot_outputs[s];
            if (dst == nullptr || plan.swa_abs_idx == nullptr) {
                continue;
            }
            const Dsv41ReplayPoolDesc& swa  = plan.pools[plan.swa_slot_pool_indices[s]];
            const int64_t              row  = static_cast<int64_t>(r) * swa.block_stride;
            const int32_t*             base = plan.swa_abs_idx + (static_cast<int64_t>(r) * Q + q) * W;
            int32_t*                   out  = dst + out_rq * W;
            for (int k = 0; k < W; ++k) {
                const int tok = base[k];
                if (tok < 0) {
                    out[k] = -1;
                    continue;
                }
                const int64_t bid_idx = tok / swa.tokens_per_block;
                if (bid_idx >= swa.max_blocks) {
                    out[k] = -1;
                    continue;
                }
                const int64_t bid = static_cast<int64_t>(swa.block_table[row + bid_idx]);
                out[k]            = (bid > 0) ?
                             static_cast<int32_t>(bid * swa.entries_per_block + tok % swa.entries_per_block) :
                             -1;
            }
        }
    }

}

}  // namespace

void invokeDsv41DecodeReplayMetadata(const Dsv41DecodeReplayMetadataPlan& plan,
                                     const int32_t* start_pos,
                                     int32_t start_offset,
                                     int32_t live_request_count,
                                     cudaStream_t stream) {
    TORCH_CHECK(plan.q_len > 0, "q_len must be positive");
    TORCH_CHECK(plan.batch_size > 0, "batch capacity must be positive");
    TORCH_CHECK(plan.window > 0, "window must be positive");
    TORCH_CHECK(start_pos != nullptr, "start_pos must be provided");
    TORCH_CHECK(plan.valid_requests != nullptr, "valid_requests buffer is required");
    TORCH_CHECK(live_request_count >= 0 && live_request_count <= plan.batch_size,
                "live request count must fit the capture batch");
    TORCH_CHECK(plan.pool_count >= 0 && plan.pool_count <= kMaxDsv41ReplayPools,
                "pool count exceeds the fused plan capacity");
    for (int p = 0; p < plan.pool_count; ++p) {
        const auto& pool = plan.pools[p];
        TORCH_CHECK(pool.block_table != nullptr, "pool block table pointer is null");
        TORCH_CHECK(pool.block_stride > 0, "pool block stride must be positive");
        if (plan.pool_block_table_mirror[p] != nullptr) {
            TORCH_CHECK(plan.pool_mirror_stride[p] > 0, "pool mirror stride must be positive");
        }
        TORCH_CHECK(pool.entries_per_block > 0, "pool entries per block must be positive");
        TORCH_CHECK(pool.tokens_per_block > 0, "pool tokens per block must be positive");
        TORCH_CHECK(pool.max_blocks > 0, "pool max blocks must be positive");
    }
    for (int s = 0; s < kMaxDsv41ReplaySwaSlotSets; ++s) {
        if (plan.swa_slot_outputs[s] != nullptr) {
            TORCH_CHECK(plan.swa_slot_pool_indices[s] >= 0 && plan.swa_slot_pool_indices[s] < plan.pool_count,
                        "SWA slot output references an invalid pool index");
        }
    }

    const int grid = plan.batch_size;
    dsv41DecodeReplayMetadataKernel<<<grid, kReplayThreads, 0, stream>>>(
        plan, start_pos, start_offset, live_request_count);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "dsv41 decode replay metadata kernel failed: ", cudaGetErrorString(result));
}

}  // namespace rtp_llm
