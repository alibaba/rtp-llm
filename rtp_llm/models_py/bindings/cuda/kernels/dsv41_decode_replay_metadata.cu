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

// Block-id lookup through the SOURCE table, reproducing exactly what the
// Python chain read through the refreshed mirror:
//  * live rows: the mirror equals the source row over [0, block_stride) and
//    holds 0 beyond (the v2 plan guarantees mirror columns >= source columns,
//    and the refresh zeroes the uncovered tail), so any clamp-domain index
//    below block_stride reads the identical value and any index at or above
//    it resolves to 0 — the same value the mirror holds there.
//  * dummy rows: the refresh zeroes the mirror row, so every lookup resolved
//    bid 0; returning 0 without a memory read is identical.
// Reading the source removes the write-then-read ordering between the mirror
// refresh and the lookups, so no cross-block fence is needed.
__device__ __forceinline__ int64_t
lookupMirrorBlockId(const Dsv41ReplayPoolDesc& pool, int row, int64_t idx, bool live_row) {
    if (!live_row || idx < 0 || idx >= static_cast<int64_t>(pool.block_stride)) {
        return 0;
    }
    return static_cast<int64_t>(pool.block_table[static_cast<int64_t>(row) * pool.block_stride + idx]);
}

// Grid = (batch, tiles). Every (row, tile) block cooperates on the mirror
// refresh for ALL pools (its slice of each mirror row), so the 8-pool column
// sweep no longer serializes on the single block the old grid=batch launch
// used at batch 1-4. Tile 0 additionally computes the per-request metadata
// for its row: thread q < Q owns query q.
//
// The window-width outputs (topk_window_idxs / swa_abs_idx, Q*W ints per row,
// and the SWA read-slot translations) were previously produced by ONE thread
// per query; they are now distributed across the whole tile-0 block by linear
// output index.
__global__ void dsv41DecodeReplayMetadataKernel(Dsv41DecodeReplayMetadataPlan plan,
                                                 const int32_t* __restrict__ start_pos,
                                                 int32_t start_offset,
                                                 int32_t live_request_count) {
    const int r     = blockIdx.x;
    const int tile  = blockIdx.y;
    const int tiles = gridDim.y;
    const bool live = r < live_request_count;

    if (tile == 0 && threadIdx.x == 0) {
        plan.valid_requests[r] = live;
    }

    // ---- Block-table mirror refresh (all blocks cooperate) ----
    // Live rows snapshot the captured per-tag source table (refreshed by the
    // C++ runner's fused strided D2D copy) into the metadata's persistent
    // pool_block_tables mirror, bounded to the intersection like the Python
    // chain (n_cols = min(src, dst); dst tail rows/columns zeroed). Rows past
    // the live batch are zeroed so a request that left a larger earlier
    // replay cannot survive in the metadata's pool_block_tables. gridDim.y
    // tiles widen each row's sweep to tiles*blockDim.x threads.
    {
        const int threads_per_row = tiles * kReplayThreads;
        const int tid_in_row      = tile * kReplayThreads + static_cast<int>(threadIdx.x);
        for (int p = 0; p < plan.pool_count; ++p) {
            int32_t* mirror = plan.pool_block_table_mirror[p];
            if (mirror == nullptr) {
                continue;
            }
            const Dsv41ReplayPoolDesc& pool        = plan.pools[p];
            const int64_t              cols        = min(pool.block_stride, plan.pool_mirror_stride[p]);
            const int64_t              mirror_cols = plan.pool_mirror_stride[p];
            const int64_t              src_row     = static_cast<int64_t>(r) * pool.block_stride;
            const int64_t              dst_row     = static_cast<int64_t>(r) * plan.pool_mirror_stride[p];
            if (live) {
                // Copy the live source columns; zero any mirror columns the
                // source does not cover (Python: dst_bt[..., n_cols:].zero_()).
                for (int64_t c = tid_in_row; c < mirror_cols; c += threads_per_row) {
                    mirror[dst_row + c] = c < cols ? pool.block_table[src_row + c] : 0;
                }
            } else {
                for (int64_t c = tid_in_row; c < mirror_cols; c += threads_per_row) {
                    mirror[dst_row + c] = 0;
                }
            }
        }
    }

    if (tile != 0) {
        // Extra tiles only join the mirror refresh above.
        return;
    }

    // Dummy rows (r >= live) are recomputed with sp=0, exactly like the
    // Python chain: its grid covered the full capture capacity, the C++ fill
    // had already zeroed the dummy length mirrors (prefix=0 / seq+1=1) and
    // block-table rows, so every pool lookup hit block id 0 and resolved the
    // write slot to -1 — which the captured SWA KV writer skips. Skipping the
    // recompute would leave the previous (larger) replay's positive slot ids
    // in these rows; a freed block reallocated to a live request would then
    // be corrupted by the dummy row's stale write.
    const int Q = plan.q_len;
    const int W = plan.window;

    // Row start position, shared by every output below (each thread computes
    // it redundantly; one cached load).
    int sp = 0;
    if (live) {
        sp = start_pos[r] - start_offset;
    }
    // Mirror _build_start_pos_from_attention_inputs' clamp so the whole
    // [sp, sp + q_len) window stays within KV/pool capacity.
    if (sp < 0) {
        sp = 0;
    }
    if (sp > plan.max_start) {
        sp = plan.max_start;
    }

    const int q = static_cast<int>(threadIdx.x);
    if (q < Q) {
        const int     abs_pos = sp + q;
        const int64_t out_rq  = static_cast<int64_t>(r) * Q + q;

        // ---- Per-request scalars ----
        if (q == 0) {
            plan.start_pos_out[r] = sp;
            if (plan.decode_seq_start_per_req != nullptr) {
                plan.decode_seq_start_per_req[r] = static_cast<int64_t>(sp);
            }
            int swa_len = sp + Q;
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
    }

    // ---- Window indices and absolute-position window ----
    // topk_window_idxs: full ring (in_ring + 1 + k) % W once the request is
    // past one full window, else partial k <= abs_pos / -1 pad.
    // swa_abs_idx: left-aligned abs positions [win_start .. abs_pos].
    // One W-wide row per (r, q): distributed across the whole block by linear
    // output index instead of a single thread's W-loop.
    if (plan.topk_window_idxs != nullptr && plan.swa_abs_idx != nullptr) {
        const int64_t rows = static_cast<int64_t>(Q) * W;
        for (int64_t i = threadIdx.x; i < rows; i += kReplayThreads) {
            const int local_q = static_cast<int>(i / W);
            const int k       = static_cast<int>(i % W);
            const int abs_pos = sp + local_q;
            const int in_ring = abs_pos % W;
            const bool is_full = abs_pos >= (W - 1);
            const int  win_start = static_cast<int>(max(abs_pos - W + 1, 0));
            const int64_t row_off = (static_cast<int64_t>(r) * Q + local_q) * W;
            plan.topk_window_idxs[row_off + k] = is_full ? (in_ring + 1 + k) % W : (k <= abs_pos ? k : -1);
            plan.swa_abs_idx[row_off + k]      = (win_start + k <= abs_pos) ? (win_start + k) : -1;
        }
    }

    // ---- CSA indexer output buffer reset ----
    if (plan.topk_buffer_compressed != nullptr) {
        const int64_t elems = static_cast<int64_t>(Q) * plan.index_topk;
        int32_t*     buf    = plan.topk_buffer_compressed + static_cast<int64_t>(r) * Q * plan.index_topk;
        for (int64_t i = threadIdx.x; i < elems; i += kReplayThreads) {
            buf[i] = -1;
        }
    }

    if (q < Q) {
        const int     abs_pos = sp + q;
        const int64_t out_rq  = static_cast<int64_t>(r) * Q + q;

        // Block-id lookups resolve through the source table with the mirror
        // width as clamp domain — the exact domain the Python chain used
        // (fused_phase2b BT_STRIDE, translate max_num_blocks_per_req,
        // state-ring % max_blocks); see lookupMirrorBlockId.
        for (int p = 0; p < plan.pool_count; ++p) {
            const Dsv41ReplayPoolDesc& pool  = plan.pools[p];
            int64_t*                   slots = plan.pool_write_slots[p];
            if (slots == nullptr || plan.pool_block_table_mirror[p] == nullptr) {
                continue;
            }
            const int64_t m_max = static_cast<int64_t>(pool.max_blocks) - 1;
            const int64_t E     = pool.entries_per_block;
            if (pool.role == kSwaKv || pool.role == kDecoderSwaKv || pool.role == kHcaKv) {
                // Ratio-1: every token writes (kernel-domain position).
                const int64_t pos     = abs_pos;
                const int64_t bis     = pos / pool.tokens_per_block;
                const int64_t clamped = min(max(bis, int64_t(0)), m_max);
                const int64_t bid     = lookupMirrorBlockId(pool, r, clamped, live);
                const int64_t in_blk  = pos % E;
                slots[out_rq]         = (bid > 0) ? bid * E + in_blk : -1;
            } else if (pool.role == kCsaKv || pool.role == kIndexerKv) {
                // Ratio-4 boundary: (abs+1) % 4 == 0 (production fused
                // phase2b Triton kernel semantics for V4.1 as well).
                const int64_t abs_p1  = static_cast<int64_t>(abs_pos) + 1;
                const bool    on_b    = (abs_p1 % 4) == 0;
                const int64_t cmp_idx = abs_p1 / 4 - 1;
                const bool    skip    = !on_b || cmp_idx < 0;
                const int64_t safe    = skip ? 0 : cmp_idx;
                const int64_t bis     = safe / pool.tokens_per_block;
                const int64_t clamped = min(max(bis, int64_t(0)), m_max);
                const int64_t bid     = lookupMirrorBlockId(pool, r, clamped, live);
                const int64_t in_blk  = safe % E;
                slots[out_rq]         = (!skip && bid > 0) ? bid * E + in_blk : -1;
            }
            // kStateRing pools write through pool_state_slots below.
        }

        // ---- Compressor state rings: cyclic (pos // tpb) % max_blocks ----
        // Mirrors _update_compressor_state_slot_mappings -> positions are the
        // raw token positions; entries_per_block is the in-block modulo domain.
        for (int p = 0; p < plan.pool_count; ++p) {
            const Dsv41ReplayPoolDesc& pool  = plan.pools[p];
            int64_t*                   slots = plan.pool_state_slots[p];
            if (slots == nullptr || plan.pool_block_table_mirror[p] == nullptr || pool.role != kStateRing) {
                continue;
            }
            const int64_t block_in_seq =
                (static_cast<int64_t>(abs_pos) / pool.tokens_per_block) % static_cast<int64_t>(pool.max_blocks);
            const int64_t bid    = lookupMirrorBlockId(pool, r, block_in_seq, live);
            const int64_t in_blk = abs_pos % pool.entries_per_block;
            slots[out_rq]        = (bid > 0) ? bid * pool.entries_per_block + in_blk : -1;
        }
    }

    // ---- SWA global read slots: translate swa_abs_idx through the SWA table ----
    // Two consumers exist: the parent metadata (SWA_KV table) and, under
    // bounded replay, the decoder-SWA child metadata (DECODER_SWA table).
    // Translation resolves the source table with the mirror width as
    // max_num_blocks_per_req, exactly like translate_local_to_global_slots.
    // Distributed across the whole block by linear output index; the input
    // token id is recomputed with the producer's exact formula.
    if (plan.swa_abs_idx != nullptr) {
        for (int s = 0; s < kMaxDsv41ReplaySwaSlotSets; ++s) {
            int32_t* dst = plan.swa_slot_outputs[s];
            if (dst == nullptr) {
                continue;
            }
            const int pool_idx = plan.swa_slot_pool_indices[s];
            if (pool_idx < 0 || pool_idx >= plan.pool_count) {
                continue;
            }
            const Dsv41ReplayPoolDesc& swa = plan.pools[pool_idx];
            if (plan.pool_block_table_mirror[pool_idx] == nullptr) {
                continue;
            }
            const int64_t rows = static_cast<int64_t>(Q) * W;
            for (int64_t i = threadIdx.x; i < rows; i += kReplayThreads) {
                const int64_t row_q = i / W;
                const int     k     = static_cast<int>(i % W);
                const int     abs_pos = sp + static_cast<int>(row_q);
                const bool    is_full = abs_pos >= (W - 1);
                const int     win_start = static_cast<int>(max(abs_pos - W + 1, 0));
                // Identical to the swa_abs_idx value written above for this
                // (row, q, k): full windows use win_start + k, partial
                // windows k <= abs_pos, else -1.
                const int tok = is_full ? win_start + k : (k <= abs_pos ? k : -1);
                const int64_t out_idx = (static_cast<int64_t>(r) * Q + row_q) * W + k;
                if (tok < 0) {
                    dst[out_idx] = -1;
                    continue;
                }
                const int64_t bid_idx = tok / swa.tokens_per_block;
                if (bid_idx >= swa.max_blocks) {
                    dst[out_idx] = -1;
                    continue;
                }
                const int64_t bid = lookupMirrorBlockId(swa, r, bid_idx, live);
                dst[out_idx]      = (bid > 0) ?
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
        TORCH_CHECK(pool.raw_tokens_per_block > 0, "pool raw tokens per block must be positive");
        TORCH_CHECK(pool.tokens_per_block > 0, "pool kernel tokens per block must be positive");
        TORCH_CHECK(pool.max_blocks > 0, "pool max blocks must be positive");
    }
    for (int s = 0; s < kMaxDsv41ReplaySwaSlotSets; ++s) {
        if (plan.swa_slot_outputs[s] != nullptr) {
            TORCH_CHECK(plan.swa_slot_pool_indices[s] >= 0 && plan.swa_slot_pool_indices[s] < plan.pool_count,
                        "SWA slot output references an invalid pool index");
        }
    }

    // Tiled 2D grid: gridDim.y blocks join the 8-pool mirror refresh so the
    // column sweep no longer serializes on gridDim.x = batch (1-4 at low
    // load); every pool row is swept by tiles*256 threads.
    int max_mirror_cols = 1;
    for (int p = 0; p < plan.pool_count; ++p) {
        if (plan.pool_mirror_stride[p] > 0) {
            max_mirror_cols = std::max(max_mirror_cols, static_cast<int>(plan.pool_mirror_stride[p]));
        }
    }
    const int tiles = std::min(max_mirror_cols / kReplayThreads + 1, 64);
    const dim3 grid(plan.batch_size, tiles);
    dsv41DecodeReplayMetadataKernel<<<grid, kReplayThreads, 0, stream>>>(
        plan, start_pos, start_offset, live_request_count);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "dsv41 decode replay metadata kernel failed: ", cudaGetErrorString(result));
}

}  // namespace rtp_llm
