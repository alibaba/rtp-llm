#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

// Fused per-replay DSv4.1 decode metadata refresh.
//
// Replays of the target-verify / decode CUDA graph must rewrite the impl's
// persistent metadata tensors in place (stable addresses baked into the
// graph) from three live inputs:
//   * start_pos           [B] int32 (clamped first-token positions)
//   * live request count  scalar
//   * per-pool block tables [B, max_blocks] int32 (already refreshed into the
//     captured attention-input mirrors by CudaGraphRunner's fused strided
//     D2D copy; this kernel also snapshots them into the metadata's own
//     persistent pool_block_tables buffers)
//
// The previous implementation entered Python per replay (GIL acquire +
// pybind tuple + ~147 aten/Triton launches from ~1 ms of interpreter time).
// This kernel computes the whole update in one launch.
//
// A Plan is built once per graph instance at capture time (see the Python
// factory in decode_fmha_impl.py). All pointers must stay alive for the
// graph instance lifetime; the plan never mutates after construction.

constexpr int kMaxDsv41ReplayPools      = 8;
constexpr int kMaxDsv41ReplaySwaSlotSets = 2;

// Pool role inside the fused kernel. One pool block table feeds several
// derived buffers; the role selects which formulas apply. All formulas
// mirror the Python chain (decode_attn_metadata.py and its fused Triton
// kernels): block-id lookups read the metadata's own pool_block_tables
// mirror — refreshed by this kernel from the captured source table — with
// the mirror's width as the clamp domain, exactly like the Triton
// BT_STRIDE constants and translate_local_to_global_slots'
// max_num_blocks_per_req.
enum class Dsv41ReplayPoolRole : int32_t {
    // Ratio-1 ring KV (swa_kv / decoder_swa_kv): every token writes; global
    // read slots translate swa_abs_idx through this pool's table.
    SWA_KV = 0,
    // Same formulas as SWA_KV but a physically distinct ring (decoder SWA).
    DECODER_SWA_KV = 1,
    CSA_KV = 2,
    // Ratio-128 compressed KV boundary writes (HCA_KV = global_kv_1).
    HCA_KV = 3,
    // Ratio-4 compressed KV boundary writes (indexer_kv).
    INDEXER_KV = 4,
    // Compressor state ring: cyclic (pos // tokens_per_block) % max_blocks,
    // driven from compressor_state_slot_mappings, not pool_write_slots.
    STATE_RING = 5,
};

struct Dsv41ReplayPoolDesc {
    const int32_t* block_table       = nullptr;  // [B, block_stride] (captured source)
    int64_t        block_stride      = 0;        // source row width (elements)
    int32_t        entries_per_block = 0;        // slot multiplier within one block
    int32_t        raw_tokens_per_block = 0;     // raw-token coverage of one table row
    int32_t        tokens_per_block  = 0;        // kernel-domain tokens per block row
    int32_t        max_blocks        = 0;        // mirror clamp domain (mirror columns)
    int32_t        role              = static_cast<int32_t>(Dsv41ReplayPoolRole::SWA_KV);
};

struct Dsv41DecodeReplayMetadataPlan {
    // Persistent outputs (stable addresses read by the captured graph).
    int32_t* position_ids             = nullptr;  // [B * Q] i32
    int64_t* position_ids_long        = nullptr;  // [B * Q] i64
    int64_t* decode_seq_start_per_req = nullptr;  // [B] i64
    bool*     valid_requests           = nullptr;  // [B] bool storage
    int32_t* start_pos_out            = nullptr;  // [B] i32
    int32_t* slot_mapping_swa         = nullptr;  // [B * Q] i32 (ring slots)
    int32_t* topk_window_idxs         = nullptr;  // [B, Q, WINDOW] i32
    int32_t* swa_abs_idx              = nullptr;  // [B, Q, WINDOW] i32
    int32_t* topk_buffer_compressed   = nullptr;  // [B, Q, INDEX_TOPK] i32 (-1 reset)
    int32_t* swa_topk_length          = nullptr;  // [B] i32
    int64_t* pool_write_slots[kMaxDsv41ReplayPools]        = {};  // [B * Q] i64
    int64_t* pool_state_slots[kMaxDsv41ReplayPools]        = {};  // [B * Q] i64 (cyclic state rings)
    int32_t* pool_block_table_mirror[kMaxDsv41ReplayPools] = {};  // [B, mirror_stride] i32
    int64_t  pool_mirror_stride[kMaxDsv41ReplayPools]      = {};  // mirror row width (elements)
    // SWA read-slot translation outputs (parent SWA + bounded-replay decoder
    // SWA child). Each references the pool whose table translates it.
    int32_t* swa_slot_outputs[kMaxDsv41ReplaySwaSlotSets]      = {};  // [B * Q, WINDOW] i32
    int32_t  swa_slot_pool_indices[kMaxDsv41ReplaySwaSlotSets] = {};

    Dsv41ReplayPoolDesc pools[kMaxDsv41ReplayPools];
    int32_t             pool_count = 0;

    int32_t index_topk = 0;  // topk_buffer_compressed last-dim width
    int32_t max_start  = 0;  // clamp bound: start_pos must fit [0, max_seq_len - q_len]

    int32_t q_len      = 1;
    int32_t batch_size = 0;  // capture capacity (max B)
    int32_t window     = 128;
};

void invokeDsv41DecodeReplayMetadata(const Dsv41DecodeReplayMetadataPlan& plan,
                                     const int32_t* start_pos,
                                     int32_t start_offset,
                                     int32_t live_request_count,
                                     cudaStream_t stream);

}  // namespace rtp_llm
