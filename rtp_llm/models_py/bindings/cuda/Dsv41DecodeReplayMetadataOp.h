#pragma once

#include "rtp_llm/models_py/bindings/cuda/kernels/dsv41_decode_replay_metadata.h"

#include <ATen/ATen.h>

#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

// Capture-time plan factory: Python hands the persistent metadata tensors and
// per-pool block-table mirrors once per graph instance; the returned plan
// holds raw device pointers valid for as long as the impl keeps the tensors
// alive (graph-instance lifetime).
std::shared_ptr<Dsv41DecodeReplayMetadataPlan>
buildDsv41DecodeReplayMetadataPlan(const std::vector<at::Tensor>& block_tables,
                                   const std::vector<std::string>& roles,
                                   const std::vector<int64_t>&     entries_per_block,
                                   const std::vector<int64_t>&     tokens_per_block,
                                   const at::Tensor&               valid_requests,
                                   const at::Tensor&               start_pos_out,
                                   const at::Tensor&               position_ids,
                                   const at::Tensor&               position_ids_long,
                                   const at::Tensor&               decode_seq_start_per_req,
                                   const at::Tensor&               slot_mapping_swa,
                                   const at::Tensor&               topk_window_idxs,
                                   const at::Tensor&               swa_abs_idx,
                                   const at::Tensor&               topk_buffer_compressed,
                                   const at::Tensor&               swa_topk_length,
                                   const std::vector<at::Tensor>&  pool_write_slots,
                                   const std::vector<at::Tensor>&  pool_block_table_mirrors,
                                   const std::vector<at::Tensor>&  swa_slot_outputs,
                                   const std::vector<int64_t>&     swa_slot_pool_indices,
                                   int64_t                          q_len,
                                   int64_t                          batch_size,
                                   int64_t                          window,
                                   int64_t                          max_start);

}  // namespace rtp_llm
