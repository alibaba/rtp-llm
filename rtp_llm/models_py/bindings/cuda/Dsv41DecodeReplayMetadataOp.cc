#include "rtp_llm/models_py/bindings/cuda/Dsv41DecodeReplayMetadataOp.h"

#include "rtp_llm/models_py/bindings/cuda/kernels/dsv41_decode_replay_metadata.h"

#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtp_llm {

namespace {

const at::Tensor& requireCudaInt32(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    return tensor;
}

const at::Tensor& requireCudaInt64(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    return tensor;
}

int32_t roleFromName(const std::string& role) {
    if (role == "swa_kv") {
        return static_cast<int32_t>(Dsv41ReplayPoolRole::SWA_KV);
    }
    if (role == "decoder_swa_kv") {
        return static_cast<int32_t>(Dsv41ReplayPoolRole::DECODER_SWA_KV);
    }
    if (role == "csa_kv") {
        return static_cast<int32_t>(Dsv41ReplayPoolRole::CSA_KV);
    }
    if (role == "hca_kv") {
        return static_cast<int32_t>(Dsv41ReplayPoolRole::HCA_KV);
    }
    if (role == "indexer_kv") {
        return static_cast<int32_t>(Dsv41ReplayPoolRole::INDEXER_KV);
    }
    if (role == "state_ring") {
        return static_cast<int32_t>(Dsv41ReplayPoolRole::STATE_RING);
    }
    throw std::runtime_error("unknown DSv4.1 replay pool role: " + role);
}

}  // namespace

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
                                   int64_t                          max_start) {
    auto plan = std::make_shared<Dsv41DecodeReplayMetadataPlan>();

    TORCH_CHECK(block_tables.size() == roles.size() && block_tables.size() == entries_per_block.size()
                    && block_tables.size() == tokens_per_block.size(),
                "pool descriptor vectors must have equal length");
    TORCH_CHECK(static_cast<int64_t>(block_tables.size()) <= kMaxDsv41ReplayPools,
                "pool count exceeds the fused plan capacity");
    TORCH_CHECK(static_cast<int64_t>(pool_write_slots.size()) <= kMaxDsv41ReplayPools,
                "pool write slot vector exceeds the fused plan capacity");
    TORCH_CHECK(static_cast<int64_t>(pool_block_table_mirrors.size()) <= kMaxDsv41ReplayPools,
                "pool mirror vector exceeds the fused plan capacity");
    TORCH_CHECK(static_cast<int64_t>(swa_slot_outputs.size()) <= kMaxDsv41ReplaySwaSlotSets,
                "SWA slot output count exceeds the fused plan capacity");
    TORCH_CHECK(swa_slot_outputs.size() == swa_slot_pool_indices.size(),
                "SWA slot output and pool-index vectors must have equal length");

    plan->pool_count = static_cast<int32_t>(block_tables.size());
    for (size_t p = 0; p < block_tables.size(); ++p) {
        const auto& table = requireCudaInt32(block_tables[p], "pool block_table");
        TORCH_CHECK(table.dim() == 2, "pool block_table must be rank 2");
        TORCH_CHECK(table.size(0) >= batch_size, "pool block_table rows must cover the capture batch");
        auto& desc  = plan->pools[p];
        desc.block_table       = table.data_ptr<int32_t>();
        desc.block_stride      = table.size(1);
        desc.entries_per_block = static_cast<int32_t>(entries_per_block[p]);
        desc.tokens_per_block  = static_cast<int32_t>(tokens_per_block[p]);
        desc.max_blocks        = static_cast<int32_t>(table.size(1));
        desc.role              = roleFromName(roles[p]);
    }
    for (size_t p = 0; p < pool_write_slots.size(); ++p) {
        if (!pool_write_slots[p].defined() || pool_write_slots[p].numel() == 0) {
            continue;  // pool without a slot buffer: kernel skips it
        }
        const auto& slots = requireCudaInt64(pool_write_slots[p], "pool write slots");
        TORCH_CHECK(slots.numel() >= batch_size * q_len, "pool write slots must cover the capture token count");
        plan->pool_write_slots[p] = slots.data_ptr<int64_t>();
    }
    for (size_t p = 0; p < pool_block_table_mirrors.size(); ++p) {
        TORCH_CHECK(pool_block_table_mirrors[p].defined(), "pool block table mirror must be defined");
        const auto& mirror = requireCudaInt32(pool_block_table_mirrors[p], "pool block table mirror");
        TORCH_CHECK(mirror.dim() == 2 && mirror.size(0) >= batch_size,
                    "pool block table mirror must be rank 2 covering the capture batch");
        TORCH_CHECK(p < block_tables.size() && mirror.size(1) >= block_tables[p].size(1),
                    "pool block table mirror columns must cover the source table");
        plan->pool_block_table_mirror[p] = mirror.data_ptr<int32_t>();
        plan->pool_mirror_stride[p]      = mirror.size(1);
        // The metadata mirror's width is the addressing domain the previous
        // Python path used for every clamp (phase2b BT_STRIDE, translate
        // max_num_blocks_per_req); keep the kernel on the same domain.
        if (static_cast<int32_t>(mirror.size(1)) < plan->pools[p].max_blocks) {
            plan->pools[p].max_blocks = static_cast<int32_t>(mirror.size(1));
        }
    }
    for (size_t s = 0; s < swa_slot_outputs.size(); ++s) {
        TORCH_CHECK(swa_slot_outputs[s].defined(), "swa slot output must be defined");
        const auto& out = requireCudaInt32(swa_slot_outputs[s], "swa slot output");
        plan->swa_slot_outputs[s]        = out.data_ptr<int32_t>();
        plan->swa_slot_pool_indices[s]   = static_cast<int32_t>(swa_slot_pool_indices[s]);
    }

    TORCH_CHECK(valid_requests.defined(), "valid_requests must be defined");
    TORCH_CHECK(valid_requests.is_cuda(), "valid_requests must be a CUDA tensor");
    TORCH_CHECK(valid_requests.scalar_type() == at::kBool, "valid_requests must be bool");
    TORCH_CHECK(valid_requests.is_contiguous(), "valid_requests must be contiguous");
    TORCH_CHECK(valid_requests.numel() >= batch_size, "valid_requests must cover the capture batch");
    TORCH_CHECK(swa_topk_length.defined() && swa_topk_length.is_cuda()
                    && swa_topk_length.scalar_type() == at::kInt && swa_topk_length.is_contiguous()
                    && swa_topk_length.numel() >= batch_size,
                "swa_topk_length must be a contiguous CUDA int32 [batch] tensor");
    plan->valid_requests           = valid_requests.data_ptr<bool>();
    plan->start_pos_out            = requireCudaInt32(start_pos_out, "start_pos").data_ptr<int32_t>();
    plan->position_ids             = requireCudaInt32(position_ids, "position_ids").data_ptr<int32_t>();
    if (position_ids_long.defined() && position_ids_long.numel() > 0) {
        plan->position_ids_long = requireCudaInt64(position_ids_long, "position_ids_long").data_ptr<int64_t>();
    }
    if (decode_seq_start_per_req.defined() && decode_seq_start_per_req.numel() > 0) {
        plan->decode_seq_start_per_req =
            requireCudaInt64(decode_seq_start_per_req, "decode_seq_start_per_req").data_ptr<int64_t>();
    }
    plan->slot_mapping_swa         = requireCudaInt32(slot_mapping_swa, "slot_mapping_swa").data_ptr<int32_t>();
    plan->topk_window_idxs         = requireCudaInt32(topk_window_idxs, "topk_window_idxs").data_ptr<int32_t>();
    plan->swa_abs_idx              = requireCudaInt32(swa_abs_idx, "swa_abs_idx").data_ptr<int32_t>();
    if (topk_buffer_compressed.defined() && topk_buffer_compressed.numel() > 0) {
        plan->topk_buffer_compressed = requireCudaInt32(topk_buffer_compressed, "topk_buffer_compressed")
                                           .data_ptr<int32_t>();
        TORCH_CHECK(topk_buffer_compressed.dim() == 3, "topk_buffer_compressed must be rank 3");
        plan->index_topk = static_cast<int32_t>(topk_buffer_compressed.size(2));
    }
    plan->swa_topk_length = requireCudaInt32(swa_topk_length, "swa_topk_length").data_ptr<int32_t>();

    TORCH_CHECK(q_len > 0, "q_len must be positive");
    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(window > 0, "window must be positive");
    plan->q_len      = static_cast<int32_t>(q_len);
    plan->batch_size = static_cast<int32_t>(batch_size);
    plan->window     = static_cast<int32_t>(window);
    plan->max_start  = static_cast<int32_t>(max_start);
    return plan;
}

}  // namespace rtp_llm
