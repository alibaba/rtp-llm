#pragma once

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm::test {

// Write one KV block (optionally per-layer) from host/device tensors for test
inline bool writeKVBlockForTest(KVCacheManager&      manager,
                                int                  block_index,
                                int                  layer_id,
                                const torch::Tensor& k_buffer,
                                const torch::Tensor& v_buffer) {
    // Basic size/type validation to prevent out-of-bounds copy
    const auto& spec             = manager.cacheConfig().topology().soleGroupForLayer(layer_id).spec;
    size_t      expected_k_bytes = spec->k_block_size_bytes();
    size_t      expected_v_bytes = spec->v_block_size_bytes();
    size_t      src_k_bytes      = k_buffer.nbytes();
    size_t      src_v_bytes      = v_buffer.nbytes();
    if (src_k_bytes < expected_k_bytes || src_v_bytes < expected_v_bytes) {
        RTP_LLM_LOG_ERROR("writeKVBlockForTest src bytes too small: k[%zu]<[%zu] or v[%zu]<[%zu]",
                          src_k_bytes,
                          expected_k_bytes,
                          src_v_bytes,
                          expected_v_bytes);
        return false;
    }

    auto dst = manager.convertIndexToBuffer(block_index, layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        !dst.empty(), "convertIndexToBuffer returned empty for layer %d, block %d", layer_id, block_index);
    if (!dst[0].addr) {
        RTP_LLM_LOG_ERROR("convertIndexToBuffer returned null for layer %d, block %d", layer_id, block_index);
        return false;
    }

    auto copyFunc = [&](const torch::Tensor& src_tensor,
                        const BlockInfo&     dst_block,
                        size_t               dst_byte_offset,
                        size_t               copy_bytes) -> bool {
        const size_t dst_bytes = dst_block.size_bytes;
        if (dst_bytes < dst_byte_offset + copy_bytes) {
            RTP_LLM_LOG_ERROR(
                "dst block bytes[%zu] < dst_offset[%zu] + copy bytes[%zu] in writeKVBlockForTest(layer=%d)",
                dst_bytes,
                dst_byte_offset,
                copy_bytes,
                layer_id);
            return false;
        }

        auto* dst_ptr    = static_cast<char*>(dst_block.addr) + dst_byte_offset;
        auto  dst_device = dst_block.is_cuda ? torch::kCUDA : torch::kCPU;
        auto  src_device = src_tensor.is_cuda() ? torch::kCUDA : torch::kCPU;
        auto  dst_t      = torch::from_blob(
            dst_ptr, {(int64_t)copy_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(dst_device));
        auto src_t = torch::from_blob(src_tensor.data_ptr(),
                                      {(int64_t)copy_bytes},
                                      torch::TensorOptions().dtype(torch::kUInt8).device(src_device));
        dst_t.copy_(src_t);
        return true;
    };

    if (!copyFunc(k_buffer, dst[0], 0, expected_k_bytes)) {
        return false;
    }

    if (!copyFunc(v_buffer, dst[0], expected_k_bytes, expected_v_bytes)) {
        return false;
    }

    cudaSyncAndCheck();
    return true;
}

inline bool writeKVBlockForTest(KVCacheManager&      manager,
                                int                  block_index,
                                const torch::Tensor& k_buffer,
                                const torch::Tensor& v_buffer) {
    if (block_index < 0 || block_index >= manager.cacheConfig().block_num) {
        RTP_LLM_LOG_WARNING(
            "Invalid block_index: %d, valid range: [0, %d)", block_index, manager.cacheConfig().block_num);
        return false;
    }

    bool all_success = true;
    for (int layer_id = 0; layer_id < manager.cacheConfig().layer_num; ++layer_id) {
        all_success = writeKVBlockForTest(manager, block_index, layer_id, k_buffer, v_buffer) && all_success;
    }
    return all_success;
}

}  // namespace rtp_llm::test
