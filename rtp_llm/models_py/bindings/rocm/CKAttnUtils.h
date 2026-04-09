#pragma once

#include "rtp_llm/models_py/bindings/common/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

namespace rtp_llm {

struct CKAttn {
    KVBlockArray  kv_block_array;
    torch::Tensor kv_cache_offset;

    torch::Tensor kv_cache_block_id_device;
    torch::Tensor kv_cache_kernel_block_id_device;

    torch::Tensor prefix_lengths;
    torch::Tensor cu_seqlens;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor input_lengths;
    torch::Tensor sequence_lengths;
    torch::Tensor padding_offset;
    int           max_seq_len;
    bool          decode_plan;

    DataType attn_type;

    static void setKvCache(KVBlockArray& kv_block_array, const KvCacheInfo& kv_cache) {
        kv_block_array.mPrimaryPoolPtr = kv_cache.kv_cache_buffer.data_ptr();
        if (kv_cache.kv_scale_buffer.defined() && kv_cache.kv_scale_buffer.numel() > 0) {
            kv_block_array.scale = kv_cache.kv_scale_buffer.data_ptr();
        }
    }
};

using CKAttnPtr = std::shared_ptr<CKAttn>;

inline ParamsPtr PrepareCKAttn(const AttentionConfigs& configs,
                               const torch::Tensor&    kv_cache_block_id,
                               int                     batch_size,
                               bool                    use_fp8_fmha) {
    if (batch_size <= 0 || !kv_cache_block_id.defined() || kv_cache_block_id.numel() == 0) {
        return nullptr;
    }
    auto            ck_attn    = std::make_shared<CKAttn>();
    KvCacheDataType cache_type = KvCacheDataType::BASE;
    if (configs.kv_cache_dtype == KvCacheDataType::INT8) {
        RTP_LLM_LOG_DEBUG("now use kv_cache int8");
        cache_type = KvCacheDataType::INT8;
    }
    const auto max_blocks_per_batch = kv_cache_block_id.size(1);
    auto const elemSize             = use_fp8_fmha ? sizeof(int8_t) : 2;  // 2 for kv cache fp16

    ck_attn->kv_cache_offset           = torch::empty({(int64_t)batch_size, 1, 2, (int64_t)max_blocks_per_batch},
                                            torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    ck_attn->kv_block_array            = KVBlockArray(batch_size,
                                           max_blocks_per_batch,
                                           configs.tokens_per_block,
                                           configs.kv_head_num * configs.size_per_head * elemSize,
                                           0,
                                           0,
                                           nullptr,
                                           nullptr,
                                           (rtp_llm::KVCacheIndex*)ck_attn->kv_cache_offset.data_ptr<int>());
    ck_attn->kv_block_array.cache_type = cache_type;
    ck_attn->kv_block_array.mScaleBytesPerBlock = configs.tokens_per_block * configs.kv_head_num * sizeof(float);
    hipStream_t stream                          = at::hip::getCurrentHIPStream().stream();
    invokeConvertOffsetToBlockArrayData(ck_attn->kv_cache_offset.data_ptr<int>(),
                                        kv_cache_block_id.data_ptr<int>(),
                                        batch_size,
                                        max_blocks_per_batch,
                                        stream);
    return ck_attn;
}

}  // namespace rtp_llm
