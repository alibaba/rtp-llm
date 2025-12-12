#pragma once

#include <memory>
#include <torch/torch.h>
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/devices/OpData.h"

namespace rtp_llm {

struct TRTAttn: public ParamsBase {
    KVBlockArray  kv_block_array;
    torch::Tensor kv_cache_offset;
    torch::Tensor kv_cache_offset_h;

    torch::Tensor padding_offset;
    torch::Tensor cu_seqlens;
    torch::Tensor cu_seqlens_without_prefix;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor input_lengths;
    torch::Tensor prefix_lengths;
    torch::Tensor sequence_lengths;
    torch::Tensor cu_mask_rows;
    int           max_seq_len;
    int           max_prefix_length;
    int           context_total_kv_length;
    bool          decode_plan;

    DataType attn_type;

    static void setKvCache(KVBlockArray& kv_block_array, const KvCacheInfo& kv_cache);
};

using TRTAttnPtr = std::shared_ptr<TRTAttn>;

}  // namespace rtp_llm
