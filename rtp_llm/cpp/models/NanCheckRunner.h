#pragma once

#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

#include <torch/extension.h>

namespace rtp_llm {

class KvCacheNanCheckRunner {
public:
    static bool run(const AttentionConfigs& attention_config,
                    DataType                cache_dtype,
                    size_t                  cache_element_size,
                    size_t                  layer_num,
                    const torch::Tensor&    layer_base_addr_buffer,
                    const GptModelInputs&   inputs,
                    torch::Tensor&          nan_flag);
};

}  // namespace rtp_llm
