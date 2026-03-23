#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

#include <vector>

namespace rtp_llm {

struct NanCheckStaging {
    std::vector<BufferPtr> buffers;

    void hold(const BufferPtr& buffer) {
        if (buffer) {
            buffers.push_back(buffer);
        }
    }
};

class KvCacheNanCheckRunner {
public:
    static bool run(DeviceBase*                 device,
                    const AttentionConfigs&     attention_config,
                    DataType                    cache_dtype,
                    size_t                      cache_element_size,
                    size_t                      layer_num,
                    const BufferPtr&            layer_base_addr_buffer,
                    const GptModelInputs&       inputs,
                    BufferPtr&                  nan_flag,
                    NanCheckStaging&            staging);
};

}  // namespace rtp_llm
