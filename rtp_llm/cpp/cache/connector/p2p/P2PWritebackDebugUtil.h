#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

bool pdKvWritebackChecksumDebugEnabled();

void logPdKvWritebackChecksum(const std::string&                          stage,
                              int64_t                                     request_id,
                              const std::string&                          unique_key,
                              const std::shared_ptr<LayerBlockConverter>& converter,
                              const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                              int                                         partition_count,
                              int                                         partition_id);

void logPdKvWritebackChecksum(const std::string&                                    stage,
                              int64_t                                               request_id,
                              const std::string&                                    unique_key,
                              const std::shared_ptr<LayerBlockConverter>&           converter,
                              const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                              int                                                   partition_count);

}  // namespace rtp_llm
