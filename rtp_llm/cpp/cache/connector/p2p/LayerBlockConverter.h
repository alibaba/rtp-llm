#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include <string>
#include <cstddef>
#include <utility>
#include <vector>

namespace rtp_llm {

class LayerBlockConverter {
public:
    virtual ~LayerBlockConverter() = default;

    virtual std::vector<BlockInfo> convertIndexToBuffer(int                layer_id,
                                                        const std::string& cache_tag,
                                                        int                block_id,
                                                        int                partition_count = 1,
                                                        int                partition_id    = 0) const = 0;

    virtual std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const = 0;
};

}  // namespace rtp_llm
