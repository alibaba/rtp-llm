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

    /// @return each element describes one KV or scale sub-buffer.
    virtual std::vector<BlockInfo> convertIndexToBuffer(int                layer_id,
                                                        const std::string& cache_tag,
                                                        int                block_id,
                                                        int                partition_count = 1,
                                                        int                partition_id    = 0,
                                                        size_t             module_id      = 0) const = 0;

    /// @brief Owning module of a globally unique group tag (0 = main, 1..N = MTP).
    virtual size_t moduleOfTag(const std::string& tag) const = 0;

    /// @brief Whether the tag's owning module has this module-local layer id.
    virtual bool tagOwnsLayer(const std::string& tag, int layer_id) const = 0;

    virtual std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const = 0;
};

}  // namespace rtp_llm
