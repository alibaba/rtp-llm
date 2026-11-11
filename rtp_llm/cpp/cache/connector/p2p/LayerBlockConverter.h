#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

/// @brief 封装 CoordinatorCacheManager 的 convertIndexToBuffer 接口，将 block_id 转换为 BlockInfo
class LayerBlockConverter {
public:
    virtual ~LayerBlockConverter() = default;

    /// @return each element describes one KV or scale sub-buffer.
    virtual std::vector<BlockInfo> convertIndexToBuffer(int                layer_id,
                                                        const std::string& tag,
                                                        int                block_id,
                                                        int                partition_count,
                                                        int                partition_id,
                                                        size_t             module_id = 0) const = 0;

    /// @brief Owning module of a globally unique group tag (0 = main, 1..N = MTP).
    virtual size_t moduleOfTag(const std::string& tag) const = 0;

    /// @brief Whether the tag's owning module has this module-local layer id.
    virtual bool tagOwnsLayer(const std::string& tag, int layer_id) const = 0;

    /// @brief 返回所有需要注册 RDMA MR 的 buffer 及其对齐大小
    virtual std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const = 0;
};

}  // namespace rtp_llm
