#pragma once

#include <memory>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"

namespace rtp_llm {

/// @brief 单一类型KV缓存分配器
/// 只负责一个KVCacheGroup的分配
class SingleTypeKVCacheAllocator : public KVCacheAllocator {
public:
    /// @brief 构造函数
    /// @param group 要管理的KV缓存组
    explicit SingleTypeKVCacheAllocator(KVCacheGroupPtr group = nullptr);
    
    ~SingleTypeKVCacheAllocator() override = default;

    /// @brief 设置要管理的KV缓存组
    /// @param group KV缓存组
    void setKVCacheGroup(KVCacheGroupPtr group);

    /// @brief 分配KV缓存
    /// @param request 分配请求
    /// @return 分配结果
    AllocResult allocate(const AllocRequest& request) override;

    /// @brief 释放KV缓存
    /// @param request 释放请求
    /// @return 释放结果
    FreeResult free(const FreeRequest& request) override;

    /// @brief 获取层缓存基地址
    /// @param layer_id 层ID
    /// @param block_id 块ID
    /// @return K和V缓存地址对
    std::pair<void*, void*> layerCacheBase(int layer_id, int block_id) const override;

private:
    KVCacheGroupPtr single_group_;  ///< 管理的单一组
};

using SingleTypeKVCacheAllocatorPtr = std::shared_ptr<SingleTypeKVCacheAllocator>;

}  // namespace rtp_llm
