#pragma once

#include <memory>
#include <map>

#include "rtp_llm/cpp/cache_new/KVCacheReaderWriter.h"
#include "rtp_llm/cpp/cache_new/MemoryKVCacheConnector.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// A simple reader/writer backed by in-memory connector.
class MemoryKVCacheReaderWriter final: public KVCacheReaderWriter {
public:
    MemoryKVCacheReaderWriter()           = default;
    ~MemoryKVCacheReaderWriter() override = default;

public:
    bool init() override;

    void asyncRead(const BatchKVCacheResourcePtr& resource, const CallBack& callback) override;
    // void asyncReadByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, const CallBack& callback) override;

    void asyncWrite(const BatchKVCacheResourcePtr& resource, const CallBack& callback) override;
    // void asyncWriteByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, const CallBack& callback) override;

private:
    bool match(int64_t key) const;
    KVCacheConnector::Buffers cacheKeyWiseLayout(int64_t cache_key, const BatchKVCacheResourcePtr& resource) const;
    void writeBuffers(const KVCacheConnector::Buffers& buffers, const CallBack& callback) const;
    // void asyncWriteLayer(int64_t cache_key, int32_t layer_idx, const std::shared_ptr<BlockIds>& block_indices_ptr, const CallBack& callback);
    // void asyncReadLayer(int64_t cache_key, int32_t layer_idx, const std::shared_ptr<BlockIds>& block_indices_ptr, const CallBack& callback);
    // Buffers layerWiseLayout(const BatchKVCacheResourcePtr& resource) const;
    // Buffers blockWiseLayout(const BatchKVCacheResourcePtr& resource) const;

private:
    // group_id->connector, 一个connector对应一个group
    std::map<int, std::shared_ptr<MemoryKVCacheConnector>> group_to_connector_;

    // group_id->group_type, 不同的group使用不同的读写方式, 决定是put还是prefixPut
    enum class KVCacheGroupType {
        FULL = 0,
        LINEAR = 1,
    };
    std::map<int, KVCacheGroupType> group_type_map_;

    // layer_id->connector, 表示该层的cache使用哪个connector读写
    std::map<int, std::shared_ptr<MemoryKVCacheConnector>> layer_to_connector_;

    // layer_id->group_id, 表示这个层属于哪个group
    std::map<int, int> layer_to_group_;
    
    std::shared_ptr<KVCacheAllocator>                      allocator_;
};

}  // namespace rtp_llm
