#pragma once

#include <memory>

#include "rtp_llm/cpp/cache_new/KVCacheReaderWriter.h"
#include "rtp_llm/cpp/cache_new/MemoryKVCacheConnector.h"
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
    void asyncReadByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, const CallBack& callback) override;

    void asyncWrite(const BatchKVCacheResourcePtr& resource, const CallBack& callback) override;
    void asyncWriteByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, const CallBack& callback) override;

private:
    void connectorWrite(const BatchKVCacheResourcePtr& resource, int layer_id, const CallBack& callback);
    bool match(int64_t key) const;

private:
    // 一个connector对应一个group
    std::map<int, std::shared_ptr<MemoryKVCacheConnector>> layer_to_connector_map_;
    std::vector<std::shared_ptr<MemoryKVCacheConnector>>   connectors_;
    std::shared_ptr<KVCacheAllocator>                      allocator_;
};

}  // namespace rtp_llm
