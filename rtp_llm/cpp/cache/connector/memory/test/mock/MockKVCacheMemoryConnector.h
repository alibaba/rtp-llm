#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

namespace rtp_llm {

class MockKVCacheMemoryConnector: public KVCacheMemoryConnector {
public:
    MockKVCacheMemoryConnector(const CacheConfig&                       cache_config,
                               const KVCacheConfig&                     kv_cache_config,
                               const std::shared_ptr<KVCacheAllocator>& allocator,
                               rtp_llm::DeviceBase*                     device,
                               const std::vector<std::string>&          worker_addrs,
                               const kmonitor::MetricsReporterPtr&      metrics_reporter):
        KVCacheMemoryConnector(cache_config, kv_cache_config, allocator, device, worker_addrs, metrics_reporter) {}
    ~MockKVCacheMemoryConnector() override = default;

public:
    MOCK_METHOD(std::shared_ptr<AsyncMatchContext>,
                asyncMatch,
                (const std::shared_ptr<KVCacheResource>& resource, const std::shared_ptr<Meta>& meta),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncRead,
                (const std::shared_ptr<KVCacheResource>&   resource,
                 const std::shared_ptr<Meta>&              meta,
                 const std::shared_ptr<AsyncMatchContext>& match_context,
                 int                                       start_read_block_index,
                 int                                       read_block_num),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncWrite,
                (const std::shared_ptr<KVCacheResource>& resource, const std::shared_ptr<Meta>& meta),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncWriteByLayer,
                (int layer_id, const std::shared_ptr<KVCacheResource>& resource, const std::shared_ptr<Meta>& meta),
                (override));
    MOCK_METHOD(bool,
                copyCache,
                (const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response),
                (override));
};

}  // namespace rtp_llm
