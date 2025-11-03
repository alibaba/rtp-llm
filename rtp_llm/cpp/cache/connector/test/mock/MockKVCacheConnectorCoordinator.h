#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

namespace rtp_llm {

class MockKVCacheConnectorCoordinator: public KVCacheConnectorCoordinator {
public:
    MockKVCacheConnectorCoordinator(const CacheConfig&                       cache_config,
                                    const KVCacheConfig&                     kv_cache_config,
                                    const RuntimeConfig&                     runtime_config,
                                    const std::shared_ptr<KVCacheAllocator>& allocator,
                                    rtp_llm::DeviceBase*                     device,
                                    const kmonitor::MetricsReporterPtr&      metrics_reporter = nullptr):
        KVCacheConnectorCoordinator(
            cache_config, kv_cache_config, runtime_config, {}, {}, allocator, device, metrics_reporter) {}
    ~MockKVCacheConnectorCoordinator() override = default;

public:
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncRead,
                (const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncWrite,
                (const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncWriteByLayer,
                (int layer_id, const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context),
                (override));
    MOCK_METHOD(bool, executeFunction, (const FunctionRequestPB& request, FunctionResponsePB& response), (override));
};

}  // namespace rtp_llm
