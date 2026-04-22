#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"

namespace rtp_llm {

class MockKVCacheConnectorReadWriteContext: public KVCacheConnectorReadWriteContext {
public:
    MockKVCacheConnectorReadWriteContext() {
        ON_CALL(*this, kvCacheResource()).WillByDefault(testing::ReturnRef(default_resource_));
        ON_CALL(*this, cacheKeys()).WillByDefault(testing::ReturnRef(default_cache_keys_));
    }
    ~MockKVCacheConnectorReadWriteContext() override = default;

public:
    MOCK_METHOD(const KVCacheResource&, kvCacheResource, (), (const));
    MOCK_METHOD(const CacheKeysType&, cacheKeys, (), (const));
    MOCK_METHOD(const std::shared_ptr<Meta>&, meta, (), (const, override));

    const ModelKVResources& modelKVResources() const override {
        cached_model_resources_.model_resources.assign(1, kvCacheResource());
        cached_model_resources_.cache_keys = cacheKeys();
        return cached_model_resources_;
    }

private:
    KVCacheResource          default_resource_;
    CacheKeysType            default_cache_keys_;
    mutable ModelKVResources cached_model_resources_;
};

}  // namespace rtp_llm
