#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"

namespace rtp_llm {

class MockKVCacheConnectorReadWriteContext: public KVCacheConnectorReadWriteContext {
public:
    MockKVCacheConnectorReadWriteContext()           = default;
    ~MockKVCacheConnectorReadWriteContext() override = default;

public:
    MOCK_METHOD(const KVCacheResource&, kvCacheResource, (), (const, override));
    MOCK_METHOD(bool, enableMemoryCache, (), (const, override));
};

}  // namespace rtp_llm
