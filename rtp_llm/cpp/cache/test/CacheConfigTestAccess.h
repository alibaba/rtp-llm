#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm::test {

class CacheConfigTestAccess {
public:
    static void
    addMtpModule(CacheConfig& config, std::shared_ptr<const CacheTopology> topology, CacheModuleMetadata metadata);
};

}  // namespace rtp_llm::test
