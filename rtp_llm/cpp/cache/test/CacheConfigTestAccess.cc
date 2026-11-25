#include "rtp_llm/cpp/cache/test/CacheConfigTestAccess.h"

#include <utility>

namespace rtp_llm::test {

void CacheConfigTestAccess::addMtpModule(CacheConfig&                         config,
                                         std::shared_ptr<const CacheTopology> topology,
                                         CacheModuleMetadata                  metadata) {
    config.addMtpModule(std::move(topology), metadata);
}

}  // namespace rtp_llm::test
