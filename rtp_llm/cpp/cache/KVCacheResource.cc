#include "rtp_llm/cpp/cache/CacheManager.h"

using namespace std;

namespace rtp_llm {

void KVCacheResource::clear() {
    block_id.clear();
}

KVCacheResource KVCacheResource::clone(std::shared_ptr<CacheManager>& cache_manager) const {
    if (!block_id.empty()) {
        cache_manager->incrRefCounter(block_id);
    }
    return *this;
}

}  // namespace rtp_llm