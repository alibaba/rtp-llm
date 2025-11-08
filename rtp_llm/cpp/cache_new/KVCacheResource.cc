#include "rtp_llm/cpp/cache_new/KVCacheResource.h"
#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"

using namespace std;

namespace rtp_llm {

void KVCacheResource::clear() {
    block_id.clear();
}

KVCacheResource KVCacheResource::clone(KVCacheGroup& kv_cache_group) const {
    if (!block_id.empty()) {
        kv_cache_group.reference(block_id);
    }
    return *this;
}

}  // namespace rtp_llm

