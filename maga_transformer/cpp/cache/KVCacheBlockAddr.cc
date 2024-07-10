#include "maga_transformer/cpp/cache/CacheManager.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

void KVCacheBlockAddr::clear() {
    offset.clear();
}

KVCacheBlockAddr KVCacheBlockAddr::clone(std::shared_ptr<CacheManager>& cache_manager) {
    if (!offset.empty()) {
        cache_manager->incrBlockRefCounter(offset);
    }
    return *this;
}

}  // namespace rtp_llm
