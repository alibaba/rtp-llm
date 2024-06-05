#include "maga_transformer/cpp/cache/CacheManager.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

void KVCacheBlockAddr::clear() {
    k_ptr.clear();
    v_ptr.clear();
    k_scale_ptr.clear();
    v_scale_ptr.clear();
}

KVCacheBlockAddr KVCacheBlockAddr::clone(std::shared_ptr<CacheManager>& cache_manager) {
    if (!k_ptr.empty()) {
        cache_manager->incrBlockRefCounter(k_ptr[0]);
    }
    return *this;
}

}  // namespace rtp_llm
