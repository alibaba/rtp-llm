#include "rtp_llm/cpp/disaggregate/cache_store/ThreadLocalScratch.h"

#include <map>

namespace rtp_llm {

StagedMemoryCopyScratch& threadLocalScratch(int device_index) {
    // No destructor cleanup — releaseStagedMemoryCopyScratch lives in
    // no_block_copy which would create a link-time dependency from
    // cache_store_base. At thread exit the CUDA context is typically
    // already torn down, so manual free is unnecessary.
    static thread_local std::map<int, StagedMemoryCopyScratch> tls;
    return tls[device_index];
}

}  // namespace rtp_llm
