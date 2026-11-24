#pragma once

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"

namespace rtp_llm::rdma_transport::detail {

// Pool quota and physical GPU OOM both permit pool shrink, but only physical
// GPU OOM can benefit from Torch reclamation. Callbacks run after the preceding
// pool operation has returned; reclaim must not hold a Barex pool lock.
template<typename Alloc, typename Shrink, typename Reclaim>
::accl::barex::BarexResult allocWithOomShrinkRetry(Alloc&& alloc, Shrink&& shrink, Reclaim&& reclaim) {
    using ::accl::barex::BAREX_ERR_OOM;
    using ::accl::barex::BAREX_ERR_GPU_OOM;
    const auto is_oom = [](auto result) { return result == BAREX_ERR_OOM || result == BAREX_ERR_GPU_OOM; };
    auto       result = alloc();
    if (!is_oom(result)) {
        return result;
    }
    shrink(0.5F);
    result = alloc();
    if (!is_oom(result)) {
        return result;
    }
    shrink(1.0F);
    result = alloc();
    if (result != BAREX_ERR_GPU_OOM || !reclaim()) {
        return result;
    }
    return alloc();
}

}  // namespace rtp_llm::rdma_transport::detail
