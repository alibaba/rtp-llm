#pragma once
#include <cute/arch/cluster_sm90.hpp>
namespace rtp_llm::engram::comm {
CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

}  // namespace rtp_llm::engram::comm
