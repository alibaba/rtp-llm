#pragma once

#include <cstdint>
#include "deep_ep.hpp"

namespace deep_ep {
inline Buffer* createDeepEPBuffer(int     world_rank,
                           int     local_world_size,
                           int     world_size,
                           int64_t num_nvl_bytes,
                           int64_t num_rdma_bytes,
                           bool    low_latency_mode,
                           bool    use_nvshmem_transport = true) {
    return new Buffer(world_rank, world_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode);
}
}  // namespace deep_ep