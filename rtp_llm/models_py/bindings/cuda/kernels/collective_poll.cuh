#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rtp_llm {

// Only sample the timer while a peer has not published its payload. Successful
// polls keep the normal fast path; a failed peer cannot pin a GPU indefinitely.
__device__ __forceinline__ void check_collective_poll_timeout(uint64_t& start, uint32_t& polls) {
    if ((polls++ & 1023u) != 0)
        return;
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (start == 0) {
        start = now;
    } else if (now - start > 30000000000ull) {
        printf("RTP collective peer timeout: block=%u thread=%u\n", blockIdx.x, threadIdx.x);
        asm volatile("trap;" ::: "memory");
    }
}

}  // namespace rtp_llm
