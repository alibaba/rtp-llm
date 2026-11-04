#include "greedy_rejection.h"

namespace {

__global__ void greedyRejection(const int32_t* draft_ids,
                               const int32_t* target_ids,
                               int32_t*       output_ids,
                               int32_t*       accepted,
                               int            proposals,
                               int            target_stride) {
    const int lane = threadIdx.x;
    const int row  = blockIdx.x;
    int target = 0;
    if (lane <= proposals) {
        target = target_ids[(int64_t(row) * (proposals + 1) + lane) * target_stride + target_stride - 1];
    }
    const bool mismatch = lane < proposals && draft_ids[int64_t(row) * proposals + lane] != target;
    const unsigned mask = __ballot_sync(0xffffffffu, mismatch);
    const int stop = mask ? __ffs(mask) - 1 : proposals;
    if (lane <= proposals) {
        output_ids[int64_t(row) * (proposals + 1) + lane] = lane <= stop ? target : 0;
    }
    if (lane == 0) {
        accepted[row] = stop + 1;
    }
}

}  // namespace

extern "C" cudaError_t rtpLlmGreedyRejection(const int32_t* draft_ids,
                                            const int32_t* target_ids,
                                            int32_t*       output_ids,
                                            int32_t*       accepted,
                                            int            batch,
                                            int            proposals,
                                            int            target_stride,
                                            cudaStream_t   stream) {
    if (batch < 0 || proposals < 0 || proposals >= 32 || target_stride <= 0) {
        return cudaErrorInvalidValue;
    }
    if (batch) {
        greedyRejection<<<batch, 32, 0, stream>>>(
            draft_ids, target_ids, output_ids, accepted, proposals, target_stride);
        return cudaGetLastError();
    }
    return cudaSuccess;
}
