#include <cuda_runtime.h>
#include "test_api.h"
#include "3rdparty/trt_beam_search/beamSearchKernelsTemplate.h"

// The reference bodies are verbatim Round 2 sources. Only their symbols are
// renamed, outside the frozen files, to coexist with the production kernels.
#define addCumLogProbsKernel frozenAddCumLogProbsKernel
#define launchAddCumLogProbs frozenLaunchAddCumLogProbs
namespace tensorrt_llm::kernels {
#include "frozen/addCumLogProbs.cuh"
}
#undef addCumLogProbsKernel
#define beamStage1Kernel frozenBeamStage1Kernel
#define beamStage2Kernel frozenBeamStage2Kernel
#define BeamStage3KernelSmem FrozenBeamStage3KernelSmem
#define beamStage3Kernel frozenBeamStage3Kernel
#define beamSearchKernelLauncher frozenBeamSearchKernelLauncher
#define MAX_BLOCK_SIZE FROZEN_MAX_BLOCK_SIZE
#define MIN_BLOCK_SIZE FROZEN_MIN_BLOCK_SIZE
#undef INSTANTIATE_BEAM_SEARCH
#include "frozen/beamSearchKernelsTemplate.h"
#undef addCumLogProbsKernel
#undef launchAddCumLogProbs
#undef beamStage1Kernel
#undef beamStage2Kernel
#undef BeamStage3KernelSmem
#undef beamStage3Kernel
#undef beamSearchKernelLauncher
#undef MAX_BLOCK_SIZE
#undef MIN_BLOCK_SIZE
#undef INSTANTIATE_BEAM_SEARCH

namespace tensorrt_llm::kernels {
template <typename T, bool V2>
void frozenInvokeTopkBeamSearch(T const* values, T const* bias, void* workspace, BeamHypotheses& bh,
    cudaStream_t stream)
{
    int const padded = padToNextPowerOfTwo(std::max(bh.nBeamWidthIn, bh.nBeamWidthOut));
#define FROZEN_CASE(P) case P: frozenBeamSearchKernelLauncher<T, P, V2>(values, bias, workspace, bh, stream); break
    switch (padded) {
    case 1:
    case 2:
    FROZEN_CASE(4);
    FROZEN_CASE(8);
    FROZEN_CASE(128);
    FROZEN_CASE(256);
    FROZEN_CASE(512);
    FROZEN_CASE(1024);
    default: throw std::runtime_error("test frozen sampler: unsupported padded beam width");
    }
#undef FROZEN_CASE
}
template void frozenInvokeTopkBeamSearch<float, true>(float const*, float const*, void*, BeamHypotheses&, cudaStream_t);
template void frozenInvokeTopkBeamSearch<float, false>(float const*, float const*, void*, BeamHypotheses&, cudaStream_t);
template void frozenInvokeTopkBeamSearch<half, true>(half const*, half const*, void*, BeamHypotheses&, cudaStream_t);
template void frozenInvokeTopkBeamSearch<half, false>(half const*, half const*, void*, BeamHypotheses&, cudaStream_t);

template <typename T, int P, bool V2>
void stage3(int const* ids, T const* scores, float* scratch, BeamHypotheses bh, bool reference, cudaStream_t stream)
{
    constexpr int threads = P < 32 ? 32 : P;
    size_t const dynamic = V2 ? 0 : sizeof(T) * bh.nBeamWidthIn * bh.nBeamWidthOut * 2;
    if (reference) {
        frozenBeamStage3Kernel<T, P, threads, true, V2><<<bh.nBatchSize, threads, dynamic, stream>>>(
            ids, scores, scratch, bh);
    } else {
        beamStage3Kernel<T, P, threads, true, V2><<<bh.nBatchSize, threads, dynamic, stream>>>(
            ids, scores, scratch, bh);
    }
}

template <typename T, bool V2>
void stage3Dispatch(int const* ids, T const* scores, float* scratch, BeamHypotheses bh, bool reference,
    cudaStream_t stream)
{
#define STAGE_CASE(P) case P: stage3<T, P, V2>(ids, scores, scratch, bh, reference, stream); break
    switch (padToNextPowerOfTwo(std::max(bh.nBeamWidthIn, bh.nBeamWidthOut))) {
    case 1:
    case 2:
    STAGE_CASE(4);
    STAGE_CASE(8);
    STAGE_CASE(128);
    STAGE_CASE(256);
    STAGE_CASE(512);
    STAGE_CASE(1024);
    default: throw std::runtime_error("test stage3: unsupported padded beam width");
    }
#undef STAGE_CASE
}

void testLaunchStage3(void const* ids, void const* scores, float* scratch, BeamHypotheses bh,
    bool fp16, bool v2, bool reference, cudaStream_t stream)
{
#define DISPATCH(T, V) stage3Dispatch<T, V>(static_cast<int const*>(ids), static_cast<T const*>(scores), scratch, bh, reference, stream)
    if (fp16) {
        if (v2) { DISPATCH(half, true); } else { DISPATCH(half, false); }
    } else {
        if (v2) { DISPATCH(float, true); } else { DISPATCH(float, false); }
    }
#undef DISPATCH
    check_cuda_error();
}
} // namespace tensorrt_llm::kernels
