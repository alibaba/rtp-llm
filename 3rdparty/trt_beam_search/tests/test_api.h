#pragma once

#include "3rdparty/trt_beam_search/beamSearchKernels.h"

namespace tensorrt_llm::kernels {
template <typename T, bool IS_V2>
void frozenInvokeTopkBeamSearch(T const*, T const*, void*, BeamHypotheses&, cudaStream_t);

template <typename T>
void frozenLaunchAddCumLogProbs(T*, float const*, FinishedState const*, int const*, float const*,
    runtime::SizeType32 const*, size_t, size_t, size_t, size_t, int, cudaStream_t);

void testLaunchStage3(void const* ids, void const* scores, float* scratch, BeamHypotheses bh,
    bool fp16, bool v2, bool reference, cudaStream_t stream);
} // namespace tensorrt_llm::kernels
