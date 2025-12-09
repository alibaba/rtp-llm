#pragma once
#include "rtp_llm/cpp/devices/GraphUtils.h"
#include <ATen/cuda/CUDAGraph.h>

namespace rtp_llm {

// Backward compatibility typedefs (for code that still uses old names)
using CudaGraphState        = GraphState;
using CudaGraphCaptureGuard = GraphCaptureGuard;

}  // namespace rtp_llm
