#pragma once
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/devices/graph_common/GraphCommonTypes.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>

using namespace torch_ext;

namespace rtp_llm {}  // namespace rtp_llm

class CudaGraphStreamLife {
public:
    CudaGraphStreamLife(at::cuda::CUDAStream capture_stream):
        origin_stream_(at::cuda::getCurrentCUDAStream(at::cuda::current_device())) {
        // Set `capture_stream` for capture. All kernels should use this stream while capturing.
        at::cuda::setCurrentCUDAStream(capture_stream);
        RTP_LLM_LOG_INFO("Set Cuda Stream: capture_stream -> %d, origin_stream -> %d",
                         capture_stream.stream(),
                         origin_stream_.stream());
    }
    ~CudaGraphStreamLife() {
        at::cuda::setCurrentCUDAStream(origin_stream_);
    }

private:
    at::cuda::CUDAStream origin_stream_;
};

// RAII guard for CUDA graph capture state
class CudaGraphCaptureGuard {
public:
    CudaGraphCaptureGuard() {
        rtp_llm::CaptureCheck::in_cuda_graph_capture = true;
    }

    ~CudaGraphCaptureGuard() {
        rtp_llm::CaptureCheck::in_cuda_graph_capture = false;
    }

    // Non-copyable, non-movable
    CudaGraphCaptureGuard(const CudaGraphCaptureGuard&)            = delete;
    CudaGraphCaptureGuard& operator=(const CudaGraphCaptureGuard&) = delete;
    CudaGraphCaptureGuard(CudaGraphCaptureGuard&&)                 = delete;
    CudaGraphCaptureGuard& operator=(CudaGraphCaptureGuard&&)      = delete;
};

namespace rtp_llm {

using CudaGraphState = GraphExecutionState;

}  // namespace rtp_llm
