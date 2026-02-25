#pragma once
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/devices/graph_common/GraphCommonTypes.h"
#include "rtp_llm/cpp/rocm/hip_capture_check.h"
#include <ATen/hip/HIPGeneratorImpl.h>
#include <ATen/hip/HIPGraph.h>

using namespace torch_ext;

namespace rtp_llm {}  // namespace rtp_llm

class HipGraphStreamLife {
public:
    HipGraphStreamLife(at::hip::HIPStream capture_stream):
        origin_stream_(at::hip::getCurrentHIPStream(at::hip::current_device())) {
        // Set `capture_stream` for capture. All kernels should use this stream while capturing.
        at::hip::setCurrentHIPStream(capture_stream);
        RTP_LLM_LOG_INFO("Set HIP Stream: capture_stream -> %d, origin_stream -> %d",
                         capture_stream.stream(),
                         origin_stream_.stream());
    }
    ~HipGraphStreamLife() {
        at::hip::setCurrentHIPStream(origin_stream_);
    }

private:
    at::hip::HIPStream origin_stream_;
};

// RAII guard for HIP graph capture state
class HipGraphCaptureGuard {
public:
    HipGraphCaptureGuard() {
        rtp_llm::rocm::CaptureCheck::in_hip_graph_capture = true;
    }

    ~HipGraphCaptureGuard() {
        rtp_llm::rocm::CaptureCheck::in_hip_graph_capture = false;
    }

    // Non-copyable, non-movable
    HipGraphCaptureGuard(const HipGraphCaptureGuard&)            = delete;
    HipGraphCaptureGuard& operator=(const HipGraphCaptureGuard&) = delete;
    HipGraphCaptureGuard(HipGraphCaptureGuard&&)                 = delete;
    HipGraphCaptureGuard& operator=(HipGraphCaptureGuard&&)      = delete;
};

namespace rtp_llm {

using HipGraphState = GraphExecutionState;

}  // namespace rtp_llm
