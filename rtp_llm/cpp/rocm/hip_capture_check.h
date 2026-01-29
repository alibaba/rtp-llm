#pragma once

namespace rtp_llm {
namespace rocm {

// Capture state check for HIP graph capture.
// This struct is separated to avoid including HIP-specific headers
// in files that only need to check capture state.
struct CaptureCheck {
    static bool in_hip_graph_capture;
};

}  // namespace rocm
}  // namespace rtp_llm
