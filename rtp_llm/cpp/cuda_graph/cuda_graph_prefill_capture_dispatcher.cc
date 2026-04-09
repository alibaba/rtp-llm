#include "rtp_llm/cpp/cuda_graph/cuda_graph_prefill_capture_dispatcher.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>

using namespace torch_ext;

namespace rtp_llm {
namespace cuda_graph {

void CudaGraphPrefillCaptureDispatcher::build(const GraphParams& graph_params) {
    capture_range_ = sequenceLengthsToCapture(graph_params);
}

std::vector<int> CudaGraphPrefillCaptureDispatcher::sequenceLengthsToCapture(const GraphParams& graph_params) {
    RTP_LLM_CHECK_WITH_INFO(!graph_params.prefill_capture_seq_lens.empty(),
                            "prefill_capture_seq_lens must be provided from Python and cannot be empty");

    RTP_LLM_LOG_INFO("Using prefill capture sequence lengths from Python: %zu lengths",
                     graph_params.prefill_capture_seq_lens.size());

    std::vector<int> result = graph_params.prefill_capture_seq_lens;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    RTP_LLM_LOG_INFO(
        "Total sequence lengths to capture: %zu (min: %d, max: %d)", result.size(), result.front(), result.back());
    return result;
}

bool CudaGraphPrefillCaptureDispatcher::tryGetRealGraphPrefillSeqLen(const PyModelInputs& inputs,
                                                                     BatchDescriptor&     batch_descriptor) {
    batch_descriptor.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("prefill cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), batch_descriptor.current_seq_len);
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("prefill seq_len %d exceeds max captured %d, fallback to normal run",
                            batch_descriptor.current_seq_len,
                            capture_range_.back());
        return false;
    }
    batch_descriptor.current_real_graph_seq_len = *it;
    batch_descriptor.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
    return true;
}

}  // namespace cuda_graph
}  // namespace rtp_llm
