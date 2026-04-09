#include "rtp_llm/cpp/cuda_graph/cuda_graph_dispatcher.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>

using namespace torch_ext;

namespace rtp_llm {
namespace cuda_graph {

void CudaGraphCaptureDispatcher::build(const GraphParams& graph_params,
                                       size_t             max_bs,
                                       bool               is_prefill_cuda_graph_mode) {
    if (is_prefill_cuda_graph_mode) {
        capture_range_ = prefillSequenceLengthsToCapture(graph_params);
    } else {
        capture_range_ = decodeBatchSizesToCapture(graph_params, max_bs);
    }
}

std::vector<int> CudaGraphCaptureDispatcher::decodeBatchSizesToCapture(const GraphParams& graph_params, size_t max_bs) {
    if (!graph_params.decode_capture_batch_sizes.empty()) {
        std::vector<int> sizes = graph_params.decode_capture_batch_sizes;
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", sizes.size());
        std::sort(sizes.begin(), sizes.end());
        return sizes;
    }

    std::vector<int> capture_bs;
    int              max_generate_batch_size = static_cast<int>(max_bs);
    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", max_generate_batch_size);
    for (int i = 1; i <= std::min(32, max_generate_batch_size); i += 1) {
        capture_bs.push_back(i);
    }
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

std::vector<int> CudaGraphCaptureDispatcher::prefillSequenceLengthsToCapture(const GraphParams& graph_params) {
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

bool CudaGraphCaptureDispatcher::tryGetRealGraphPrefillSeqLen(const PyModelInputs& inputs,
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

bool CudaGraphCaptureDispatcher::tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs,
                                                                BatchDescriptor&     batch_descriptor) {
    int cuda_graph_bs                   = inputs.attention_inputs.input_lengths.size(0);
    batch_descriptor.current_batch_size = cuda_graph_bs;
    RTP_LLM_LOG_DEBUG("canRun judge for batch size: %d", cuda_graph_bs);
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("decode cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), batch_descriptor.current_batch_size);
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("decode batch size %d exceeds max captured %d, fallback to normal run",
                            batch_descriptor.current_batch_size,
                            capture_range_.back());
        return false;
    }
    batch_descriptor.current_real_graph_bs = *it;
    RTP_LLM_LOG_DEBUG("batch size used in replay: %d (graph key %d)",
                      batch_descriptor.current_batch_size,
                      batch_descriptor.current_real_graph_bs);

    if (inputs.attention_inputs.is_prefill) {
        batch_descriptor.seq_len_sum = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    } else {
        batch_descriptor.seq_len_sum = cuda_graph_bs;
    }
    RTP_LLM_LOG_DEBUG("can run cuda graph for decode");
    return true;
}

}  // namespace cuda_graph
}  // namespace rtp_llm
