#pragma once

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <vector>

namespace rtp_llm {
namespace cuda_graph {

/// Owns sorted CUDA graph capture keys (seq_len or batch size) and runtime resolution into `BatchDescriptor`.
class CudaGraphCaptureDispatcher {
public:
    void build(const GraphParams& graph_params, size_t max_bs, bool is_prefill_cuda_graph_mode);

    const std::vector<int>& captureRange() const {
        return capture_range_;
    }

    bool tryGetRealGraphPrefillSeqLen(const torch_ext::PyModelInputs& inputs, BatchDescriptor& batch_descriptor);
    bool tryGetRealGraphDecodeBatchSize(const torch_ext::PyModelInputs& inputs, BatchDescriptor& batch_descriptor);

private:
    static std::vector<int> decodeBatchSizesToCapture(const GraphParams& graph_params, size_t max_bs);
    static std::vector<int> prefillSequenceLengthsToCapture(const GraphParams& graph_params);

    std::vector<int> capture_range_;
};

}  // namespace cuda_graph
}  // namespace rtp_llm
