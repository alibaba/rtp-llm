#pragma once

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <vector>

namespace rtp_llm {
namespace cuda_graph {

/// Sorted prefill capture sequence lengths and runtime resolution into `BatchDescriptor`.
class CudaGraphPrefillCaptureDispatcher {
public:
    void build(const GraphParams& graph_params);

    const std::vector<int>& captureRange() const {
        return capture_range_;
    }

    bool tryGetRealGraphPrefillSeqLen(const torch_ext::PyModelInputs& inputs, BatchDescriptor& batch_descriptor);

private:
    static std::vector<int> sequenceLengthsToCapture(const GraphParams& graph_params);

    std::vector<int> capture_range_;
};

}  // namespace cuda_graph
}  // namespace rtp_llm
