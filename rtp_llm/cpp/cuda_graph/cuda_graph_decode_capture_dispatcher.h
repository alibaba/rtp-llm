#pragma once

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <vector>

namespace rtp_llm {
namespace cuda_graph {

/// Sorted decode capture batch sizes and runtime resolution into `BatchDescriptor`.
class CudaGraphDecodeCaptureDispatcher {
public:
    void build(const GraphParams& graph_params, size_t max_bs);

    const std::vector<int>& captureRange() const {
        return capture_range_;
    }

    bool tryGetRealGraphDecodeBatchSize(const torch_ext::PyModelInputs& inputs, BatchDescriptor& batch_descriptor);

private:
    static std::vector<int> batchSizesToCapture(const GraphParams& graph_params, size_t max_bs);

    std::vector<int> capture_range_;
};

}  // namespace cuda_graph
}  // namespace rtp_llm
