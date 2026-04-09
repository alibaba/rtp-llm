#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

namespace rtp_llm {

CudaGraphRunnerBase* CudaGraphRunner::createForPrefill(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph          = true;
    params.is_prefill_cuda_graph_mode = true;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = params.max_seq_len;
    }
    return new CudaGraphPrefillRunner(std::move(params), std::move(py_instance));
}

CudaGraphRunnerBase* CudaGraphRunner::createForDecode(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph          = true;
    params.is_prefill_cuda_graph_mode = false;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = 1;
    }
    return new CudaGraphDecodeRunner(std::move(params), std::move(py_instance));
}

}  // namespace rtp_llm
