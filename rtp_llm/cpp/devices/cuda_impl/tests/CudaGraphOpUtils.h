#pragma once
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cuda_graph {
using namespace rtp_llm;
using CudaGraphRunnerPtr = CudaGraphRunner*;

struct CudaGraphRunnerConfig {
    bool              is_prefill_mode            = false;
    bool              enable_debug_mode          = false;
    int64_t           max_seq_len                = 0;
    int64_t           tokens_per_block           = 0;
    int64_t           max_context_batch_size     = 1;
    int64_t           concurrency_limit          = 128;
    std::vector<int>  decode_capture_batch_sizes = {};
    std::vector<int>  prefill_capture_seq_lens   = {};
    torch::ScalarType data_type                  = torch::kFloat16;
};

inline CudaGraphRunnerPtr createCudaGraphRunner(py::object py_instance, const CudaGraphRunnerConfig& config) {
    GraphParams graph_params;
    graph_params.enable_cuda_graph            = true;
    graph_params.enable_cuda_graph_debug_mode = config.enable_debug_mode;
    graph_params.is_prefill_cuda_graph_mode   = config.is_prefill_mode;
    graph_params.max_seq_len                  = config.max_seq_len;
    graph_params.tokens_per_block             = config.tokens_per_block;
    graph_params.concurrency_limit            = config.concurrency_limit;
    graph_params.max_context_batch_size       = config.max_context_batch_size;
    graph_params.decode_capture_batch_sizes   = config.decode_capture_batch_sizes;
    graph_params.prefill_capture_seq_lens     = config.prefill_capture_seq_lens;

    CudaGraphRunnerPtr cuda_graph_runner_ptr = CudaGraphRunner::create(graph_params, std::move(py_instance));
    cuda_graph_runner_ptr->setModelDataType(torch::scalarTypeToTypeMeta(config.data_type));
    return cuda_graph_runner_ptr;
}

}  // namespace cuda_graph
