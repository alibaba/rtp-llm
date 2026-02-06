#pragma once
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cuda_graph {
using namespace rtp_llm;
using CudaGraphRunnerPtr = CudaGraphRunner*;

// Simple wrapper that creates a CudaGraphRunner from GraphParams
inline CudaGraphRunnerPtr createCudaGraphRunner(py::object py_instance, GraphParams graph_params) {
    // Ensure cuda graph is enabled
    graph_params.enable_cuda_graph = true;

    // Set num_tokens_per_bs based on mode if not already set
    if (graph_params.num_tokens_per_bs == 0) {
        if (graph_params.is_prefill_cuda_graph_mode) {
            graph_params.num_tokens_per_bs = graph_params.max_seq_len;
        } else {
            graph_params.num_tokens_per_bs = 1;  // Decode mode
        }
    }

    return CudaGraphRunner::create(graph_params, std::move(py_instance));
}

}  // namespace cuda_graph
