#pragma once

#include "rtp_llm/cpp/cuda_graph/cuda_graph_decode_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_prefill_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner_base.h"

namespace rtp_llm {

/// Factory helpers for tests and internal use (no instances).
struct CudaGraphRunner {
    static CudaGraphRunnerBase* createForPrefill(py::object py_instance, GraphParams params);
    static CudaGraphRunnerBase* createForDecode(py::object py_instance, GraphParams params);
};

}  // namespace rtp_llm
