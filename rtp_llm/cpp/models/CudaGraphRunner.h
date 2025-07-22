#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/models/CudaGraphUtils.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>

namespace py = pybind11;
namespace rtp_llm {
class CudaGraphRunner {
public:
    CudaGraphRunner(const DeviceInitParams& params,
                    py::object              py_instance,
                    int                     kv_cache_block_offset,
                    DeviceBase*             device,
                    bool                    in_test = false):
        py_instance_(std::move(py_instance)),
        enable_cuda_graph(params.hw_kernel_config.enable_cuda_graph),
        concurrency_limit(params.concurrency_config.concurrency_limit),
        capture_stream(at::cuda::getStreamFromPool()),
        disable_padding(params.hw_kernel_config.disable_padding),
        hidden_size_(params.hidden_size),
        max_seq_len(params.max_seq_len),
        seq_size_per_block(params.tokens_per_block),
        kv_cache_block_offset_(kv_cache_block_offset),
        device_(device),
        in_test_(in_test) {
        py::gil_scoped_acquire gil;
        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("CudaGraphRunner constructor: Python instance is null or none.");
        }
        std::cout << "max_seq_len: " << max_seq_len << std::endl;
        std::cout << "seq_size_per_block: " << seq_size_per_block << std::endl;
        std::cout << "kv_cache_block_offset: " << kv_cache_block_offset_ << std::endl;
        std::cout << "hidden_size_: " << hidden_size_ << std::endl;
        py_forward_method = py_instance_.attr("forward");
    }

    void           Capture();
    void           CaptureOneBatchSize(int bs);
    void           PrepareInputs(PyModelInputs& inputs);
    bool           CanRun(PyModelInputs& inputs);
    void           replay(int bs);
    void           init_capture();
    void           init_kernel_internal_memory();
    int            get_current_real_graph_bs();
    PyModelOutputs forward(PyModelInputs& inputs);
    py::object     py_instance_;

private:
    std::vector<int>                       get_batch_sizes_to_capture(int concurrency_limit);
    py::object                             py_forward_method;
    bool                                   enable_cuda_graph{false};
    int                                    concurrency_limit{32};
    at::cuda::CUDAStream                   capture_stream;
    bool                                   disable_padding{true};
    int                                    hidden_size_;
    size_t                                 max_bs{1};
    int                                    num_tokens_per_bs{1};
    int                                    max_num_token{1};
    int                                    current_batch_size{1};
    int                                    current_real_graph_bs{1};
    int                                    max_seq_len{0};
    int                                    seq_size_per_block{0};
    int                                    kv_cache_block_offset_{0};
    std::vector<int>                       capture_range;
    std::unordered_map<int, GraphInstance> graph_instances;
    CaptureMemoryHold                      capture_mem_hold;

public:
    DeviceBase* device_{nullptr};

private:
    bool             in_test_{false};
    static const int MIN_CACHE_INPUT_TOKEN_NUM = 512;
};
}  // namespace rtp_llm
