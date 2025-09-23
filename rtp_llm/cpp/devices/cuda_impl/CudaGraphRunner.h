#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphUtils.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include "rtp_llm/cpp/devices/GraphBase.h"

namespace py = pybind11;
namespace rtp_llm {
class CudaGraphRunner: public GraphBase {
public:
    CudaGraphRunner(const DeviceInitParams& params,
                    py::object              py_instance,
                    int                     kv_cache_block_offset,
                    DeviceBase*             device,
                    bool                    is_prefill_cuda_graph_mode = false):
        GraphBase(std::move(py_instance)),
        enable_cuda_graph_(params.hw_kernel_config.enable_cuda_graph),
        is_prefill_cuda_graph_mode_(is_prefill_cuda_graph_mode),
        concurrency_limit_(params.concurrency_config.concurrency_limit),
        capture_stream_(at::cuda::getStreamFromPool()),
        enable_cuda_graph_debug_mode_(params.hw_kernel_config.enable_cuda_graph_debug_mode),
        hidden_size_(params.hidden_size),
        max_seq_len_(params.max_seq_len),
        seq_size_per_block_(params.tokens_per_block),
        kv_cache_block_offset_(kv_cache_block_offset),
        device_(device) {
        py::gil_scoped_acquire gil;
        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("CudaGraphRunner constructor: Python instance is null or none.");
        }

        py_forward_method_   = py_instance_.attr("forward");
        py_fmha_type_method_ = py_instance_.attr("get_fmha_type");
        RTP_LLM_LOG_INFO("Initialize CudaGraphRunner with parameters below: \n \
            enable_cuda_graph_: %d, concurrency_limit_: %d, enable_cuda_graph_debug_mode_: %d, hidden_size_: %d, max_seq_len_: %d, seq_size_per_block_: %d, kv_cache_block_offset_: %d, is_prefill_cuda_graph_mode_: %d",
                         enable_cuda_graph_,
                         concurrency_limit_,
                         enable_cuda_graph_debug_mode_,
                         hidden_size_,
                         max_seq_len_,
                         seq_size_per_block_,
                         kv_cache_block_offset_,
                         is_prefill_cuda_graph_mode_);
    }
    ~CudaGraphRunner() {
        RTP_LLM_LOG_INFO("Release CudaGraphRunner .....");
        py::gil_scoped_acquire gil;
        py_instance_.release();
        RTP_LLM_LOG_INFO("Release CudaGraphRunner Successfully");
    }
    void           capture();
    void           captureOneBatchSize(int bs);
    void           prepareInputs(PyModelInputs& inputs);
    bool           canRun(PyModelInputs& inputs);
    void           replay(int bs);
    void           initCapture() override;
    void           initKernelInternalMemory();
    int            getCurrentRealGraphBs();
    PyModelOutputs forward(PyModelInputs& inputs) override;
    py::object     normalForward(PyModelInputs& inputs);

private:
    void             copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor);
    std::vector<int> getBatchSizesToCapture(int concurrency_limit);
    bool             tryGetRealGraphBatchSize(PyModelInputs& inputs);
    void extractValidHiddenStates(PyModelOutputs& outputs, const PyModelInputs& inputs, int32_t total_valid_tokens);
    py::object                             py_forward_method_;
    py::object                             py_fill_params_method_;
    bool                                   enable_cuda_graph_{false};
    bool                                   is_prefill_cuda_graph_mode_{false};
    int                                    concurrency_limit_{32};
    at::cuda::CUDAStream                   capture_stream_;
    bool                                   enable_cuda_graph_debug_mode_{false};
    int                                    hidden_size_;
    size_t                                 max_bs_{1};
    int                                    num_tokens_per_bs_{1};
    int                                    max_num_token_{1};
    int                                    current_batch_size_{1};
    int                                    current_real_graph_bs_{1};
    int                                    max_seq_len_{0};
    int                                    seq_size_per_block_{0};
    int                                    kv_cache_block_offset_{0};
    int                                    seq_len_sum_{0};
    std::vector<int>                       capture_range_;
    std::unordered_map<int, GraphInstance> graph_instances_;
    CaptureMemoryHold                      capture_mem_hold_;

public:
    DeviceBase* device_{nullptr};

private:
    std::deque<std::shared_ptr<FlashInferAttnParams>> PRIVATE_DECODE_PARAMS_CACHE;
    std::deque<std::shared_ptr<FlashInferAttnParams>> PRIVATE_PREFILL_PARAMS_CACHE;
};
}  // namespace rtp_llm
