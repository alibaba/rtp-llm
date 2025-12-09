#pragma once
#include "rtp_llm/cpp/devices/GraphBase.h"
#include "rtp_llm/cpp/devices/DeviceData.h"
#include <ATen/hip/HIPGraph.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace py = pybind11;
namespace rtp_llm {

class HipGraphRunner: public GraphBase {
public:
    HipGraphRunner(const DeviceInitParams& params,
                   py::object              py_instance,
                   int                     kv_cache_block_offset,
                   at::hip::HIPStream      capture_stream,
                   bool                    is_prefill_hip_graph_mode = false):
        GraphBase(std::move(py_instance)), capture_stream_storage_(capture_stream) {
        // Initialize base class members
        enable_graph_               = params.hw_kernel_config.enable_cuda_graph;  // Uses same config as CUDA
        is_prefill_graph_mode_      = is_prefill_hip_graph_mode;
        enable_graph_debug_mode_    = params.hw_kernel_config.enable_cuda_graph_debug_mode;
        max_seq_len_                = params.max_seq_len;
        seq_size_per_block_         = params.tokens_per_block;
        kv_cache_block_offset_      = kv_cache_block_offset;
        prefill_capture_seq_lens_   = params.hw_kernel_config.prefill_capture_seq_lens;
        decode_capture_batch_sizes_ = params.hw_kernel_config.decode_capture_batch_sizes;

        // Set capture stream pointer for base class
        capture_stream_ = &capture_stream_storage_;

        py::gil_scoped_acquire gil;
        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("HipGraphRunner constructor: Python instance is null or none.");
        }
        if (is_prefill_graph_mode_) {
            max_bs_ = params.fifo_scheduler_config.max_context_batch_size;
        } else {
            max_bs_ = params.concurrency_config.concurrency_limit;
        }
        py_forward_method_     = py_instance_.attr("forward");
        py_fill_params_method_ = py_instance_.attr("fill_params");

        // Initialize HIP-specific tensor options (torch::kCUDA works for HIP in PyTorch)
        options_device_int32_ = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        options_cpu_int32_    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);

        RTP_LLM_LOG_INFO("Initialize HipGraphRunner with parameters below: \n \
            enable_graph_: %d, concurrency_limit_: %d, enable_graph_debug_mode_: %d, max_seq_len_: %d, seq_size_per_block_: %d, kv_cache_block_offset_: %d, is_prefill_graph_mode_: %d",
                         enable_graph_,
                         max_bs_,
                         enable_graph_debug_mode_,
                         max_seq_len_,
                         seq_size_per_block_,
                         kv_cache_block_offset_,
                         is_prefill_graph_mode_);
    }

    ~HipGraphRunner() {
        RTP_LLM_LOG_INFO("Release HipGraphRunner .....");
        py::gil_scoped_acquire gil;
        py_instance_.release();
        RTP_LLM_LOG_INFO("Release HipGraphRunner Successfully");
    }

protected:
    // Override GPU-specific methods
    void optimizedCopy(const torch::Tensor& src, torch::Tensor& dst, size_t size) override;
    void syncDevice() override;
    void replayGraphImpl(int key) override;
    void performGraphCapture(int key, const char* key_type) override;

private:
    // Store the capture stream (base class has pointer to this)
    at::hip::HIPStream capture_stream_storage_;
};

}  // namespace rtp_llm
