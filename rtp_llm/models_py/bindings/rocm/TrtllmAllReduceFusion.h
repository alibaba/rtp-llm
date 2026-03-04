#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace rtp_llm {

/// RAII wrapper around the CommWorkspace opaque pointer.
/// Exposes IPC lifecycle management as methods, keeping them
/// out of the flat ops namespace.
class TrtllmArFusionHandle {
public:
    TrtllmArFusionHandle(int64_t device_id,
                         int64_t rank,
                         int64_t world_size,
                         int64_t max_size_in_bytes,
                         int64_t comm_ptrs_buf_len);
    ~TrtllmArFusionHandle();

    // IPC handle exchange
    torch::Tensor get_barrier_handle();
    torch::Tensor get_data_handle();
    void open_barrier_handles(std::vector<torch::Tensor> handles);
    void open_data_handles(std::vector<torch::Tensor> handles);

    // CUDA Graph capture support
    void capture_clear();
    std::vector<torch::Tensor> get_captured_handles();
    torch::Tensor get_captured_offsets();
    void open_captured_handles(std::vector<torch::Tensor> handles,
                               std::vector<int64_t> offsets,
                               int64_t ptr_idx);

    // Compute ops
    void allreduce_rms(torch::Tensor& allreduce_in,
                       torch::Tensor& residual_in,
                       torch::Tensor& rms_gamma,
                       torch::Tensor& residual_out,
                       torch::Tensor& norm_out,
                       torch::Tensor& scale_out,
                       double eps,
                       int64_t quant_type);

    void allreduce(torch::Tensor& allreduce_in,
                   torch::Tensor& allreduce_out);

private:
    int64_t fptr_;
};

/// Register TrtllmArFusionHandle as a pybind11 class on the given module.
void registerTrtllmArFusionHandle(py::module& module);

}  // namespace rtp_llm
