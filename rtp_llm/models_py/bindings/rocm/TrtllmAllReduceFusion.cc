#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>

#include "rtp_llm/models_py/bindings/rocm/TrtllmAllReduceFusion.h"
#include "rtp_llm/cpp/kernels/rocm/trtllm_allreduce_fusion.h"

namespace py = pybind11;

namespace rtp_llm {

// ---------------------------------------------------------------------------
// TrtllmArFusionHandle implementation
// ---------------------------------------------------------------------------

TrtllmArFusionHandle::TrtllmArFusionHandle(int64_t device_id,
                                           int64_t rank,
                                           int64_t world_size,
                                           int64_t max_size_in_bytes,
                                           int64_t comm_ptrs_buf_len)
    : fptr_(init_ar_fusion(device_id, rank, world_size,
                           max_size_in_bytes, comm_ptrs_buf_len)) {}

TrtllmArFusionHandle::~TrtllmArFusionHandle() {
    if (fptr_ != 0) {
        destroy_ar_fusion(fptr_);
        fptr_ = 0;
    }
}

torch::Tensor TrtllmArFusionHandle::get_barrier_handle() {
    return get_ar_fusion_barrier_handle(fptr_);
}

torch::Tensor TrtllmArFusionHandle::get_data_handle() {
    return get_ar_fusion_data_handle(fptr_);
}

void TrtllmArFusionHandle::open_barrier_handles(std::vector<torch::Tensor> handles) {
    open_ar_fusion_barrier_handles(fptr_, handles);
}

void TrtllmArFusionHandle::open_data_handles(std::vector<torch::Tensor> handles) {
    open_ar_fusion_data_handles(fptr_, handles);
}

void TrtllmArFusionHandle::capture_clear() {
    ar_fusion_capture_clear(fptr_);
}

std::vector<torch::Tensor> TrtllmArFusionHandle::get_captured_handles() {
    return get_ar_fusion_captured_handles(fptr_);
}

torch::Tensor TrtllmArFusionHandle::get_captured_offsets() {
    return get_ar_fusion_captured_offsets(fptr_);
}

void TrtllmArFusionHandle::open_captured_handles(std::vector<torch::Tensor> handles,
                                                  std::vector<int64_t> offsets,
                                                  int64_t ptr_idx) {
    open_ar_fusion_captured_handles(fptr_, handles, offsets, ptr_idx);
}

void TrtllmArFusionHandle::allreduce_rms(torch::Tensor& allreduce_in,
                                          torch::Tensor& residual_in,
                                          torch::Tensor& rms_gamma,
                                          torch::Tensor& residual_out,
                                          torch::Tensor& norm_out,
                                          torch::Tensor& scale_out,
                                          double eps,
                                          int64_t quant_type) {
    rtp_llm::allreduce_rms(fptr_, allreduce_in, residual_in, rms_gamma,
                           residual_out, norm_out, scale_out, eps, quant_type);
}

void TrtllmArFusionHandle::allreduce(torch::Tensor& allreduce_in,
                                      torch::Tensor& allreduce_out) {
    rtp_llm::allreduce(fptr_, allreduce_in, allreduce_out);
}

// ---------------------------------------------------------------------------
// pybind11 registration
// ---------------------------------------------------------------------------

void registerTrtllmArFusionHandle(py::module& module) {
    py::class_<TrtllmArFusionHandle>(module, "TrtllmArFusionHandle")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t>(),
             py::arg("device_id"),
             py::arg("rank"),
             py::arg("world_size"),
             py::arg("max_size_in_bytes"),
             py::arg("comm_ptrs_buf_len"))
        // IPC handle exchange
        .def("get_barrier_handle", &TrtllmArFusionHandle::get_barrier_handle)
        .def("get_data_handle", &TrtllmArFusionHandle::get_data_handle)
        .def("open_barrier_handles", &TrtllmArFusionHandle::open_barrier_handles,
             py::arg("handles"))
        .def("open_data_handles", &TrtllmArFusionHandle::open_data_handles,
             py::arg("handles"))
        // CUDA Graph capture support
        .def("capture_clear", &TrtllmArFusionHandle::capture_clear)
        .def("get_captured_handles", &TrtllmArFusionHandle::get_captured_handles)
        .def("get_captured_offsets", &TrtllmArFusionHandle::get_captured_offsets)
        .def("open_captured_handles", &TrtllmArFusionHandle::open_captured_handles,
             py::arg("handles"),
             py::arg("offsets"),
             py::arg("ptr_idx"))
        // Compute ops
        .def("allreduce_rms", &TrtllmArFusionHandle::allreduce_rms,
             py::arg("allreduce_in"),
             py::arg("residual_in"),
             py::arg("rms_gamma"),
             py::arg("residual_out"),
             py::arg("norm_out"),
             py::arg("scale_out"),
             py::arg("eps"),
             py::arg("quant_type"))
        .def("allreduce", &TrtllmArFusionHandle::allreduce,
             py::arg("allreduce_in"),
             py::arg("allreduce_out"));
}

}  // namespace rtp_llm
