#include "rtp_llm/models_py/bindings/rocm/RocmQuickReduce.h"

#include <torch/extension.h>

#include <optional>

#include "rtp_llm/models_py/bindings/rocm/kernels/quickreduce/quick_reduce.h"

namespace py = pybind11;

namespace rtp_llm {

quickreduce::fptr_t
              rocm_quick_reduce_init_custom_qr(int64_t rank, int64_t world_size, std::optional<int64_t> qr_max_size);
void          rocm_quick_reduce_destroy(quickreduce::fptr_t handle);
torch::Tensor rocm_quick_reduce_get_handle(quickreduce::fptr_t handle);
void          rocm_quick_reduce_open_handles(quickreduce::fptr_t handle, const std::vector<torch::Tensor>& handles);
void          rocm_quick_reduce_all_reduce(
             quickreduce::fptr_t handle, torch::Tensor& inp, torch::Tensor& out, int64_t quant_level, bool cast_bf2half);
int64_t rocm_quick_reduce_max_size();

class RocmQuickReduceHandle {
public:
    RocmQuickReduceHandle(int64_t rank, int64_t world_size, std::optional<int64_t> qr_max_size = std::nullopt) {
        handle_ = rocm_quick_reduce_init_custom_qr(rank, world_size, qr_max_size);
    }

    ~RocmQuickReduceHandle() {
        if (handle_ != 0) {
            rocm_quick_reduce_destroy(handle_);
            handle_ = 0;
        }
    }

    quickreduce::fptr_t ptr() const {
        return handle_;
    }

private:
    quickreduce::fptr_t handle_ = 0;
};

void registerRocmQuickReduce(py::module& m) {
    py::class_<RocmQuickReduceHandle>(m, "RocmQuickReduceHandle")
        .def(py::init<int64_t, int64_t, std::optional<int64_t>>(),
             py::arg("rank"),
             py::arg("world_size"),
             py::arg("qr_max_size") = std::nullopt)
        .def("ptr", &RocmQuickReduceHandle::ptr);
    m.def("rocm_quick_reduce_get_handle", &rocm_quick_reduce_get_handle);
    m.def("rocm_quick_reduce_open_handles", &rocm_quick_reduce_open_handles);
    m.def("rocm_quick_reduce_all_reduce", &rocm_quick_reduce_all_reduce);
    m.def("rocm_quick_reduce_max_size", &rocm_quick_reduce_max_size);
}

}  // namespace rtp_llm
