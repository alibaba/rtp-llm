#include "rtp_llm/models_py/bindings/rocm/VllmCustomAllReduce.h"

#include <torch/extension.h>

namespace py = pybind11;

namespace rtp_llm {

using fptr_t = int64_t;

fptr_t init_vllm_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                           torch::Tensor&             rank_data,
                           int64_t                    rank,
                           bool                       fully_connected);
bool   vllm_custom_ar_is_weak_contiguous(torch::Tensor& t);
void   vllm_custom_ar_all_reduce(
      fptr_t handle, torch::Tensor& inp, torch::Tensor& out, fptr_t reg_buffer, int64_t reg_buffer_size_bytes);
void    vllm_custom_ar_dispose(fptr_t handle);
int64_t vllm_custom_ar_meta_size();
void    vllm_custom_ar_register_buffer(fptr_t handle, const std::vector<fptr_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>> vllm_custom_ar_get_graph_buffer_ipc_meta(fptr_t handle);
void                                                   vllm_custom_ar_register_graph_buffers(fptr_t                                   handle,
                                                                                             const std::vector<std::vector<int64_t>>& handles,
                                                                                             const std::vector<std::vector<int64_t>>& offsets);
std::tuple<fptr_t, torch::Tensor>                      vllm_custom_ar_allocate_shared_buffer_and_handle(int64_t size);
fptr_t                                                 vllm_custom_ar_open_mem_handle(torch::Tensor& mem_handle);
void                                                   vllm_custom_ar_free_shared_buffer(fptr_t buffer);

class VllmCustomAllReduceHandle {
public:
    VllmCustomAllReduceHandle(const std::vector<fptr_t>& fake_ipc_ptrs,
                              torch::Tensor&             rank_data,
                              int64_t                    rank,
                              bool                       fully_connected) {
        handle_ = init_vllm_custom_ar(fake_ipc_ptrs, rank_data, rank, fully_connected);
    }

    ~VllmCustomAllReduceHandle() {
        if (handle_ != 0) {
            vllm_custom_ar_dispose(handle_);
            handle_ = 0;
        }
    }

    fptr_t ptr() const {
        return handle_;
    }

private:
    fptr_t handle_ = 0;
};

void registerVllmCustomAllReduce(py::module& m) {
    py::class_<VllmCustomAllReduceHandle>(m, "VllmCustomAllReduceHandle")
        .def(py::init<const std::vector<fptr_t>&, torch::Tensor&, int64_t, bool>())
        .def("ptr", &VllmCustomAllReduceHandle::ptr);
    m.def("vllm_custom_ar_is_weak_contiguous", &vllm_custom_ar_is_weak_contiguous);
    m.def("vllm_custom_ar_all_reduce", &vllm_custom_ar_all_reduce);
    m.def("vllm_custom_ar_meta_size", &vllm_custom_ar_meta_size);
    m.def("vllm_custom_ar_register_buffer", &vllm_custom_ar_register_buffer);
    m.def("vllm_custom_ar_get_graph_buffer_ipc_meta", &vllm_custom_ar_get_graph_buffer_ipc_meta);
    m.def("vllm_custom_ar_register_graph_buffers", &vllm_custom_ar_register_graph_buffers);
    m.def("vllm_custom_ar_allocate_shared_buffer_and_handle", &vllm_custom_ar_allocate_shared_buffer_and_handle);
    m.def("vllm_custom_ar_open_mem_handle", &vllm_custom_ar_open_mem_handle);
    m.def("vllm_custom_ar_free_shared_buffer", &vllm_custom_ar_free_shared_buffer);
}

}  // namespace rtp_llm
