// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM csrc custom all-reduce implementation.

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "rtp_llm/models_py/bindings/rocm/kernels/vllm_custom_allreduce.cuh"

namespace rtp_llm {

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

fptr_t init_vllm_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                           torch::Tensor&             rank_data,
                           int64_t                    rank,
                           bool                       fully_connected) {
    int world_size = fake_ipc_ptrs.size();
    if (world_size > 8) {
        throw std::invalid_argument("world size > 8 is not supported");
    }
    if (world_size % 2 != 0) {
        throw std::invalid_argument("Odd num gpus is not supported for now");
    }
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument("invalid rank passed in");
    }

    vllm::Signal* ipc_ptrs[8];
    for (int i = 0; i < world_size; ++i) {
        ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
    }
    return reinterpret_cast<fptr_t>(new vllm::CustomAllreduce(
        ipc_ptrs, rank_data.mutable_data_ptr(), rank_data.numel(), rank, world_size, fully_connected));
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice.
 */
bool vllm_custom_ar_is_weak_contiguous(torch::Tensor& t) {
    if (t.is_contiguous()) {
        return true;
    }
    const int64_t storage_nbytes = static_cast<int64_t>(t.storage().nbytes());
    return storage_nbytes - t.storage_offset() * t.element_size() == static_cast<int64_t>(t.numel() * t.element_size());
}

void vllm_custom_ar_all_reduce(
    fptr_t handle, torch::Tensor& inp, torch::Tensor& out, fptr_t reg_buffer_ptr, int64_t reg_buffer_size_bytes) {
    auto*            fa = reinterpret_cast<vllm::CustomAllreduce*>(handle);
    c10::DeviceGuard device_guard(inp.device());
    auto             stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

    TORCH_CHECK(inp.scalar_type() == out.scalar_type());
    TORCH_CHECK(inp.numel() == out.numel());
    TORCH_CHECK(vllm_custom_ar_is_weak_contiguous(out));
    TORCH_CHECK(vllm_custom_ar_is_weak_contiguous(inp));

    auto  input_size = inp.numel() * inp.element_size();
    void* reg_buffer = reinterpret_cast<void*>(reg_buffer_ptr);
    if (reg_buffer != nullptr) {
        TORCH_CHECK(input_size <= reg_buffer_size_bytes);
        CUDACHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
    } else {
        reg_buffer = inp.data_ptr();
    }

    switch (out.scalar_type()) {
        case at::ScalarType::Float:
            fa->allreduce<float>(
                stream, reinterpret_cast<float*>(reg_buffer), reinterpret_cast<float*>(out.data_ptr()), out.numel());
            break;
        case at::ScalarType::Half:
            fa->allreduce<half>(
                stream, reinterpret_cast<half*>(reg_buffer), reinterpret_cast<half*>(out.data_ptr()), out.numel());
            break;
        case at::ScalarType::BFloat16:
            fa->allreduce<nv_bfloat16>(stream,
                                       reinterpret_cast<nv_bfloat16*>(reg_buffer),
                                       reinterpret_cast<nv_bfloat16*>(out.data_ptr()),
                                       out.numel());
            break;
        default:
            throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }
}

void vllm_custom_ar_dispose(fptr_t handle) {
    delete reinterpret_cast<vllm::CustomAllreduce*>(handle);
}

int64_t vllm_custom_ar_meta_size() {
    return sizeof(vllm::Signal);
}

void vllm_custom_ar_register_buffer(fptr_t handle, const std::vector<fptr_t>& fake_ipc_ptrs) {
    auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(handle);
    TORCH_CHECK(fake_ipc_ptrs.size() == static_cast<size_t>(fa->world_size_));
    void* ipc_ptrs[8];
    for (size_t i = 0; i < fake_ipc_ptrs.size(); ++i) {
        ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
    }
    fa->register_buffer(ipc_ptrs);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> vllm_custom_ar_get_graph_buffer_ipc_meta(fptr_t handle) {
    auto* fa                   = reinterpret_cast<vllm::CustomAllreduce*>(handle);
    auto [ipc_handle, offsets] = fa->get_graph_buffer_ipc_meta();
    std::vector<int64_t> bytes(ipc_handle.begin(), ipc_handle.end());
    return std::make_tuple(bytes, offsets);
}

void vllm_custom_ar_register_graph_buffers(fptr_t                                   handle,
                                           const std::vector<std::vector<int64_t>>& handles,
                                           const std::vector<std::vector<int64_t>>& offsets) {
    auto*                    fa = reinterpret_cast<vllm::CustomAllreduce*>(handle);
    std::vector<std::string> bytes;
    bytes.reserve(handles.size());
    for (const auto& handle_bytes : handles) {
        bytes.emplace_back(handle_bytes.begin(), handle_bytes.end());
    }
    fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, torch::Tensor> vllm_custom_ar_allocate_shared_buffer_and_handle(int64_t size) {
    int device_index = 0;
    CUDACHECK(cudaGetDevice(&device_index));
    c10::DeviceGuard device_guard(c10::Device(c10::DeviceType::CUDA, device_index));
    void*            buffer = nullptr;

    cudaStreamCaptureMode mode   = cudaStreamCaptureModeRelaxed;
    auto                  stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

#if defined(USE_ROCM)
    CUDACHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&buffer), size, hipDeviceMallocUncached));
#else
    CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&buffer), size));
#endif
    CUDACHECK(cudaMemsetAsync(buffer, 0, size, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

    auto options           = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto ipc_handle_tensor = torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
    CUDACHECK(cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(ipc_handle_tensor.data_ptr()), buffer));

    return std::make_tuple(reinterpret_cast<fptr_t>(buffer), ipc_handle_tensor);
}

fptr_t vllm_custom_ar_open_mem_handle(torch::Tensor& mem_handle) {
    void* ipc_ptr = nullptr;
    CUDACHECK(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&ipc_ptr),
                                   *reinterpret_cast<const cudaIpcMemHandle_t*>(mem_handle.data_ptr()),
                                   cudaIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<fptr_t>(ipc_ptr);
}

void vllm_custom_ar_free_shared_buffer(fptr_t buffer) {
    CUDACHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}

}  // namespace rtp_llm
