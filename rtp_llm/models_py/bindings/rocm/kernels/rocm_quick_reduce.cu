// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM csrc custom QuickReduce implementation.

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "rtp_llm/models_py/bindings/rocm/kernels/quickreduce/quick_reduce.h"

namespace rtp_llm {

quickreduce::fptr_t
rocm_quick_reduce_init_custom_qr(int64_t rank, int64_t world_size, std::optional<int64_t> qr_max_size) {
    if (world_size > 8) {
        throw std::invalid_argument("world size > 8 is not supported");
    }
    if (world_size == 6) {
        throw std::invalid_argument("world size == 6 is not supported");
    }
    if (world_size % 2 != 0) {
        throw std::invalid_argument("Odd num gpus is not supported for now");
    }
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument("invalid rank passed in");
    }
    quickreduce::DeviceComms* fptr = new quickreduce::DeviceComms();
    fptr->init(world_size, rank, qr_max_size);
    return reinterpret_cast<quickreduce::fptr_t>(fptr);
}

void rocm_quick_reduce_destroy(quickreduce::fptr_t handle) {
    if (handle) {
        auto* fa = reinterpret_cast<quickreduce::DeviceComms*>(handle);
        fa->destroy();
        delete fa;
    }
}

torch::Tensor rocm_quick_reduce_get_handle(quickreduce::fptr_t handle) {
    auto*             fa          = reinterpret_cast<quickreduce::DeviceComms*>(handle);
    hipIpcMemHandle_t ipc_handle  = fa->get_handle();
    auto              options     = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto              data_handle = torch::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))}, options);
    std::memcpy(data_handle.data_ptr(), &ipc_handle, sizeof(hipIpcMemHandle_t));
    return data_handle;
}

void rocm_quick_reduce_open_handles(quickreduce::fptr_t handle, const std::vector<torch::Tensor>& handles) {
    auto*                          fa = reinterpret_cast<quickreduce::DeviceComms*>(handle);
    std::vector<hipIpcMemHandle_t> ipc_handles;
    ipc_handles.reserve(handles.size());
    for (const auto& handle_tensor : handles) {
        hipIpcMemHandle_t ipc_handle;
        std::memcpy(&ipc_handle, handle_tensor.data_ptr(), sizeof(hipIpcMemHandle_t));
        ipc_handles.push_back(ipc_handle);
    }
    fa->open_ipc_handles(ipc_handles);
}

void rocm_quick_reduce_all_reduce(
    quickreduce::fptr_t handle, torch::Tensor& inp, torch::Tensor& out, int64_t quant_level, bool cast_bf2half) {
    auto*            fa = reinterpret_cast<quickreduce::DeviceComms*>(handle);
    c10::DeviceGuard device_guard(inp.device());
    auto             stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

    TORCH_CHECK(inp.scalar_type() == out.scalar_type());
    TORCH_CHECK(inp.numel() == out.numel());
    TORCH_CHECK(out.numel() <= fa->kMaxProblemSize);

    if (out.scalar_type() == at::ScalarType::Half) {
        fa->allreduce<half, false>(reinterpret_cast<half*>(inp.data_ptr()),
                                   reinterpret_cast<half*>(out.data_ptr()),
                                   out.numel(),
                                   quant_level,
                                   stream);
    } else if (out.scalar_type() == at::ScalarType::BFloat16) {
        if (cast_bf2half) {
            fa->allreduce<half, true>(reinterpret_cast<half*>(inp.data_ptr()),
                                      reinterpret_cast<half*>(out.data_ptr()),
                                      out.numel(),
                                      quant_level,
                                      stream);
        } else {
            fa->allreduce<quickreduce::nv_bfloat16, false>(reinterpret_cast<quickreduce::nv_bfloat16*>(inp.data_ptr()),
                                                           reinterpret_cast<quickreduce::nv_bfloat16*>(out.data_ptr()),
                                                           out.numel(),
                                                           quant_level,
                                                           stream);
        }
    } else {
        throw std::runtime_error("quick allreduce only supports float16 and bfloat16");
    }
}

int64_t rocm_quick_reduce_max_size() {
    return static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
}

}  // namespace rtp_llm

namespace quickreduce {

#define INSTANTIATE_FOR_WORLDSIZE(T, Codec, cast_bf2half)                                                              \
    template struct AllReduceTwoshot<T, Codec<T, 2>, cast_bf2half>;                                                    \
    template struct AllReduceTwoshot<T, Codec<T, 4>, cast_bf2half>;                                                    \
    template struct AllReduceTwoshot<T, Codec<T, 8>, cast_bf2half>;

INSTANTIATE_FOR_WORLDSIZE(nv_bfloat16, CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(nv_bfloat16, CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(nv_bfloat16, CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(nv_bfloat16, CodecQ8, false)
INSTANTIATE_FOR_WORLDSIZE(nv_bfloat16, CodecFP, true)
INSTANTIATE_FOR_WORLDSIZE(nv_bfloat16, CodecQ4, true)
INSTANTIATE_FOR_WORLDSIZE(nv_bfloat16, CodecQ6, true)
INSTANTIATE_FOR_WORLDSIZE(nv_bfloat16, CodecQ8, true)

INSTANTIATE_FOR_WORLDSIZE(half, CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecQ8, false)

#undef INSTANTIATE_FOR_WORLDSIZE

}  // namespace quickreduce
