/*
 * Copyright (c) 2026, Alibaba Group. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
// Correctness-first TP=2 FC2/finalize plus one-shot FP8 IPC all-reduce.
// This intentionally uses the scalar fc2_reference contract. It is a protocol
// baseline, not an MMA implementation or a performance claim.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "fc2_reference.cuh"
#include "moe_tp_fused_fp8_transport.cuh"

namespace {
namespace fused = rtp_llm::moe_tp_fused_fp8;

constexpr int      kDefaultBlocks    = 16;
constexpr uint64_t kWireMagic        = 0x5254504D4F454631ULL;  // RTPMOEF1
constexpr uint32_t kWireVersion      = 1;
constexpr uint64_t kSpinLimit        = 1ULL << 26;
constexpr uint64_t kErrorPeerTimeout = 1ULL << 63;
constexpr uint64_t kErrorNumeric     = 1ULL << 62;

inline void cuda_check(cudaError_t status, char const* operation) {
    TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

inline size_t div_up(size_t numerator, size_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

struct IpcWireHandle {
    uint64_t           magic;
    uint32_t           version;
    int32_t            device;
    int32_t            rank;
    int32_t            blocks;
    uint64_t           max_numel;
    uint64_t           packet_count;
    cudaIpcMemHandle_t packets;
    cudaIpcMemHandle_t ready;
    cudaIpcMemHandle_t ack;
    cudaIpcMemHandle_t error;
};
static_assert(std::is_trivially_copyable_v<IpcWireHandle>);

struct DeviceParams {
    fused::TransportWorkspace local;
    fused::TransportWorkspace peer;
    uint64_t                  epoch;
    size_t                    packet_count;
    size_t                    numel;
    int                       rank;
};

__device__ __forceinline__ void set_timeout(uint64_t* error) {
    atomicOr(reinterpret_cast<unsigned long long*>(error), static_cast<unsigned long long>(kErrorPeerTimeout));
}

__device__ __forceinline__ void write_packet(fused::Packet*       packet,
                                             __nv_bfloat16 const* source,
                                             size_t               numel,
                                             size_t               packet_index,
                                             float*               warp_maxima,
                                             uint64_t*            error) {
    const size_t base      = packet_index * fused::kPacketValues;
    float        local_max = 0.f;
    for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
        const size_t index = base + item;
        const float  value = index < numel ? __bfloat162float(source[index]) : 0.f;
        if (!isfinite(value))
            atomicOr(reinterpret_cast<unsigned long long*>(error), static_cast<unsigned long long>(kErrorNumeric));
        local_max = fused::max_abs(value, local_max);
    }
    const float max_abs = fused::block_max_abs(local_max, warp_maxima);
    const float scale   = max_abs == 0.f ? 0.f : 448.f / max_abs;
    if (threadIdx.x == 0 && !isfinite(scale))
        atomicOr(reinterpret_cast<unsigned long long*>(error), static_cast<unsigned long long>(kErrorNumeric));
    for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
        const size_t index   = base + item;
        const float  value   = index < numel ? __bfloat162float(source[index]) : 0.f;
        packet->values[item] = static_cast<__nv_fp8_e4m3>(scale == 0.f ? value : value * scale);
    }
    if (threadIdx.x == 0) {
        packet->scale       = scale;
        packet->reserved[0] = 0;
        packet->reserved[1] = 0;
        packet->reserved[2] = 0;
    }
}

__device__ __forceinline__ bool publish_reduce_and_ack(
    DeviceParams const& params, __nv_bfloat16* out, size_t packet_index, float* warp_maxima, int* protocol_status) {
    fused::Packet* local_packet = params.local.packets + packet_index;
    write_packet(local_packet, out, params.numel, packet_index, warp_maxima, params.local.error);
    // The system fence is issued by every packet writer. A block barrier alone
    // cannot make another thread's global stores system-visible to a PCIe peer.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        fused::store_release_sys(params.epoch, params.local.ready + packet_index);
    __syncthreads();

    if (threadIdx.x == 0)
        *protocol_status = fused::wait_epoch(params.peer.ready + packet_index, params.epoch, kSpinLimit) ? 1 : 0;
    __syncthreads();
    if (!*protocol_status) {
        if (threadIdx.x == 0)
            set_timeout(params.local.error);
        return false;
    }

    const fused::Packet* peer_packet = params.peer.packets + packet_index;
    const size_t         base        = packet_index * fused::kPacketValues;
    for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
        const size_t index = base + item;
        if (index < params.numel) {
            const float own   = local_packet->scale == 0.f ?
                                    static_cast<float>(local_packet->values[item]) :
                                    static_cast<float>(local_packet->values[item]) / local_packet->scale;
            const float other = peer_packet->scale == 0.f ?
                                    static_cast<float>(peer_packet->values[item]) :
                                    static_cast<float>(peer_packet->values[item]) / peer_packet->scale;
            // Rank ordering is part of the numeric contract, even though a
            // two-addend finite sum is usually commutative.
            out[index] = __float2bfloat16_rn(params.rank == 0 ? own + other : other + own);
            if (!isfinite(__bfloat162float(out[index])))
                atomicOr(reinterpret_cast<unsigned long long*>(params.local.error),
                         static_cast<unsigned long long>(kErrorNumeric));
        }
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0)
        fused::store_release_sys(params.epoch, params.local.ack + packet_index);
    __syncthreads();
    if (threadIdx.x == 0)
        *protocol_status = fused::wait_epoch(params.peer.ack + packet_index, params.epoch, kSpinLimit) ? 1 : 0;
    __syncthreads();
    if (!*protocol_status && threadIdx.x == 0)
        set_timeout(params.local.error);
    return *protocol_status != 0;
}

__global__ void one_shot_all_reduce_kernel(DeviceParams params, __nv_bfloat16 const* input, __nv_bfloat16* output) {
    __shared__ float warp_maxima[fused::kPacketThreads / 32];
    __shared__ int   protocol_status;
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    for (size_t packet = blockIdx.x; packet < params.packet_count; packet += gridDim.x) {
        const size_t base = packet * fused::kPacketValues;
        for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
            const size_t index = base + item;
            if (index < params.numel)
                output[index] = input[index];
        }
        __syncthreads();
        if (!publish_reduce_and_ack(params, output, packet, warp_maxima, &protocol_status))
            return;
    }
}

__global__ void local_fc2_kernel(fused::Fc2ReferenceParams fc2, __nv_bfloat16* output, size_t numel) {
    if (reinterpret_cast<uint64_t const*>(fc2.error_flags) != nullptr
        && fused::load_acquire_sys(reinterpret_cast<uint64_t const*>(fc2.error_flags)) != 0)
        return;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < numel;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        output[index] = fused::fc2_token_h_reference(
            fc2, index / fused::kFc2ReferenceHiddenSize, index % fused::kFc2ReferenceHiddenSize);
    }
}

__global__ void fused_fc2_kernel(DeviceParams params, fused::Fc2ReferenceParams fc2, __nv_bfloat16* output) {
    __shared__ float warp_maxima[fused::kPacketThreads / 32];
    __shared__ int   protocol_status;
    if (threadIdx.x == 0)
        protocol_status = fused::load_acquire_sys(params.local.error) == 0 ? 1 : 0;
    __syncthreads();
    if (!protocol_status)
        return;
    for (size_t packet = blockIdx.x; packet < params.packet_count; packet += gridDim.x) {
        const size_t base = packet * fused::kPacketValues;
        for (int item = threadIdx.x; item < fused::kPacketValues; item += blockDim.x) {
            const size_t index = base + item;
            if (index < params.numel)
                output[index] = fused::fc2_token_h_reference(
                    fc2, index / fused::kFc2ReferenceHiddenSize, index % fused::kFc2ReferenceHiddenSize);
        }
        __syncthreads();
        if (!publish_reduce_and_ack(params, output, packet, warp_maxima, &protocol_status))
            return;
    }
}

class MoeTpFusedFp8 {
public:
    MoeTpFusedFp8(uint64_t max_numel, int device_index, int rank, int blocks = kDefaultBlocks):
        max_numel_(max_numel),
        device_(device_index),
        rank_(rank),
        blocks_(blocks),
        packet_capacity_(div_up(max_numel, fused::kPacketValues)) {
        TORCH_CHECK(max_numel_ > 0, "max_numel must be positive");
        TORCH_CHECK(rank_ >= 0 && rank_ < fused::kTpSize, "rank must be 0 or 1");
        TORCH_CHECK(blocks_ > 0, "blocks must be positive");
        c10::cuda::CUDAGuard guard(device_);
        ensure_resident_grid();
        try {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&packets_), packet_capacity_ * sizeof(fused::Packet)),
                       "cudaMalloc(packets)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&ready_), packet_capacity_ * sizeof(uint64_t)),
                       "cudaMalloc(ready)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&ack_), packet_capacity_ * sizeof(uint64_t)),
                       "cudaMalloc(ack)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&error_), sizeof(uint64_t)), "cudaMalloc(error)");
            cuda_check(cudaMemsetAsync(ready_, 0, packet_capacity_ * sizeof(uint64_t), current_stream()),
                       "cudaMemsetAsync(ready)");
            cuda_check(cudaMemsetAsync(ack_, 0, packet_capacity_ * sizeof(uint64_t), current_stream()),
                       "cudaMemsetAsync(ack)");
            cuda_check(cudaMemsetAsync(error_, 0, sizeof(uint64_t), current_stream()), "cudaMemsetAsync(error)");
        } catch (...) {
            release_noexcept();
            throw;
        }
    }

    ~MoeTpFusedFp8() {
        release_noexcept();
    }

    py::bytes get_ipc_handle() {
        ensure_open();
        c10::cuda::CUDAGuard guard(device_);
        IpcWireHandle        wire{};
        wire.magic        = kWireMagic;
        wire.version      = kWireVersion;
        wire.device       = device_;
        wire.rank         = rank_;
        wire.blocks       = blocks_;
        wire.max_numel    = max_numel_;
        wire.packet_count = packet_capacity_;
        cuda_check(cudaIpcGetMemHandle(&wire.packets, packets_), "cudaIpcGetMemHandle(packets)");
        cuda_check(cudaIpcGetMemHandle(&wire.ready, ready_), "cudaIpcGetMemHandle(ready)");
        cuda_check(cudaIpcGetMemHandle(&wire.ack, ack_), "cudaIpcGetMemHandle(ack)");
        cuda_check(cudaIpcGetMemHandle(&wire.error, error_), "cudaIpcGetMemHandle(error)");
        return py::bytes(reinterpret_cast<char const*>(&wire), sizeof(wire));
    }

    void open_peer(py::bytes raw) {
        ensure_open();
        const std::string bytes = raw;
        TORCH_CHECK(bytes.size() == sizeof(IpcWireHandle), "invalid fused MoE IPC handle size");
        IpcWireHandle wire{};
        std::memcpy(&wire, bytes.data(), sizeof(wire));
        TORCH_CHECK(wire.magic == kWireMagic && wire.version == kWireVersion, "invalid fused MoE IPC wire version");
        TORCH_CHECK(wire.rank == 1 - rank_ && wire.max_numel == max_numel_ && wire.blocks == blocks_
                        && wire.packet_count == packet_capacity_,
                    "peer fused MoE workspace mismatch");
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
        try {
            cuda_check(cudaIpcOpenMemHandle(
                           reinterpret_cast<void**>(&peer_packets_), wire.packets, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(packets; requires CUDA IPC P2P)");
            cuda_check(cudaIpcOpenMemHandle(
                           reinterpret_cast<void**>(&peer_ready_), wire.ready, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(ready; requires CUDA IPC P2P)");
            cuda_check(
                cudaIpcOpenMemHandle(reinterpret_cast<void**>(&peer_ack_), wire.ack, cudaIpcMemLazyEnablePeerAccess),
                "cudaIpcOpenMemHandle(ack; requires CUDA IPC P2P)");
            cuda_check(cudaIpcOpenMemHandle(
                           reinterpret_cast<void**>(&peer_error_), wire.error, cudaIpcMemLazyEnablePeerAccess),
                       "cudaIpcOpenMemHandle(error; requires CUDA IPC P2P)");
        } catch (...) {
            close_peer_noexcept();
            throw;
        }
    }

    void close_peer() {
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
    }

    void all_reduce(torch::Tensor input, torch::Tensor output) {
        validate_bf16_pair(input, output, "all_reduce");
        launch_common(input.numel(), [&](DeviceParams params, cudaStream_t stream) {
            one_shot_all_reduce_kernel<<<blocks_, fused::kPacketThreads, 0, stream>>>(
                params, bf16_ptr(input), bf16_ptr(output));
            cuda_check(cudaGetLastError(), "one_shot_all_reduce_kernel launch");
        });
    }

    void local_fc2(torch::Tensor activation_fp8,
                   torch::Tensor activation_scale,
                   torch::Tensor weight_fp8,
                   torch::Tensor weight_scale,
                   torch::Tensor route_ids,
                   torch::Tensor route_weights,
                   py::object    gated_shared,
                   torch::Tensor output) {
        const auto   fc2   = validate_fc2(activation_fp8,
                                          activation_scale,
                                          weight_fp8,
                                          weight_scale,
                                          route_ids,
                                          route_weights,
                                          gated_shared,
                                          output,
                                          /*require_peer=*/false);
        const size_t numel = output.numel();
        bind_stream();
        c10::cuda::CUDAGuard guard(device_);
        local_fc2_kernel<<<blocks_, fused::kPacketThreads, 0, current_stream()>>>(fc2, bf16_ptr(output), numel);
        cuda_check(cudaGetLastError(), "local_fc2_kernel launch");
    }

    void fused_fc2(torch::Tensor activation_fp8,
                   torch::Tensor activation_scale,
                   torch::Tensor weight_fp8,
                   torch::Tensor weight_scale,
                   torch::Tensor route_ids,
                   torch::Tensor route_weights,
                   py::object    gated_shared,
                   torch::Tensor output) {
        const auto fc2 = validate_fc2(activation_fp8,
                                      activation_scale,
                                      weight_fp8,
                                      weight_scale,
                                      route_ids,
                                      route_weights,
                                      gated_shared,
                                      output,
                                      /*require_peer=*/true);
        launch_common(output.numel(), [&](DeviceParams params, cudaStream_t stream) {
            fused_fc2_kernel<<<blocks_, fused::kPacketThreads, 0, stream>>>(params, fc2, bf16_ptr(output));
            cuda_check(cudaGetLastError(), "fused_fc2_kernel launch");
        });
    }

    torch::Tensor copy_local_packets() {
        ensure_open();
        TORCH_CHECK(last_packets_ != 0, "no fused/all_reduce invocation has published packets");
        c10::cuda::CUDAGuard guard(device_);
        auto                 options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_);
        auto result = torch::empty({static_cast<int64_t>(last_packets_ * sizeof(fused::Packet))}, options);
        cuda_check(cudaMemcpyAsync(result.data_ptr(),
                                   packets_,
                                   last_packets_ * sizeof(fused::Packet),
                                   cudaMemcpyDeviceToDevice,
                                   current_stream()),
                   "cudaMemcpyAsync(copy_local_packets)");
        return result;
    }

    // Debug-only asynchronous device scalar. Callers must first synchronize the
    // context's dedicated stream, then may consume this tensor on their current
    // stream. It deliberately does not bind that debug copy to the launch stream.
    torch::Tensor error_status() {
        ensure_open();
        c10::cuda::CUDAGuard guard(device_);
        auto result = torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA, device_));
        cuda_check(
            cudaMemcpyAsync(result.data_ptr(), error_, sizeof(uint64_t), cudaMemcpyDeviceToDevice, current_stream()),
            "cudaMemcpyAsync(error_status)");
        return result;
    }

    int blocks() const {
        return blocks_;
    }

    void close() {
        c10::cuda::CUDAGuard guard(device_);
        close_peer_impl();
        free_local_checked();
        bound_stream_ = nullptr;
        stream_bound_ = false;
        last_packets_ = 0;
    }

private:
    cudaStream_t current_stream() const {
        return at::cuda::getCurrentCUDAStream(device_).stream();
    }

    void ensure_open() const {
        TORCH_CHECK(packets_ && ready_ && ack_ && error_, "MoeTpFusedFp8 context is closed");
    }

    void ensure_peer_open() const {
        ensure_open();
        TORCH_CHECK(peer_packets_ && peer_ready_ && peer_ack_ && peer_error_, "open_peer must be called first");
    }

    void bind_stream() {
        const cudaStream_t stream = current_stream();
        if (stream_bound_) {
            TORCH_CHECK(bound_stream_ == stream, "MoeTpFusedFp8 context may only use one CUDA stream");
        } else {
            bound_stream_ = stream;
            stream_bound_ = true;
        }
        cudaStreamCaptureStatus status{};
        cuda_check(cudaStreamIsCapturing(stream, &status), "cudaStreamIsCapturing");
        TORCH_CHECK(status == cudaStreamCaptureStatusNone, "MoeTpFusedFp8 does not support CUDA graph capture");
    }

    void ensure_resident_grid() {
        int active = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fused_fc2_kernel, fused::kPacketThreads, 0),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor(fused_fc2_kernel)");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        const int64_t limit = static_cast<int64_t>(active) * properties.multiProcessorCount;
        TORCH_CHECK(blocks_ <= limit, "fused MoE blocks=", blocks_, " exceeds resident limit=", limit);
    }

    fused::Fc2ReferenceParams validate_fc2(torch::Tensor activation_fp8,
                                           torch::Tensor activation_scale,
                                           torch::Tensor weight_fp8,
                                           torch::Tensor weight_scale,
                                           torch::Tensor route_ids,
                                           torch::Tensor route_weights,
                                           py::object    gated_shared,
                                           torch::Tensor output,
                                           bool          require_peer) const {
        if (require_peer)
            ensure_peer_open();
        else
            ensure_open();
        validate_bf16_output(output, "fused/local FC2 output");
        auto check_cuda_contiguous = [&](torch::Tensor const& tensor, char const* name) {
            TORCH_CHECK(tensor.is_cuda() && tensor.device().index() == device_ && tensor.is_contiguous(),
                        name,
                        " must be a contiguous CUDA tensor on the context device");
        };
        check_cuda_contiguous(activation_fp8, "activation_fp8");
        check_cuda_contiguous(activation_scale, "activation_scale");
        check_cuda_contiguous(weight_fp8, "weight_fp8");
        check_cuda_contiguous(weight_scale, "weight_scale");
        check_cuda_contiguous(route_ids, "route_ids");
        check_cuda_contiguous(route_weights, "route_weights");
        TORCH_CHECK(activation_fp8.scalar_type() == at::kFloat8_e4m3fn
                        && weight_fp8.scalar_type() == at::kFloat8_e4m3fn,
                    "activation_fp8 and weight_fp8 must be Float8_e4m3fn");
        TORCH_CHECK(activation_scale.scalar_type() == at::kFloat && weight_scale.scalar_type() == at::kFloat
                        && route_weights.scalar_type() == at::kFloat && route_ids.scalar_type() == at::kInt,
                    "scales/route_weights must be float32 and route_ids int32");
        TORCH_CHECK(activation_fp8.dim() == 3 && activation_fp8.size(1) == fused::kFc2ReferenceTopK
                        && activation_fp8.size(2) == fused::kFc2ReferenceIntermediateSize,
                    "activation_fp8 must be [T,8,256]");
        const int tokens = activation_fp8.size(0);
        TORCH_CHECK(activation_scale.dim() == 3 && activation_scale.size(0) == tokens
                        && activation_scale.size(1) == fused::kFc2ReferenceTopK
                        && activation_scale.size(2) == fused::kFc2ReferenceKBlocks,
                    "activation_scale must be [T,8,2]");
        TORCH_CHECK(weight_fp8.dim() == 3 && weight_fp8.size(1) == fused::kFc2ReferenceHiddenSize
                        && weight_fp8.size(2) == fused::kFc2ReferenceIntermediateSize,
                    "weight_fp8 must be [E,2048,256]");
        const int experts = weight_fp8.size(0);
        TORCH_CHECK(experts > 0 && experts <= fused::kFc2ReferenceExperts, "weight expert count must be in [1,256]");
        TORCH_CHECK(weight_scale.dim() == 3 && weight_scale.size(0) == experts
                        && weight_scale.size(1) == fused::kFc2ReferenceHBlocks
                        && weight_scale.size(2) == fused::kFc2ReferenceKBlocks,
                    "weight_scale must be [E,16,2]");
        TORCH_CHECK(route_ids.dim() == 2 && route_ids.size(0) == tokens && route_ids.size(1) == fused::kFc2ReferenceTopK
                        && route_weights.dim() == 2 && route_weights.size(0) == tokens
                        && route_weights.size(1) == fused::kFc2ReferenceTopK,
                    "route ids/weights must be [T,8]");
        TORCH_CHECK(output.dim() == 2 && output.size(0) == tokens && output.size(1) == fused::kFc2ReferenceHiddenSize
                        && static_cast<uint64_t>(output.numel()) <= max_numel_,
                    "output must be [T,2048] and fit the context workspace");
        __nv_bfloat16 const* shared_ptr = nullptr;
        if (!gated_shared.is_none()) {
            auto shared = gated_shared.cast<torch::Tensor>();
            check_cuda_contiguous(shared, "gated_shared");
            TORCH_CHECK(shared.scalar_type() == at::kBFloat16 && shared.sizes() == output.sizes(),
                        "gated_shared must be BF16 [T,2048]");
            shared_ptr = bf16_ptr(shared);
        }
        return {reinterpret_cast<__nv_fp8_e4m3 const*>(activation_fp8.data_ptr()),
                activation_scale.data_ptr<float>(),
                reinterpret_cast<__nv_fp8_e4m3 const*>(weight_fp8.data_ptr()),
                weight_scale.data_ptr<float>(),
                route_ids.data_ptr<int32_t>(),
                route_weights.data_ptr<float>(),
                shared_ptr,
                reinterpret_cast<uint32_t*>(error_),
                tokens,
                experts};
    }

    void validate_bf16_output(torch::Tensor const& output, char const* name) const {
        TORCH_CHECK(output.is_cuda() && output.device().index() == device_ && output.is_contiguous()
                        && output.scalar_type() == at::kBFloat16,
                    name,
                    " must be contiguous BF16 CUDA tensor on context device");
    }

    void validate_bf16_pair(torch::Tensor const& input, torch::Tensor const& output, char const* name) const {
        ensure_peer_open();
        validate_bf16_output(input, name);
        validate_bf16_output(output, name);
        TORCH_CHECK(input.numel() == output.numel() && input.numel() > 0
                        && static_cast<uint64_t>(input.numel()) <= max_numel_,
                    name,
                    " requires matching nonempty tensors within max_numel");
    }

    template<class Launch>
    void launch_common(size_t numel, Launch&& launch) {
        ensure_peer_open();
        bind_stream();
        c10::cuda::CUDAGuard guard(device_);
        const size_t         packets = div_up(numel, fused::kPacketValues);
        TORCH_CHECK(packets <= packet_capacity_, "workspace packet capacity exceeded");
        const uint64_t epoch = ++epoch_;
        TORCH_CHECK(epoch != 0, "fused MoE epoch overflow");
        last_packets_ = packets;
        DeviceParams params{{packets_, ready_, ack_, error_},
                            {peer_packets_, peer_ready_, peer_ack_, peer_error_},
                            epoch,
                            packets,
                            numel,
                            rank_};
        launch(params, current_stream());
    }

    static __nv_bfloat16 const* bf16_ptr(torch::Tensor const& tensor) {
        return reinterpret_cast<__nv_bfloat16 const*>(tensor.data_ptr());
    }
    static __nv_bfloat16* bf16_ptr(torch::Tensor& tensor) {
        return reinterpret_cast<__nv_bfloat16*>(tensor.data_ptr());
    }

    void close_peer_impl() {
        if (peer_packets_)
            cuda_check(cudaIpcCloseMemHandle(peer_packets_), "cudaIpcCloseMemHandle(packets)");
        if (peer_ready_)
            cuda_check(cudaIpcCloseMemHandle(peer_ready_), "cudaIpcCloseMemHandle(ready)");
        if (peer_ack_)
            cuda_check(cudaIpcCloseMemHandle(peer_ack_), "cudaIpcCloseMemHandle(ack)");
        if (peer_error_)
            cuda_check(cudaIpcCloseMemHandle(peer_error_), "cudaIpcCloseMemHandle(error)");
        peer_packets_ = nullptr;
        peer_ready_   = nullptr;
        peer_ack_     = nullptr;
        peer_error_   = nullptr;
    }

    void close_peer_noexcept() noexcept {
        if (peer_packets_)
            cudaIpcCloseMemHandle(peer_packets_);
        if (peer_ready_)
            cudaIpcCloseMemHandle(peer_ready_);
        if (peer_ack_)
            cudaIpcCloseMemHandle(peer_ack_);
        if (peer_error_)
            cudaIpcCloseMemHandle(peer_error_);
        peer_packets_ = nullptr;
        peer_ready_   = nullptr;
        peer_ack_     = nullptr;
        peer_error_   = nullptr;
    }

    void free_local_checked() {
        if (packets_) {
            cuda_check(cudaFree(packets_), "cudaFree(packets)");
            packets_ = nullptr;
        }
        if (ready_) {
            cuda_check(cudaFree(ready_), "cudaFree(ready)");
            ready_ = nullptr;
        }
        if (ack_) {
            cuda_check(cudaFree(ack_), "cudaFree(ack)");
            ack_ = nullptr;
        }
        if (error_) {
            cuda_check(cudaFree(error_), "cudaFree(error)");
            error_ = nullptr;
        }
    }

    void release_noexcept() noexcept {
        int        old_device  = -1;
        const bool have_device = cudaGetDevice(&old_device) == cudaSuccess;
        if (have_device && old_device != device_)
            cudaSetDevice(device_);
        close_peer_noexcept();
        if (packets_)
            cudaFree(packets_);
        if (ready_)
            cudaFree(ready_);
        if (ack_)
            cudaFree(ack_);
        if (error_)
            cudaFree(error_);
        packets_ = nullptr;
        ready_   = nullptr;
        ack_     = nullptr;
        error_   = nullptr;
        if (have_device && old_device != device_)
            cudaSetDevice(old_device);
    }

    uint64_t       max_numel_;
    int            device_;
    int            rank_;
    int            blocks_;
    size_t         packet_capacity_;
    fused::Packet* packets_{};
    uint64_t*      ready_{};
    uint64_t*      ack_{};
    uint64_t*      error_{};
    fused::Packet* peer_packets_{};
    uint64_t*      peer_ready_{};
    uint64_t*      peer_ack_{};
    uint64_t*      peer_error_{};
    uint64_t       epoch_{};
    size_t         last_packets_{};
    cudaStream_t   bound_stream_{};
    bool           stream_bound_{};
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MoeTpFusedFp8>(m, "MoeTpFusedFp8")
        .def(py::init<uint64_t, int, int, int>(),
             py::arg("max_numel"),
             py::arg("device_index"),
             py::arg("rank"),
             py::arg("blocks") = kDefaultBlocks)
        .def("get_ipc_handle", &MoeTpFusedFp8::get_ipc_handle)
        .def("open_peer", &MoeTpFusedFp8::open_peer)
        .def("close_peer", &MoeTpFusedFp8::close_peer)
        .def("all_reduce", &MoeTpFusedFp8::all_reduce)
        .def("local_fc2", &MoeTpFusedFp8::local_fc2)
        .def("fused_fc2", &MoeTpFusedFp8::fused_fc2)
        .def("copy_local_packets", &MoeTpFusedFp8::copy_local_packets)
        .def("debug_packet_bytes", &MoeTpFusedFp8::copy_local_packets)
        .def("error_status", &MoeTpFusedFp8::error_status)
        .def("blocks", &MoeTpFusedFp8::blocks)
        .def("close", &MoeTpFusedFp8::close);
}
