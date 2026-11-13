// SPDX-License-Identifier: Apache-2.0
// Build with CUDA 13.2 and matching NCCL 2.28 or newer headers and runtime.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using Bf16                          = __nv_bfloat16;
constexpr int            kMaxRows   = 32;
constexpr int            kThreads   = 128;
constexpr int            kMaxBlocks = 64;
thread_local std::string error_message;

void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void check(ncclResult_t status) {
    if (status != ncclSuccess)
        throw std::runtime_error(ncclGetErrorString(status));
}

struct Context {
    int          device;
    int          hidden;
    ncclComm_t   comm = nullptr;
    ncclDevComm  dev{};
    ncclWindow_t window{};
    void*        input       = nullptr;
    bool         dev_created = false;
    bool         registered  = false;

    Context(int ordinal, int hidden_size): device(ordinal), hidden(hidden_size) {}
    ~Context() {
        // This path is also used after partial construction. No kernel may
        // consume the mapping until construction succeeds on both ranks.
        cudaSetDevice(device);
        if (comm) {
            if (dev_created)
                ncclDevCommDestroy(comm, &dev);
            if (registered)
                ncclCommWindowDeregister(comm, window);
            ncclCommAbort(comm);
        }
        if (input)
            ncclMemFree(input);
    }
};

struct alignas(16) Pack {
    Bf16 values[8];
};

__global__ void reduce_gate(
    Bf16* output, const Bf16* shared, const Bf16* gate, int count, int hidden, ncclDevComm comm, ncclWindow_t window) {
    const auto*                        local = static_cast<const Pack*>(ncclGetLsaPointer(window, 0, comm.lsaRank));
    const auto*                        peer  = static_cast<const Pack*>(ncclGetLsaPointer(window, 0, 1 - comm.lsaRank));
    const auto*                        shared_packs = reinterpret_cast<const Pack*>(shared);
    auto*                              output_packs = reinterpret_cast<Pack*>(output);
    ncclCoopCta                        coop;
    ncclLsaBarrierSession<ncclCoopCta> barrier{coop, comm, ncclTeamTagLsa(), blockIdx.x};
    // The caller stages its local partial on this same CUDA stream.
    barrier.sync(coop, cuda::memory_order_acq_rel);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count / 8; i += blockDim.x * gridDim.x) {
        const Pack  x          = local[i];
        const Pack  y          = peer[i];
        const Pack  addend     = shared_packs[i];
        const float gate_value = __bfloat162float(gate[(i * 8) / hidden]);
        const float scale      = __fdividef(1.0f, 1.0f + __expf(-gate_value));
        Pack        result;
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const Bf16 routed = __float2bfloat16(__bfloat162float(x.values[lane]) + __bfloat162float(y.values[lane]));
            result.values[lane] =
                __float2bfloat16(fmaf(scale, __bfloat162float(addend.values[lane]), __bfloat162float(routed)));
        }
        output_packs[i] = result;
    }
    // All peers finish reading before a later layer can reuse the window.
    barrier.sync(coop, cuda::memory_order_acq_rel);
}

template<typename Function>
int protect(Function function) {
    try {
        function();
        error_message.clear();
        return 0;
    } catch (const std::exception& error) {
        error_message = error.what();
        return 1;
    }
}
}  // namespace

extern "C" int qwen38_lsa_abi() {
    return 3;
}
extern "C" const char* qwen38_lsa_error() {
    return error_message.c_str();
}
extern "C" int qwen38_lsa_uid_size() {
    return sizeof(ncclUniqueId);
}

extern "C" int qwen38_lsa_unique_id(void* output) {
    return protect([&] { check(ncclGetUniqueId(static_cast<ncclUniqueId*>(output))); });
}

extern "C" int qwen38_lsa_create(int rank, int device, int hidden, const void* unique_id, void** output) {
    return protect([&] {
        *output = nullptr;
        if (rank < 0 || rank > 1)
            throw std::runtime_error("LSA requires two ranks");
        if (hidden <= 0 || hidden % 8 != 0 || hidden > std::numeric_limits<int>::max() / kMaxRows)
            throw std::runtime_error("LSA hidden size must be positive, divisible by 8 and fit its row capacity");
        check(cudaSetDevice(device));
        int major, minor;
        check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        if (major != 12 || minor != 0)
            throw std::runtime_error("LSA MoE requires SM120");
        int runtime_version;
        check(ncclGetVersion(&runtime_version));
        if (runtime_version / 100 != NCCL_VERSION_CODE / 100)
            throw std::runtime_error("Build the LSA library with matching NCCL major/minor headers");
        auto context = std::make_unique<Context>(device, hidden);
        check(ncclCommInitRank(&context->comm, 2, *static_cast<const ncclUniqueId*>(unique_id), rank));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
        ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
        check(ncclCommQueryProperties(context->comm, &properties));
        if (!properties.deviceApiSupport || properties.nLsaTeams != 1)
            throw std::runtime_error("NCCL device LSA is unavailable");
#endif
        const size_t window_bytes = static_cast<size_t>(kMaxRows) * hidden * sizeof(Bf16);
        check(ncclMemAlloc(&context->input, window_bytes));
        check(ncclCommWindowRegister(
            context->comm, context->input, window_bytes, &context->window, NCCL_WIN_COLL_SYMMETRIC));
        context->registered = true;
#ifdef NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
        ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#else
        ncclDevCommRequirements requirements{};
#endif
        requirements.lsaBarrierCount = kMaxBlocks;
        check(ncclDevCommCreate(context->comm, &requirements, &context->dev));
        context->dev_created = true;
        if (context->dev.lsaSize != 2)
            throw std::runtime_error("Expected two LSA peers");
        *output = context.release();
    });
}

extern "C" int qwen38_lsa_apply(void*       handle,
                                const void* input,
                                const void* shared,
                                const void* gate,
                                void*       output,
                                int         rows,
                                void*       stream_pointer) {
    return protect([&] {
        auto* context = static_cast<Context*>(handle);
        if (!context || rows < 1 || rows > kMaxRows)
            throw std::runtime_error("Invalid LSA context or row count");
        int device;
        check(cudaGetDevice(&device));
        if (device != context->device)
            throw std::runtime_error("LSA device mismatch");
        const auto   stream = static_cast<cudaStream_t>(stream_pointer);
        const size_t bytes  = rows * context->hidden * sizeof(Bf16);
        check(cudaMemcpyAsync(context->input, input, bytes, cudaMemcpyDeviceToDevice, stream));
        const int blocks = std::min(kMaxBlocks, (rows * context->hidden - 1) / 256 + 1);
        reduce_gate<<<blocks, kThreads, 0, stream>>>(static_cast<Bf16*>(output),
                                                     static_cast<const Bf16*>(shared),
                                                     static_cast<const Bf16*>(gate),
                                                     rows * context->hidden,
                                                     context->hidden,
                                                     context->dev,
                                                     context->window);
        check(cudaGetLastError());
    });
}

// Out-of-place NCCL reference for the paired integration benchmark.
extern "C" int qwen38_lsa_reference(void* handle, const void* input, void* output, int rows, void* stream_pointer) {
    return protect([&] {
        auto* context = static_cast<Context*>(handle);
        if (!context || rows < 1 || rows > kMaxRows)
            throw std::runtime_error("Invalid NCCL reference context or rows");
        check(ncclAllReduce(input,
                            output,
                            rows * context->hidden,
                            ncclBfloat16,
                            ncclSum,
                            context->comm,
                            static_cast<cudaStream_t>(stream_pointer)));
    });
}

extern "C" void qwen38_lsa_destroy(void* handle) {
    delete static_cast<Context*>(handle);
}
