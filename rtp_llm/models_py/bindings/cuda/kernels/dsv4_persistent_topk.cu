// Persistent TopK kernel for the DeepSeek-V4 sparse-attention indexer.
// Vendored from vLLM (csrc/topk.cu + csrc/persistent_topk.cuh).
// See dsv4_persistent_topk.h for contract.

#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_persistent_topk.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifndef USE_ROCM
#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_persistent_topk.cuh"
#endif

namespace torch_ext {

namespace {

#ifndef USE_ROCM
struct DevicePropInfo {
    int num_sms            = 0;
    int max_smem_per_block = 0;
};

DevicePropInfo get_device_prop_info(int device) {
    static std::mutex                              mu;
    static std::unordered_map<int, DevicePropInfo> cache;
    std::lock_guard<std::mutex>                    lock(mu);
    auto                                           it = cache.find(device);
    if (it != cache.end()) {
        return it->second;
    }
    DevicePropInfo info;
    cudaDeviceGetAttribute(&info.num_sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&info.max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cache.emplace(device, info);
    return info;
}

class CaptureRelaxedGuard {
public:
    explicit CaptureRelaxedGuard(bool enabled): active_(enabled) {
        if (active_) {
            mode_ = cudaStreamCaptureModeRelaxed;
            cudaThreadExchangeStreamCaptureMode(&mode_);
        }
    }
    ~CaptureRelaxedGuard() {
        if (active_) {
            cudaThreadExchangeStreamCaptureMode(&mode_);
        }
    }

private:
    bool                  active_ = false;
    cudaStreamCaptureMode mode_   = cudaStreamCaptureModeGlobal;
};

vllm::persistent::RadixRowState*
get_radix_row_states(int device, size_t needed_groups, size_t hw_resident_cap, cudaStream_t stream) {
    namespace P = vllm::persistent;
    struct DeviceRadixState {
        P::RadixRowState*  ptr             = nullptr;
        size_t             capacity_groups = 0;
        cudaStream_t       init_stream     = nullptr;
        std::vector<void*> retired_ptrs;
    };

    static std::mutex                                mu;
    static std::unordered_map<int, DeviceRadixState> states;
    std::lock_guard<std::mutex>                      lock(mu);
    auto&                                            st = states[device];

    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    cudaError_t             cap_err    = cudaStreamIsCapturing(stream, &cap_status);
    TORCH_CHECK(cap_err == cudaSuccess, "cudaStreamIsCapturing failed: ", cudaGetErrorString(cap_err));
    const bool is_capturing = (cap_status != cudaStreamCaptureStatusNone);

    if (!is_capturing && !st.retired_ptrs.empty() && st.ptr != nullptr && st.capacity_groups >= needed_groups) {
        cudaDeviceSynchronize();
        for (void* old_ptr : st.retired_ptrs) {
            cudaFree(old_ptr);
        }
        st.retired_ptrs.clear();
    }

    if (st.ptr != nullptr && st.capacity_groups >= needed_groups) {
        return st.ptr;
    }

    CaptureRelaxedGuard guard(is_capturing);
    int                 prev_device = device;
    cudaGetDevice(&prev_device);
    if (prev_device != device) {
        cudaSetDevice(device);
    }
    struct DeviceRestore {
        int  prev;
        bool restore;
        ~DeviceRestore() {
            if (restore) {
                cudaSetDevice(prev);
            }
        }
    } dev_restore{prev_device, prev_device != device};

    if (st.init_stream == nullptr) {
        cudaError_t s_err = cudaStreamCreateWithFlags(&st.init_stream, cudaStreamNonBlocking);
        TORCH_CHECK(s_err == cudaSuccess, "radix init stream creation failed: ", cudaGetErrorString(s_err));
    }

    const size_t target_groups = std::max(needed_groups, hw_resident_cap);
    const size_t target_bytes  = target_groups * sizeof(P::RadixRowState);

    void*       new_raw_ptr = nullptr;
    cudaError_t alloc_err   = cudaMalloc(&new_raw_ptr, target_bytes);
    TORCH_CHECK(alloc_err == cudaSuccess, "radix state allocation failed: ", cudaGetErrorString(alloc_err));

    cudaError_t mz_err = cudaMemsetAsync(new_raw_ptr, 0, target_bytes, st.init_stream);
    TORCH_CHECK(mz_err == cudaSuccess, "radix state pre-zero failed: ", cudaGetErrorString(mz_err));

    cudaError_t sync_err = cudaStreamSynchronize(st.init_stream);
    TORCH_CHECK(sync_err == cudaSuccess, "radix state sync failed: ", cudaGetErrorString(sync_err));

    if (st.ptr != nullptr) {
        if (is_capturing) {
            st.retired_ptrs.push_back(st.ptr);
        } else {
            cudaDeviceSynchronize();
            for (void* old_ptr : st.retired_ptrs) {
                cudaFree(old_ptr);
            }
            st.retired_ptrs.clear();
            cudaFree(st.ptr);
        }
    }

    st.ptr             = reinterpret_cast<P::RadixRowState*>(new_raw_ptr);
    st.capacity_groups = target_groups;
    return st.ptr;
}

template<int TopK>
void launch_persistent_topk(const torch::Tensor& logits,
                            const torch::Tensor& lengths,
                            torch::Tensor&       output,
                            torch::Tensor&       workspace,
                            int64_t              max_seq_len) {
    namespace P = vllm::persistent;

    const int64_t num_rows = logits.size(0);
    const int64_t stride   = logits.stride(0);
    cudaStream_t  stream   = at::cuda::getCurrentCUDAStream();

    // Launch-geometry bound: the caller-supplied upper bound on the per-row
    // ``lengths`` values, clamped to the physical row stride. The chunk grid
    // below used to be derived from ``stride`` — the full padded row width —
    // so decode launched CTAs for pool-capacity columns that no row ever
    // reads (real compressed indexer rows are several times shorter). Chunk
    // geometry follows this bound while addressing keeps the real ``stride``
    // (row pointers, ``params.stride`` and the FilteredTopK fallback below
    // all still use ``stride``). Callers without a host-known length bound
    // pass the full row width, which reproduces the legacy full-width launch.
    const int64_t  bound64     = std::min<int64_t>(std::max<int64_t>(max_seq_len, 1), std::max<int64_t>(stride, 1));
    const uint32_t chunk_bound = static_cast<uint32_t>(bound64);

    const int            device             = logits.get_device();
    const DevicePropInfo prop_info          = get_device_prop_info(device);
    const int            num_sms            = prop_info.num_sms;
    const int            max_smem_per_block = prop_info.max_smem_per_block;

    if (num_rows > 32 && max_smem_per_block >= 128 * 1024) {
        cudaError_t status = vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(logits.data_ptr<float>(),
                                                                                     output.data_ptr<int32_t>(),
                                                                                     lengths.data_ptr<int32_t>(),
                                                                                     static_cast<uint32_t>(num_rows),
                                                                                     static_cast<uint32_t>(TopK),
                                                                                     static_cast<uint32_t>(stride),
                                                                                     stream);
        TORCH_CHECK(status == cudaSuccess, "FilteredTopK failed: ", cudaGetErrorString(status));
    } else {
        TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
        TORCH_CHECK(workspace.dtype() == torch::kUInt8, "workspace must be uint8");

        const bool needs_cooperative = chunk_bound > P::RADIX_THRESHOLD;

        uint32_t ctas_per_group = 1;
        uint32_t chunk_size     = chunk_bound;
        size_t   smem_size      = P::kSmemMedium;
        if (needs_cooperative) {
            int effective_max_smem;
            if (num_rows <= 4) {
                effective_max_smem = std::min(max_smem_per_block, static_cast<int>(P::kSmemMedium));
            } else if (num_rows <= 8) {
                constexpr int kSmemCapMedium = 48 * 1024;
                effective_max_smem           = std::min(max_smem_per_block, kSmemCapMedium);
            } else {
                effective_max_smem = max_smem_per_block;
            }

            size_t   available_for_ordered = static_cast<size_t>(effective_max_smem) - P::kFixedSmemLarge;
            uint32_t max_chunk_elements    = static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));

            uint32_t vec_size = 1;
            if (stride % 4 == 0)
                vec_size = 4;
            else if (stride % 2 == 0)
                vec_size = 2;

            max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
            uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
            if (max_chunk_elements < min_chunk)
                max_chunk_elements = min_chunk;

            ctas_per_group = (chunk_bound + max_chunk_elements - 1) / max_chunk_elements;
            chunk_size     = (chunk_bound + ctas_per_group - 1) / ctas_per_group;
            chunk_size     = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
            if (chunk_size > max_chunk_elements)
                chunk_size = max_chunk_elements;

            smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
            if (smem_size < P::kSmemMedium)
                smem_size = P::kSmemMedium;

            int         occupancy = 1;
            cudaError_t occ_err   = cudaSuccess;
            if (vec_size == 4) {
                occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occupancy, P::persistent_topk_kernel<TopK, 4>, P::kThreadsPerBlock, smem_size);
            } else if (vec_size == 2) {
                occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occupancy, P::persistent_topk_kernel<TopK, 2>, P::kThreadsPerBlock, smem_size);
            } else {
                occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occupancy, P::persistent_topk_kernel<TopK, 1>, P::kThreadsPerBlock, smem_size);
            }
            TORCH_CHECK(
                occ_err == cudaSuccess, "persistent_topk occupancy query failed: ", cudaGetErrorString(occ_err));
            if (occupancy < 1)
                occupancy = 1;

            const uint32_t hw_resident_cap   = static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(occupancy);
            uint32_t       max_resident_ctas = hw_resident_cap;
            uint32_t       headroom          = (occupancy > 1) ? static_cast<uint32_t>(num_sms) : 1u;
            if (max_resident_ctas >= headroom + ctas_per_group) {
                max_resident_ctas -= headroom;
            }
            uint32_t num_groups = std::min(max_resident_ctas / ctas_per_group, static_cast<uint32_t>(num_rows));
            if (num_groups == 0)
                num_groups = 1;
            uint32_t total_ctas = num_groups * ctas_per_group;

            if (total_ctas > hw_resident_cap) {
                TORCH_CHECK(max_smem_per_block >= 128 * 1024,
                            "persistent_topk would oversubscribe and the FilteredTopK "
                            "fallback requires >=128KB smem per block (have ",
                            max_smem_per_block,
                            "). total_ctas=",
                            total_ctas,
                            " > num_sms*occupancy=",
                            hw_resident_cap,
                            " (TopK=",
                            TopK,
                            ", vec_size=",
                            vec_size,
                            ", ctas_per_group=",
                            ctas_per_group,
                            ", smem=",
                            smem_size,
                            ").");
                cudaError_t status =
                    vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(logits.data_ptr<float>(),
                                                                            output.data_ptr<int32_t>(),
                                                                            lengths.data_ptr<int32_t>(),
                                                                            static_cast<uint32_t>(num_rows),
                                                                            static_cast<uint32_t>(TopK),
                                                                            static_cast<uint32_t>(stride),
                                                                            stream);
                TORCH_CHECK(status == cudaSuccess, "FilteredTopK fallback failed: ", cudaGetErrorString(status));
                return;
            }

            size_t state_bytes = num_groups * sizeof(P::RadixRowState);
            TORCH_CHECK(workspace.size(0) >= static_cast<int64_t>(state_bytes),
                        "workspace too small, need ",
                        state_bytes,
                        " bytes");

            P::PersistentTopKParams params;
            params.input          = logits.data_ptr<float>();
            params.output         = output.data_ptr<int32_t>();
            params.lengths        = lengths.data_ptr<int32_t>();
            params.num_rows       = static_cast<uint32_t>(num_rows);
            params.stride         = static_cast<uint32_t>(stride);
            params.top_k          = static_cast<uint32_t>(TopK);
            params.chunk_size     = chunk_size;
            params.row_states     = get_radix_row_states(device, num_groups, hw_resident_cap, stream);
            params.ctas_per_group = ctas_per_group;
            params.max_seq_len    = chunk_bound;

#define LAUNCH_PERSISTENT(TOPK_VAL, VS)                                                                                \
    do {                                                                                                               \
        auto        kernel = &P::persistent_topk_kernel<TOPK_VAL, VS>;                                                 \
        cudaError_t err    = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);     \
        TORCH_CHECK(err == cudaSuccess, "Failed to set smem: ", cudaGetErrorString(err));                              \
        kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params);                                        \
    } while (0)

            if (vec_size == 4) {
                LAUNCH_PERSISTENT(TopK, 4);
            } else if (vec_size == 2) {
                LAUNCH_PERSISTENT(TopK, 2);
            } else {
                LAUNCH_PERSISTENT(TopK, 1);
            }
#undef LAUNCH_PERSISTENT
        } else {
            // Non-cooperative path: every row fits within RADIX_THRESHOLD, so only
            // histogram_2048_topk / histogram_256_topk run (one CTA per group, <=
            // kSmemMedium shared memory) and row_states is never touched.
            TORCH_CHECK(max_smem_per_block >= static_cast<int>(P::kSmemMedium),
                        "persistent_topk requires >= ",
                        P::kSmemMedium,
                        " bytes smem per block (have ",
                        max_smem_per_block,
                        ")");
            int         occupancy = 1;
            cudaError_t occ_err   = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &occupancy, P::persistent_topk_kernel<TopK, 1>, P::kThreadsPerBlock, smem_size);
            TORCH_CHECK(
                occ_err == cudaSuccess, "persistent_topk occupancy query failed: ", cudaGetErrorString(occ_err));
            if (occupancy < 1)
                occupancy = 1;

            const uint32_t hw_resident_cap = static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(occupancy);
            uint32_t       num_groups      = std::min(hw_resident_cap, static_cast<uint32_t>(num_rows));
            if (num_groups == 0)
                num_groups = 1;

            P::PersistentTopKParams params;
            params.input          = logits.data_ptr<float>();
            params.output         = output.data_ptr<int32_t>();
            params.lengths        = lengths.data_ptr<int32_t>();
            params.num_rows       = static_cast<uint32_t>(num_rows);
            params.stride         = static_cast<uint32_t>(stride);
            params.top_k          = static_cast<uint32_t>(TopK);
            params.chunk_size     = chunk_size;
            params.row_states     = reinterpret_cast<P::RadixRowState*>(workspace.data_ptr<uint8_t>());
            params.ctas_per_group = 1;
            params.max_seq_len    = chunk_bound;

            auto        kernel = &P::persistent_topk_kernel<TopK, 1>;
            cudaError_t err    = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            TORCH_CHECK(err == cudaSuccess, "Failed to set smem: ", cudaGetErrorString(err));
            kernel<<<num_groups, P::kThreadsPerBlock, smem_size, stream>>>(params);
        }
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "persistent_topk failed: ", cudaGetErrorString(err));
}
#endif

}  // namespace

void dsv4_persistent_topk(const torch::Tensor& logits,
                          const torch::Tensor& lengths,
                          torch::Tensor&       output,
                          torch::Tensor&       workspace,
                          int64_t              k,
                          int64_t              max_seq_len) {
#ifndef USE_ROCM
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
    TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");
    TORCH_CHECK(output.dtype() == torch::kInt32, "output must be int32");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
    TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2, "lengths must be 1D or 2D");
    TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
    TORCH_CHECK(logits.stride(1) == 1, "logits rows must be contiguous");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");

    const int64_t num_rows = logits.size(0);

    TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
    TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k, "output size mismatch");
    TORCH_CHECK(
        k == 512 || k == 1024 || k == 2048, "dsv4_persistent_topk supports k=512, k=1024, or k=2048, got k=", k);

    if (k == 512) {
        launch_persistent_topk<512>(logits, lengths, output, workspace, max_seq_len);
    } else if (k == 1024) {
        launch_persistent_topk<1024>(logits, lengths, output, workspace, max_seq_len);
    } else {
        launch_persistent_topk<2048>(logits, lengths, output, workspace, max_seq_len);
    }
#else
    TORCH_CHECK(false, "dsv4_persistent_topk is not supported on ROCm");
#endif
}

}  // namespace torch_ext
