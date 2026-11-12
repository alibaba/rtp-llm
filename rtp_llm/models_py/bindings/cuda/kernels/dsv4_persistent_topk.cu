// Persistent TopK kernel for the DeepSeek-V4 sparse-attention indexer.
// Vendored from vLLM (csrc/topk.cu + csrc/persistent_topk.cuh).
// See dsv4_persistent_topk.h for contract.

#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_persistent_topk.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#ifndef USE_ROCM
#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_persistent_topk.cuh"
#endif

namespace torch_ext {

namespace {

#ifndef USE_ROCM
template<int TopK, int Rounds = 4>
void launch_persistent_topk(const torch::Tensor& logits,
                            const torch::Tensor& lengths,
                            torch::Tensor&       output,
                            torch::Tensor&       workspace,
                            int64_t              max_seq_len) {
    namespace P = vllm::persistent;

    const int64_t num_rows = logits.size(0);
    const int64_t stride   = logits.stride(0);
    cudaStream_t  stream   = at::cuda::getCurrentCUDAStream();

    static int num_sms            = 0;
    static int max_smem_per_block = 0;
    if (num_sms == 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
        cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    }

    if (Rounds == 4 && num_rows > 32 && max_smem_per_block >= 128 * 1024) {
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
        const auto input_address = reinterpret_cast<uintptr_t>(logits.data_ptr<float>());
        if (stride % 4 == 0 && (Rounds == 4 || input_address % alignof(float4) == 0))
            vec_size = 4;
        else if (stride % 2 == 0 && (Rounds == 4 || input_address % alignof(float2) == 0))
            vec_size = 2;

        max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
        uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
        if (max_chunk_elements < min_chunk)
            max_chunk_elements = min_chunk;

        uint32_t ctas_per_group = (static_cast<uint32_t>(stride) + max_chunk_elements - 1) / max_chunk_elements;
        uint32_t chunk_size     = (static_cast<uint32_t>(stride) + ctas_per_group - 1) / ctas_per_group;
        chunk_size              = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
        if (chunk_size > max_chunk_elements)
            chunk_size = max_chunk_elements;

        size_t smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
        if (smem_size < P::kSmemMedium)
            smem_size = P::kSmemMedium;

        int         occupancy = 1;
        cudaError_t occ_err   = cudaSuccess;
        if (vec_size == 4) {
            occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &occupancy, P::persistent_topk_kernel<TopK, 4, Rounds>, P::kThreadsPerBlock, smem_size);
        } else if (vec_size == 2) {
            occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &occupancy, P::persistent_topk_kernel<TopK, 2, Rounds>, P::kThreadsPerBlock, smem_size);
        } else {
            occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &occupancy, P::persistent_topk_kernel<TopK, 1, Rounds>, P::kThreadsPerBlock, smem_size);
        }
        TORCH_CHECK(occ_err == cudaSuccess, "persistent_topk occupancy query failed: ", cudaGetErrorString(occ_err));
        if (occupancy < 1)
            occupancy = 1;

        const bool needs_cooperative = static_cast<uint32_t>(max_seq_len) >
                                       (Rounds == 2 ? static_cast<uint32_t>(TopK) : P::RADIX_THRESHOLD);

        const uint32_t hw_resident_cap   = static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(occupancy);
        uint32_t       max_resident_ctas = hw_resident_cap;
        if (needs_cooperative) {
            uint32_t headroom = (occupancy > 1) ? static_cast<uint32_t>(num_sms) : 1u;
            if (max_resident_ctas >= headroom + ctas_per_group) {
                max_resident_ctas -= headroom;
            }
        }
        uint32_t num_groups = std::min(max_resident_ctas / ctas_per_group, static_cast<uint32_t>(num_rows));
        if (num_groups == 0)
            num_groups = 1;
        uint32_t total_ctas = num_groups * ctas_per_group;

        if (needs_cooperative && total_ctas > hw_resident_cap) {
            TORCH_CHECK(Rounds == 4,
                        "BF16 persistent_topk would oversubscribe its cooperative grid: ",
                        total_ctas, " > num_sms*occupancy=", hw_resident_cap);
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

        {
            cudaError_t mz_err = cudaMemsetAsync(workspace.data_ptr<uint8_t>(), 0, state_bytes, stream);
            TORCH_CHECK(mz_err == cudaSuccess,
                        "row_states memset failed: ",
                        cudaGetErrorString(mz_err));
        }

        P::PersistentTopKParams params;
        params.input          = logits.data_ptr<float>();
        params.output         = output.data_ptr<int32_t>();
        params.lengths        = lengths.data_ptr<int32_t>();
        params.num_rows       = static_cast<uint32_t>(num_rows);
        params.stride         = static_cast<uint32_t>(stride);
        params.top_k          = static_cast<uint32_t>(TopK);
        params.chunk_size     = chunk_size;
        params.row_states     = reinterpret_cast<P::RadixRowState*>(workspace.data_ptr<uint8_t>());
        params.ctas_per_group = ctas_per_group;
        params.max_seq_len    = static_cast<uint32_t>(max_seq_len);

#define LAUNCH_PERSISTENT(TOPK_VAL, VS)                                                                                \
    do {                                                                                                               \
        auto        kernel = &P::persistent_topk_kernel<TOPK_VAL, VS, Rounds>;                                         \
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

void dsv4_persistent_topk_bf16_scores(const torch::Tensor& logits,
                                      const torch::Tensor& lengths,
                                      torch::Tensor&       output,
                                      torch::Tensor&       workspace,
                                      int64_t              k,
                                      int64_t              max_seq_len) {
#ifndef USE_ROCM
    // BF16 provenance and device length bounds are producer preconditions.
    // Never scan or round arbitrary FP32 values to qualify this entry point.
    TORCH_CHECK(logits.is_cuda() && lengths.is_cuda() && output.is_cuda() && workspace.is_cuda(),
                "BF16 persistent_topk tensors must be CUDA tensors");
    TORCH_CHECK(lengths.device() == logits.device() && output.device() == logits.device()
                    && workspace.device() == logits.device(),
                "BF16 persistent_topk tensors must share a device");
    TORCH_CHECK(logits.dtype() == torch::kFloat32 && lengths.dtype() == torch::kInt32
                    && output.dtype() == torch::kInt32 && workspace.dtype() == torch::kUInt8,
                "BF16 persistent_topk requires FP32 logits, int32 lengths/output and uint8 workspace");
    TORCH_CHECK(logits.dim() == 2 && logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
                "BF16 persistent_topk requires nonoverlapping contiguous logits rows");
    const int64_t rows = logits.size(0);
    TORCH_CHECK(rows >= 1 && rows <= 32 && k == 512,
                "BF16 persistent_topk supports 1..32 rows and K512");
    TORCH_CHECK((lengths.dim() == 1 || lengths.dim() == 2) && lengths.is_contiguous()
                    && lengths.numel() == rows,
                "BF16 persistent_topk lengths must be contiguous with one value per row");
    TORCH_CHECK(output.dim() == 2 && output.is_contiguous() && output.size(0) == rows && output.size(1) == k,
                "BF16 persistent_topk output must be contiguous [rows,512]");
    TORCH_CHECK(workspace.dim() == 1 && workspace.is_contiguous() && workspace.numel() >= 1024 * 1024
                    && reinterpret_cast<uintptr_t>(workspace.data_ptr<uint8_t>())
                           % alignof(vllm::persistent::RadixRowState) == 0,
                "BF16 persistent_topk workspace must be aligned contiguous uint8 with at least 1MiB");
    TORCH_CHECK(max_seq_len > 0 && max_seq_len <= logits.size(1)
                    && logits.stride(0) <= std::numeric_limits<int32_t>::max()
                    && logits.stride(0) <= std::numeric_limits<uint32_t>::max() / rows,
                "BF16 persistent_topk requires 0 < max_seq_len <= width and bounded uint32 row offsets");
    const c10::cuda::CUDAGuard device_guard(logits.device());
    launch_persistent_topk<512, 2>(logits, lengths, output, workspace, max_seq_len);
#else
    TORCH_CHECK(false, "dsv4_persistent_topk_bf16_scores is not supported on ROCm");
#endif
}

}  // namespace torch_ext
