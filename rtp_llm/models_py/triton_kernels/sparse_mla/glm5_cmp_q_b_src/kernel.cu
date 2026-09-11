#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "epilogue.cuh"

namespace {
template<int M, int STAGES>
constexpr int smem_bytes =
    2 * 16 * 128 * 2 + STAGES * (M / 2 * 128 + 128 * 128 + 2 * 128 * 4) + 32 * 8 * 3 + 2 * 8 * 2 + 8 + sizeof(uint32_t);

CUtensorMap tma(void*               pointer,
                CUtensorMapDataType dtype,
                uint64_t            inner,
                uint64_t            outer,
                uint32_t            box_inner,
                uint32_t            box_outer,
                uint64_t            stride,
                CUtensorMapSwizzle  swizzle) {
    CUtensorMap      descriptor{};
    const cuuint64_t dimensions[2] = {inner, outer}, strides[1] = {stride};
    const cuuint32_t box[2] = {box_inner, box_outer}, element_strides[2] = {1, 1};
    C10_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(&descriptor,
                                                 dtype,
                                                 2,
                                                 pointer,
                                                 dimensions,
                                                 strides,
                                                 box,
                                                 element_strides,
                                                 CU_TENSOR_MAP_INTERLEAVE_NONE,
                                                 swizzle,
                                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return descriptor;
}

template<int M, int STAGES, int EWG>
void launch(const at::Tensor& a,
            const at::Tensor& sa,
            const at::Tensor& b,
            const at::Tensor& sb,
            const at::Tensor& cos,
            const at::Tensor& positions,
            const at::Tensor& nope,
            const at::Tensor& query,
            bool              neox,
            bool              pdl) {
    const int  rows = a.size(0), aligned = (rows + 3) / 4 * 4;
    auto       kernel = rtp_llm::glm5_h8::q_b_h8_kernel<M, STAGES, EWG>;
    const auto ta =
        tma(a.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2048, rows, 128, M / 2, 2048, CU_TENSOR_MAP_SWIZZLE_128B);
    const auto tb =
        tma(b.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2048, 2048, 128, 128, 2048, CU_TENSOR_MAP_SWIZZLE_128B);
    const auto tsa =
        tma(sa.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, aligned, 4, M, 1, aligned * 4, CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tsb =
        tma(sb.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 2048, 4, 128, 1, 2048 * 4, CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto tcd = tma(
        nope.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 1536, rows, 64, 16, 1536 * 2, CU_TENSOR_MAP_SWIZZLE_128B);
    rtp_llm::glm5_h8::Args args{cos.data_ptr<float>(),
                                positions.data_ptr(),
                                reinterpret_cast<__nv_bfloat16*>(nope.data_ptr()),
                                reinterpret_cast<__nv_bfloat16*>(query.data_ptr()),
                                rows,
                                positions.scalar_type() == at::kLong,
                                neox};
    cudaLaunchAttribute    attrs[2]{};
    attrs[0].id                                         = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim                             = {2, 1, 1};
    attrs[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(((rows + M - 1) / M) * 16);
    config.blockDim         = dim3(128 * (1 + EWG));
    config.dynamicSmemBytes = smem_bytes<M, STAGES>;
    config.stream           = at::cuda::getCurrentCUDAStream();
    config.attrs            = attrs;
    config.numAttrs         = pdl ? 2 : 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, static_cast<uint32_t>(rows), ta, tb, tsa, tsb, tcd, args));
}
}  // namespace

void initialize_q_b_h8() {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
                "fused H8 Q-B requires SM100/SM103");
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        rtp_llm::glm5_h8::q_b_h8_kernel<16, 12, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<16, 12>));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        rtp_llm::glm5_h8::q_b_h8_kernel<64, 10, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<64, 10>));
}

void q_b_proj_h8(const at::Tensor& a,
                 const at::Tensor& sa,
                 const at::Tensor& b,
                 const at::Tensor& sb,
                 const at::Tensor& cos,
                 const at::Tensor& positions,
                 const at::Tensor& nope,
                 const at::Tensor& query,
                 bool              neox,
                 bool              pdl) {
    TORCH_CHECK(a.dim() == 2 && a.size(0) > 0 && a.size(0) <= 256 && a.size(1) == 2048,
                "activation must have shape [M,2048], 1 <= M <= 256");
    const auto rows = a.size(0), aligned = (rows + 3) / 4 * 4;
    for (const auto& t : {a, sa, b, sb, cos, positions, nope, query}) {
        TORCH_CHECK(t.is_cuda() && t.device() == a.device(), "Q-B operands must share one CUDA device");
    }
    TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && a.is_contiguous(), "activation must be contiguous FP8");
    TORCH_CHECK(b.scalar_type() == at::kFloat8_e4m3fn && b.is_contiguous()
                    && b.sizes() == at::IntArrayRef({2048, 2048}),
                "weight must be contiguous FP8 [2048,2048]");
    TORCH_CHECK(sa.scalar_type() == at::kInt && sa.sizes() == at::IntArrayRef({rows, 4}) && sa.stride(0) == 1
                    && sa.stride(1) == aligned,
                "activation scales require packed column-major [M,4]");
    TORCH_CHECK(sb.scalar_type() == at::kInt && sb.sizes() == at::IntArrayRef({2048, 4}) && sb.stride(0) == 1
                    && sb.stride(1) == 2048,
                "weight scales require packed column-major [2048,4]");
    for (const auto& t : {a, sa, b, sb, nope, query}) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "Q-B GEMM buffers require 16-byte alignment");
    }
    TORCH_CHECK(sa.storage().nbytes() >= static_cast<size_t>(sa.storage_offset() * 4 + aligned * 16),
                "activation scales require padded storage for all aligned rows");
    TORCH_CHECK(cos.scalar_type() == at::kFloat && cos.dim() == 2 && cos.size(0) > 0 && cos.size(1) == 64
                    && cos.is_contiguous(),
                "cos_sin must be contiguous FP32 [S,64]");
    TORCH_CHECK((positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong)
                    && positions.sizes() == at::IntArrayRef({rows}) && positions.is_contiguous(),
                "positions must be contiguous int32/int64 [M]");
    TORCH_CHECK(nope.scalar_type() == at::kBFloat16 && nope.is_contiguous()
                    && nope.sizes() == at::IntArrayRef({rows, 8, 192}),
                "NoPE must be contiguous BF16 [M,8,192]");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && query.is_contiguous()
                    && query.sizes() == at::IntArrayRef({rows, 8, 576}),
                "query must be contiguous BF16 [M,8,576]");
    const c10::cuda::CUDAGuard guard(a.device());
    const auto*                properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
                "fused H8 Q-B requires SM100/SM103");
    if (rows <= 16)
        launch<16, 12, 2>(a, sa, b, sb, cos, positions, nope, query, neox, pdl);
    else
        launch<64, 10, 4>(a, sa, b, sb, cos, positions, nope, query, neox, pdl);
}
