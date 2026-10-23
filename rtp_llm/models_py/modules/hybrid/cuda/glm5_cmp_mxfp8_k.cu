#include <torch/extension.h>
#include "glm5_cmp_mxfp8_k.cuh"
#include <c10/cuda/driver_api.h>

#include <rtp_kernel/glm5/indexer_k_cache_types.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace rtp_llm {
namespace detail = rtp_kernel::glm5::detail;
namespace {

constexpr int kHidden        = 6144;
constexpr int kOutput        = 128;
constexpr int kBlockK        = 128;
constexpr int kWorkers       = 148;
constexpr int kCachePageSize = 64;
constexpr int kCacheStride   = 132;
constexpr int kRopeDim       = 64;
// Exact CUDA-DAG B1 producer configuration.
constexpr int kDynamicSmem = 223020;

using Config = deep_gemm::Fp8GemmConfig<cute::UMMA::Major::K,
                                        cute::UMMA::Major::K,
                                        128,
                                        128,
                                        128,
                                        0,
                                        128,
                                        6144,
                                        16,
                                        128,
                                        128,
                                        1,
                                        128,
                                        128,
                                        128,
                                        11,
                                        128,
                                        256,
                                        1,
                                        false,
                                        IndexerKMxfp8Epilogue,
                                        false,
                                        false,
                                        148,
                                        true,
                                        true,
                                        deep_gemm::GemmType::Normal,
                                        false,
                                        cutlass::float_e4m3_t,
                                        cutlass::float_e4m3_t,
                                        cutlass::bfloat16_t,
                                        deep_gemm::epilogue::transform::EpilogueIdentity>;

CUtensorMap make_tma_2d(void*               pointer,
                        CUtensorMapDataType dtype,
                        std::uint64_t       inner,
                        std::uint64_t       outer,
                        std::uint32_t       box_inner,
                        std::uint32_t       box_outer,
                        std::uint64_t       outer_stride_bytes,
                        CUtensorMapSwizzle  swizzle) {
    CUtensorMap      descriptor{};
    const cuuint64_t dimensions[2]      = {inner, outer};
    const cuuint64_t strides[1]         = {outer_stride_bytes};
    const cuuint32_t box[2]             = {box_inner, box_outer};
    const cuuint32_t element_strides[2] = {1, 1};
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

void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
}

void validate(const torch::Tensor& activation,
              const torch::Tensor& activation_scale,
              const torch::Tensor& weight,
              const torch::Tensor& weight_scale,
              const torch::Tensor& norm_weight,
              const torch::Tensor& norm_bias,
              const torch::Tensor& cos_sin,
              const torch::Tensor& positions,
              const torch::Tensor& slot_mapping,
              const torch::Tensor& cache) {
    check_tensor(activation, at::kFloat8_e4m3fn, "activation");
    check_tensor(activation_scale, at::kInt, "activation_scale");
    check_tensor(weight, at::kFloat8_e4m3fn, "weight");
    check_tensor(weight_scale, at::kInt, "weight_scale");
    check_tensor(norm_weight, at::kBFloat16, "norm_weight");
    check_tensor(norm_bias, at::kBFloat16, "norm_bias");
    check_tensor(cos_sin, at::kFloat, "cos_sin");
    check_tensor(positions, at::kInt, "positions");
    check_tensor(slot_mapping, at::kLong, "slot_mapping");
    check_tensor(cache, at::kByte, "cache");

    const int rows         = static_cast<int>(activation.size(0));
    const int aligned_rows = (rows + 3) / 4 * 4;
    TORCH_CHECK(rows > 0 && activation.dim() == 2 && activation.size(1) == kHidden && activation.is_contiguous(),
                "activation must be non-empty contiguous FP8 [M,6144]");
    TORCH_CHECK(activation_scale.sizes() == torch::IntArrayRef({rows, kHidden / 512}) && activation_scale.stride(0) == 1
                    && activation_scale.stride(1) == aligned_rows,
                "activation_scale must be native packed INT32 [M,12]");
    TORCH_CHECK(weight.sizes() == torch::IntArrayRef({kOutput, kHidden}) && weight.is_contiguous(),
                "weight must be contiguous FP8 [128,6144]");
    TORCH_CHECK(weight_scale.sizes() == torch::IntArrayRef({kOutput, kHidden / 512}) && weight_scale.stride(0) == 1
                    && weight_scale.stride(1) == kOutput,
                "weight_scale must be native packed INT32 [128,12]");
    TORCH_CHECK(norm_weight.numel() == kOutput && norm_weight.is_contiguous() && norm_bias.numel() == kOutput
                    && norm_bias.is_contiguous(),
                "norm_weight and norm_bias must be contiguous BF16 [128]");
    TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == kRopeDim && cos_sin.is_contiguous(),
                "cos_sin must be contiguous FP32 [max_position,64]");
    TORCH_CHECK(positions.sizes() == torch::IntArrayRef({rows}) && positions.is_contiguous(),
                "positions must be contiguous INT32 [M]");
    TORCH_CHECK(slot_mapping.sizes() == torch::IntArrayRef({rows}) && slot_mapping.is_contiguous(),
                "slot_mapping must be contiguous INT64 [M]");
    TORCH_CHECK(cache.dim() == 4 && cache.size(1) == kCachePageSize && cache.size(2) == 1
                    && cache.size(3) == kCacheStride && cache.is_contiguous(),
                "cache must be contiguous UINT8 [blocks,64,1,132]");
    TORCH_CHECK(activation.device() == activation_scale.device() && activation.device() == weight.device()
                    && activation.device() == weight_scale.device() && activation.device() == norm_weight.device()
                    && activation.device() == norm_bias.device() && activation.device() == cos_sin.device()
                    && activation.device() == positions.device() && activation.device() == slot_mapping.device()
                    && activation.device() == cache.device(),
                "indexer_k_cache tensors must share one CUDA device");
}

void launch(const torch::Tensor& activation,
            const torch::Tensor& activation_scale,
            const torch::Tensor& weight,
            const torch::Tensor& weight_scale,
            const torch::Tensor& norm_weight,
            const torch::Tensor& norm_bias,
            const torch::Tensor& cos_sin,
            const torch::Tensor& positions,
            const torch::Tensor& slot_mapping,
            const torch::Tensor& cache,
            double               epsilon,
            cudaStream_t         stream,
            bool                 pdl) {
    const auto                kernel         = deep_gemm::fp8_gemm_epilogue_impl<Config, detail::IndexerKCacheArgs>;
    const int                 rows           = static_cast<int>(activation.size(0));
    const int                 aligned_rows   = (rows + 3) / 4 * 4;
    CUtensorMap               tensor_map_a   = make_tma_2d(activation.data_ptr(),
                                           CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                           kHidden,
                                           rows,
                                           kBlockK,
                                           16,
                                           kHidden,
                                           CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap               tensor_map_b   = make_tma_2d(weight.data_ptr(),
                                           CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                           kHidden,
                                           kOutput,
                                           kBlockK,
                                           kOutput,
                                           kHidden,
                                           CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap               tensor_map_sfa = make_tma_2d(activation_scale.data_ptr(),
                                             CU_TENSOR_MAP_DATA_TYPE_INT32,
                                             aligned_rows,
                                             kHidden / 512,
                                             16,
                                             1,
                                             static_cast<std::uint64_t>(aligned_rows) * sizeof(std::int32_t),
                                             CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap               tensor_map_sfb = make_tma_2d(weight_scale.data_ptr(),
                                             CU_TENSOR_MAP_DATA_TYPE_INT32,
                                             kOutput,
                                             kHidden / 512,
                                             kOutput,
                                             1,
                                             static_cast<std::uint64_t>(kOutput) * sizeof(std::int32_t),
                                             CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap               unused_tensor_map_cd{};
    detail::IndexerKCacheArgs epilogue_args{
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(norm_bias.data_ptr()),
        cos_sin.data_ptr<float>(),
        positions.data_ptr<std::int32_t>(),
        slot_mapping.data_ptr<std::int64_t>(),
        cache.data_ptr<std::uint8_t>(),
        rows,
        static_cast<std::int32_t>(cache.size(0)),
        static_cast<float>(epsilon),
        0u,
    };

    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmem));
    CUdeviceptr   grouped_layout = 0;
    std::uint32_t shape_m        = rows;
    std::uint32_t shape_n        = kOutput;
    std::uint32_t shape_k        = kHidden;
    void*         arguments[]    = {
        &grouped_layout,
        &epilogue_args,
        &shape_m,
        &shape_n,
        &shape_k,
        &tensor_map_a,
        &tensor_map_b,
        &tensor_map_sfa,
        &tensor_map_sfb,
        &unused_tensor_map_cd,
    };
    cudaLaunchAttribute attribute{};
    cudaLaunchConfig_t  config{};
    const int           logical_tasks = (rows + 15) / 16;
    const int           workers       = std::min(kWorkers, logical_tasks);
    config.gridDim                    = dim3(workers);
    config.blockDim                   = dim3(384);
    config.dynamicSmemBytes           = kDynamicSmem;
    config.stream                     = stream;
    if (pdl) {
        attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.attrs                                         = &attribute;
        config.numAttrs                                      = 1;
    }
    C10_CUDA_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), arguments));
}

}  // namespace

void glm5CmpMxfp8IndexerK(const torch::Tensor& activation,
                          const torch::Tensor& activation_scale,
                          const torch::Tensor& weight,
                          const torch::Tensor& weight_scale,
                          const torch::Tensor& norm_weight,
                          const torch::Tensor& norm_bias,
                          const torch::Tensor& cos_sin,
                          const torch::Tensor& positions,
                          const torch::Tensor& slot_mapping,
                          const torch::Tensor& cache,
                          double               epsilon,
                          bool                 pdl) {
    validate(activation,
             activation_scale,
             weight,
             weight_scale,
             norm_weight,
             norm_bias,
             cos_sin,
             positions,
             slot_mapping,
             cache);
    const c10::cuda::CUDAGuard guard(activation.device());
    launch(activation,
           activation_scale,
           weight,
           weight_scale,
           norm_weight,
           norm_bias,
           cos_sin,
           positions,
           slot_mapping,
           cache,
           epsilon,
           at::cuda::getCurrentCUDAStream(),
           pdl);
}

}  // namespace rtp_llm
