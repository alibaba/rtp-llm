#include "rtp_llm/cpp/kernels/atex/ops/f8/quant.h"
#include "rtp_llm/cpp/kernels/atex/cooperative/reduce.cuh"
#include "rtp_llm/cpp/kernels/atex/cooperative/common.cuh"
#include "rtp_llm/cpp/kernels/atex/thread/reduce.cuh"
#include "rtp_llm/cpp/kernels/atex/copy/vec.cuh"

namespace atex {
namespace impl {

template<typename f16x2_t, typename f8x2_t, uint32_t TPB, uint32_t VPT>
__global__ void
device_minmax_pertensor_quant_f16_fp8e4m3(const f16x2_t* const x, const uint32_t numel, uint8x2_t* y, fp32_t* scale) {
    // 统计 x 函数中的最大值，并以最大值的 448 分之一计算scale，从而对给定的 tensor 进行 fp8 量化
    // 量化后的值存入 y

    uint32_t tid       = blockIdx.x * TPB + threadIdx.x;
    uint32_t n_threads = gridDim.x * TPB;

    f16x2_t local_x[VPT];
    f8x2_t  local_y[VPT];
    f16x2_t local_max = {0.0f, 0.0f};

    if (tid == 0) {
        *scale = 0.0f;
    }
    atex::cooperative::sync();

    for (uint32_t i = tid * VPT; i < numel; i += n_threads * VPT) {
        atex::copy<sizeof(f16x2_t) * VPT>(x + i, local_x);
        local_max = atex::thread::reduce_absmax<VPT>(local_max, local_x);
    }

    fp32_t            local_max_ = cvt_f16_to_f32(__hmax(local_max.x, local_max.y));
    __shared__ fp32_t smem[TPB / warpSize];

    // reduce acorss blocks
    fp32_t global_max = atex::cooperative::reduce_max<TPB>(local_max_, smem, scale);
    fp32_t scale_     = fmaxf(global_max / atex::FP8_E4M3_MAX, atex::SCALE_MIN);
    fp32_t inv_scale  = 1 / scale_;

    for (uint32_t i = tid * VPT; i < numel; i += n_threads * VPT) {
        atex::copy<sizeof(f16x2_t) * VPT>(x + i, local_x);

#pragma unroll
        for (uint32_t j = 0; j < VPT; j++) {
            fp32x2_t value = cvt_f16x2_to_f32x2(local_x[j]);
            value.x *= inv_scale;
            value.y *= inv_scale;
            f8x2_t out{value};
            local_y[j] = out;
        }

        atex::copy<sizeof(f8x2_t) * VPT>(local_y, y + i);
    }
}

std::tuple<Tensor, Tensor> launch_minmax_pertensor_quant_fp16_fp8e4m3(const Tensor& x) {
    TORCH_CHECK(x.is_contiguous(), "Input tensor x must be contiguous.");
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a Cuda Tensor.");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, "Input tensor x must be a FP16 Tensor.");

    const uint32_t numel = (uint32_t)x.numel();
    TORCH_CHECK(numel > 0, "Input tensor x is empty.");
    TORCH_CHECK(numel % 8 == 0, "Elements of tensor x must be a multiple of 8.");

    Tensor output_y     = at::empty(x.sizes(), x.options().dtype(at::kChar));
    Tensor output_scale = at::empty({1}, x.options().dtype(at::kFloat));

    constexpr uint32_t TPB = 256;
    constexpr uint32_t VPT = 4;

    uint32_t elements_per_block_pass = TPB * VPT;
    uint32_t grid_size =
        min((numel + elements_per_block_pass - 1) / elements_per_block_pass, atex::cooperative::CooperativeMaxBlocks);

    void* _x = x.data_ptr();
    void* _y = output_y.data_ptr();
    void* _s = output_scale.data_ptr();

    void* kernel_args[] = {&(_x), (void*)&numel, &(_y), &(_s)};

    cudaError_t result = cudaLaunchCooperativeKernel(
        (const void*)device_minmax_pertensor_quant_f16_fp8e4m3<fp16x2_t, fp8x2_e4m3_t, TPB, VPT>,
        grid_size,
        TPB,
        kernel_args,
        0,
        at::cuda::getCurrentCUDAStream());

    return {output_y, output_scale};
}

std::tuple<Tensor, Tensor> launch_minmax_pertensor_quant_bf16_fp8e4m3(const Tensor& x) {
    TORCH_CHECK(x.is_contiguous(), "Input tensor x must be contiguous.");
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a Cuda Tensor.");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "Input tensor x must be a BF16 Tensor.");

    const uint32_t numel = (uint32_t)x.numel();
    TORCH_CHECK(numel > 0, "Input tensor x is empty.");
    TORCH_CHECK(numel % 8 == 0, "Elements of tensor x must be a multiple of 8.");

    Tensor output_y     = at::empty(x.sizes(), x.options().dtype(at::kChar));
    Tensor output_scale = at::empty({1}, x.options().dtype(at::kFloat));

    constexpr uint32_t TPB = 256;
    constexpr uint32_t VPT = 4;

    uint32_t elements_per_block_pass = TPB * VPT;
    uint32_t grid_size =
        min((numel + elements_per_block_pass - 1) / elements_per_block_pass, atex::cooperative::CooperativeMaxBlocks);

    void* _x = x.data_ptr();
    void* _y = output_y.data_ptr();
    void* _s = output_scale.data_ptr();

    void* kernel_args[] = {&(_x), (void*)&numel, &(_y), &(_s)};

    cudaError_t result = cudaLaunchCooperativeKernel(
        (const void*)device_minmax_pertensor_quant_f16_fp8e4m3<bf16x2_t, fp8x2_e4m3_t, TPB, VPT>,
        grid_size,
        TPB,
        kernel_args,
        0,
        at::cuda::getCurrentCUDAStream());

    return {output_y, output_scale};
}

}  // namespace impl
}  // namespace atex
