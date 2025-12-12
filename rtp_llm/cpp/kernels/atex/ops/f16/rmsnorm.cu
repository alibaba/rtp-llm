#include "rtp_llm/cpp/kernels/atex/ops/f16/rmsnorm.h"
#include "rtp_llm/cpp/kernels/atex/block/reduce.cuh"
#include "rtp_llm/cpp/kernels/atex/copy/vec.cuh"

namespace atex {
namespace impl {

template<typename f16x2_t, uint32_t TPB, uint32_t VPT, uint32_t LOOP>
__global__ void device_rmsnorm_f16(const f16x2_t* const __restrict__ x,  // [m, n]
                                   const f16x2_t* const __restrict__ w,  // [n]
                                   f16x2_t* __restrict__ o,              // [m, n]
                                   const fp32_t eps) {
    f16x2_t local_x[VPT * LOOP];
    f16x2_t local_w[VPT * LOOP];

    const uint32_t     tid   = threadIdx.x;
    const uint32_t     bid   = blockIdx.x;
    const uint32_t     index = tid * VPT;
    constexpr uint32_t n     = TPB * VPT * LOOP;

    auto _x = x + bid * n;
    auto _o = o + bid * n;
    auto _w = w;

    fp32_t local_sum = 0.0f;

#pragma unroll
    for (uint32_t loop = 0; loop < LOOP; loop++) {
        atex::copy<sizeof(f16x2_t) * VPT>(_x + index + loop * TPB * VPT, local_x + loop * VPT);
        atex::copy<sizeof(f16x2_t) * VPT>(_w + index + loop * TPB * VPT, local_w + loop * VPT);

#pragma unroll
        for (uint32_t i = 0; i < VPT; i++) {
            fp32x2_t value = atex::cvt_f16x2_to_f32x2(local_x[i + loop * VPT]);
            local_sum += value.y * value.y + value.x * value.x;
        }
    }

    __shared__ fp32_t workspace[TPB / atex::warpSize];
    fp32_t            block_sum        = atex::block::reduce_sum<TPB, true>(local_sum, workspace) / (n * 2);
    fp32_t            normalize_factor = rsqrt(block_sum + eps);
    f16x2_t           local_o[VPT];

#pragma unroll
    for (uint32_t loop = 0; loop < LOOP; loop++) {
#pragma unroll
        for (uint32_t i = 0; i < VPT; i++) {
            fp32x2_t value  = atex::cvt_f16x2_to_f32x2(local_x[i + loop * VPT]);
            fp32x2_t weight = atex::cvt_f16x2_to_f32x2(local_w[i + loop * VPT]);
            value.x         = value.x * weight.x * normalize_factor;
            value.y         = value.y * weight.y * normalize_factor;
            local_o[i]      = atex::cvt_f32x2_to_f16x2<f16x2_t>(value);
        }

        atex::copy<sizeof(f16x2_t) * VPT>(local_o, _o + index + loop * TPB * VPT);
    }
}

template<typename f16x2_t, uint32_t TPB, uint32_t VPT, uint32_t LOOP>
__global__ void device_skip_rmsnorm_f16(const f16x2_t* const x,         // [m, n]
                                        const f16x2_t* const r,         // [m, n]
                                        const f16x2_t* const w,         // [n]
                                        f16x2_t*             o,         // [m, n]
                                        f16x2_t*             skip_out,  // [m, n]
                                        const fp32_t         eps) {
    f16x2_t local_x[VPT * LOOP];
    f16x2_t local_w[VPT * LOOP];

    const uint32_t     tid   = threadIdx.x;
    const uint32_t     bid   = blockIdx.x;
    const uint32_t     index = tid * VPT;
    constexpr uint32_t n     = TPB * VPT * LOOP;

    auto _x        = x + bid * n;
    auto _r        = r + bid * n;
    auto _o        = o + bid * n;
    auto _skip_out = skip_out + bid * n;
    auto _w        = w;

    fp32_t local_sum = 0.0f;

#pragma unroll
    for (uint32_t loop = 0; loop < LOOP; loop++) {
        atex::copy<sizeof(f16x2_t) * VPT>(_x + index + loop * TPB * VPT, local_x + loop * VPT);
        atex::copy<sizeof(f16x2_t) * VPT>(_r + index + loop * TPB * VPT, local_w + loop * VPT);

#pragma unroll
        for (uint32_t i = 0; i < VPT; i++) {
            local_x[i + loop * VPT] = local_x[i + loop * VPT] + local_w[i + loop * VPT];
            fp32x2_t value          = atex::cvt_f16x2_to_f32x2(local_x[i + loop * VPT]);
            local_sum += value.y * value.y + value.x * value.x;
        }

        atex::copy<sizeof(f16x2_t) * VPT>(&local_x[loop * VPT], _skip_out + index + loop * TPB * VPT);
    }

    __shared__ fp32_t workspace[TPB / atex::warpSize];
    fp32_t            block_sum        = atex::block::reduce_sum<TPB, true>(local_sum, workspace) / (n * 2);
    fp32_t            normalize_factor = rsqrt(block_sum + eps);
    f16x2_t           local_o[VPT];

#pragma unroll
    for (uint32_t loop = 0; loop < LOOP; loop++) {
        atex::copy<sizeof(f16x2_t) * VPT>(_w + index + loop * TPB * VPT, local_w + loop * VPT);
#pragma unroll
        for (uint32_t i = 0; i < VPT; i++) {
            fp32x2_t value  = atex::cvt_f16x2_to_f32x2(local_x[i + loop * VPT]);
            fp32x2_t weight = atex::cvt_f16x2_to_f32x2(local_w[i + loop * VPT]);
            value.x         = value.x * weight.x * normalize_factor;
            value.y         = value.y * weight.y * normalize_factor;
            local_o[i]      = atex::cvt_f32x2_to_f16x2<f16x2_t>(value);
        }

        atex::copy<sizeof(f16x2_t) * VPT>(local_o, _o + index + loop * TPB * VPT);
    }
}

Tensor launch_rmsnorm_fp16(const Tensor& x, const Tensor& weight, const float eps) {
    TORCH_CHECK(x.is_contiguous(), "x is not contiguous.");
    TORCH_CHECK(weight.is_contiguous(), "w is not contiguous.");

    TORCH_CHECK(x.is_cuda(), "Input Tensor x must be a Cuda Tensor.");
    TORCH_CHECK(weight.is_cuda(), "Input Tensor weight must be a Cuda Tensor.");

    TORCH_CHECK(x.scalar_type() == c10::ScalarType::Half, "Input Tensor x must be a FP16 Tensor.");
    TORCH_CHECK(weight.scalar_type() == c10::ScalarType::Half, "Input Tensor weight must be a FP16 Tensor.");

    Tensor output = at::empty_like(x);

    // 假设 fp16x2_t 是 4 字节，那么 VPT = 16 / 4 = 4
    // 假设 VPTx2 = 16 / 4 * 2 = 8
    constexpr int32_t VPT             = 16 / sizeof(fp16x2_t);
    constexpr int32_t VPTx2           = 16 / sizeof(fp16x2_t) * 2;
    const int32_t     normalize_shape = x.sizes()[x.ndimension() - 1];
    const int32_t     grid_size       = x.numel() / normalize_shape;

    TORCH_CHECK(normalize_shape % VPTx2 == 0,
                c10::str("Normalize shape (", normalize_shape, ") should be a multiple of ", VPT * 2));

    auto stream = at::cuda::getCurrentCUDAStream();
    switch (normalize_shape) {
        case 256:
            device_rmsnorm_f16<fp16x2_t, 256 / VPTx2, VPT, 1><<<grid_size, 256 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        case 512:
            device_rmsnorm_f16<fp16x2_t, 512 / VPTx2, VPT, 1><<<grid_size, 512 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        case 768:
            device_rmsnorm_f16<fp16x2_t, 768 / VPTx2, VPT, 1><<<grid_size, 768 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        case 1024:
            device_rmsnorm_f16<fp16x2_t, 1024 / VPTx2, VPT, 1><<<grid_size, 1024 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        case 1536:
            device_rmsnorm_f16<fp16x2_t, 1536 / VPTx2, VPT, 1><<<grid_size, 1536 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        case 2048:
            device_rmsnorm_f16<fp16x2_t, 2048 / VPTx2, VPT, 1><<<grid_size, 2048 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        case 4096:
            device_rmsnorm_f16<fp16x2_t, 4096 / VPTx2, VPT, 1><<<grid_size, 4096 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        case 8192:
            device_rmsnorm_f16<fp16x2_t, 8192 / VPTx2, VPT, 1><<<grid_size, 8192 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        case 16384:
            device_rmsnorm_f16<fp16x2_t, 8192 / VPTx2, VPT, 2><<<grid_size, 8192 / VPTx2, 0, stream>>>(
                PTR<fp16x2_t>(x), PTR<fp16x2_t>(weight), PTR<fp16x2_t>(output), eps);
            break;
        default:
            throw std::runtime_error(c10::str("Normalize shape (",
                                              normalize_shape,
                                              ") is not supported yet, check rmsnorm.cu for more information."));
    };
    return output;
}

Tensor launch_rmsnorm_bf16(const Tensor& x, const Tensor& weight, const float eps) {
    TORCH_CHECK(x.is_contiguous(), "x is not contiguous.");
    TORCH_CHECK(weight.is_contiguous(), "w is not contiguous.");

    TORCH_CHECK(x.is_cuda(), "Input Tensor x must be a Cuda Tensor.");
    TORCH_CHECK(weight.is_cuda(), "Input Tensor weight must be a Cuda Tensor.");

    TORCH_CHECK(x.scalar_type() == c10::ScalarType::BFloat16, "Input Tensor x must be a BF16 Tensor.");
    TORCH_CHECK(weight.scalar_type() == c10::ScalarType::BFloat16, "Input Tensor weight must be a BF16 Tensor.");

    Tensor output = at::empty_like(x);

    // 假设 bf16x2_t 是 4 字节，那么 VPT = 16 / 4 = 4
    // 假设 VPTx2 = 16 / 4 * 2 = 8
    constexpr int32_t VPT             = 16 / sizeof(bf16x2_t);
    constexpr int32_t VPTx2           = 16 / sizeof(bf16x2_t) * 2;
    const int32_t     normalize_shape = x.sizes()[x.ndimension() - 1];
    const int32_t     grid_size       = x.numel() / normalize_shape;

    TORCH_CHECK(normalize_shape % VPTx2 == 0,
                c10::str("Normalize shape (", normalize_shape, ") should be a multiple of ", VPT * 2));

    auto stream = at::cuda::getCurrentCUDAStream();
    switch (normalize_shape) {
        case 256:
            device_rmsnorm_f16<bf16x2_t, 256 / VPTx2, VPT, 1><<<grid_size, 256 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        case 512:
            device_rmsnorm_f16<bf16x2_t, 512 / VPTx2, VPT, 1><<<grid_size, 512 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        case 768:
            device_rmsnorm_f16<bf16x2_t, 768 / VPTx2, VPT, 1><<<grid_size, 768 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        case 1024:
            device_rmsnorm_f16<bf16x2_t, 1024 / VPTx2, VPT, 1><<<grid_size, 1024 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        case 1536:
            device_rmsnorm_f16<bf16x2_t, 1536 / VPTx2, VPT, 1><<<grid_size, 1536 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        case 2048:
            device_rmsnorm_f16<bf16x2_t, 2048 / VPTx2, VPT, 1><<<grid_size, 2048 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        case 4096:
            device_rmsnorm_f16<bf16x2_t, 4096 / VPTx2, VPT, 1><<<grid_size, 4096 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        case 8192:
            device_rmsnorm_f16<bf16x2_t, 8192 / VPTx2, VPT, 1><<<grid_size, 8192 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        case 16384:
            device_rmsnorm_f16<bf16x2_t, 8192 / VPTx2, VPT, 2><<<grid_size, 8192 / VPTx2, 0, stream>>>(
                PTR<bf16x2_t>(x), PTR<bf16x2_t>(weight), PTR<bf16x2_t>(output), eps);
            break;
        default:
            throw std::runtime_error(c10::str("Normalize shape (",
                                              normalize_shape,
                                              ") is not supported yet, check rmsnorm.cu for more information."));
    };
    return output;
}

std::tuple<Tensor, Tensor>
launch_skiprmsnorm_fp16(const Tensor& x, const Tensor& skip, const Tensor& weight, const float eps) {
    TORCH_CHECK(x.is_contiguous(), "x is not contiguous.");
    TORCH_CHECK(weight.is_contiguous(), "w is not contiguous.");

    TORCH_CHECK(x.is_cuda(), "Input Tensor x must be a Cuda Tensor.");
    TORCH_CHECK(weight.is_cuda(), "Input Tensor weight must be a Cuda Tensor.");

    TORCH_CHECK(x.scalar_type() == c10::ScalarType::Half, "Input Tensor x must be a FP16 Tensor.");
    TORCH_CHECK(weight.scalar_type() == c10::ScalarType::Half, "Input Tensor weight must be a FP16 Tensor.");

    Tensor output      = at::empty_like(x);
    Tensor skip_output = at::empty_like(x);

    // 假设 fp16x2_t 是 4 字节，那么 VPT = 16 / 4 = 4
    // 假设 VPTx2 = 16 / 4 * 2 = 8
    constexpr int32_t VPT             = 16 / sizeof(fp16x2_t);
    constexpr int32_t VPTx2           = 16 / sizeof(fp16x2_t) * 2;
    const int32_t     normalize_shape = x.sizes()[x.ndimension() - 1];
    const int32_t     grid_size       = x.numel() / normalize_shape;

    TORCH_CHECK(normalize_shape % VPTx2 == 0,
                c10::str("Normalize shape (", normalize_shape, ") should be a multiple of ", VPT * 2));

    auto stream = at::cuda::getCurrentCUDAStream();
    switch (normalize_shape) {
        case 256:
            device_skip_rmsnorm_f16<fp16x2_t, 256 / VPTx2, VPT, 1>
                <<<grid_size, 256 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                        PTR<fp16x2_t>(skip),
                                                        PTR<fp16x2_t>(weight),
                                                        PTR<fp16x2_t>(output),
                                                        PTR<fp16x2_t>(skip_output),
                                                        eps);
            break;
        case 512:
            device_skip_rmsnorm_f16<fp16x2_t, 512 / VPTx2, VPT, 1>
                <<<grid_size, 512 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                        PTR<fp16x2_t>(skip),
                                                        PTR<fp16x2_t>(weight),
                                                        PTR<fp16x2_t>(output),
                                                        PTR<fp16x2_t>(skip_output),
                                                        eps);
            break;
        case 768:
            device_skip_rmsnorm_f16<fp16x2_t, 768 / VPTx2, VPT, 1>
                <<<grid_size, 768 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                        PTR<fp16x2_t>(skip),
                                                        PTR<fp16x2_t>(weight),
                                                        PTR<fp16x2_t>(output),
                                                        PTR<fp16x2_t>(skip_output),
                                                        eps);
            break;
        case 1024:
            device_skip_rmsnorm_f16<fp16x2_t, 1024 / VPTx2, VPT, 1>
                <<<grid_size, 1024 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                         PTR<fp16x2_t>(skip),
                                                         PTR<fp16x2_t>(weight),
                                                         PTR<fp16x2_t>(output),
                                                         PTR<fp16x2_t>(skip_output),
                                                         eps);
            break;
        case 1536:
            device_skip_rmsnorm_f16<fp16x2_t, 1536 / VPTx2, VPT, 1>
                <<<grid_size, 1536 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                         PTR<fp16x2_t>(skip),
                                                         PTR<fp16x2_t>(weight),
                                                         PTR<fp16x2_t>(output),
                                                         PTR<fp16x2_t>(skip_output),
                                                         eps);
            break;
        case 2048:
            device_skip_rmsnorm_f16<fp16x2_t, 2048 / VPTx2, VPT, 1>
                <<<grid_size, 2048 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                         PTR<fp16x2_t>(skip),
                                                         PTR<fp16x2_t>(weight),
                                                         PTR<fp16x2_t>(output),
                                                         PTR<fp16x2_t>(skip_output),
                                                         eps);
            break;
        case 4096:
            device_skip_rmsnorm_f16<fp16x2_t, 4096 / VPTx2, VPT, 1>
                <<<grid_size, 4096 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                         PTR<fp16x2_t>(skip),
                                                         PTR<fp16x2_t>(weight),
                                                         PTR<fp16x2_t>(output),
                                                         PTR<fp16x2_t>(skip_output),
                                                         eps);
            break;
        case 8192:
            device_skip_rmsnorm_f16<fp16x2_t, 8192 / VPTx2, VPT, 1>
                <<<grid_size, 8192 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                         PTR<fp16x2_t>(skip),
                                                         PTR<fp16x2_t>(weight),
                                                         PTR<fp16x2_t>(output),
                                                         PTR<fp16x2_t>(skip_output),
                                                         eps);
            break;
        case 16384:
            device_skip_rmsnorm_f16<fp16x2_t, 8192 / VPTx2, VPT, 2>
                <<<grid_size, 8192 / VPTx2, 0, stream>>>(PTR<fp16x2_t>(x),
                                                         PTR<fp16x2_t>(skip),
                                                         PTR<fp16x2_t>(weight),
                                                         PTR<fp16x2_t>(output),
                                                         PTR<fp16x2_t>(skip_output),
                                                         eps);
            break;
        default:
            throw std::runtime_error(c10::str("Normalize shape (",
                                              normalize_shape,
                                              ") is not supported yet, check rmsnorm.cu for more information."));
    };
    return {output, skip_output};
}

std::tuple<Tensor, Tensor>
launch_skiprmsnorm_bf16(const Tensor& x, const Tensor& skip, const Tensor& weight, const float eps) {
    TORCH_CHECK(x.is_contiguous(), "x is not contiguous.");
    TORCH_CHECK(weight.is_contiguous(), "w is not contiguous.");

    TORCH_CHECK(x.is_cuda(), "Input Tensor x must be a Cuda Tensor.");
    TORCH_CHECK(weight.is_cuda(), "Input Tensor weight must be a Cuda Tensor.");

    TORCH_CHECK(x.scalar_type() == c10::ScalarType::BFloat16, "Input Tensor x must be a BF16 Tensor.");
    TORCH_CHECK(weight.scalar_type() == c10::ScalarType::BFloat16, "Input Tensor weight must be a BF16 Tensor.");

    Tensor output      = at::empty_like(x);
    Tensor skip_output = at::empty_like(x);

    // 假设 bf16x2_t 是 4 字节，那么 VPT = 16 / 4 = 4
    // 假设 VPTx2 = 16 / 4 * 2 = 8
    constexpr int32_t VPT             = 16 / sizeof(bf16x2_t);
    constexpr int32_t VPTx2           = 16 / sizeof(bf16x2_t) * 2;
    const int32_t     normalize_shape = x.sizes()[x.ndimension() - 1];
    const int32_t     grid_size       = x.numel() / normalize_shape;

    TORCH_CHECK(normalize_shape % VPTx2 == 0,
                c10::str("Normalize shape (", normalize_shape, ") should be a multiple of ", VPT * 2));

    auto stream = at::cuda::getCurrentCUDAStream();
    switch (normalize_shape) {
        case 256:
            device_skip_rmsnorm_f16<bf16x2_t, 256 / VPTx2, VPT, 1>
                <<<grid_size, 256 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                        PTR<bf16x2_t>(skip),
                                                        PTR<bf16x2_t>(weight),
                                                        PTR<bf16x2_t>(output),
                                                        PTR<bf16x2_t>(skip_output),
                                                        eps);
            break;
        case 512:
            device_skip_rmsnorm_f16<bf16x2_t, 512 / VPTx2, VPT, 1>
                <<<grid_size, 512 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                        PTR<bf16x2_t>(skip),
                                                        PTR<bf16x2_t>(weight),
                                                        PTR<bf16x2_t>(output),
                                                        PTR<bf16x2_t>(skip_output),
                                                        eps);
            break;
        case 768:
            device_skip_rmsnorm_f16<bf16x2_t, 768 / VPTx2, VPT, 1>
                <<<grid_size, 768 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                        PTR<bf16x2_t>(skip),
                                                        PTR<bf16x2_t>(weight),
                                                        PTR<bf16x2_t>(output),
                                                        PTR<bf16x2_t>(skip_output),
                                                        eps);
            break;
        case 1024:
            device_skip_rmsnorm_f16<bf16x2_t, 1024 / VPTx2, VPT, 1>
                <<<grid_size, 1024 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                         PTR<bf16x2_t>(skip),
                                                         PTR<bf16x2_t>(weight),
                                                         PTR<bf16x2_t>(output),
                                                         PTR<bf16x2_t>(skip_output),
                                                         eps);
            break;
        case 1536:
            device_skip_rmsnorm_f16<bf16x2_t, 1536 / VPTx2, VPT, 1>
                <<<grid_size, 1536 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                         PTR<bf16x2_t>(skip),
                                                         PTR<bf16x2_t>(weight),
                                                         PTR<bf16x2_t>(output),
                                                         PTR<bf16x2_t>(skip_output),
                                                         eps);
            break;
        case 2048:
            device_skip_rmsnorm_f16<bf16x2_t, 2048 / VPTx2, VPT, 1>
                <<<grid_size, 2048 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                         PTR<bf16x2_t>(skip),
                                                         PTR<bf16x2_t>(weight),
                                                         PTR<bf16x2_t>(output),
                                                         PTR<bf16x2_t>(skip_output),
                                                         eps);
            break;
        case 4096:
            device_skip_rmsnorm_f16<bf16x2_t, 4096 / VPTx2, VPT, 1>
                <<<grid_size, 4096 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                         PTR<bf16x2_t>(skip),
                                                         PTR<bf16x2_t>(weight),
                                                         PTR<bf16x2_t>(output),
                                                         PTR<bf16x2_t>(skip_output),
                                                         eps);
            break;
        case 8192:
            device_skip_rmsnorm_f16<bf16x2_t, 8192 / VPTx2, VPT, 1>
                <<<grid_size, 8192 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                         PTR<bf16x2_t>(skip),
                                                         PTR<bf16x2_t>(weight),
                                                         PTR<bf16x2_t>(output),
                                                         PTR<bf16x2_t>(skip_output),
                                                         eps);
            break;
        case 16384:
            device_skip_rmsnorm_f16<bf16x2_t, 8192 / VPTx2, VPT, 2>
                <<<grid_size, 8192 / VPTx2, 0, stream>>>(PTR<bf16x2_t>(x),
                                                         PTR<bf16x2_t>(skip),
                                                         PTR<bf16x2_t>(weight),
                                                         PTR<bf16x2_t>(output),
                                                         PTR<bf16x2_t>(skip_output),
                                                         eps);
            break;
        default:
            throw std::runtime_error(c10::str("Normalize shape (",
                                              normalize_shape,
                                              ") is not supported yet, check rmsnorm.cu for more information."));
    };
    return {output, skip_output};
}

}  // namespace impl
}  // namespace atex