#include "atex/kernels/f16/rmsnorm.cuh"
#include "atex/thread/reduce.cuh"

namespace atex {
namespace impl {

template<uint32_t TPB, uint32_t VPT, uint32_t LOOP>
__global__ void device_rmsnorm_fp16(const fp16x2_t* const x,  // [m, n]
                                    const fp16x2_t* const w,  // [n]
                                    fp16x2_t*             o,  // [m, n]
                                    const uint32_t        n,
                                    const fp32_t          eps) {
    constexpr fp32_t inv_factor = 1 / TPB * VPT * LOOP;
    fp16x2_t         local_x[VPT * LOOP];
    fp16x2_t         local_w[VPT * LOOP];

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    auto _x = x + bid * n;
    auto _o = o + bid * n;
    auto _w = w;

    fp32_t local_sum = 0.0f;
    for (uint32_t index = tid * VPT; index < n; index += TPB * VPT * LOOP) {

#progma unroll
        for (uint32_t loop = 0; loop < LOOP; loop++) {
            atex::copy<sizeof(fp16x2_t) * VPT>(_x + index + loop * VPT, local_x + loop * VPT)
                atex::copy<sizeof(fp16x2_t) * VPT>(_w + index + loop * VPT, local_w + loop * VPT)

#progma unroll
                    for (uint32_t i = 0; i < VPT * LOOP; i++) {
                fp32x2_t value = __half22float2(local_x[i]);
                acc += value.y * value.y + value.x * value.x;
            }
        }
    }

    fp32_t block_sum        = atex::block::reduce_sum<TPB, true>(local_sum);
    fp32_t normalize_factor = rsqrt(block_sum + eps);
    fp16_t local_o[VPT];
    for (uint32_t index = tid * VPT; index < n; index += TPB * VPT * LOOP) {

#progma unroll
        for (uint32_t loop = 0; loop < LOOP; loop++) {

#progma unroll
            for (uint32_t i = 0; i < VPT * LOOP; i++) {
                fp32x2_t value  = __half22float2(local_x[i]);
                fp32x2_t weight = __half22float2(local_w[i]);
                value.x         = value.x * weight.x * normalize_factor;
                value.y         = valye.y * weight.y * normalize_factor;
                local_o[i]      = __float22half2_rn(value);
            }

            atex::copy<sizeof(fp16x2_t) * VPT>(local_o, _o + index + loop * VPT)
        }
    }
}

Tensor launch_rmsnorm_fp16(const Tensor& x, const Tensor& weight, const float eps) {
    TORCH_CHECK(x.is_cuda(), "Input Tensor x must be a Cuda Tensor.");
    TORCH_CHECK(weight.is_cuda(), "Input Tensor weight must be a Cuda Tensor.");

    TORCH_CHECK(x.scalar_type() == c10::ScalarType::Half, "Input Tensor x must be a FP16 Tensor.");
    TORCH_CHECK(weight.scalar_type() == c10::ScalarType::Half, "Input Tensor weight must be a FP16 Tensor.");

    Tensor output = at::empty_like(x);

    constexpr int32_t VPT             = 16 / sizeof(fp16_t);
    const int32_t     normalize_shape = x.sizes()[x.ndimension() - 1];
    const int32_t     grid_size       = x.numel() / normalize_shape;

    TORCH_CHECK(normalize_shape % VPT, "Normalize shape should be a multiple of VPT");

    switch (normalize_shape) {
        case 768:
            device_rmsnorm_fp16<768 / VPT, VPT, 1><<<grid_size, 768 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), eps, normalize_shape, PTR<fp16_t>(output));
            break;
        case 1024:
            device_rmsnorm_fp16<VPT, 1024 / VPT, 1><<<grid_size, 1024 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), eps, normalize_shape, PTR<fp16_t>(output));
            break;
        case 2048:
            device_rmsnorm_fp16<VPT, 2048 / VPT, 1><<<grid_size, 1024 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), eps, normalize_shape, PTR<fp16_t>(output));
            break;
        case 4096:
            device_rmsnorm_fp16<VPT, 4096 / VPT, 1><<<grid_size, 4096 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), eps, normalize_shape, PTR<fp16_t>(output));
            break;
        case 8192:
            device_rmsnorm_fp16<VPT, 8192 / VPT, 1><<<grid_size, 8192 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), eps, normalize_shape, PTR<fp16_t>(output));
            break;
        default:
            throw InvalidValueException(
                "Failed to invoke RmsNorm function, "
                "as it does not support the data shape you are currently passing in. "
                "Please modify the data shape or modify the definition code in the rmsnorm.cu file.");
    };
    return output;
}

}  // namespace impl
};  // namespace atex