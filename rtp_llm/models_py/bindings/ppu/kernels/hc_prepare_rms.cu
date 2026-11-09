#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace {
// Reuse the installed FP32 MeanOps combine/shuffle/project arithmetic.
// Only load scalar type changes; the logical config is obtained from FP32.
struct PrepareOps: at::native::MeanOps<float, float, float, float> {
    float* xf;
    int    width;
    bool   mean;
    float* inverse;
    float  epsilon;
    PrepareOps(float* converted, int k, bool average, float* inv, float eps):
        at::native::MeanOps<float, float, float, float>(1.0f / 4096),
        xf(converted),
        width(k),
        mean(average),
        inverse(inv),
        epsilon(eps) {}
    inline C10_DEVICE float reduce(float a, c10::BFloat16 b, int64_t index) const {
        const float value = static_cast<float>(b);
        // Guarded config: output_idx<1>() == blockIdx.x, no global split.
        xf[static_cast<int64_t>(blockIdx.x) * width + index] = value;
        return __fadd_rn(a, __fmul_rn(value, value));
    }
    inline C10_DEVICE float project(float a) const {
        const float statistic = mean ? __fmul_rn(a, 1.0f / 4096) : a;
        // The installed SDK float rsqrt overload uses __ppumath_rsqrt_f32.
        // Mean stays unchanged; inverse is a distinct FP32 output.
        if (inverse)
            inverse[blockIdx.x] = ::rsqrt(__fadd_rn(statistic, epsilon));
        return statistic;
    }
};
std::tuple<at::Tensor, at::Tensor, at::Tensor> prepare_rms(const at::Tensor& input, double eps) {
    const bool mean = true;
    TORCH_CHECK(eps == 1e-6, "Only frozen HC RMS epsilon is qualified for this probe");
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.size(0) == 24
                    && input.size(1) == (mean ? 4096 : 16384) && input.is_contiguous()
                    && reinterpret_cast<uintptr_t>(input.data_ptr()) % 8 == 0,
                "HC Q3 prepare requires aligned contiguous BF16 [24,16384|4096]");
    const c10::cuda::CUDAGuard guard(input.device());
    auto                       xf           = at::empty(input.sizes(), input.options().dtype(at::kFloat));
    auto                       output       = at::empty({24, 1}, xf.options());
    auto                       inverse      = at::empty({24, 1}, xf.options());
    auto                       first_output = output.narrow(0, 0, 1), first_input = xf.narrow(0, 0, 1);
    auto                       single = at::TensorIterator::reduce_op(first_output, first_input);
    using namespace at::native;
    auto config = setReduceConfig<float, float, 4, 4>(single);
    TORCH_CHECK(config.num_outputs == 1 && config.num_inputs == input.size(1) && config.block_width == 512
                    && config.block_height == 1 && config.vectorize_input && config.output_vec_size == 1
                    && config.step_input == 512 && config.step_output == 1 && config.input_mult[0] == 1
                    && config.output_mult[0] == 0 && (config.output_mult[1] == 0 || config.output_mult[1] == 1)
                    && config.should_block_x_reduce() && !config.should_global_reduce() && config.ctas_per_output == 1,
                "Unsupported FP32 singleton geometry");
    // Mixed BF16 input / FP32 output retains actual byte strides in calculators.
    auto batch = at::TensorIterator::reduce_op(output, input);
    TORCH_CHECK(batch.num_output_elements() == 24 && batch.can_use_32bit_indexing() && batch.num_reduce_dims() == 1
                    && batch.input_dtype() == at::kBFloat16,
                "Unexpected BF16 iterator");
    config.num_outputs = 24;
    auto reduce        = ReduceOp<c10::BFloat16, PrepareOps, uint32_t, float, 4, 4>(
        PrepareOps(xf.data_ptr<float>(), input.size(1), mean, inverse.data_ptr<float>(), static_cast<float>(eps)),
        config,
        make_input_calculator<uint32_t>(batch),
        make_output_calculator<uint32_t>(batch),
        batch.data_ptr(1),
        static_cast<char*>(batch.data_ptr(0)),
        std::nullopt,
        nullptr,
        nullptr,
        nullptr,
        0.0f,
        1,
        0);
    reduce.accumulate   = false;
    reduce.final_output = true;
    launch_reduce_kernel<512>(config, reduce);
    return {xf, output, inverse};
}
}  // namespace
TORCH_LIBRARY(hc_prepare_q3_v3_probe, m) {
    m.def("prepare_rms(Tensor input, float eps) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(hc_prepare_q3_v3_probe, CUDA, m) {
    m.impl("prepare_rms", prepare_rms);
}
