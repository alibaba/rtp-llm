#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace {

template<typename Ops>
at::Tensor reduce_rows(const at::Tensor& input, Ops ops) {
    using namespace at::native;
    auto output       = at::empty({input.size(0), 1}, input.options());
    auto first_output = output.narrow(0, 0, 1);
    auto first_input  = input.narrow(0, 0, 1);
    auto single       = at::TensorIterator::reduce_op(first_output, first_input);
    auto batch        = at::TensorIterator::reduce_op(output, input);
    auto config       = setReduceConfig<float, float, 4, 4>(single);
    TORCH_CHECK(config.num_outputs == 1 && config.num_inputs == input.size(1) && config.block_width == 512
                    && config.block_height == 1 && config.vectorize_input && config.output_vec_size == 1
                    && config.step_input == 512 && config.step_output == 1 && config.should_block_x_reduce()
                    && !config.should_global_reduce() && config.ctas_per_output == 1,
                "Unsupported singleton reduction geometry");
    TORCH_CHECK(batch.num_output_elements() == input.size(0) && batch.can_use_32bit_indexing()
                    && batch.num_reduce_dims() == 1,
                "Unexpected batched reduction iterator");

    // Each CTA uses the singleton reduction geometry and reads one row.
    config.num_outputs  = input.size(0);
    auto reduce         = ReduceOp<float, Ops, uint32_t, float, 4, 4>(ops,
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
    return output;
}

struct AddFloat {
    C10_HOST_DEVICE float operator()(float a, float b) const {
        return a + b;
    }
};

at::Tensor row_reduce(const at::Tensor& input, bool mean) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat && input.dim() == 2 && input.is_contiguous()
                    && input.size(0) > 0 && input.size(0) <= 48 && input.size(1) == (mean ? 4096 : 16384)
                    && reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
                "Expected aligned FP32 HC rows");
    const c10::cuda::CUDAGuard guard(input.device());
    if (mean) {
        return reduce_rows(input, at::native::MeanOps<float, float, float, float>(1.0f / 4096));
    }
    return reduce_rows(input, at::native::func_wrapper<float>(AddFloat{}));
}

}  // namespace

TORCH_LIBRARY(rtp_llm_ppu, m) {
    m.def("hc_row_reduce(Tensor input, bool mean) -> Tensor");
}

TORCH_LIBRARY_IMPL(rtp_llm_ppu, CUDA, m) {
    m.impl("hc_row_reduce", row_reduce);
}
