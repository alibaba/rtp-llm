#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/cuda_impl/Dispatch.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"

using namespace std;

namespace fastertransformer {

LayernormOutput CudaDevice::layernorm(const LayernormParams& params) {
    const auto& input = params.input;
    auto& output = params.norm_output;
    const auto& weights = params.weights;
    const auto& gamma = weights ? weights->get().gamma.get()->data() : nullptr;
    const auto& beta = (weights && weights->get().beta) ? weights->get().beta.get()->data() : nullptr;

    const auto norm_type = params.norm_type;
    const auto data_type = input.type();
    const auto m = input.shape()[0];
    const auto n = input.shape()[1];
    const auto eps = params.eps;

    if (!weights.has_value()) {
        if (!params.add_bias_output.has_value()) {
            throw OpException({OpErrorType::ERROR_UNIMPLEMENTED,
                               "Layernorm without weight must provide bias output."});
        }
        if (params.alpha.has_value() || (norm_type == NormType::alphanorm)) {
            const auto alpha = params.alpha.value_or(1.0f);
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeAlphaAddBiasResidual,
                output.data(),
                input.data(),
                params.residual1 ? params.residual1.value().get().data() : nullptr,
                params.bias ? params.bias.value().get().data() : nullptr,
                alpha,
                m,
                n,
                stream_
            );
        } else if (params.bias.has_value()) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeAddBiasResidual,
                output.data(),
                input.data(),
                params.residual1 ? params.residual1.value().get().data() : nullptr,
                params.residual2 ? params.residual2.value().get().data() : nullptr,
                params.bias.value().get().data(),
                nullptr, // scale_inter
                nullptr, // scale_out
                m,
                n,
                stream_
            );
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
        sync_check_cuda_error();
        return;
    }

    if (!(norm_type == NormType::layernorm || norm_type == NormType::rmsnorm)) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    if (params.residual1.has_value() || params.bias.has_value()) {
        const auto& add_bias_output = params.add_bias_output ? params.add_bias_output.value().get() : output;
        if (params.norm_type == NormType::layernorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeGeneralAddBiasResidualLayerNorm,
                add_bias_output.data(),
                output.data(),
                input.data(),
                params.bias ? params.bias.value().get().data() : nullptr,
                params.residual1 ? params.residual1.value().get().data() : nullptr,
                gamma,
                beta,
                eps,
                m,
                n,
                stream_,
                true, // use_diff_of_squares
                nullptr, // scale
                nullptr, // dynamic_scale
                nullptr  // out_quant
            );
        } else if (params.norm_type == NormType::rmsnorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeAddBiasResidualRmsNorm,
                add_bias_output.data(),
                output.data(),
                input.data(),
                params.bias ? params.bias.value().get().data() : nullptr,
                params.residual1 ? params.residual1.value().get().data() : nullptr,
                gamma,
                beta,
                eps,
                m,
                n,
                stream_,
                nullptr, // scale
                nullptr, // dynamic_scale
                nullptr  // out_quant
            );
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    } else {
        if (params.norm_type == NormType::layernorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeGeneralLayerNorm,
                output.data(),
                input.data(),
                gamma,
                beta,
                eps,
                m,
                n,
                stream_,
                true, // use_diff_of_squares
                nullptr, // scale
                nullptr, // dynamic_scale
                nullptr  // out_quant
            );
        } else if (params.norm_type == NormType::rmsnorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeGeneralRmsNorm,
                output.data(),
                input.data(),
                gamma,
                beta,
                eps,
                m,
                n,
                stream_,
                nullptr, // scale
                nullptr, // dynamic_scale
                nullptr  // out_quant
            );
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }
    sync_check_cuda_error();
    return;
}


} // namespace fastertransformer
