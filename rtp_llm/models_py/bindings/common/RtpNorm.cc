#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/common/RtpNorm.h"
#include "rtp_llm/models_py/bindings/common/kernels/activation_kernels.h"
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
using namespace std;
namespace th = torch;
using namespace rtp_llm;
namespace torch_ext {
void fused_bias_add(at::Tensor& input, at::Tensor& bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    CHECK_DIM(2, input);
    TORCH_CHECK(bias.dim() == 1 || (bias.dim() == 2 && bias.size(0) == 1),
                "bias must have shape [hidden_size] or [1, hidden_size]");
    CHECK_EQ(input.device(), bias.device());
    CHECK_EQ(input.scalar_type(), bias.scalar_type());
    CHECK_EQ(input.size(1), bias.numel());
    StreamType stream = GET_CURRENT_STREAM();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        invokeAddBias(static_cast<c_type*>(input.data_ptr()),
                      static_cast<c_type*>(bias.data_ptr()),
                      input.numel(),
                      input.size(1),
                      stream);
        return true;
    });
}

void fused_bias_gelu(at::Tensor& input, at::Tensor& bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    CHECK_DIM(2, input);
    TORCH_CHECK(bias.dim() == 1 || (bias.dim() == 2 && bias.size(0) == 1),
                "bias must have shape [hidden_size] or [1, hidden_size]");
    CHECK_EQ(input.device(), bias.device());
    CHECK_EQ(input.scalar_type(), bias.scalar_type());
    CHECK_EQ(input.size(1), bias.numel());
    StreamType stream = GET_CURRENT_STREAM();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        invokeAddBiasGelu(static_cast<c_type*>(input.data_ptr()),
                          static_cast<c_type*>(bias.data_ptr()),
                          input.numel(),
                          input.size(1),
                          stream);
        return true;
    });
}

void fused_bias_gelu_quant_fp8(at::Tensor& input, at::Tensor& bias, at::Tensor& output, at::Tensor& scales) {
#if USING_CUDA
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    CHECK_INPUT(output);
    CHECK_CUDA(scales);
    CHECK_DIM(2, input);
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    CHECK_DIM(2, output);
    CHECK_DIM(2, scales);
    CHECK_EQ(input.device(), bias.device());
    CHECK_EQ(input.device(), output.device());
    CHECK_EQ(input.device(), scales.device());
    CHECK_EQ(input.scalar_type(), bias.scalar_type());
    CHECK_EQ(input.size(0), output.size(0));
    CHECK_EQ(input.size(1), output.size(1));
    CHECK_EQ(input.size(1), bias.numel());
    TORCH_CHECK(input.size(1) % 128 == 0, "hidden size must be divisible by 128");
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Float8_e4m3fn, "output must be float8_e4m3fn");
    TORCH_CHECK(scales.scalar_type() == at::ScalarType::Int, "scales must be int32 UE8M0 packs");
    TORCH_CHECK(scales.stride(0) == 1, "scales must use column-major TMA layout");
    TORCH_CHECK(scales.size(0) == input.size(0), "scale row count mismatch");
    TORCH_CHECK(scales.size(1) * 4 >= input.size(1) / 128, "scale column count mismatch");
    StreamType stream = GET_CURRENT_STREAM();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        invokeAddBiasGeluQuantFp8(static_cast<c_type*>(input.data_ptr()),
                                  static_cast<c_type*>(bias.data_ptr()),
                                  output.data_ptr(),
                                  static_cast<uint32_t*>(scales.data_ptr()),
                                  input.size(0),
                                  input.size(1),
                                  scales.stride(1),
                                  stream);
        return true;
    });
#else
    TORCH_CHECK(false, "fused_bias_gelu_quant_fp8 is CUDA-only");
#endif
}

void layernorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& beta, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    auto device = input.device();
    CHECK_EQ(weight.device(), device);
    CHECK_EQ(beta.device(), device);
    CHECK_DIM(2, input);   // input: (batch_size, hidden_size)
    CHECK_DIM(1, weight);  // weight: (hidden_size)
    CHECK_DIM(1, beta);    // weight: (hidden_size)
    CHECK_EQ(input.size(1), weight.size(0));
    unsigned int batch_size  = input.size(0);
    unsigned int hidden_size = input.size(1);
    CHECK_EQ(output.size(0), batch_size);
    CHECK_EQ(output.size(1), hidden_size);

    StreamType stream = GET_CURRENT_STREAM();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        invokeGeneralLayerNorm(static_cast<c_type*>(nullptr),
                               static_cast<c_type*>(output.data_ptr()),
                               static_cast<c_type*>(input.data_ptr()),
                               static_cast<c_type*>(weight.data_ptr()),
                               static_cast<c_type*>(beta.data_ptr()),
                               eps,
                               batch_size,
                               hidden_size,
                               stream);
        // true, (const float*)nullptr, (float*)nullptr, static_cast<c_type*>(nullptr), true);
        return true;
    });
}

void fused_add_layernorm(
    at::Tensor& input, at::Tensor& residual, at::Tensor& bias, at::Tensor& weight, at::Tensor& beta, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(residual);
    CHECK_INPUT(weight);
    CHECK_INPUT(beta);
    auto device = input.device();
    CHECK_EQ(residual.device(), device);
    CHECK_EQ(weight.device(), device);
    CHECK_EQ(beta.device(), device);
    CHECK_EQ(residual.scalar_type(), input.scalar_type());
    CHECK_EQ(weight.scalar_type(), input.scalar_type());
    CHECK_EQ(beta.scalar_type(), input.scalar_type());
    CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
    CHECK_DIM(2, residual);  // input: (batch_size, hidden_size)
    CHECK_DIM(1, weight);    // weight: (hidden_size)
    CHECK_DIM(1, beta);      // weight: (hidden_size)
    CHECK_EQ(input.size(1), weight.size(0));
    CHECK_EQ(input.size(0), residual.size(0));
    CHECK_EQ(input.size(1), residual.size(1));
    CHECK_EQ(input.size(1), beta.size(0));
    // for bert model, bias is none
    if (bias.numel() != 0) {
        CHECK_INPUT(bias);
        CHECK_EQ(bias.device(), device);
        CHECK_EQ(bias.scalar_type(), input.scalar_type());
        TORCH_CHECK(bias.dim() == 1 || (bias.dim() == 2 && bias.size(0) == 1),
                    "bias must have shape [hidden_size] or [1, hidden_size]");
        CHECK_EQ(input.size(1), bias.numel());
    }
    unsigned int batch_size  = input.size(0);
    unsigned int hidden_size = input.size(1);

    StreamType stream = GET_CURRENT_STREAM();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        c_type* bias_ptr = nullptr;
        if (bias.numel() != 0) {
            bias_ptr = static_cast<c_type*>(bias.data_ptr());
        }
        invokeGeneralAddBiasResidualLayerNorm(static_cast<c_type*>(residual.data_ptr()),
                                              static_cast<c_type*>(input.data_ptr()),
                                              static_cast<c_type*>(input.data_ptr()),
                                              bias_ptr,
                                              static_cast<c_type*>(residual.data_ptr()),
                                              static_cast<c_type*>(weight.data_ptr()),
                                              static_cast<c_type*>(beta.data_ptr()),
                                              eps,
                                              batch_size,
                                              hidden_size,
                                              stream);
        // true, (const float*)nullptr, (float*)nullptr, static_cast<c_type*>(nullptr), true);
        return true;
    });
}

void fused_add_layernorm_quant_fp8(at::Tensor& input,
                                   at::Tensor& residual,
                                   at::Tensor& bias,
                                   at::Tensor& weight,
                                   at::Tensor& beta,
                                   at::Tensor& output,
                                   at::Tensor& scales,
                                   double      eps) {
#if USING_CUDA
    CHECK_INPUT(input);
    CHECK_INPUT(residual);
    CHECK_INPUT(weight);
    CHECK_INPUT(beta);
    CHECK_INPUT(output);
    CHECK_CUDA(scales);
    CHECK_DIM(2, input);
    CHECK_DIM(2, residual);
    CHECK_DIM(1, weight);
    CHECK_DIM(1, beta);
    CHECK_DIM(2, output);
    CHECK_DIM(2, scales);
    CHECK_EQ(input.device(), residual.device());
    CHECK_EQ(input.device(), weight.device());
    CHECK_EQ(input.device(), beta.device());
    CHECK_EQ(input.device(), output.device());
    CHECK_EQ(input.device(), scales.device());
    CHECK_EQ(input.scalar_type(), residual.scalar_type());
    CHECK_EQ(input.scalar_type(), weight.scalar_type());
    CHECK_EQ(input.scalar_type(), beta.scalar_type());
    CHECK_EQ(input.sizes(), residual.sizes());
    CHECK_EQ(input.sizes(), output.sizes());
    CHECK_EQ(input.size(1), weight.numel());
    CHECK_EQ(input.size(1), beta.numel());
    TORCH_CHECK(input.size(1) > 0 && input.size(1) <= 1024 && input.size(1) % 128 == 0,
                "hidden size must be in (0, 1024] and divisible by 128");
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Float8_e4m3fn, "output must be float8_e4m3fn");
    TORCH_CHECK(scales.scalar_type() == at::ScalarType::Int, "scales must be int32 UE8M0 packs");
    TORCH_CHECK(scales.stride(0) == 1, "scales must use column-major TMA layout");
    TORCH_CHECK(scales.size(0) == input.size(0), "scale row count mismatch");
    TORCH_CHECK(scales.size(1) * 4 >= input.size(1) / 128, "scale column count mismatch");
    if (bias.numel() != 0) {
        CHECK_INPUT(bias);
        CHECK_EQ(input.device(), bias.device());
        CHECK_EQ(input.scalar_type(), bias.scalar_type());
        TORCH_CHECK(bias.dim() == 1 && bias.numel() == input.size(1), "bias must have shape [hidden_size]");
    }
    if (input.numel() == 0) {
        return;
    }

    StreamType stream = GET_CURRENT_STREAM();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        const c_type* bias_ptr = bias.numel() == 0 ? nullptr : static_cast<const c_type*>(bias.data_ptr());
        invokeGeneralAddBiasResidualLayerNormQuantFp8(static_cast<c_type*>(residual.data_ptr()),
                                                      static_cast<c_type*>(input.data_ptr()),
                                                      static_cast<const c_type*>(input.data_ptr()),
                                                      bias_ptr,
                                                      static_cast<const c_type*>(residual.data_ptr()),
                                                      static_cast<const c_type*>(weight.data_ptr()),
                                                      static_cast<const c_type*>(beta.data_ptr()),
                                                      static_cast<float>(eps),
                                                      input.size(0),
                                                      input.size(1),
                                                      output.data_ptr(),
                                                      static_cast<uint32_t*>(scales.data_ptr()),
                                                      scales.stride(1),
                                                      stream);
        return true;
    });
#else
    TORCH_CHECK(false, "fused_add_layernorm_quant_fp8 is CUDA-only");
#endif
}
}  // namespace torch_ext
