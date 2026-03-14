#include "rtp_llm/models_py/bindings/cuda/DeepGemmPluginOp.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/cuda/deep_gemm/DeepGemmPlugin.h"
#include "rtp_llm/cpp/kernels/scaled_fp8_quant.h"

namespace rtp_llm {

static BufferPtr makeDummyZeros(MemoryType mem_type) {
    return std::make_shared<Buffer>(mem_type, DataType::TYPE_INVALID, std::vector<size_t>{0}, nullptr);
}

torch::Tensor
deep_gemm_fp8(torch::Tensor lhs_bf16, torch::Tensor rhs_data, torch::Tensor rhs_scale, int user_deep_gemm_num_sm) {
    CHECK_INPUT(lhs_bf16);
    CHECK_INPUT(rhs_data);
    CHECK_INPUT(rhs_scale);

    const int64_t m = lhs_bf16.size(0);
    const int64_t k = lhs_bf16.size(1);
    const int64_t n = rhs_data.size(0);

    auto          padding_size = DeepGemmPlugin::getPaddingSize(m, DeepGemmType::Normal);
    const int64_t padded_m     = static_cast<int64_t>((m + padding_size - 1) / padding_size * padding_size);

    auto opts_fp8  = torch::TensorOptions().dtype(TORCH_FP8_E4M3_TYPE).device(lhs_bf16.device());
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(lhs_bf16.device());
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(lhs_bf16.device());

    auto fp8_data  = torch::empty({padded_m, k}, opts_fp8);
    auto fp8_scale = torch::empty({k / 128, padded_m}, opts_f32);

    StreamType stream = GET_CURRENT_STREAM();

    invokeComputeFP8Quantize128(reinterpret_cast<__nv_fp8_e4m3*>(fp8_data.data_ptr()),
                                fp8_scale.data_ptr<float>(),
                                reinterpret_cast<const __nv_bfloat16*>(lhs_bf16.data_ptr()),
                                padded_m,
                                k,
                                static_cast<int64_t>(m * k),
                                true,
                                stream);

    // lhs: quantized fp8 [padded_m, k] + col-major scale [k/128, padded_m]
    auto lhs_k = torchTensor2Buffer(fp8_data);
    auto lhs_s = torchTensor2Buffer(fp8_scale);
    auto lhs   = std::make_shared<QBuffer>(std::move(lhs_k), std::move(lhs_s), makeDummyZeros(MemoryType::MEMORY_GPU));

    // rhs: fp8 [n, k], scale: fp32 [n/128, k/128]
    auto rhs_k = torchTensor2Buffer(rhs_data);
    auto rhs_s = torchTensor2Buffer(rhs_scale);
    auto rhs   = std::make_shared<QBuffer>(std::move(rhs_k), std::move(rhs_s), makeDummyZeros(MemoryType::MEMORY_GPU));

    if (padded_m > m) {
        auto padded_output = torch::empty({padded_m, n}, opts_bf16);
        auto out_buf       = torchTensor2Buffer(padded_output);
        DeepGemmPlugin::gemmFp8(*lhs, *rhs, *out_buf, user_deep_gemm_num_sm, stream);
        return padded_output.slice(0, 0, m).contiguous();
    } else {
        auto output  = torch::empty({m, n}, opts_bf16);
        auto out_buf = torchTensor2Buffer(output);
        DeepGemmPlugin::gemmFp8(*lhs, *rhs, *out_buf, user_deep_gemm_num_sm, stream);
        return output;
    }
}

void deep_gemm_grouped_fp8_masked(torch::Tensor lhs_data,
                                  torch::Tensor lhs_scale,
                                  torch::Tensor rhs_data,
                                  torch::Tensor rhs_scale,
                                  torch::Tensor output,
                                  torch::Tensor masked_m,
                                  int           expected_m,
                                  int           user_deep_gemm_num_sm) {
    CHECK_INPUT(lhs_data);
    CHECK_CUDA(lhs_scale);
    CHECK_INPUT(rhs_data);
    CHECK_INPUT(rhs_scale);
    CHECK_INPUT(output);
    CHECK_INPUT(masked_m);

    // lhs: fp8 [num_groups, m, k], scale: fp32 [num_groups, m, k/128] (row-major; V2 converts to col-major internally)
    auto lhs_k = torchTensor2Buffer(lhs_data);
    auto lhs_s = torchTensor2Buffer(lhs_scale);
    auto lhs   = std::make_shared<QBuffer>(std::move(lhs_k), std::move(lhs_s), makeDummyZeros(MemoryType::MEMORY_GPU));

    // rhs: fp8 [num_groups, n, k], scale: fp32 [num_groups, n/128, k/128]
    auto rhs_k = torchTensor2Buffer(rhs_data);
    auto rhs_s = torchTensor2Buffer(rhs_scale);
    auto rhs   = std::make_shared<QBuffer>(std::move(rhs_k), std::move(rhs_s), makeDummyZeros(MemoryType::MEMORY_GPU));

    auto out_buf    = torchTensor2Buffer(output);
    auto masked_buf = torchTensor2Buffer(masked_m);

    StreamType stream = GET_CURRENT_STREAM();
    DeepGemmPlugin::groupedGemmFp8Masked(*lhs, *rhs, *out_buf, *masked_buf, expected_m, user_deep_gemm_num_sm, stream);
}

void registerDeepGemmPluginOp(py::module& m) {
    m.def("deep_gemm_fp8",
          &deep_gemm_fp8,
          "DeepGemmPlugin FP8 GEMM: quantize bf16 lhs internally then A * B^T, returns output tensor",
          py::arg("lhs_bf16"),
          py::arg("rhs_data"),
          py::arg("rhs_scale"),
          py::arg("user_deep_gemm_num_sm") = -1);

    m.def("deep_gemm_grouped_fp8_masked",
          &deep_gemm_grouped_fp8_masked,
          "DeepGemmPlugin Grouped FP8 GEMM with masked layout",
          py::arg("lhs_data"),
          py::arg("lhs_scale"),
          py::arg("rhs_data"),
          py::arg("rhs_scale"),
          py::arg("output"),
          py::arg("masked_m"),
          py::arg("expected_m"),
          py::arg("user_deep_gemm_num_sm") = -1);
}

}  // namespace rtp_llm
