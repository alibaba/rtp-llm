#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/interface.h"
#include "rtp_llm/cpp/pybind/th_utils.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

namespace torch_ext {

namespace tkc = tensorrt_llm::kernels::cutlass_kernels;
namespace tc  = tensorrt_llm::cutlass_extensions;

template<typename T>
Tensor int8_gemm_helper(Tensor                  input_activations,
                        Tensor                  weight,
                        Tensor                  alphaCol,
                        Tensor                  alphaRow,
                        torch::optional<Tensor> bias,
                        const int64_t           timing_iterations,
                        float&                  avg_time,
                        bool                    select_config) {

    const int m      = input_activations.size(0);
    const int n      = weight.size(0);
    const int k      = input_activations.size(1);
    auto      stream = at::cuda::getCurrentCUDAStream().stream();

    const int8_t* input_act_ptr = get_ptr<const int8_t>(input_activations);
    const int8_t* weight_ptr    = get_ptr<const int8_t>(weight);
    const float*  alpha_col_ptr = get_ptr<const float>(alphaCol);
    const float*  alpha_row_ptr = get_ptr<const float>(alphaRow);

    T* bias_ptr = bias.has_value() ? (T*)bias.value().data_ptr() : nullptr;

    auto output_tensor = torch::empty({m, n}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    T*   output_tensor_ptr = get_ptr<T>(output_tensor);

    using SqGemmRunnerPtr  = std::shared_ptr<tkc::CutlassInt8GemmRunnerInterface>;
    SqGemmRunnerPtr runner = std::make_shared<tkc::CutlassInt8GemmRunner<T>>();

    const int ws_bytes = runner->getWorkspaceSize(m, n, k);
    auto  ws_tensor = torch::empty({ws_bytes}, torch::dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));
    char* ws_ptr    = get_ptr<char>(ws_tensor);

    tk::QuantMode quant_mode = tk::QuantMode::fromDescription(true, true, true, true, false, false, false, false);

    if (select_config) {
        tc::CutlassGemmConfig              best_config;
        std::vector<tc::CutlassGemmConfig> configs  = runner->getValidConfigs(input_act_ptr,
                                                                             weight_ptr,
                                                                             quant_mode,
                                                                             alpha_col_ptr,
                                                                             alpha_row_ptr,
                                                                             output_tensor_ptr,
                                                                             m,
                                                                             n,
                                                                             k,
                                                                             ws_ptr,
                                                                             ws_bytes,
                                                                             stream);
        float                              min_time = 100000.0f;

        for (int i = 0; i < configs.size(); i++) {
            float total_time_fpaintb = 0;
            if (tkc::is_valid_split_k_factor(m, n, k, configs[i], ws_bytes, true) == false) {
                continue;
            }

            auto runner_operation = [&](cudaStream_t stream) {
                runner->gemm(input_act_ptr,
                             weight_ptr,
                             quant_mode,
                             alpha_col_ptr,
                             alpha_row_ptr,
                             output_tensor_ptr,
                             bias_ptr,
                             tkc::CutlassActivationType::IDENTITY,
                             m,
                             n,
                             k,
                             configs[i],
                             ws_ptr,
                             ws_bytes,
                             stream);
            };
            float cur_avg_time = rtp_llm::timing_function(runner_operation, timing_iterations, stream);

            if (cur_avg_time < min_time) {
                min_time    = cur_avg_time;
                best_config = configs[i];
            }
        }
        avg_time = min_time;
        tkc::print_config_file(best_config, m, n, k, min_time, "./config.ini", std::ios::app);
    } else {
        tc::CutlassGemmConfig config = runner->getChosenConfig(input_act_ptr,
                                                               weight_ptr,
                                                               quant_mode,
                                                               alpha_col_ptr,
                                                               alpha_row_ptr,
                                                               output_tensor_ptr,
                                                               m,
                                                               n,
                                                               k,
                                                               ws_ptr,
                                                               ws_bytes,
                                                               stream);

        auto runner_operation = [&](cudaStream_t stream) {
            runner->gemm(input_act_ptr,
                         weight_ptr,
                         quant_mode,
                         alpha_col_ptr,
                         alpha_row_ptr,
                         output_tensor_ptr,
                         bias_ptr,
                         tkc::CutlassActivationType::IDENTITY,
                         m,
                         n,
                         k,
                         config,
                         ws_ptr,
                         ws_bytes,
                         stream);
        };
        avg_time = rtp_llm::timing_function(runner_operation, timing_iterations, stream);
    }

    return output_tensor;
}

Tensor _int8_gemm(Tensor                  input_activations,
                  Tensor                  weight,
                  Tensor                  alphaCol,
                  Tensor                  alphaRow,
                  torch::optional<Tensor> bias,
                  int64_t                 timing_iterations,
                  float&                  avg_time,
                  bool                    select_config) {
    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank for weight");

    TORCH_CHECK(input_activations.size(1) == weight.size(1), "dim 1 of act and dim 0 of weight must be equal");

    Tensor output_tensor;
    output_tensor = int8_gemm_helper<half>(
        input_activations, weight, alphaCol, alphaRow, bias, timing_iterations, avg_time, select_config);
    return output_tensor;
}

Tensor
int8_gemm(Tensor input_activations, Tensor weight, Tensor alphaCol, Tensor alphaRow, torch::optional<Tensor> bias) {
    float dummy = 0.f;
    return _int8_gemm(input_activations, weight, alphaCol, alphaRow, bias, 1, dummy, false);
}

Tensor int8_gemm_config_select(
    Tensor input_activations, Tensor weight, Tensor alphaCol, Tensor alphaRow, const int64_t timing_iterations) {
    float avg_time = 0;
    return _int8_gemm(
        input_activations, weight, alphaCol, alphaRow, torch::optional<Tensor>(), timing_iterations, avg_time, true);
}

TORCH_LIBRARY(int8_gemm_ops, m) {
    m.def("int8_gemm", int8_gemm);
    m.def("int8_gemm_config_select", int8_gemm_config_select);
}

}  // namespace torch_ext