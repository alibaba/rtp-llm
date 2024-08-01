#pragma once
#include <torch/torch.h>

#include "src/fastertransformer/devices/testing/TestBase.h"

namespace fastertransformer {

namespace torch_impl {

torch::Tensor GeGluNoneApproximate(const torch::Tensor& tensor) {
    return torch::gelu(tensor, "tanh");
}

torch::Tensor gelu(const torch::Tensor& tensor) {
    return torch::gelu(tensor);
}

// torch::Tensor geglu(const torch::Tensor& tensor) {
//     FT_CHECK(tensor.shape()[tensor.dim() - 1] % 2 == 0);
//     auto chunk = tensor.chunk(2, tensor.dim() - 1);
//     return torch::gelu(tensor);
// }

torch::Tensor identity(const torch::Tensor& tensor) {
    return tensor;
}

static std::unordered_map<ActivationType, std::function<torch::Tensor(const torch::Tensor&)>> ACT2FUN {
    {ActivationType::Gelu , gelu},
    {ActivationType::Geglu , gelu},
    {ActivationType::GeGluNoneApproximate , GeGluNoneApproximate},
    {ActivationType::Swiglu , torch::silu},
    {ActivationType::Silu, torch::silu},
    {ActivationType::Sigmoid, torch::sigmoid},
    {ActivationType::Identity, identity},

};


struct LoraLinearLayerImpl : torch::nn::Module {

    LoraLinearLayerImpl(int input_hidden_size, int output_hidden_size) :
    f(torch::nn::LinearOptions(input_hidden_size, output_hidden_size).bias(false)) {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("f", f);
    }

    torch::nn::Linear f;

    torch::Tensor forward(torch::Tensor input) {
        return f(input);
    }

    torch::Tensor forwardLora(torch::Tensor input,
                              std::vector<int> input_lengths,
                              std::vector<torch::Tensor> lora_a,
                              std::vector<torch::Tensor> lora_b)
    {
        auto output = f(input);
        torch::Tensor lora_output = torch::empty(0);
        size_t start = 0;
        for (int i = 0; i < input_lengths.size(); i++) {
            auto lora_input = input.index({torch::indexing::Slice(start, start + input_lengths[i])}).contiguous();
            auto lora_output_slice = torch::matmul(lora_input.to(torch::kFloat), lora_a[i].to(torch::kFloat));
            lora_output_slice = torch::matmul(lora_output_slice.to(torch::kFloat), lora_b[i].to(torch::kFloat));
            lora_output = torch::cat({lora_output, lora_output_slice}, 0);
            start = start + input_lengths[i];
        }
        output = output + lora_output;
        return output;
    }
};

TORCH_MODULE(LoraLinearLayer);

struct FfnLayerImpl : torch::nn::Module {
    FfnLayerImpl(int hidden_size, int intermediate_size, ActivationType act_t) :
                                    gate_proj(hidden_size, intermediate_size),
                                    up_proj(hidden_size, intermediate_size),
                                    down_proj(intermediate_size, hidden_size),
                                    act_t(act_t)
    {
        register_module("gate_proj", gate_proj);
        register_module("up_proj", up_proj);
        register_module("down_proj", down_proj);
   }

    torch::Tensor forward(torch::Tensor x) {
        if (isGatedActivation(act_t)) {
            return down_proj->forward(ACT2FUN[act_t](gate_proj->forward(x)) * up_proj->forward(x));
        }
        return down_proj->forward(ACT2FUN[act_t](up_proj->forward(x)));
    }

    torch::Tensor forwardLora(torch::Tensor input,
                              std::vector<int> input_lengths,
                              std::vector<torch::Tensor> gate_lora_a,
                              std::vector<torch::Tensor> gate_lora_b,
                              std::vector<torch::Tensor> up_lora_a,
                              std::vector<torch::Tensor> up_lora_b,
                              std::vector<torch::Tensor> down_lora_a,
                              std::vector<torch::Tensor> down_lora_b)
    {
        if (isGatedActivation(act_t)) {
            auto gate_lora_output = gate_proj->forwardLora(input,
                                                          input_lengths,
                                                          gate_lora_a,
                                                          gate_lora_b);
            auto up_lora_output = up_proj->forwardLora(input,
                                                      input_lengths,
                                                      up_lora_a,
                                                      up_lora_b);
            return down_proj->forwardLora(ACT2FUN[act_t](gate_lora_output) * up_lora_output,
                                         input_lengths,
                                         down_lora_a,
                                         down_lora_b);
        }
        auto up_lora_output = up_proj->forwardLora(input,
                                                  input_lengths,
                                                  up_lora_a,
                                                  up_lora_b);
        return down_proj->forwardLora(ACT2FUN[act_t](up_lora_output),
                                     input_lengths,
                                     down_lora_a,
                                     down_lora_b);

    }

    LoraLinearLayer gate_proj, up_proj, down_proj;
    ActivationType act_t;
};

TORCH_MODULE(FfnLayer);

};

};