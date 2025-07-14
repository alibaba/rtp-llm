#pragma once
#include <torch/torch.h>

#include "rtp_llm/cpp/devices/testing/TestBase.h"

namespace rtp_llm {

namespace torch_impl {

torch::Tensor GeGluNoneApproximate(const torch::Tensor& tensor) {
    return torch::gelu(tensor, "tanh");
}

torch::Tensor gelu(const torch::Tensor& tensor) {
    return torch::gelu(tensor);
}

// torch::Tensor geglu(const torch::Tensor& tensor) {
//     RTP_LLM_CHECK(tensor.shape()[tensor.dim() - 1] % 2 == 0);
//     auto chunk = tensor.chunk(2, tensor.dim() - 1);
//     return torch::gelu(tensor);
// }

torch::Tensor identity(const torch::Tensor& tensor) {
    return tensor;
}

static std::unordered_map<ActivationType, std::function<torch::Tensor(const torch::Tensor&)>> ACT2FUN{
    {ActivationType::Gelu, gelu},
    {ActivationType::Geglu, gelu},
    {ActivationType::GeGluNoneApproximate, GeGluNoneApproximate},
    {ActivationType::Swiglu, torch::silu},
    {ActivationType::Silu, torch::silu},
    {ActivationType::Sigmoid, torch::sigmoid},
    {ActivationType::Identity, identity},

};

struct LoraLinearLayerImpl: torch::nn::Module {

    LoraLinearLayerImpl(int input_hidden_size, int output_hidden_size):
        f(torch::nn::LinearOptions(input_hidden_size, output_hidden_size).bias(false)) {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("f", f);
    }

    torch::nn::Linear f;

    torch::Tensor forward(torch::Tensor input) {
        return f(input);
    }

    torch::Tensor forwardLora(torch::Tensor              input,
                              std::vector<int>           input_lengths,
                              std::vector<torch::Tensor> lora_a,
                              std::vector<torch::Tensor> lora_b) {
        auto          output      = f(input);
        torch::Tensor lora_output = torch::empty(0);
        size_t        start       = 0;
        for (int i = 0; i < input_lengths.size(); i++) {
            auto lora_input = input.index({torch::indexing::Slice(start, start + input_lengths[i])}).contiguous();
            auto lora_output_slice = torch::matmul(lora_input.to(torch::kFloat), lora_a[i].to(torch::kFloat));
            lora_output_slice      = torch::matmul(lora_output_slice.to(torch::kFloat), lora_b[i].to(torch::kFloat));
            lora_output            = torch::cat({lora_output, lora_output_slice}, 0);
            start                  = start + input_lengths[i];
        }
        output = output + lora_output;
        return output;
    }
};

TORCH_MODULE(LoraLinearLayer);

struct FfnLayerImpl: torch::nn::Module {
    FfnLayerImpl(int hidden_size, int intermediate_size, ActivationType act_t):
        gate_proj(hidden_size, intermediate_size),
        up_proj(hidden_size, intermediate_size),
        down_proj(intermediate_size, hidden_size),
        act_t(act_t) {
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

    torch::Tensor forwardLora(torch::Tensor              input,
                              std::vector<int>           input_lengths,
                              std::vector<torch::Tensor> gate_lora_a,
                              std::vector<torch::Tensor> gate_lora_b,
                              std::vector<torch::Tensor> up_lora_a,
                              std::vector<torch::Tensor> up_lora_b,
                              std::vector<torch::Tensor> down_lora_a,
                              std::vector<torch::Tensor> down_lora_b) {
        if (isGatedActivation(act_t)) {
            auto gate_lora_output = gate_proj->forwardLora(input, input_lengths, gate_lora_a, gate_lora_b);
            auto up_lora_output   = up_proj->forwardLora(input, input_lengths, up_lora_a, up_lora_b);
            return down_proj->forwardLora(
                ACT2FUN[act_t](gate_lora_output) * up_lora_output, input_lengths, down_lora_a, down_lora_b);
        }
        auto up_lora_output = up_proj->forwardLora(input, input_lengths, up_lora_a, up_lora_b);
        return down_proj->forwardLora(ACT2FUN[act_t](up_lora_output), input_lengths, down_lora_a, down_lora_b);
    }

    LoraLinearLayer gate_proj, up_proj, down_proj;
    ActivationType  act_t;
};

TORCH_MODULE(FfnLayer);

//// MOE Layer
struct Expert: torch::nn::Module {
    Expert(int64_t tok_dim, int64_t inter_size, int64_t output_size, ActivationType act_t): act_t(act_t) {
        gate = register_module("gate", torch::nn::Linear(tok_dim, inter_size));
        up   = register_module("up", torch::nn::Linear(tok_dim, inter_size));
        down = register_module("down", torch::nn::Linear(inter_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x, int64_t id) {
        if (isGatedActivation(act_t)) {
            auto ret = down(ACT2FUN[act_t](gate(x)) * up(x));
            {
                std::cout << "gate: expert_idx:" << id
                          << "\n"
                          //   << gate->weight.sizes() << "\n"
                          //   << gate->weight.index_select(0, torch::arange(0, 8)).index_select(1, torch::arange(0, 8))
                          //   << "\n"
                          << gate(x).sizes() << "\n"
                          << (ACT2FUN[act_t](gate(x))).index_select(1, torch::arange(0, 8)) << std::endl;
                std::cout
                    << "up: expert_idx:" << id
                    << "\n"
                    //   << up->weight.sizes() << "\n"
                    //   << up->weight.index_select(0, torch::arange(0, 8)).index_select(1, torch::arange(0, 8)) <<"\n"
                    << up(x).sizes() << "\n"
                    << (up(x)).index_select(1, torch::arange(0, 8)) << std::endl;
                std::cout << "down: expert_idx:" << id
                          << "\n"
                          //   << down->weight.sizes() << "\n"
                          //   << down->weight.index_select(0, torch::arange(0, 8)).index_select(1, torch::arange(0,8))
                          //   << "\n"
                          << ret.sizes() << "\n"
                          << (ret).index_select(1, torch::arange(0, 8)) << std::endl;
                std::cout << "test: expert_idx:" << id << "\n"
                          << ret.sizes() << "\n"
                          << (down(ACT2FUN[act_t](gate(x)))).index_select(1, torch::arange(0, 8)) << std::endl;
            }
            return ret;
        }
        auto ret = down(ACT2FUN[act_t](up(x)));
        {
            std::cout
                << "up: expert_idx:" << id
                << "\n"
                //   << up->weight.sizes() << "\n"
                //   << up->weight.index_select(0, torch::arange(0, 8)).index_select(1, torch::arange(0, 8)) << "\n"
                //   << up(x).sizes() << "\n"
                << (ACT2FUN[act_t](up(x))).index_select(1, torch::arange(0, 8)) << std::endl;
            std::cout
                << "down: expert_idx:" << id
                << "\n"
                //   << down->weight.sizes() << "\n"
                //   << down->weight.index_select(0, torch::arange(0, 8)).index_select(1, torch::arange(0, 8)) << "\n"
                << ret.sizes() << "\n"
                << (ret).index_select(1, torch::arange(0, 8)) << std::endl;
        }
        return ret;
    }

    torch::nn::Linear gate{nullptr}, up{nullptr}, down{nullptr};
    ActivationType    act_t;
};

struct GatingNetwork: torch::nn::Module {
    GatingNetwork(int64_t tok_dim, int64_t num_experts) {
        gating = register_module("gating", torch::nn::Linear(tok_dim, num_experts));
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::softmax(gating->forward(x), /*dim=*/1);
    }

    torch::nn::Linear gating{nullptr};
};

struct MoEImpl: torch::nn::Module {
    MoEImpl(int64_t tok_dim, int64_t inter_size, int64_t num_experts, int64_t topK, ActivationType act_t): topK(topK) {
        for (int64_t i = 0; i < num_experts; ++i) {
            experts.push_back(register_module("expert" + std::to_string(i),
                                              std::make_shared<Expert>(tok_dim, inter_size, tok_dim, act_t)));
        }
        gating_network = register_module("gating_network", std::make_shared<GatingNetwork>(tok_dim, num_experts));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto gate_outputs = gating_network->forward(x);

        auto topk         = gate_outputs.topk(topK, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
        auto topk_values  = std::get<0>(topk);
        auto topk_indices = std::get<1>(topk);
        std::cout << "topk_values:\n" << topk_values << std::endl;
        std::cout << "topk_indices:\n" << topk_indices << std::endl;

        auto output = torch::zeros({x.size(0), x.size(1)});

        for (int64_t expert_idx = 0; expert_idx < experts.size(); expert_idx++) {
            auto indices = torch::nonzero(topk_indices.eq(expert_idx));
            std::cout << "sub_indices:" << expert_idx << "\n" << indices.sizes() << "\n" << indices << std::endl;

            auto subTokens = x.index_select(0, indices.index({"...", 0}));
            // std::cout << "subTokens:" << expert_idx << "\n"
            //           << subTokens.sizes() << "\n"
            //           << subTokens.index_select(1, torch::arange(0, 8)) << std::endl;

            auto exp_out = experts[expert_idx]->forward(subTokens, expert_idx);
            std::cout << "exp_out:" << expert_idx << "\n"
                      << exp_out.sizes() << "\n"
                      << exp_out.index_select(1, torch::arange(0, 8)) << std::endl;
            for (int64_t i = 0; i < subTokens.size(0); ++i) {
                auto  row     = indices[i];
                auto  tokenID = row[0].item<int64_t>();
                auto  kID     = row[1].item<int64_t>();
                float weight  = topk_values.index({tokenID, kID}).item<float>();
                output[tokenID] += weight * exp_out[i];
            }
        }
        std::cout << "output:" << "\n"
                  << output.sizes() << "\n"
                  << output.index_select(1, torch::arange(0, 8)) << std::endl;
        return output;
    }

    std::vector<std::shared_ptr<Expert>> experts;
    std::shared_ptr<GatingNetwork>       gating_network;
    int64_t                              topK;
};
TORCH_MODULE(MoE);

};  // namespace torch_impl

};  // namespace rtp_llm