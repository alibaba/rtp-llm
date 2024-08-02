#pragma once
#include "src/fastertransformer/devices/testing/TestBase.h"
#include <torch/torch.h>
#include <iostream>
torch::Tensor GeGluNoneApproximate(const torch::Tensor& tensor) {
    return torch::gelu(tensor, "tanh");
}

torch::Tensor gelu(const torch::Tensor& tensor) {
    return torch::gelu(tensor);
}

std::unordered_map<ActivationType,
                   std::function<torch::Tensor(const torch::Tensor&)>> ACT2FUN {
    {ActivationType::Geglu , gelu},
    {ActivationType::GeGluNoneApproximate , GeGluNoneApproximate},
    {ActivationType::Swiglu , torch::silu},
    {ActivationType::Silu, torch::silu}
};

struct MLPImpl : torch::nn::Module {
    MLPImpl(int hidden_size, int intermediate_size, ActivationType act_t) :
        gate_proj(torch::nn::LinearOptions(hidden_size, intermediate_size)
            .bias(false)),
        up_proj(torch::nn::LinearOptions(hidden_size, intermediate_size)
            .bias(false)),
        down_proj(torch::nn::LinearOptions(intermediate_size, hidden_size)
            .bias(false)),
        act_t(act_t) {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("gate_proj", gate_proj);
        register_module("up_proj", up_proj);
        register_module("down_proj", down_proj);
   }

    torch::Tensor forward(torch::Tensor x) {
        if (isGatedActivation(act_t)) {
            return down_proj(ACT2FUN[act_t](gate_proj(x)) * up_proj(x));
        }
        return down_proj(ACT2FUN[act_t](up_proj(x)));
    }

    torch::Tensor forward_test(torch::Tensor x) {
        return ACT2FUN[act_t](gate_proj(x));
    }

    torch::Tensor forward_up(torch::Tensor x) {
        return up_proj(x);
    }

    torch::nn::Linear gate_proj, up_proj, down_proj;
    ActivationType act_t;
};

TORCH_MODULE(MLP);

class FfnLayerTest : public DeviceTestBase {
public:

    struct FfnLayerTestInput {
        torch::Tensor input;
        torch::Tensor gate_proj;
        torch::Tensor up_proj;
        torch::Tensor down_proj;
    };

    struct FfnLayerTestOutput {
        torch::Tensor out;
    };

    FfnLayerTestInput PrepareFfnLayerInput(size_t token_num,
                                             size_t hidden_size,
                                             size_t inter_size,
                                             DataType type)
    {
        auto dtype = dataTypeToTorchType(type);
        auto input = 0.01 * torch::rand(
            {(int)token_num, (int)hidden_size}, torch::Device(torch::kCPU)).to(dtype);
        auto gate_proj = 0.01 * torch::rand(
            {(int)hidden_size, (int)inter_size}, torch::Device(torch::kCPU)).to(dtype);
        auto up_proj = 0.01 * torch::rand(
            {(int)hidden_size, (int)inter_size}, torch::Device(torch::kCPU)).to(dtype);
        auto down_proj = 0.01 * torch::rand(
            {(int)inter_size, (int)hidden_size}, torch::Device(torch::kCPU)).to(dtype);
        return FfnLayerTestInput({input, gate_proj, up_proj, down_proj});
    }

    FfnLayerTestOutput FfnOpRun(FfnLayerTestInput& params,
                                ActivationType Atype)
    {
        bool is_cpu     = (this->device_->getDeviceProperties().type == DeviceType::Cpu);
        auto alloc_type = is_cpu ? AllocationType::HOST : AllocationType::DEVICE;
        auto input      = tensorToBuffer(params.input, alloc_type);
        auto gate_proj  = tensorToBuffer(params.gate_proj, alloc_type);
        auto up_proj    = tensorToBuffer(params.up_proj, alloc_type);
        auto down_proj  = tensorToBuffer(params.down_proj, alloc_type);

        FfnLayerWeights weights;
        weights.up_weight = std::make_unique<const DenseWeights>(DenseWeights(up_proj));
        weights.down_weight = std::make_unique<const DenseWeights>(DenseWeights(down_proj));
        weights.gate_weight = std::make_unique<const DenseWeights>(DenseWeights(gate_proj));

        FfnConfigs ffn_configs({Atype});
        FfnLayerParams Opparams(*input,
                                ffn_configs,
                                weights);

        auto output  = this->device_->ffnLayer(Opparams);
        return FfnLayerTestOutput({bufferToTensor(*(output.hidden_states))});

    }

    FfnLayerTestOutput FfnTorchRefRun(FfnLayerTestInput& params,
                                      ActivationType Atype)
    {
        MLP mlp(params.input.sizes()[1], params.gate_proj.sizes()[0], Atype);
        mlp.ptr()->to(torch::Device(torch::kCPU));
        auto state_dict = mlp.ptr()->named_parameters();
        torch::NoGradGuard no_grad;
        state_dict["gate_proj.weight"].set_data(params.gate_proj.t().to(torch::kFloat));
        state_dict["up_proj.weight"].set_data(params.up_proj.t().to(torch::kFloat));
        state_dict["down_proj.weight"].set_data(params.down_proj.t().to(torch::kFloat));
        return FfnLayerTestOutput({mlp->forward(params.input.to(torch::kFloat))});

    }

    void FfnOpTest(size_t token_num,
                   size_t hidden_size,
                   size_t inter_size,
                   ActivationType act,
                   DataType type)
    {
        auto input = PrepareFfnLayerInput(token_num, hidden_size, inter_size, type);
        auto result = FfnOpRun(input, act);
        auto result_ref = FfnTorchRefRun(input, act);
        assertTensorClose(result.out.to(result_ref.out.scalar_type()), result_ref.out);
    }
};

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

class MoELayerTest: public DeviceTestBase {
public:
    struct MoELayerTestInput {
        torch::Tensor input;
        torch::Tensor gating;
        torch::Tensor gate;
        torch::Tensor up;
        torch::Tensor down;
    };

    struct MoELayerTestOutput {
        torch::Tensor out;
    };

    MoELayerTestInput
    PrepareMoELayerInput(size_t token_num, size_t tok_dim, size_t inter_size, size_t expertNum, DataType type) {
        auto dtype  = dataTypeToTorchType(type);
        auto input  = torch::randn({(int)token_num, (int)tok_dim}, torch::Device(torch::kCPU)).to(dtype);
        auto gating = torch::randn({(int)tok_dim, (int)expertNum}, torch::Device(torch::kCPU)).to(dtype);
        auto gate = torch::randn({(int)expertNum, (int)tok_dim, (int)inter_size}, torch::Device(torch::kCPU)).to(dtype);
        auto up   = torch::randn({(int)expertNum, (int)tok_dim, (int)inter_size}, torch::Device(torch::kCPU)).to(dtype);
        auto down = torch::randn({(int)expertNum, (int)inter_size, (int)tok_dim}, torch::Device(torch::kCPU)).to(dtype);

        return MoELayerTestInput({input, gating, gate, up, down});
    }

    MoELayerTestOutput MoELayerRun(MoELayerTestInput& params, size_t expertNum, size_t topK, ActivationType Atype) {
        bool is_cpu     = (this->device_->getDeviceProperties().type == DeviceType::Cpu);
        auto alloc_type = is_cpu ? AllocationType::HOST : AllocationType::DEVICE;
        auto input      = tensorToBuffer(params.input, alloc_type);
        auto gating     = tensorToBuffer(params.gating, alloc_type);
        auto gate       = tensorToBuffer(params.gate, alloc_type);
        auto up         = tensorToBuffer(params.up, alloc_type);
        auto down       = tensorToBuffer(params.down, alloc_type);

        FfnLayerWeights weights;
        weights.moe_gating_weight = std::make_unique<const DenseWeights>(DenseWeights(gating));
        weights.moe_up_weight     = std::make_unique<const DenseWeights>(DenseWeights(up));
        weights.moe_down_weight   = std::make_unique<const DenseWeights>(DenseWeights(down));
        weights.moe_gate_weight   = std::make_unique<const DenseWeights>(DenseWeights(gate));

        MoeConfigs     moe_configs({expertNum, topK});
        FfnConfigs     ffn_configs({Atype, moe_configs});
        FfnLayerParams Opparams(*input, ffn_configs, weights);

        auto output = this->device_->ffnLayer(Opparams);
        return MoELayerTestOutput({bufferToTensor(*(output.hidden_states))});
    }

    MoELayerTestOutput MoETorchRefRun(MoELayerTestInput& params, size_t expertNum, size_t topK, ActivationType Atype) {
        MoE moe(params.gate.sizes()[1], params.gate.sizes()[2], params.gate.sizes()[0], topK, Atype);
        moe.ptr()->to(torch::Device(torch::kCPU));
        auto state_dict = moe.ptr()->named_parameters();

        torch::NoGradGuard no_grad;
        state_dict["gating_network.gating.weight"].set_data(params.gating.t().to(torch::kFloat));
        for (size_t i = 0; i < expertNum; i++) {
            auto expertStr = "expert" + std::to_string(i);
            state_dict[expertStr + ".gate.weight"].set_data(params.gate[i].t().to(torch::kFloat));
            state_dict[expertStr + ".up.weight"].set_data(params.up[i].t().to(torch::kFloat));
            state_dict[expertStr + ".down.weight"].set_data(params.down[i].t().to(torch::kFloat));
        }

        return MoELayerTestOutput({moe->forward(params.input.to(torch::kFloat))});
    }

    void MoEOpTest(size_t         token_num,
                   size_t         tok_dim,
                   size_t         inter_size,
                   size_t         expertNum,
                   size_t         topK,
                   ActivationType act,
                   DataType       type) {
        auto input      = PrepareMoELayerInput(token_num, tok_dim, inter_size, expertNum, type);
        auto result     = MoELayerRun(input, expertNum, topK, act);
        auto result_ref = MoETorchRefRun(input, expertNum, topK, act);
        assertTensorClose(result.out.to(result_ref.out.type()), result_ref.out, 1e2, 1e2);
    }
};
