#pragma once
#include "src/fastertransformer/devices/testing/TestBase.h"
#include <torch/torch.h>



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

        FfnLayerWeights weights (
            std::make_unique<const DenseWeights>(
                DenseWeights(up_proj)),
            std::make_unique<const DenseWeights>(
                DenseWeights(gate_proj)),
            std::make_unique<const DenseWeights>(
                DenseWeights(down_proj))
        );

        FfnLayerParams Opparams(*input,
                                weights,
                                Atype);

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
        auto OpResult = FfnOpRun(input, act);
        auto RefResult = FfnTorchRefRun(input, act);
        assertTensorClose(OpResult.out.to(RefResult.out.type()), RefResult.out);
    }
    
    

};