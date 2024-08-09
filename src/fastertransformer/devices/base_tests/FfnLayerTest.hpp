#pragma once
#include "src/fastertransformer/devices/torch_impl/FfnLayer.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include <torch/torch.h>

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
        torch_impl::FfnLayer mlp(params.input.sizes()[1], params.gate_proj.sizes()[0], Atype);
        mlp.ptr()->to(torch::Device(torch::kCPU));
        auto state_dict = mlp.ptr()->named_parameters();
        torch::NoGradGuard no_grad;
        state_dict["gate_proj.f.weight"].set_data(params.gate_proj.t().to(torch::kFloat));
        state_dict["up_proj.f.weight"].set_data(params.up_proj.t().to(torch::kFloat));
        state_dict["down_proj.f.weight"].set_data(params.down_proj.t().to(torch::kFloat));
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

    struct FfnLoraLayerTestInput {
        FfnLayerTestInput input;
        std::vector<int> input_lengths;
        std::vector<torch::Tensor> gate_lora_a;
        std::vector<torch::Tensor> gate_lora_b;
        std::vector<torch::Tensor> up_lora_a;
        std::vector<torch::Tensor> up_lora_b;
        std::vector<torch::Tensor> down_lora_a;
        std::vector<torch::Tensor> down_lora_b;
    };

    FfnLoraLayerTestInput PrepareFfnLayerLoraInput(std::vector<int> input_lengths,
                                                    std::vector<int> gate_ranks,
                                                    std::vector<int> up_ranks,
                                                    std::vector<int> down_ranks,
                                                    int hidden_size,
                                                    int inter_size,
                                                    DataType input_type,
                                                    DataType lora_type)
    {
        auto input_tensor_options = torch::TensorOptions(dataTypeToTorchType(input_type)).device(torch::Device(torch::kCPU));
        auto lora_tensor_options = torch::TensorOptions(dataTypeToTorchType(lora_type)).device(torch::Device(torch::kCPU));
        auto token_num =  std::accumulate(input_lengths.begin(), input_lengths.end(), 0);
        auto batch_size = input_lengths.size();
        auto input = 0.01 * torch::rand({(int)token_num, (int)hidden_size}, input_tensor_options);
        auto gate_proj = 0.01 * torch::rand({(int)hidden_size, (int)inter_size}, input_tensor_options);
        auto up_proj = 0.01 * torch::rand({(int)hidden_size, (int)inter_size}, input_tensor_options);
        auto down_proj = 0.01 * torch::rand({(int)inter_size, (int)hidden_size}, input_tensor_options);
        std::vector<torch::Tensor> gate_lora_a(batch_size);
        std::vector<torch::Tensor> gate_lora_b(batch_size);
        std::vector<torch::Tensor> up_lora_a(batch_size);
        std::vector<torch::Tensor> up_lora_b(batch_size);
        std::vector<torch::Tensor> down_lora_a(batch_size);
        std::vector<torch::Tensor> down_lora_b(batch_size);
        for (int i = 0; i < batch_size; i++) {
            gate_lora_a[i] = torch::rand({hidden_size, gate_ranks[i]}, lora_tensor_options);
            gate_lora_b[i] = torch::rand({gate_ranks[i], inter_size}, lora_tensor_options);
            up_lora_a[i] = torch::rand({hidden_size, up_ranks[i]}, lora_tensor_options);
            up_lora_b[i] = torch::rand({up_ranks[i], inter_size}, lora_tensor_options);
            down_lora_a[i] = torch::rand({inter_size, down_ranks[i]}, lora_tensor_options);
            down_lora_b[i] = torch::rand({down_ranks[i], hidden_size}, lora_tensor_options);
        }
        return FfnLoraLayerTestInput({FfnLayerTestInput{input, gate_proj, up_proj, down_proj},
                                      input_lengths,
                                      gate_lora_a,
                                      gate_lora_b,
                                      up_lora_a,
                                      up_lora_b,
                                      down_lora_a,
                                      down_lora_b});
    }

    FfnLayerTestOutput FfnLayerLoraOpRun(FfnLoraLayerTestInput& params,
                                         ActivationType Atype)
    {
        bool is_cpu     = (this->device_->getDeviceProperties().type == DeviceType::Cpu);
        auto alloc_type = is_cpu ? AllocationType::HOST : AllocationType::DEVICE;
        auto input      = tensorToBuffer(params.input.input, alloc_type);
        auto gate_proj  = tensorToBuffer(params.input.gate_proj, alloc_type);
        auto up_proj    = tensorToBuffer(params.input.up_proj, alloc_type);
        auto down_proj  = tensorToBuffer(params.input.down_proj, alloc_type);

        FfnLayerWeights weights;
        weights.up_weight = std::make_unique<const DenseWeights>(DenseWeights(up_proj));
        weights.down_weight = std::make_unique<const DenseWeights>(DenseWeights(down_proj));
        weights.gate_weight = std::make_unique<const DenseWeights>(DenseWeights(gate_proj));

        FfnConfigs ffn_configs({Atype});
        FfnLayerParams Opparams(*input,
                                ffn_configs,
                                weights);

        // lora
        auto lora_input_lengths = createHostBuffer<int32_t>({(size_t)params.input_lengths.size()}, params.input_lengths.data());
        std::vector<ConstBufferPtr> gate_lora_as;
        std::vector<ConstBufferPtr> gate_lora_bs;
        std::vector<ConstBufferPtr> up_lora_as;
        std::vector<ConstBufferPtr> up_lora_bs;
        std::vector<ConstBufferPtr> down_lora_as;
        std::vector<ConstBufferPtr> down_lora_bs;
        for (int i = 0; i < params.input_lengths.size(); i++) {
            gate_lora_as.push_back(tensorToBuffer(params.gate_lora_a[i]));
            gate_lora_bs.push_back(tensorToBuffer(params.gate_lora_b[i]));
            up_lora_as.push_back(tensorToBuffer(params.up_lora_a[i]));
            up_lora_bs.push_back(tensorToBuffer(params.up_lora_b[i]));
            down_lora_as.push_back(tensorToBuffer(params.down_lora_a[i]));
            down_lora_bs.push_back(tensorToBuffer(params.down_lora_b[i]));
        }
        Opparams.lora_input = lora::FfnLayerLoraInput({
            std::make_shared<lora::LoraOpInput>(lora_input_lengths, gate_lora_as, gate_lora_bs),
            std::make_shared<lora::LoraOpInput>(lora_input_lengths, up_lora_as, up_lora_bs),
            std::make_shared<lora::LoraOpInput>(lora_input_lengths, down_lora_as, down_lora_bs)
        });

        auto output  = this->device_->ffnLayer(Opparams);
        return FfnLayerTestOutput({bufferToTensor(*(output.hidden_states))});
    }

    FfnLayerTestOutput FfnLayerLoraTorchRefRun(FfnLoraLayerTestInput& params,
                                               ActivationType Atype)
    {
        torch_impl::FfnLayer mlp(params.input.input.sizes()[1], params.input.gate_proj.sizes()[0], Atype);
        mlp.ptr()->to(torch::Device(torch::kCPU));
        auto state_dict = mlp.ptr()->named_parameters();
        torch::NoGradGuard no_grad;
        state_dict["gate_proj.f.weight"].set_data(params.input.gate_proj.t().to(torch::kFloat));
        state_dict["up_proj.f.weight"].set_data(params.input.up_proj.t().to(torch::kFloat));
        state_dict["down_proj.f.weight"].set_data(params.input.down_proj.t().to(torch::kFloat));
        return FfnLayerTestOutput({mlp->forwardLora(params.input.input.to(torch::kFloat),
                                                    params.input_lengths,
                                                    params.gate_lora_a,
                                                    params.gate_lora_b,
                                                    params.up_lora_a,
                                                    params.up_lora_b,
                                                    params.down_lora_a,
                                                    params.down_lora_b)});
    }

    void FfnLayerLoraTest(std::vector<int> input_lengths,
                          std::vector<int> gate_ranks,
                          std::vector<int> up_ranks,
                          std::vector<int> down_ranks,
                          size_t hidden_size,
                          size_t inter_size,
                          DataType input_type,
                          DataType lora_type,
                          ActivationType act)
    {
        auto input = PrepareFfnLayerLoraInput(input_lengths,
                                              gate_ranks,
                                              up_ranks,
                                              down_ranks,
                                              hidden_size,
                                              inter_size,
                                              input_type,
                                              lora_type);
        auto result = FfnLayerLoraOpRun(input, act);
        auto result_ref = FfnLayerLoraTorchRefRun(input, act);
        assertTensorClose(result.out.to(result_ref.out.scalar_type()), result_ref.out);
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

    MoELayerTestOutput MoELayerRun(MoELayerTestInput& params, size_t inter_size, size_t expertNum, size_t topK, ActivationType Atype) {
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

        MoeConfigs     moe_configs({expertNum, topK, false, (int64_t)inter_size, false});
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
        auto result     = MoELayerRun(input, inter_size, expertNum, topK, act);
        auto result_ref = MoETorchRefRun(input, expertNum, topK, act);
        assertTensorClose(result.out.to(result_ref.out.type()), result_ref.out, 1e2, 1e2);
    }
};
