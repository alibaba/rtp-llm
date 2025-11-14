#pragma once
#include "rtp_llm/cpp/devices/torch_impl/FfnLayer.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <torch/torch.h>
#include <functional>
#include <iostream>

using namespace rtp_llm;

class FfnLayerTest: public DeviceTestBase {
public:
    struct FfnLayerTestInput {
        torch::Tensor input;
        torch::Tensor gate_proj;
        torch::Tensor up_proj;
        torch::Tensor down_proj;
        torch::Tensor gate_up_proj;
    };

    struct FfnLayerTestOutput {
        torch::Tensor out;
    };

    FfnLayerTestInput
    PrepareFfnLayerInput(size_t token_num, size_t hidden_size, size_t inter_size, DataType type, ActivationType act) {
        auto dtype     = dataTypeToTorchType(type);
        auto input     = 0.01 * torch::rand({(int)token_num, (int)hidden_size}, torch::Device(torch::kCPU)).to(dtype);
        auto gate_proj = 0.01 * torch::rand({(int)hidden_size, (int)inter_size}, torch::Device(torch::kCPU)).to(dtype);
        auto up_proj   = 0.01 * torch::rand({(int)hidden_size, (int)inter_size}, torch::Device(torch::kCPU)).to(dtype);
        auto down_proj = 0.01 * torch::rand({(int)inter_size, (int)hidden_size}, torch::Device(torch::kCPU)).to(dtype);
        if (isGatedActivation(act)) {
            auto gate_up_proj = torch::cat({gate_proj, up_proj}, 1);
            return FfnLayerTestInput({input, gate_proj, up_proj, down_proj, gate_up_proj});
        } else {
            return FfnLayerTestInput({input, gate_proj, up_proj, down_proj});
        }
    }

    FfnLayerTestOutput FfnOpRun(FfnLayerTestInput& params, ActivationType Atype) {
        bool is_cpu     = (this->device_->getDeviceProperties().type == DeviceType::Cpu);
        auto alloc_type = is_cpu ? AllocationType::HOST : AllocationType::DEVICE;
        auto input      = tensorToBuffer(params.input, alloc_type);
        auto gate_proj  = tensorToBuffer(params.gate_proj, alloc_type);
        auto up_proj    = tensorToBuffer(params.up_proj, alloc_type);
        auto down_proj  = tensorToBuffer(params.down_proj, alloc_type);

        FfnLayerWeights weights;
        weights.up_weight   = std::make_unique<const DenseWeights>(DenseWeights(up_proj));
        weights.down_weight = std::make_unique<const DenseWeights>(DenseWeights(down_proj));
        weights.gate_weight = std::make_unique<const DenseWeights>(DenseWeights(gate_proj));
        if (isGatedActivation(Atype)) {
            auto gate_up_proj      = tensorToBuffer(params.gate_up_proj, alloc_type);
            weights.gate_up_weight = std::make_unique<const DenseWeights>(DenseWeights(gate_up_proj));
        }
        FfnConfigs     ffn_configs({Atype});
        FfnLayerParams Opparams(*input, ffn_configs, weights);

        auto output = this->device_->ffnLayer(Opparams);
        return FfnLayerTestOutput({bufferToTensor(*(output.hidden_states))});
    }

    FfnLayerTestOutput FfnTorchRefRun(FfnLayerTestInput& params, ActivationType Atype) {
        torch_impl::FfnLayer mlp(params.input.sizes()[1], params.gate_proj.sizes()[0], Atype);
        mlp.ptr()->to(torch::Device(torch::kCPU));
        auto               state_dict = mlp.ptr()->named_parameters();
        torch::NoGradGuard no_grad;
        state_dict["gate_proj.f.weight"].set_data(params.gate_proj.t().to(torch::kFloat));
        state_dict["up_proj.f.weight"].set_data(params.up_proj.t().to(torch::kFloat));
        state_dict["down_proj.f.weight"].set_data(params.down_proj.t().to(torch::kFloat));
        return FfnLayerTestOutput({mlp->forward(params.input.to(torch::kFloat))});
    }

    void FfnOpTest(size_t token_num, size_t hidden_size, size_t inter_size, ActivationType act, DataType type) {
        auto input      = PrepareFfnLayerInput(token_num, hidden_size, inter_size, type, act);
        auto result     = FfnOpRun(input, act);
        auto result_ref = FfnTorchRefRun(input, act);
        assertTensorClose(result.out.to(result_ref.out.scalar_type()), result_ref.out);
    }

    struct FfnLoraLayerTestInput {
        FfnLayerTestInput          input;
        std::vector<int>           input_lengths;
        std::vector<torch::Tensor> gate_lora_a;
        std::vector<torch::Tensor> gate_lora_b;
        std::vector<torch::Tensor> up_lora_a;
        std::vector<torch::Tensor> up_lora_b;
        std::vector<torch::Tensor> down_lora_a;
        std::vector<torch::Tensor> down_lora_b;
        std::vector<torch::Tensor> gate_up_lora_a;
        std::vector<torch::Tensor> gate_up_lora_b;
    };

    FfnLoraLayerTestInput PrepareFfnLayerLoraInput(std::vector<int> input_lengths,
                                                   std::vector<int> gate_ranks,
                                                   std::vector<int> up_ranks,
                                                   std::vector<int> down_ranks,
                                                   int              hidden_size,
                                                   int              inter_size,
                                                   DataType         input_type,
                                                   DataType         lora_type,
                                                   ActivationType   act_type) {
        auto input_tensor_options =
            torch::TensorOptions(dataTypeToTorchType(input_type)).device(torch::Device(torch::kCPU));
        auto lora_tensor_options =
            torch::TensorOptions(dataTypeToTorchType(lora_type)).device(torch::Device(torch::kCPU));
        auto token_num  = std::accumulate(input_lengths.begin(), input_lengths.end(), 0);
        auto batch_size = input_lengths.size();
        auto input      = 0.01 * torch::rand({(int)token_num, (int)hidden_size}, input_tensor_options);
        auto gate_proj  = 0.01 * torch::rand({(int)hidden_size, (int)inter_size}, input_tensor_options);
        auto up_proj    = 0.01 * torch::rand({(int)hidden_size, (int)inter_size}, input_tensor_options);
        auto down_proj  = 0.01 * torch::rand({(int)inter_size, (int)hidden_size}, input_tensor_options);
        if (isGatedActivation(act_type)) {
            auto                       gate_up_proj = torch::cat({gate_proj, up_proj}, 1);
            std::vector<torch::Tensor> gate_lora_a(batch_size);
            std::vector<torch::Tensor> gate_lora_b(batch_size);
            std::vector<torch::Tensor> up_lora_a(batch_size);
            std::vector<torch::Tensor> up_lora_b(batch_size);
            std::vector<torch::Tensor> down_lora_a(batch_size);
            std::vector<torch::Tensor> down_lora_b(batch_size);
            std::vector<torch::Tensor> gate_up_lora_a(batch_size);
            std::vector<torch::Tensor> gate_up_lora_b(batch_size);
            for (int i = 0; i < batch_size; i++) {
                gate_lora_a[i]    = torch::rand({hidden_size, gate_ranks[i]}, lora_tensor_options);
                gate_lora_b[i]    = torch::rand({gate_ranks[i], inter_size}, lora_tensor_options);
                up_lora_a[i]      = torch::rand({hidden_size, up_ranks[i]}, lora_tensor_options);
                up_lora_b[i]      = torch::rand({up_ranks[i], inter_size}, lora_tensor_options);
                down_lora_a[i]    = torch::rand({inter_size, down_ranks[i]}, lora_tensor_options);
                down_lora_b[i]    = torch::rand({down_ranks[i], hidden_size}, lora_tensor_options);
                gate_up_lora_a[i] = torch::cat({gate_lora_a[i], up_lora_a[i]}, 1);
                auto zeros1       = torch::zeros_like(up_lora_b[i]);
                auto zeros2       = torch::zeros_like(gate_lora_b[i]);
                auto column1      = torch::cat({gate_lora_b[i], zeros1}, 0);
                auto column2      = torch::cat({zeros2, up_lora_b[i]}, 0);
                gate_up_lora_b[i] = torch::cat({column1, column2}, 1);
            }
            return FfnLoraLayerTestInput({FfnLayerTestInput{input, gate_proj, up_proj, down_proj, gate_up_proj},
                                          input_lengths,
                                          gate_lora_a,
                                          gate_lora_b,
                                          up_lora_a,
                                          up_lora_b,
                                          down_lora_a,
                                          down_lora_b,
                                          gate_up_lora_a,
                                          gate_up_lora_b});
        } else {
            std::vector<torch::Tensor> gate_lora_a(batch_size);
            std::vector<torch::Tensor> gate_lora_b(batch_size);
            std::vector<torch::Tensor> up_lora_a(batch_size);
            std::vector<torch::Tensor> up_lora_b(batch_size);
            std::vector<torch::Tensor> down_lora_a(batch_size);
            std::vector<torch::Tensor> down_lora_b(batch_size);
            for (int i = 0; i < batch_size; i++) {
                gate_lora_a[i] = torch::rand({hidden_size, gate_ranks[i]}, lora_tensor_options);
                gate_lora_b[i] = torch::rand({gate_ranks[i], inter_size}, lora_tensor_options);
                up_lora_a[i]   = torch::rand({hidden_size, up_ranks[i]}, lora_tensor_options);
                up_lora_b[i]   = torch::rand({up_ranks[i], inter_size}, lora_tensor_options);
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
    }

    FfnLayerTestOutput FfnLayerLoraOpRun(FfnLoraLayerTestInput& params, ActivationType Atype) {
        bool is_cpu     = (this->device_->getDeviceProperties().type == DeviceType::Cpu);
        auto alloc_type = is_cpu ? AllocationType::HOST : AllocationType::DEVICE;
        auto input      = tensorToBuffer(params.input.input, alloc_type);
        auto gate_proj  = tensorToBuffer(params.input.gate_proj, alloc_type);
        auto up_proj    = tensorToBuffer(params.input.up_proj, alloc_type);
        auto down_proj  = tensorToBuffer(params.input.down_proj, alloc_type);

        FfnLayerWeights weights;
        weights.up_weight   = std::make_unique<const DenseWeights>(DenseWeights(up_proj));
        weights.down_weight = std::make_unique<const DenseWeights>(DenseWeights(down_proj));
        weights.gate_weight = std::make_unique<const DenseWeights>(DenseWeights(gate_proj));

        if (isGatedActivation(Atype)) {
            auto gate_up_proj      = tensorToBuffer(params.input.gate_up_proj, alloc_type);
            weights.gate_up_weight = std::make_unique<const DenseWeights>(DenseWeights(gate_up_proj));
        }

        FfnConfigs     ffn_configs({Atype});
        FfnLayerParams Opparams(*input, ffn_configs, weights);

        // lora
        auto lora_input_lengths =
            createHostBuffer<int32_t>({(size_t)params.input_lengths.size()}, params.input_lengths.data());
        std::vector<ConstBufferPtr> gate_lora_as;
        std::vector<ConstBufferPtr> gate_lora_bs;
        std::vector<ConstBufferPtr> up_lora_as;
        std::vector<ConstBufferPtr> up_lora_bs;
        std::vector<ConstBufferPtr> gate_up_lora_as;
        std::vector<ConstBufferPtr> gate_up_lora_bs;
        std::vector<ConstBufferPtr> down_lora_as;
        std::vector<ConstBufferPtr> down_lora_bs;
        for (int i = 0; i < params.input_lengths.size(); i++) {
            gate_lora_as.push_back(tensorToBuffer(params.gate_lora_a[i]));
            gate_lora_bs.push_back(tensorToBuffer(params.gate_lora_b[i]));
            up_lora_as.push_back(tensorToBuffer(params.up_lora_a[i]));
            up_lora_bs.push_back(tensorToBuffer(params.up_lora_b[i]));
            down_lora_as.push_back(tensorToBuffer(params.down_lora_a[i]));
            down_lora_bs.push_back(tensorToBuffer(params.down_lora_b[i]));
            if (isGatedActivation(Atype)) {
                gate_up_lora_as.push_back(tensorToBuffer(params.gate_up_lora_a[i]));
                gate_up_lora_bs.push_back(tensorToBuffer(params.gate_up_lora_b[i]));
            }
        }
        if (isGatedActivation(Atype)) {
            Opparams.lora_input = lora::FfnLayerLoraInput(
                {std::make_shared<lora::LoraOpInput>(lora_input_lengths, gate_up_lora_as, gate_up_lora_bs),
                 std::make_shared<lora::LoraOpInput>(lora_input_lengths, gate_up_lora_as, gate_up_lora_bs),
                 std::make_shared<lora::LoraOpInput>(lora_input_lengths, down_lora_as, down_lora_bs)});
        } else {
            Opparams.lora_input = lora::FfnLayerLoraInput(
                {std::make_shared<lora::LoraOpInput>(lora_input_lengths, gate_lora_as, gate_lora_bs),
                 std::make_shared<lora::LoraOpInput>(lora_input_lengths, up_lora_as, up_lora_bs),
                 std::make_shared<lora::LoraOpInput>(lora_input_lengths, down_lora_as, down_lora_bs)});
        }

        auto output = this->device_->ffnLayer(Opparams);
        return FfnLayerTestOutput({bufferToTensor(*(output.hidden_states))});
    }

    FfnLayerTestOutput FfnLayerLoraTorchRefRun(FfnLoraLayerTestInput& params, ActivationType Atype) {
        torch_impl::FfnLayer mlp(params.input.input.sizes()[1], params.input.gate_proj.sizes()[0], Atype);
        mlp.ptr()->to(torch::Device(torch::kCPU));
        auto               state_dict = mlp.ptr()->named_parameters();
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
                          size_t           hidden_size,
                          size_t           inter_size,
                          DataType         input_type,
                          DataType         lora_type,
                          ActivationType   act) {
        auto input = PrepareFfnLayerLoraInput(
            input_lengths, gate_ranks, up_ranks, down_ranks, hidden_size, inter_size, input_type, lora_type, act);
        auto result     = FfnLayerLoraOpRun(input, act);
        auto result_ref = FfnLayerLoraTorchRefRun(input, act);
        assertTensorClose(result.out.to(result_ref.out.scalar_type()), result_ref.out);
    }
};

class MoEGateSelectTest: public DeviceTestBase {
public:
    using Bencher = std::function<void(const char*, const std::function<void()>&)>;

    struct MoEGateSelectTestInput {
        at::Tensor                input;
        at::Tensor                moe_gating_weight;
        std::optional<at::Tensor> e_score_correction_bias;
    };

    struct MoEGateSelectTestOutput {
        at::Tensor expert_ids;
        at::Tensor expert_scales;
    };

    MoEGateSelectTestInput
    prepareMoEGateSelectInput(size_t token_num, size_t hidden_dim, size_t expert_num, bool has_bias) {
        auto input =
            torch::rand({(int64_t)token_num, (int64_t)hidden_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
            * 1000.0 / (float)hidden_dim;
        auto moe_gating_weight =
            torch::rand({(int64_t)hidden_dim, (int64_t)expert_num}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        std::optional<at::Tensor> e_score_correction_bias = std::nullopt;
        if (has_bias) {
            e_score_correction_bias = std::make_optional(torch::rand(
                {(int64_t)token_num, (int64_t)expert_num}, torch::dtype(torch::kFloat32).device(torch::kCUDA)));
        }

        return {input, moe_gating_weight, e_score_correction_bias};
    }

    MoEGateSelectTestOutput MoEGateSelectRun(const MoEGateSelectTestInput& params,
                                             size_t                        topk,
                                             size_t                        group_num,
                                             size_t                        group_topk,
                                             int                           scoring_func,
                                             bool                          has_moe_norm,
                                             const Bencher*                bencher   = nullptr,
                                             const char*                   case_name = nullptr) {
        case_name = case_name != nullptr ? case_name : "unnamed case";

        size_t expert_num = params.moe_gating_weight.size(1);

        auto input_buf = torchTensor2Buffer(params.input);

        MoeConfigs moe_configs{expert_num, 0, topk};
        moe_configs.has_moe_norm = has_moe_norm;
        moe_configs.scoring_func = scoring_func;
        if (params.e_score_correction_bias.has_value()) {
            moe_configs.n_group    = group_num;
            moe_configs.topk_group = group_topk;
        }

        FfnConfigs ffn_configs{
            ActivationType::Swiglu,
            std::make_optional(moe_configs),
        };

        BufferPtr moe_gating_weight_buf = torchTensor2Buffer(params.moe_gating_weight);

        FfnLayerWeights weights;
        weights.moe_gating_weight = std::make_unique<const DenseWeights>(DenseWeights(moe_gating_weight_buf));
        if (params.e_score_correction_bias.has_value()) {
            weights.e_score_correction_bias = torchTensor2Buffer(params.e_score_correction_bias.value());
        }

        FfnLayerParams ffn_params(*input_buf, ffn_configs, weights);

        auto result = device_->moeGateSelect(ffn_params);

        if (bencher != nullptr) {
            (*bencher)(case_name, [&, this] { this->device_->moeGateSelect(ffn_params); });
        }

        return {
            Buffer2torchTensor(*result.expert_ids, false).clone(),
            Buffer2torchTensor(*result.expert_scales, false).clone(),
        };
    }

    MoEGateSelectTestOutput MoEGateSelectRefRun(const MoEGateSelectTestInput& params,
                                                size_t                        topk,
                                                size_t                        group_num,
                                                size_t                        group_topk,
                                                int                           scoring_func,
                                                bool                          has_moe_norm) {
        size_t token_num  = params.input.size(0);
        size_t expert_num = params.moe_gating_weight.size(1);

        auto gate = params.input.matmul(params.moe_gating_weight);

        if (scoring_func == 1) {
            gate.sigmoid_();
        }

        at::Tensor gate_with_bias;
        if (params.e_score_correction_bias.has_value()) {
            const auto& e_score_correction_bias = params.e_score_correction_bias.value();

            auto scores_for_choice = gate.add(e_score_correction_bias);
            auto reshaped_scores   = scores_for_choice.view({(int)token_num, (int)group_num, -1});
            auto topk_result       = reshaped_scores.topk(2, /*dim=*/-1);
            auto group_scores      = std::get<0>(topk_result).sum(-1);
            auto group_topk_result = group_scores.topk(
                /*k=*/group_topk,
                /*dim=*/-1,
                /*largest=*/true,
                /*sorted=*/false);
            auto group_idx  = std::get<1>(group_topk_result);
            auto group_mask = torch::zeros_like(group_scores, torch::kCUDA);
            group_mask.scatter_(
                /*dim=*/1,
                /*index=*/group_idx,
                /*src=*/1.0f);
            int64_t experts_per_group = expert_num / group_num;
            auto    score_mask        = group_mask.unsqueeze(-1)
                                  .expand({(int)token_num, (int)group_num, experts_per_group})
                                  .reshape({(int)token_num, -1});

            gate_with_bias = scores_for_choice.masked_fill(torch::logical_not(score_mask.to(torch::kBool)), 0.0);
        } else {
            gate           = gate.softmax(-1);
            gate_with_bias = gate;
        }

        auto selected_result = gate_with_bias.topk(
            /*k=*/topk,
            /*dim=*/-1,
            /*largest=*/true,
            /*sorted=*/false);

        auto expert_ids   = std::get<1>(selected_result);
        auto expert_scale = gate.gather(-1, expert_ids);

        if (has_moe_norm) {
            auto expert_scale_sum = (expert_scale.sum(-1) + 1e-20).unsqueeze(-1).expand({(int)token_num, (int)topk});
            expert_scale.div_(expert_scale_sum);
        }

        return {expert_ids, expert_scale};
    }

    MoEGateSelectTestOutput processOutput(const MoEGateSelectTestOutput& output, bool mask_lowest = false) {
        auto expert_ids    = output.expert_ids;
        auto expert_scales = output.expert_scales;

        // id as secondary sorting key
        const auto id_order = expert_ids.argsort(false, -1);
        expert_ids          = expert_ids.gather(-1, id_order);
        expert_scales       = expert_scales.gather(-1, id_order);

        // scale as primary sorting key
        const auto scale_order = expert_scales.argsort(true, -1);  // stable sort, preserving id order
        expert_ids             = expert_ids.gather(-1, scale_order);
        expert_scales          = expert_scales.gather(-1, scale_order);

        // mask expert ids of the lowest scale to avoid ambiguity
        if (mask_lowest) {
            const auto min_scales = std::get<0>(torch::min(expert_scales, -1, true)).expand(expert_scales.sizes());
            expert_ids            = expert_ids.masked_fill(expert_scales.eq(min_scales), -1);
        }

        return {expert_ids, expert_scales};
    }

    void assertEquivalentOutput(const MoEGateSelectTestOutput& out_a, const MoEGateSelectTestOutput& out_b) {
        auto a = processOutput(out_a, true);
        auto b = processOutput(out_b, true);

        assertTensorClose(a.expert_scales, b.expert_scales, 1e-5, 1e-8);
        assertTensorClose(a.expert_ids, b.expert_ids, 1e-5, 1e-8);
    }

    void MoEGateSelTest(size_t         token_num,
                        size_t         hidden_dim,
                        size_t         expert_num,
                        size_t         topk,
                        bool           has_bias,
                        size_t         group_num,
                        size_t         group_topk,
                        int            scoring_func,
                        bool           has_moe_norm,
                        const Bencher* bencher = nullptr) {
        std::stringstream case_name_ss;
        case_name_ss << "token_num=" << token_num
                     << ", "
                        "hidden_dim="
                     << hidden_dim
                     << ", "
                        "expert_num="
                     << expert_num
                     << ", "
                        "topk="
                     << topk
                     << ", "
                        "has_bias="
                     << has_bias
                     << ", "
                        "group_num="
                     << group_num
                     << ", "
                        "group_topk="
                     << group_topk
                     << ", "
                        "scoring_func="
                     << scoring_func
                     << ", "
                        "has_moe_norm="
                     << has_moe_norm;
        auto case_name = case_name_ss.str();
        std::cout << "-------------------- " << case_name << " --------------------\n";

        auto params  = prepareMoEGateSelectInput(token_num, hidden_dim, expert_num, has_bias);
        auto ref_res = MoEGateSelectRefRun(params, topk, group_num, group_topk, scoring_func, has_moe_norm);
        auto res     = MoEGateSelectRun(
            params, topk, group_num, group_topk, scoring_func, has_moe_norm, bencher, case_name.c_str());

        assertEquivalentOutput(ref_res, res);
    }
};

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

    std::pair<torch::Tensor, torch::Tensor> QuantizeFP8(const torch::Tensor& x) {
        int64_t       m            = x.size(0);
        int64_t       n            = x.size(1);
        int64_t       padded_m     = m;
        int64_t       padded_n     = n;
        torch::Tensor x_padded     = x;
        int64_t       num_blocks_m = padded_m / 128;
        int64_t       num_blocks_n = padded_n / 128;
        auto          x_view       = x_padded.view({num_blocks_m, 128, num_blocks_n, 128});
        // 计算每个块的最大绝对值
        auto x_abs  = x_view.abs().to(torch::kFloat32);
        auto x_amax = torch::amax(x_abs, /*dim=*/{1, 3}, /*keepdim=*/true).clamp_min_(1e-4f);

        // 缩放并转换为FP8
        auto x_scaled = (x_view * (448.0 / x_amax)).to(torch::kFloat8_e4m3fn);

        // 重塑结果并切片回原始尺寸
        auto output = x_scaled.view({padded_m, padded_n}).contiguous();

        // 计算缩放因子并调整形状
        auto scale_factors = (x_amax / 448.0).view({num_blocks_m, num_blocks_n});

        return {output, scale_factors};
    }

    BufferPtr QuantizeFP8Weights(const torch::Tensor& w) {
        std::vector<size_t> w_shape;
        for (auto& i : w.sizes()) {
            w_shape.push_back(i);
        }
        assert(w_shape.size() == 3);
        BufferPtr w_fp8   = this->device_->allocateBuffer({DataType::TYPE_FP8_E4M3, w_shape}, {"fc1_w_fp8"});
        BufferPtr w_scale = this->device_->allocateBuffer(
            {DataType::TYPE_FP32, {w_shape[0], w_shape[1] / 128, w_shape[2] / 128}}, {"fc1_w_scale"});
        torch::Tensor w_fp8_t   = Buffer2torchTensor(w_fp8, false);
        torch::Tensor w_scale_t = Buffer2torchTensor(w_scale, false);
        for (int i = 0; i < w_shape[0]; ++i) {
            auto res     = QuantizeFP8(w[i]);
            w_fp8_t[i]   = res.first;
            w_scale_t[i] = res.second;
        }
        auto zeros_type = w_scale->where();
        return BufferPtr(
            new QBuffer(std::move(w_fp8),
                        std::move(w_scale),
                        std::move(BufferPtr(new Buffer(zeros_type, DataType::TYPE_INVALID, {0}, nullptr)))));
    }

    MoELayerTestInput
    PrepareMoELayerInput(size_t token_num, size_t tok_dim, size_t inter_size, size_t expertNum, DataType type) {
        if (type == DataType::TYPE_FP8_E4M3) {
            type = DataType::TYPE_BF16;
        }
        auto dtype  = dataTypeToTorchType(type);
        auto input  = torch::randn({(int)token_num, (int)tok_dim}, torch::Device(torch::kCPU)).to(dtype);
        auto gating = torch::randn({(int)tok_dim, (int)expertNum}, torch::Device(torch::kCPU)).to(dtype);
        auto gate = torch::randn({(int)expertNum, (int)tok_dim, (int)inter_size}, torch::Device(torch::kCPU)).to(dtype);
        auto up   = torch::randn({(int)expertNum, (int)tok_dim, (int)inter_size}, torch::Device(torch::kCPU)).to(dtype);
        auto down = torch::randn({(int)expertNum, (int)inter_size, (int)tok_dim}, torch::Device(torch::kCPU)).to(dtype);
        return MoELayerTestInput({input, gating, gate, up, down});
    }

    MoELayerTestOutput MoELayerRun(MoELayerTestInput& params,
                                   size_t             inter_size,
                                   size_t             expertNum,
                                   size_t             topK,
                                   ActivationType     Atype,
                                   DataType           type) {
        FfnLayerWeights weights;
        bool            is_cpu     = (this->device_->getDeviceProperties().type == DeviceType::Cpu);
        auto            alloc_type = is_cpu ? AllocationType::HOST : AllocationType::DEVICE;
        auto            input      = tensorToBuffer(params.input, alloc_type);
        auto            gating     = tensorToBuffer(params.gating, alloc_type);
        if (type == DataType::TYPE_FP8_E4M3) {
            torch::Tensor gate_t;
            if (isGatedActivation(Atype)) {
                gate_t = torch::cat({params.up.transpose(2, 1), params.gate.transpose(2, 1)}, 1).contiguous();
            } else {
                gate_t = params.up.transpose(2, 1).contiguous();
            }
            auto down_t               = params.down.transpose(2, 1).contiguous();
            auto down                 = QuantizeFP8Weights(down_t);
            auto gate                 = QuantizeFP8Weights(gate_t);
            weights.moe_gating_weight = std::make_unique<const DenseWeights>(DenseWeights(gating));
            weights.moe_down_weight   = std::make_unique<DenseWeights>(DenseWeights(down));
            weights.moe_gate_weight   = std::make_unique<DenseWeights>(DenseWeights(gate));
        } else {
            torch::Tensor gate_t;
            if (isGatedActivation(Atype)) {
                gate_t = torch::cat({params.up, params.gate}, 1).contiguous();
            } else {
                gate_t = params.up;
            }
            auto gate                 = tensorToBuffer(gate_t, alloc_type);
            auto down                 = tensorToBuffer(params.down, alloc_type);
            weights.moe_gating_weight = std::make_unique<const DenseWeights>(DenseWeights(gating));
            weights.moe_down_weight   = std::make_unique<DenseWeights>(DenseWeights(down));
            weights.moe_gate_weight   = std::make_unique<DenseWeights>(DenseWeights(gate));
        }
        MoeConfigs     moe_configs({expertNum, 0, topK});
        FfnConfigs     ffn_configs({Atype, moe_configs});
        FfnLayerParams Opparams(*input,
                                ffn_configs,
                                weights,
                                std::nullopt,
                                type == DataType::TYPE_FP8_E4M3 ? QScheme::Qfp8PerTokenBlock : QScheme::NoQuantize);
        auto           output = this->device_->ffnLayer(Opparams);
        return MoELayerTestOutput({bufferToTensor(*(output.hidden_states))});
    }

    MoELayerTestOutput MoETorchRefRun(MoELayerTestInput& params, size_t expertNum, size_t topK, ActivationType Atype) {
        torch_impl::MoE moe(params.gate.sizes()[1], params.gate.sizes()[2], params.gate.sizes()[0], topK, Atype);
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
                   DataType       type,
                   float          rel_diff = 1e2,
                   float          abs_diff = 1e2) {
        auto input      = PrepareMoELayerInput(token_num, tok_dim, inter_size, expertNum, type);
        auto result     = MoELayerRun(input, inter_size, expertNum, topK, act, type);
        auto result_ref = MoETorchRefRun(input, expertNum, topK, act);
        assertTensorClose(result.out.to(result_ref.out.type()), result_ref.out, rel_diff, abs_diff);
    }
};
