#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <cstddef>
#include <numeric>
#include <optional>

using namespace std;

namespace rtp_llm {

FfnLayerOutput DeviceBase::ffnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(!params.residual, "default FFN implementation does not support residual!");
    BufferPtr output;
    BufferPtr moe_gating;
    if (params.weights.moe_gating_weight) {
        RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
        auto moe_output = moeFfnLayer(params);
        output          = moe_output.hidden_states;
        moe_gating      = moe_output.moe_gating;

        auto shared_expert_output = moeSharedExpert(params).hidden_states;

        // for deep ep ll, the gather should be defered afater shared expert.
        if (moe_output.moe_combine_output) {
            moe_output.comm_barrier_hook->hook_sync();
            moe_output = gatherCombineOutput(moe_output.moe_combine_output.value());
            output     = moe_output.hidden_states;
        }

        printBufferData(*output, "moe_out_after_barrier");
        if (shared_expert_output) {
            // just add bias to output
            layernorm({output, nullptr, nullopt, mayGetRef(shared_expert_output)}).output;
        }
    } else {
        BufferPtr up_output;
        if (isGatedActivation(params.configs.activation_type)) {
            BufferPtr ffn_input_ptr = nullptr;
            RTP_LLM_LOG_DEBUG("enable_sp %d ffn_tp_size %d", params.enable_sp, init_params_.ffn_tp_size);
            if (params.enable_sp && init_params_.ffn_tp_size > 1) {
                BufferPtr ag_recv_buffer = nullptr;
                size_t    pad_token_num  = params.input.shape()[0] * init_params_.ffn_tp_size;
                if (params.qscheme == NoQuantize) {
                    ffn_input_ptr  = params.input.slice(0, params.input.shape()[0]);
                    ag_recv_buffer = allocateBuffer({ffn_input_ptr->type(), {pad_token_num, ffn_input_ptr->shape()[1]}},
                                                    {"ag_recv_buffer"});
                } else if (params.qscheme == Qint8PerToken) {
                    ffn_input_ptr = reinterpret_cast<const QBuffer&>(params.input).qslice(0, params.input.shape()[0]);
                    BufferPtr kernel = allocateBuffer(
                        {ffn_input_ptr->type(), {pad_token_num, ffn_input_ptr->shape()[1]}}, {"ag_recv_buffer"});
                    BufferPtr scales = allocateBuffer({DataType::TYPE_FP32, {pad_token_num}, AllocationType::DEVICE},
                                                      {"ag_recv_buffer_scale"});
                    ag_recv_buffer =
                        BufferPtr(new QBuffer(std::move(kernel),
                                              std::move(scales),
                                              std::move(BufferPtr(new Buffer(
                                                  MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
                } else if (params.qscheme == Qfp8PerTensor) {
                    ffn_input_ptr =
                        reinterpret_cast<const QBuffer&>(params.input).qslicePerTensor(0, params.input.shape()[0]);
                    BufferPtr kernel = allocateBuffer(
                        {ffn_input_ptr->type(), {pad_token_num, ffn_input_ptr->shape()[1]}}, {"ag_recv_buffer"});
                    BufferPtr scales = reinterpret_cast<const QBuffer&>(params.input).scalesPtr();
                    ag_recv_buffer =
                        BufferPtr(new QBuffer(std::move(kernel),
                                              std::move(scales),
                                              std::move(BufferPtr(new Buffer(
                                                  MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
                } else {
                    throw OpException(
                        {OpErrorType::ERROR_UNIMPLEMENTED, "allGatherloraLinear qscheme type not supported"});
                }
                printBufferData(*ffn_input_ptr, "ffn_ag_input");

                GemmParams up_gemm_params(*ag_recv_buffer,
                                          *(params.weights.gate_up_weight->kernel),
                                          std::nullopt,
                                          nullptr,
                                          DataType::TYPE_INVALID,
                                          params.compute_type);

                AllGatherLoraLinearOutput all_gather_output =
                    allGatherloraLinear({LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input),
                                         ffn_input_ptr,
                                         ag_recv_buffer,
                                         params.qscheme,
                                         params.output->type(),
                                         ParallelMode::FFN_TP});
                // syncAndCheck();
                ffn_input_ptr = all_gather_output.all_gather_recv_buffer;
                up_output     = all_gather_output.output;
                printBufferData(*ffn_input_ptr, "ffn_ag_inter_output");
                printBufferData(*up_output, "ffn_ag_final_output");
            } else {
                printBufferData(params.input, "input");
                GemmParams up_gemm_params(params.input,
                                          *(params.weights.gate_up_weight->kernel),
                                          std::nullopt,
                                          nullptr,
                                          DataType::TYPE_INVALID,
                                          params.compute_type);
                up_output = loraLinear(LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input)).output;
                printBufferData(*up_output, "ffn_up");
            }
            printBufferData(*up_output, "ffn_up_gate");
            bool is_cuda =
                (init_params_.device_type == DeviceType::Cuda) || (init_params_.device_type == DeviceType::ROCm);
            if (is_cuda
                && (params.configs.activation_type == ActivationType::Swiglu
                    || params.configs.activation_type == ActivationType::Silu
                    || params.configs.activation_type == ActivationType::Gelu)) {
                auto act_output = allocateBuffer(
                    {up_output->type(), {up_output->shape()[0], up_output->shape()[1] / 2}, AllocationType::DEVICE});
                up_output = activation({params.configs.activation_type,
                                        up_output,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        act_output,
                                        true,
                                        params.qscheme});
            } else {
                printBufferData(*up_output, "gate_up_output buffer");
                torch::Tensor              gate_up_output_torch_tensor = Buffer2torchTensor(up_output, false);
                std::vector<torch::Tensor> split_tensors = torch::chunk(gate_up_output_torch_tensor, 2, -1);
                torch::Tensor              first_half    = split_tensors[0].clone();
                torch::Tensor              second_half   = split_tensors[1].clone();
                BufferPtr                  gate_output   = torchTensor2Buffer(first_half);
                BufferPtr                  up_out        = torchTensor2Buffer(second_half);
                activation({params.configs.activation_type,
                            up_out,
                            std::nullopt,
                            *gate_output,
                            std::nullopt,
                            mayGetRef(params.weights.act_scale)});

                up_output = std::move(up_out);
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO(!params.enable_sp, "enable_sp is not supported for non-gated activation");
            auto up_gemm_params     = GemmParams(params.input,
                                             *(params.weights.up_weight->kernel),
                                             std::nullopt,
                                             nullptr,
                                             DataType::TYPE_INVALID,
                                             params.compute_type);
            auto lora_linear_params = LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input);
            auto activation_params  = ActivationParams(params.configs.activation_type,
                                                      nullptr,
                                                      mayGetRef(params.weights.up_weight->bias),
                                                      std::nullopt,
                                                      std::nullopt,
                                                      mayGetRef(params.weights.act_scale));
            up_output               = loraLinearWithActivation({lora_linear_params, activation_params});
        }

        if (params.qscheme != QScheme::NoQuantize && params.qscheme != QScheme::Qfp8PerTokenBlock
            && params.qscheme != QScheme::Qfp8PerToken) {
            DataType quant_out_data_type =
                params.qscheme == QScheme::Qfp8PerTensor || params.qscheme == QScheme::Qfp8PerTokenBlock ?
                    DataType::TYPE_FP8_E4M3 :
                    DataType::TYPE_INT8;
            auto quant_params = QuantizeParams(
                *up_output,
                quant_out_data_type,
                1,
                params.qscheme,
                params.weights.smoother_weight ? (OptionalConstBufferRef) * (params.weights.smoother_weight->kernel) :
                                                 std::nullopt,
                std::nullopt,
                params.weights.intermediate_weight2_static_scale_weight ?
                    (OptionalConstBufferRef) * (params.weights.intermediate_weight2_static_scale_weight->kernel) :
                    std::nullopt,
                params.weights.intermediate_weight2_static_scale_reciprocal_weight ?
                    (OptionalConstBufferRef)
                        * (params.weights.intermediate_weight2_static_scale_reciprocal_weight->kernel) :
                    std::nullopt);
            up_output = quantize(quant_params);
        }

        printBufferData(*up_output, "ffn_act");
        if (params.enable_sp && init_params_.ffn_tp_size > 1) {
            BufferPtr gemm_output = allocateBuffer(
                {params.output->type(), {up_output->shape()[0], params.weights.down_weight->kernel->shape()[1]}},
                {"ffn_rs_input"});
            GemmParams                    down_gemm_params = GemmParams(*(up_output),
                                                     *(params.weights.down_weight->kernel),
                                                     nullopt,
                                                     gemm_output,
                                                     DataType::TYPE_INVALID,
                                                     params.compute_type);
            ReduceScatterLoraLinearOutput reduce_scatter_output =
                loraLinearReduceScatter({LoraLinearParams(down_gemm_params, params.lora_input.down_lora_input),
                                         params.output,
                                         params.qscheme,
                                         params.output->type(),
                                         ParallelMode::FFN_TP});
            // syncAndCheck();
            gemm_output = reduce_scatter_output.reduce_scatter_recv_buffer;
            output      = reduce_scatter_output.output;
            printBufferData(*gemm_output, "ffn_rs_inter_output");
            printBufferData(*output, "ffn_rs_final_output");
        } else {
            auto down_gemm_params    = GemmParams(*(up_output),
                                               *(params.weights.down_weight->kernel),
                                               nullopt,
                                               params.output,
                                               DataType::TYPE_INVALID,
                                               params.compute_type);
            down_gemm_params.qscheme = params.qscheme;
            output = loraLinear(LoraLinearParams(down_gemm_params, params.lora_input.down_lora_input)).output;
        }
    }

    printBufferData(*output, "ffn_out");
    if (moe_gating != nullptr) {
        printBufferData(*moe_gating, "moe_gating");
    } else {
        RTP_LLM_LOG_TRACE("no moe_gating");
    }
    return FfnLayerOutput({std::move(output), std::move(moe_gating)});
}

FfnLayerOutput DeviceBase::epMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    const auto&       moe_conf          = params.configs.moe_configs.value();
    MoeDispatchOutput dispatched_output = epDispatch({params.input,
                                                      *gate_output.expert_ids,
                                                      *gate_output.expert_scales,
                                                      moe_conf,
                                                      false,
                                                      params.qscheme,
                                                      params.expert_stats});
    auto              hidden_states     = dispatched_output.hidden;
    auto              moe_ffn_params =
        FfnLayerParams({*hidden_states, params.configs, params.weights, params.residual, params.qscheme});
    moe_ffn_params.expert_stats = params.expert_stats;
    hidden_states               = moeFfn(moe_ffn_params,
                                         {dispatched_output.expert_ids,
                                          dispatched_output.expert_scales,
                                          nullptr,
                                          dispatched_output.deep_ep_ll_output})
                        .hidden_states;
    auto combine_out = epCombine({hidden_states,
                                  dispatched_output.indices,
                                  params.output,
                                  dispatched_output.input_split_sizes,
                                  dispatched_output.output_split_sizes,
                                  moe_conf,
                                  params.input.shape()[0],
                                  init_params_.enable_comm_overlap,
                                  dispatched_output.deep_ep_output,
                                  dispatched_output.deep_ep_ll_output,
                                  std::make_shared<MoeGateSelectOutput>(gate_output),
                                  dispatched_output.expert_ids,
                                  dispatched_output.expert_scales});
    // TODO(wangyin.yx): refact this defered combine.
    if (combine_out.comm_barrier_hook) {
        return {combine_out.all_output, nullptr, combine_out.comm_barrier_hook, combine_out};
    } else {
        auto out = gatherCombineOutput(combine_out);
        printBufferData(*out.hidden_states, "moe_ffn_ep_out");
        return out;
    }
}

FfnLayerOutput DeviceBase::moeFfnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    const auto&         moe_conf    = params.configs.moe_configs.value();
    MoeGateSelectOutput gate_output = moeGateSelect(params);
    FfnLayerOutput      output;
    if (moe_conf.ep_size > 1 && !moe_conf.use_all_gather) {
        output = epMoeFfnLayer(params, gate_output);
    } else {
        output = moeFfn(params, gate_output);
    }
    output.moe_gating = std::move(gate_output.moe_gating);
    return output;
}

FfnLayerOutput DeviceBase::moeSharedExpert(const FfnLayerParams& params) {
    if (params.weights.shared_expert) {
        RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
        const auto& moe_conf = params.configs.moe_configs.value();
        BufferPtr   shared_expert_output =
            (moe_conf.tp_size > 1) ? allocateBufferLike({params.input}, AllocationType::DEVICE, {"shared_expert_buf"}) :
                                       nullptr;
        shared_expert_output  = prepareAllReduce({std::move(shared_expert_output), ReduceOp::Sum}).buffer;
        auto ffn_params       = FfnLayerParams({params.input,
                                                params.configs,
                                                *(params.weights.shared_expert),
                                                params.residual,
                                                params.qscheme,
                                                params.compute_type,
                                                shared_expert_output});
        ffn_params.lora_input = params.lora_input;
        shared_expert_output  = ffnLayer(ffn_params).hidden_states;

        // for qwen moe
        // See
        // https://github.com/huggingface/transformers/blob/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L803
        if (params.weights.shared_expert_gate) {
            auto shared_gate = gemm(GemmParams{params.input, *(params.weights.shared_expert_gate->kernel)}.withSharedGateGemm());
            activation({ActivationType::Sigmoid, shared_gate});
            shared_expert_output =
                multiply({shared_gate->reshape({shared_gate->size()}), *shared_expert_output, shared_expert_output});
        }
        if (moe_conf.tp_size > 1 && !moe_conf.use_all_gather) {
            auto wrapper = DevicePerfWrapper(
                this, "shared_expert_all_reduce, sizeBytes=%ld", (long)shared_expert_output->sizeBytes());
            shared_expert_output = allReduce({shared_expert_output, ReduceOp::Sum}).buffer;
        }
        return {shared_expert_output};
    } else {
        return {nullptr};
    }
}

void DeviceBase::setMoEInsertion(const MoEInsertionParams& params) {
    RUNTIME_ASSERT_OP_ARG(!moe_insertion_params_, "moe insertion params set twice!");
    moe_insertion_params_ = std::make_unique<MoEInsertionParams>(params);
}

std::unique_ptr<MoEInsertionReturns> DeviceBase::stealMoEInsertionRet() {
    auto ret           = move(moe_insertion_ret_);
    moe_insertion_ret_ = nullptr;
    return ret;
}

const std::unique_ptr<MoEInsertionReturns>& DeviceBase::getMoEInsertionRet() {
    return moe_insertion_ret_;
}

void DeviceBase::computeInsertedMoE() {
    if (moe_insertion_params_) {
        RUNTIME_ASSERT_OP_ARG(!moe_insertion_ret_, "moe insertion return not fetched!");
        auto moe_insertion_params = move(moe_insertion_params_);
        moe_insertion_params_     = nullptr;

        const auto& dispatched_output = moe_insertion_params->dispatched_output;
        if (dispatched_output.comm_barrier_hook) {
            dispatched_output.comm_barrier_hook->hook_sync();
        }

        auto hidden = dispatched_output.hidden;
        printBufferData(*hidden, "layer_combine_input");

        const auto ffn_params     = moe_insertion_params->ffn_params;
        auto       moe_ffn_params = FfnLayerParams({*dispatched_output.hidden,
                                                    ffn_params.configs,
                                                    ffn_params.weights,
                                                    ffn_params.residual,
                                                    ffn_params.qscheme});

        hidden = moeFfn(moe_ffn_params,
                        {dispatched_output.expert_ids,
                         dispatched_output.expert_scales,
                         nullptr,
                         dispatched_output.deep_ep_ll_output})
                     .hidden_states;

        auto combine_out = epCombine({
            hidden,
            dispatched_output.indices,
            ffn_params.output,
            dispatched_output.input_split_sizes,
            dispatched_output.output_split_sizes,
            ffn_params.configs.moe_configs.value(),
            moe_insertion_params->origin_token_num,
            init_params_.enable_comm_overlap,
            dispatched_output.deep_ep_output,
            dispatched_output.deep_ep_ll_output,
            moe_insertion_params->gate_output,
            dispatched_output.expert_ids,
            dispatched_output.expert_scales,
        });

        moe_insertion_ret_ = std::unique_ptr<MoEInsertionReturns>(new MoEInsertionReturns({combine_out}));
    }
}

};  // namespace rtp_llm
