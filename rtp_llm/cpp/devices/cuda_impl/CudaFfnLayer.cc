#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/kernels/no_aux_tc_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/core/torch_utils/TorchEvent.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/kernels/eplb/experts_stats_kernels.h"
#include "rtp_llm/cpp/kernels/moe_kernels.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cstring>
#include <numeric>

using namespace std;

namespace rtp_llm {

void hackMoeExpert(const MoeDispatchParams& params, BufferPtr& experts_ids_host) {
    auto       elementNum      = experts_ids_host->size();
    auto const ep_size         = params.moe_configs.ep_size;
    auto const expert_num      = params.moe_configs.expert_num;
    auto const top_k           = params.moe_configs.top_k;
    auto const token_num       = params.expert_ids.shape()[0];
    auto       expert_per_rank = expert_num / ep_size;
    if (token_num == 0 || top_k == 0) {
        return;
    }
    for (int i = 0; i < elementNum; ++i) {
        *(experts_ids_host->dataWithOffset<int32_t>(i)) = i % ep_size * expert_per_rank + i / ep_size % expert_per_rank;
    }
}

MoeDispatchOutput CudaDevice::epDispatch(const MoeDispatchParams& params) {
    DevicePerfWrapper wrapper(this, "epDispatch");
    if (init_params_.use_deepep_moe) {
        if (init_params_.use_deepep_low_latency) {
            return deepEpLLDispatch(params);
        } else {
            return deepEpDispatch(params);
        }
    }
    const auto& moe_conf = params.moe_configs;

    // note: expert_num is physical expert number, including extra experts
    auto const expert_num = moe_conf.expert_num + moe_conf.extra_expert_num;
    auto const top_k      = moe_conf.top_k;
    auto const ep_size    = moe_conf.ep_size;
    auto const tp_size    = moe_conf.tp_size;
    size_t     token_num  = params.expert_ids.shape()[0];

    assert(moe_conf.ep_size == tp_size * moe_conf.dp_size);
    size_t    tp_token_size = (token_num + tp_size - 1) / tp_size;
    size_t    slice_begin   = std::min(tp_token_size * moe_conf.tp_rank, token_num);
    size_t    slice_size    = std::min(token_num - slice_begin, tp_token_size);
    BufferPtr hidden        = params.input.slice(slice_begin, slice_size);
    auto      expert_ids    = params.expert_ids.slice(slice_begin, slice_size);
    auto      expert_scales = params.expert_scales.slice(slice_begin, slice_size);
    token_num               = hidden->shape()[0];
    BufferPtr token_nums_per_rank =
        allocateBuffer({DataType::TYPE_INT32, {ep_size}, AllocationType::HOST}, {"token_nums_per_rank"});
    bufMemset(*token_nums_per_rank, 0);
    int32_t*                          token_nums_per_rank_ptr = token_nums_per_rank->data<int32_t>();
    std::vector<std::vector<int32_t>> token_idx_per_rank;
    token_idx_per_rank.resize(ep_size);
    for (int i = 0; i < ep_size; ++i) {
        token_idx_per_rank[i].reserve(token_num);
    }

    BufferPtr experts_ids_host = clone({*expert_ids, AllocationType::HOST});
    if (hack_moe_expert_) {
        hackMoeExpert(params, experts_ids_host);
    }

    int const num_experts_per_node = expert_num / ep_size;
    for (int i = 0; i < token_num; ++i) {
        for (int j = 0; j < top_k; ++j) {
            size_t expert_id = *(experts_ids_host->dataWithOffset<int32_t>(i * top_k + j));
            assert(expert_id < expert_num);
            const auto ep_rank = expert_id / num_experts_per_node;
            if (token_idx_per_rank[ep_rank].empty() || *token_idx_per_rank[ep_rank].rbegin() != i) {
                token_nums_per_rank_ptr[ep_rank] += 1;
                token_idx_per_rank[ep_rank].push_back(i);
            }
        }
    }

    if (params.overlapped && params.compute_stream_event) {
        // TOOD(wangyin.yx): encapsulate this event stream with device-independent interface
        // and implement epDispatch method in DeviceBase.
        const auto casted_event = dynamic_cast<TorchEvent*>(params.compute_stream_event.get());
        if (!casted_event) {
            throw OpException({OpErrorType::ERROR_INTERNAL, "compute_stream_event is not TorchEvent"});
        }
        casted_event->event->block(*torch_comm_stream_);
    }

    printBufferData(*token_nums_per_rank, "token_nums_per_rank");
    auto      token_nums_per_rank_gpu     = clone({*token_nums_per_rank});
    auto      all_token_nums_per_rank_gpu = allToAll({{token_nums_per_rank_gpu}}).outputs[0];
    size_t    total_size = std::accumulate(token_nums_per_rank_ptr, token_nums_per_rank_ptr + ep_size, 0);
    BufferPtr all_token_indices_cpu =
        allocateBuffer({DataType::TYPE_INT32, {total_size}, AllocationType::HOST}, {"token_nums_per_rank"});

    size_t offset = 0;
    for (int i = 0; i < ep_size; ++i) {
        memcpy(all_token_indices_cpu->dataWithOffset<int32_t>(offset),
               token_idx_per_rank[i].data(),
               token_idx_per_rank[i].size() * sizeof(int32_t));
        offset += token_idx_per_rank[i].size();
    }

    printBufferData(*all_token_indices_cpu, "all_token_indices_cpu");
    BufferPtr all_token_indices = clone({*all_token_indices_cpu});
    // sync allToAll all_token_nums_per_rank_gpu
    check_cuda_value(cudaStreamSynchronize(stream_));
    syncCommunication(false);
    auto                all_token_nums_per_rank = clone({*all_token_nums_per_rank_gpu, AllocationType::HOST});
    std::vector<size_t> input_split_sizes;
    std::vector<size_t> output_split_sizes;
    input_split_sizes.resize(ep_size);
    output_split_sizes.resize(ep_size);
    for (int i = 0; i < ep_size; ++i) {
        input_split_sizes[i]  = token_nums_per_rank_ptr[i];
        output_split_sizes[i] = *(all_token_nums_per_rank->dataWithOffset<int32_t>(i));
    }
    vector<BufferPtr> selected_buffers;
    BufferPtr         select_hidden = select({*hidden, *all_token_indices});
    BufferPtr         hidden_fp8;
    // TODO: remove or fix it
    // bool use_fp8_all2all = params.qscheme == QScheme::Qfp8PerTokenBlock
    bool use_fp8_all2all = false;
    if (use_fp8_all2all) {
        hidden_fp8 = quantize({*select_hidden, DataType::TYPE_QFP8_E4M3, 1, params.qscheme});
        selected_buffers.emplace_back(std::dynamic_pointer_cast<QBuffer>(hidden_fp8)->kernelPtr());
        selected_buffers.emplace_back(std::dynamic_pointer_cast<QBuffer>(hidden_fp8)->scalesPtr());
    } else {
        selected_buffers.emplace_back(select_hidden);
    }
    selected_buffers.emplace_back(select({*expert_ids, *all_token_indices}));
    selected_buffers.emplace_back(select({*expert_scales, *all_token_indices}));

    auto        all2all_output = allToAll({selected_buffers, input_split_sizes, output_split_sizes, params.overlapped});
    const auto& global_buffers = all2all_output.outputs;
    if (use_fp8_all2all) {
        BufferPtr hidden_fp8_out(new QBuffer(
            std::move(global_buffers[0]),
            std::move(global_buffers[1]),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));

        updateExpertGpuLoads(moe_conf, params.expert_stats, global_buffers[2]);

        return {hidden_fp8_out,
                global_buffers[2],
                global_buffers[3],
                all_token_indices,
                input_split_sizes,
                output_split_sizes,
                selected_buffers,
                all2all_output.concated_input,
                all2all_output.output_to_split,
                move(all2all_output.comm_barrier_hook)};
    } else {
        updateExpertGpuLoads(moe_conf, params.expert_stats, global_buffers[1]);

        return {global_buffers[0],
                global_buffers[1],
                global_buffers[2],
                all_token_indices,
                input_split_sizes,
                output_split_sizes,
                selected_buffers,
                all2all_output.concated_input,
                all2all_output.output_to_split,
                move(all2all_output.comm_barrier_hook)};
    }
}

MoeCombineOutput CudaDevice::epCombine(const MoeCombineParams& params) {
    DevicePerfWrapper wrapper(this, "epCombine");
    if (init_params_.use_deepep_moe) {
        if (init_params_.use_deepep_low_latency) {
            return deepEpLLCombine(params);
        } else {
            return deepEpCombine(params);
        }
    }
    auto all2all_ret = allToAll({{params.input},
                                 params.output_split_sizes,
                                 params.input_split_sizes,
                                 params.overlapped,
                                 ParallelMode::DP_AND_TP,
                                 params.compute_stream_event});
    return MoeCombineOutput({all2all_ret.outputs[0], nullptr, params, move(all2all_ret.comm_barrier_hook)});
}

FfnLayerOutput CudaDevice::gatherCombineOutput(const MoeCombineOutput& combine_outputs) {
    auto&       all_output     = combine_outputs.all_output;
    auto        scatter_output = combine_outputs.scatter_output;
    const auto& params         = combine_outputs.params;

    torch::Tensor indices_tensor;

    if (params.moe_configs.tp_size > 1) {
        // TODO: can use torch all gather unequal size to avoid copy
        const size_t tp_token_size =
            (params.origin_token_num + params.moe_configs.tp_size - 1) / params.moe_configs.tp_size;
        size_t dim1_size = all_output->shape()[1];
        size_t current_token_num =
            std::max(0,
                     std::min((int)params.origin_token_num - int(tp_token_size * params.moe_configs.tp_rank),
                              (int)tp_token_size));

        if (scatter_output == nullptr) {
            scatter_output = allocateBuffer({all_output->type(), {current_token_num, dim1_size}});
            // TODO: why this assertion?
            // assert(all_output->shape()[0] == current_token_num);
            if (scatter_output->shape()[0] > 0) {
                check_cuda_value(cudaMemsetAsync(scatter_output->data(), 0, scatter_output->sizeBytes(), stream_));
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(scatter_output->type(),
                                                 invokeScatterAdd,
                                                 all_output->data(),
                                                 all_output->shape()[0],
                                                 all_output->shape()[1],
                                                 params.indices->data<int32_t>(),
                                                 scatter_output->data(),
                                                 this->use_stable_scatter_add,
                                                 stream_);
            }
        }
        BufferPtr padding_output;
        if (params.origin_token_num == tp_token_size * params.moe_configs.tp_size && params.output) {
            padding_output = params.output;
        } else {
            padding_output =
                allocateBuffer({all_output->type(), {tp_token_size * params.moe_configs.tp_size, dim1_size}});
        }
        if (scatter_output->shape()[0] > 0) {
            copy({padding_output->view(tp_token_size * params.moe_configs.tp_rank, scatter_output->shape()[0]),
                  *scatter_output});
        }
        allGather({{padding_output}, ParallelMode::TP, {}, true});
        if (params.origin_token_num == tp_token_size * params.moe_configs.tp_size) {
            return {padding_output};
        } else {
            BufferPtr output =
                params.output ? params.output : allocateBufferLike(padding_output->view(0, params.origin_token_num));
            copy({*output, padding_output->view(0, params.origin_token_num)});
            return {output};
        }
    } else {
        if (scatter_output == nullptr) {
            vector<size_t> new_shape = all_output->shape();
            new_shape[0]             = params.origin_token_num;
            BufferPtr output         = params.output ? params.output : allocateBuffer({all_output->type(), new_shape});
            printBufferData(*output, "scatter_add_input");
            if (output->shape()[0] > 0 && params.origin_token_num > 0) {
                check_cuda_value(cudaMemsetAsync(output->data(), 0, output->sizeBytes(), stream_));
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(output->type(),
                                                 invokeScatterAdd,
                                                 all_output->data(),
                                                 all_output->shape()[0],
                                                 all_output->shape()[1],
                                                 params.indices->data<int32_t>(),
                                                 output->data(),
                                                 this->use_stable_scatter_add,
                                                 stream_);
            }
            printBufferData(*output, "scatter_add_output");
            return {output};
        } else {
            return {all_output};
        }
    }
}

MoeGateSelectOutput CudaDevice::moeGateSelect(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    const auto& moe_conf = params.configs.moe_configs.value();
    const auto& hidden   = params.input;

    const auto token_num = hidden.shape()[0];

    // note: num_expert is real expert number, not including extra expert
    const auto num_expert = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto top_k      = moe_conf.top_k;

    const auto gate = gemm({hidden,
                            *params.weights.moe_gating_weight->kernel,
                            nullopt,
                            nullptr,
                            DataType::TYPE_FP32,
                            DataType::TYPE_FP32});
    BufferPtr  moe_gating;

    const auto expert_scales = allocateBuffer({DataType::TYPE_FP32, {token_num, top_k}}, {"moe_expert_scale"});
    DataType   topk_t        = DataType::TYPE_INT32;
    if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
        // currently use_deepep_moe and use_all_gather may coexist, so we need to check both
        // when use_all_gather = 1, we should use TYPE_INT32 for genSourceRowRevert func
        if (init_params_.use_deepep_moe && !init_params_.use_all_gather) {
            topk_t = DataType::TYPE_INT64;
        }
    }
    const auto expert_for_source_row = allocateBuffer({topk_t, {token_num, top_k}}, {"moe_expert_for_src"});
    const auto softmax_out = allocateBuffer({DataType::TYPE_FP32, {token_num, num_expert}}, {"moe_softmax_out"});
    const auto source_rows = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"source_rows"});

    auto normalization_mode = moe_conf.has_moe_norm ?
                                  tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE :
                                  tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;

    prepareMoEGate(params, gate);

    if (params.weights.e_score_correction_bias) {
        moe_gating = clone({*gate});
        moeGateSelectWithBias(params, gate, moe_gating, expert_scales, expert_for_source_row, (int)normalization_mode);
    } else {
        if (topk_t == DataType::TYPE_INT64) {
            moe_plugin_->selectExpertsForTokens<int64_t>(gate->data<float>(),
                                                         gate->data<float>(),
                                                         expert_scales->data<float>(),
                                                         nullptr,  // sparse_mixer_out
                                                         softmax_out->data<float>(),
                                                         expert_for_source_row->data<int64_t>(),
                                                         source_rows->data<int>(),
                                                         token_num,
                                                         num_expert,
                                                         top_k,
                                                         0,
                                                         num_expert,
                                                         0,
                                                         normalization_mode,
                                                         stream_);
        } else {
            moe_plugin_->selectExpertsForTokens<int>(gate->data<float>(),
                                                     gate->data<float>(),
                                                     expert_scales->data<float>(),
                                                     nullptr,  // sparse_mixer_out
                                                     softmax_out->data<float>(),
                                                     expert_for_source_row->data<int>(),
                                                     source_rows->data<int>(),
                                                     token_num,
                                                     num_expert,
                                                     top_k,
                                                     0,
                                                     num_expert,
                                                     0,
                                                     normalization_mode,
                                                     stream_);
        }
        moe_gating = std::move(gate);
    }
    if (init_params_.moe_config.fake_balance_expert) {
        if (topk_t == DataType::TYPE_INT64) {
            fake_balance_expert(expert_for_source_row->data<int64_t>(),
                                expert_scales->data<float>(),
                                moe_conf.dp_rank,
                                moe_conf.dp_size,
                                moe_conf.ep_size,
                                num_expert,
                                token_num * top_k,
                                stream_);
        } else {
            fake_balance_expert(expert_for_source_row->data<int>(),
                                expert_scales->data<float>(),
                                moe_conf.dp_rank,
                                moe_conf.dp_size,
                                moe_conf.ep_size,
                                num_expert,
                                token_num * top_k,
                                stream_);
        }
    }

    // EPLB
    balanceExperts(expert_for_source_row, params.expert_stats, params.configs.moe_configs.value(), params.weights);

    printBufferData(*expert_for_source_row, "expert_for_source_row");
    printBufferData(*expert_scales, "expert_scales");
    return {expert_for_source_row, expert_scales, moe_gating};
}

void CudaDevice::moeGateSelectWithBias(const FfnLayerParams& params,
                                       BufferPtr             gate,
                                       BufferPtr             gate_with_bias,
                                       BufferPtr             expert_scales,
                                       BufferPtr             expert_for_source_row,
                                       int                   normalization_mode) {
    RTP_LLM_CHECK_WITH_INFO(normalization_mode == 0 || normalization_mode == 1, "Unsupported normalization_mode");

    const auto& moe_conf   = params.configs.moe_configs.value();
    const auto  token_num  = params.input.shape()[0];
    const auto  num_expert = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto  top_k      = moe_conf.top_k;

    at::Tensor gate_with_bias_tensor = Buffer2torchTensor(gate_with_bias, false);
    at::Tensor e_score_correction_bias_tensor =
        Buffer2torchTensor(params.weights.e_score_correction_bias, false).to(torch::kFloat32);
    gate_with_bias_tensor.add_(e_score_correction_bias_tensor);
    at::Tensor group_scores =
        torch::empty({(int64_t)token_num, moe_conf.n_group}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    if (expert_for_source_row->type() == DataType::TYPE_INT64) {
        invokeNoAuxTc<float, int64_t>(gate->data<float>(),
                                      reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
                                      expert_scales->data<float>(),
                                      expert_for_source_row->data<int64_t>(),
                                      gate_with_bias->data<float>(),
                                      token_num,
                                      num_expert,
                                      moe_conf.n_group,
                                      moe_conf.topk_group,
                                      top_k,
                                      normalization_mode,
                                      moe_conf.routed_scaling_factor,
                                      stream_);
    } else if (expert_for_source_row->type() == DataType::TYPE_INT32) {
        invokeNoAuxTc<float, int32_t>(gate->data<float>(),
                                      reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
                                      expert_scales->data<float>(),
                                      expert_for_source_row->data<int32_t>(),
                                      gate_with_bias->data<float>(),
                                      token_num,
                                      num_expert,
                                      moe_conf.n_group,
                                      moe_conf.topk_group,
                                      top_k,
                                      normalization_mode,
                                      moe_conf.routed_scaling_factor,
                                      stream_);
    } else {
        RTP_LLM_LOG_ERROR("Unsupported expert_for_source_row type: %d", int(expert_for_source_row->type()));
        throw OpException({OpErrorType::ERROR_INTERNAL, "Unsupported expert_for_source_row type"});
    }
}

void CudaDevice::prepareMoEGate(const FfnLayerParams& params, BufferPtr gate) {
    auto const& moe_conf = params.configs.moe_configs.value();

    if (moe_conf.scoring_func == 1) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(DataType::TYPE_FP32, invokeSigmoid, gate->data(), gate->size(), 1.0f, stream_);
    }
}

FfnLayerOutput CudaDevice::moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    // deepseek deepep low latency only
    if (init_params_.use_deepep_moe && init_params_.use_deepep_low_latency) {
        return deepEpLLMoeFfn(params, gate_outputs);
    }
    const auto& moe_conf = params.configs.moe_configs.value();

    if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
        return moeFfnFp8(params, gate_outputs);
    }

    const auto& hidden  = params.input;
    const auto& weights = params.weights;
    const auto  type    = hidden.type();

    const auto weight_type = weights.moe_down_weight->kernel->type();
    const auto token_num   = hidden.shape()[0];
    const auto hidden_dim  = hidden.shape()[1];

    // note: num_expert is physical expert number, including extra expert
    const auto num_expert             = moe_conf.expert_num + moe_conf.extra_expert_num;
    const auto top_k                  = moe_conf.top_k;
    bool       is_gated_activation    = isGatedActivation(params.configs.activation_type);
    auto       moe_inter_size         = is_gated_activation ? weights.moe_gate_weight->kernel->shape()[1] / 2 :
                                                              weights.moe_gate_weight->kernel->shape()[1];
    const auto normalize_expert_scale = moe_conf.normalize_expert_scale;

    // TODO group_size
    auto group_size = 0;
    if (params.weights.moe_gate_weight->kernel->isQBuffer()) {
        if (dynamic_cast<const QBuffer*>(params.weights.moe_gate_weight->kernel.get())->zerosData() != nullptr) {
            group_size =
                params.weights.moe_gate_weight->kernel->shape()[1]
                / dynamic_cast<const QBuffer*>(params.weights.moe_gate_weight->kernel.get())->zeros().shape()[1];
        }
    }
    auto      normalization_mode = moe_conf.has_moe_norm ?
                                       tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE :
                                       tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    BufferPtr output             = nullptr;
    if (params.output) {
        output = params.output;
    } else {
        output = allocateBuffer({type, {token_num, hidden_dim}});
    }
    if (token_num == 0) {
        return {output};
    }
    moe_plugin_->init(num_expert,
                      top_k,
                      normalize_expert_scale,
                      hidden_dim,
                      moe_inter_size,
                      params.configs.activation_type,
                      nvinfer1DtypeConvert(type),
                      nvinfer1DtypeConvert(weight_type),
                      group_size > 0,
                      group_size,
                      normalization_mode,
                      moe_conf.ep_size,
                      moe_conf.ep_rank);
    const auto new_ws_size   = moe_plugin_->getWorkspaceSize(token_num);
    const auto new_worksapce = allocateBuffer({DataType::TYPE_BYTES, {new_ws_size}}, {"moe_workspace"});
    auto       fc2_result    = allocateBuffer({type, {token_num, top_k, hidden_dim}}, {"moe_fc2_result"});
    const auto new_expanded_source_row_to_dest =
        allocateBuffer({DataType::TYPE_INT32, {top_k, token_num}}, {"moe_expand_src_to_dst"});
    moe_plugin_->enqueue(hidden.data(),
                         nullptr,  // gate->data<float>(),
                         nullptr,  // gate_with_bias->data<float>(),
                         weights.moe_gate_weight->kernel->data(),
                         BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
                         BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
                         OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_gate_weight->bias),
                         weights.moe_down_weight->kernel->data(),
                         BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_down_weight->kernel),
                         BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_down_weight->kernel),
                         OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_down_weight->bias),
                         token_num,
                         new_worksapce->data<char>(),
                         // output
                         output->data(),
                         fc2_result->data(),
                         nullptr,  // finished
                         gate_outputs.expert_scales->data<float>(),
                         new_expanded_source_row_to_dest->data<int>(),
                         gate_outputs.expert_ids->data<int>(),
                         stream_);
    printBufferData(*output, "moe_ffn_out");
    return FfnLayerOutput({move(output)});
}

}  // namespace rtp_llm
