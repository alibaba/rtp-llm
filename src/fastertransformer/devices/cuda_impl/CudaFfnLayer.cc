#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/utils/DevicePerfWrapper.h"
#include <cstring>
#include <numeric>

using namespace std;

namespace fastertransformer {

MoeDispatchOutput CudaDevice::epDispatch(const MoeDispatchParams& params) {
    DevicePerfWrapper wrapper(this, "epDispatch");
    const auto& moe_conf = params.moe_configs;

    auto const ep_size    = moe_conf.ep_size;
    auto const tp_size    = moe_conf.tp_size;
    auto const expert_num = moe_conf.expert_num;
    auto const top_k      = moe_conf.top_k;
    size_t     token_num  = params.expert_ids.shape()[0];

    assert(moe_conf.ep_size == tp_size * moe_conf.dp_size);
    size_t    tp_token_size = (token_num + tp_size - 1) / tp_size;
    size_t    slice_begin = std::min(tp_token_size * moe_conf.tp_rank, token_num);
    size_t    slice_size = std::min(token_num - slice_begin, tp_token_size);
    BufferPtr hidden        = params.input.slice(slice_begin, slice_size);
    auto      expert_ids    = params.expert_ids.slice(slice_begin, slice_size);
    auto      expert_scales = params.expert_scales.slice(slice_begin, slice_size);
    token_num = hidden->shape()[0];
    BufferPtr experts_ids_host = clone({*expert_ids, AllocationType::HOST});
    BufferPtr token_nums_per_rank =
        allocateBuffer({DataType::TYPE_INT32, {ep_size}, AllocationType::HOST}, {"token_nums_per_rank"});
    bufMemset(*token_nums_per_rank, 0);
    int32_t*                          token_nums_per_rank_ptr = token_nums_per_rank->data<int32_t>();
    std::vector<std::vector<int32_t>> token_idx_per_rank;
    token_idx_per_rank.resize(ep_size);
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
    printBufferData(*token_nums_per_rank, "token_nums_per_rank");
    auto token_nums_per_rank_gpu     = clone({*token_nums_per_rank});
    auto all_token_nums_per_rank_gpu = allToAll({{token_nums_per_rank_gpu}}).outputs[0];
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
    selected_buffers.emplace_back(select({*hidden, *all_token_indices}));
    selected_buffers.emplace_back(select({*expert_ids, *all_token_indices}));
    selected_buffers.emplace_back(select({*expert_scales, *all_token_indices}));

    auto all2all_output = allToAll({selected_buffers, input_split_sizes, output_split_sizes, params.overlapped});
    const auto& global_buffers = all2all_output.outputs;
    return {global_buffers[0],
            global_buffers[1],
            global_buffers[2],
            all_token_indices,
            input_split_sizes,
            output_split_sizes,
            selected_buffers,
            all2all_output.concated_input,
            all2all_output.output_to_split,
            move(all2all_output.comm_barrier_hook)
        };
}

FfnLayerOutput CudaDevice::epCombine(const MoeCombineParams& params) {
    DevicePerfWrapper wrapper(this, "epCombine");
    if (params.overlapped) {
        overlap_hold_buffers_.clear();
    }
    auto all2all_ret = allToAll({
        {params.input}, params.output_split_sizes, params.input_split_sizes, params.overlapped});
    auto all_output = all2all_ret.outputs[0];
    if (params.overlapped) {
        overlap_hold_buffers_.emplace_back(all_output);
        overlap_hold_buffers_.emplace_back(params.input);
        overlap_hold_buffers_.emplace_back(params.indices);
    }
    torch::Tensor indices_tensor;
    cudaStream_t  stream = params.overlapped ? communication_stream_ : stream_;
    if (params.moe_configs.tp_size > 1) {
        // TODO: can use torch all gather unequal size to avoid copy
        const size_t tp_token_size =
            (params.origin_token_num + params.moe_configs.tp_size - 1) / params.moe_configs.tp_size;
        size_t dim1_size = all_output->shape()[1];
        assert(params.origin_token_num >= tp_token_size * params.moe_configs.tp_rank);
        size_t current_token_num = std::max(0, std::min((int)params.origin_token_num - int(tp_token_size * params.moe_configs.tp_rank), (int)tp_token_size));

        BufferPtr scatter_output = allocateBuffer(
            {all_output->type(),
             {current_token_num,
              dim1_size}});
        if (params.overlapped) {
            overlap_hold_buffers_.emplace_back(scatter_output);
        }
        assert(scatter_output->shape()[0] == params.indices->shape()[0]);
        if (scatter_output->shape()[0] > 0) {
            cudaMemsetAsync(scatter_output->data(), 0, scatter_output->sizeBytes(), stream);
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(scatter_output->type(),
                                             invokeScatterAdd,
                                             all_output->data(),
                                             all_output->shape()[0],
                                             all_output->shape()[1],
                                             params.indices->data<int32_t>(),
                                             scatter_output->data(),
                                             stream);
        }

        BufferPtr padding_output;
        if (params.origin_token_num == tp_token_size * params.moe_configs.tp_size && params.output) {
            padding_output = params.output;
        } else {
            padding_output =
                allocateBuffer({all_output->type(), {tp_token_size * params.moe_configs.tp_size, dim1_size}});
            if (params.overlapped) {
                overlap_hold_buffers_.emplace_back(padding_output);
            }
        }
        if (scatter_output->shape()[0] > 0) {
            copy({padding_output->view(tp_token_size * params.moe_configs.tp_rank, scatter_output->shape()[0]),
                    *scatter_output,
                    params.overlapped});
        }
        allGather({{padding_output}, ParallelMode::TP, params.overlapped});
        if (params.origin_token_num == tp_token_size * params.moe_configs.tp_size) {
            return {padding_output, createCommHook()};
        } else {
            BufferPtr output = params.output ? params.output : allocateBufferLike(padding_output->view(0, params.origin_token_num));
            copy({*output, padding_output->view(0, params.origin_token_num), params.overlapped});
            return {output, createCommHook()};
        }
    } else {
        vector<size_t> new_shape = all_output->shape();
        new_shape[0]             = params.origin_token_num;
        BufferPtr output         = params.output ? params.output : allocateBuffer({all_output->type(), new_shape});
        if (output->shape()[0] > 0) {
            cudaMemsetAsync(output->data(), 0, output->sizeBytes(), stream);
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(output->type(),
                                             invokeScatterAdd,
                                             all_output->data(),
                                             all_output->shape()[0],
                                             all_output->shape()[1],
                                             params.indices->data<int32_t>(),
                                             output->data(),
                                             stream);
        }
        return {output, createCommHook()};
    }
}

MoeGateSelectOutput CudaDevice::moeGateSelect(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    const auto& moe_conf = params.configs.moe_configs.value();
    const auto& hidden   = params.input;

    const auto token_num  = hidden.shape()[0];
    const auto num_expert = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto top_k      = moe_conf.top_k;

    const auto gate = gemm({hidden, *params.weights.moe_gating_weight->kernel, nullopt, nullptr, DataType::TYPE_FP32});

    const auto expert_scales = allocateBuffer({DataType::TYPE_FP32, {token_num, top_k}}, {"moe_expert_scale"});
    const auto expert_for_source_row =
        allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"moe_expert_for_src"});
    const auto softmax_out      = allocateBuffer({DataType::TYPE_FP32, {token_num, num_expert}}, {"moe_softmax_out"});
    const auto source_rows      = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"source_rows"});

    auto       normalization_mode = moe_conf.has_moe_norm ?
                                        tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE :
                                        tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    auto       gate_with_bias     = gate;
    at::Tensor gate_with_bias_tensor;  // hold the tensor to prevent it from being released
    prepareMoEGate(params, gate, gate_with_bias_tensor, gate_with_bias);

    moe_plugin_->selectExpertsForTokens(gate->data<float>(),
                                        gate_with_bias->data<float>(),
                                        expert_scales->data<float>(),
                                        nullptr, // sparse_mixer_out
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
    return {expert_for_source_row, expert_scales};
}

FfnLayerOutput CudaDevice::moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");

    const auto& moe_conf = params.configs.moe_configs.value();
    const auto& hidden   = params.input;
    const auto& weights  = params.weights;
    const auto  type     = hidden.type();

    const auto weight_type            = weights.moe_down_weight->kernel->type();
    const auto token_num              = hidden.shape()[0];
    const auto hidden_dim             = hidden.shape()[1];
    const auto num_expert             = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto top_k                  = moe_conf.top_k;
    const auto moe_inter_size         = moe_conf.moe_inter_padding_size;
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

void CudaDevice::prepareMoEGate(const FfnLayerParams& params,
                                BufferPtr             gate,
                                torch::Tensor&        gate_with_bias_tensor,
                                BufferPtr&            gate_with_bias) {
    auto const& moe_conf   = params.configs.moe_configs.value();
    const auto& hidden     = params.input;
    const auto  token_num  = hidden.shape()[0];
    const auto  num_expert = params.weights.moe_gating_weight->kernel->shape()[1];

    if (moe_conf.scoring_func == 1) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(DataType::TYPE_FP32, invokeSigmoid, gate->data(), gate->size(), 1.0f, stream_);
    }
    if (params.weights.e_score_correction_bias) {
        const int n_routed_experts = num_expert;
        const int n_group          = moe_conf.n_group;
        const int topk_group       = moe_conf.topk_group;

        torch::Tensor gate_tensor = Buffer2torchTensor(gate, false);
        torch::Tensor e_score_correction_bias_tensor =
            Buffer2torchTensor(params.weights.e_score_correction_bias, false).to(torch::kFloat32);
        auto scores_for_choice = gate_tensor.add(e_score_correction_bias_tensor);
        auto reshaped_scores   = scores_for_choice.view({(int)token_num, n_group, -1});
        auto topk_result       = reshaped_scores.topk(2, /*dim=*/-1);
        auto group_scores      = std::get<0>(topk_result).sum(-1);
        auto group_topk_result = group_scores.topk(
            /*k=*/topk_group,
            /*dim=*/-1,
            /*largest=*/true,
            /*sorted=*/false);
        auto group_idx  = std::get<1>(group_topk_result);
        auto group_mask = torch::zeros_like(group_scores);
        group_mask.scatter_(
            /*dim=*/1,
            /*index=*/group_idx,
            /*src=*/1.0f);
        int64_t experts_per_group = n_routed_experts / n_group;
        auto    score_mask =
            group_mask.unsqueeze(-1).expand({(int)token_num, n_group, experts_per_group}).reshape({(int)token_num, -1});
        gate_with_bias_tensor = scores_for_choice.masked_fill(torch::logical_not(score_mask.to(torch::kBool)), 0.0);
        gate_with_bias        = torchTensor2Buffer(gate_with_bias_tensor);
        printBufferData(*gate_with_bias, "gate_with_bias");
    }
}

} // namespace fastertransformer
