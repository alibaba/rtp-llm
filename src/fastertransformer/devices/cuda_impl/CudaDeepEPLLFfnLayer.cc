#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cuda/Dispatch.h"

namespace fastertransformer {

MoeDispatchOutput CudaDevice::deepEpLLDispatch(const MoeDispatchParams& params) {
    const auto& moe_conf                         = params.moe_configs;
    auto const  tp_size                          = moe_conf.tp_size;
    auto const  expert_num                       = moe_conf.expert_num;
    size_t      token_num                        = params.expert_ids.shape()[0];
    size_t      tp_token_size                    = (token_num + tp_size - 1) / tp_size;

    size_t      slice_begin                      = std::min(tp_token_size * moe_conf.tp_rank, token_num);
    size_t      slice_size                       = std::min(token_num - slice_begin, tp_token_size);

    BufferPtr   hidden                           = params.input.slice(slice_begin, slice_size);
    auto        expert_ids                       = params.expert_ids.slice(slice_begin, slice_size);
    auto        expert_scales                    = params.expert_scales.slice(slice_begin, slice_size);
    token_num                                    = hidden->shape()[0];

    // prepare tensors for call deepep dispatch layout
    auto input_tensor = Buffer2torchTensorWithDstType(*hidden, false, c10::kBFloat16); // [num_token, hidden_size]
    torch::Tensor topk_idx_tensor = Buffer2torchTensorWithDstType(*expert_ids, false, c10::kLong); // [num_token, top_k]
    cudaDeviceSynchronize();

    auto dispatch_output = deepep_buffer_->lowLatencyDispatch(input_tensor,
                                                              topk_idx_tensor,
                                                              ll_num_max_token_per_rank,
                                                              expert_num,
                                                              false /*use_fp8*/,
                                                              false /*async_finish*/,
                                                              false /*return_recv_hook*/
    );
    cudaDeviceSynchronize();

    BufferPtr packed_recv_x_buffer = torchTensor2BufferWithDstType(dispatch_output.packed_recv_x, torch::kHalf);

    MoeDispatchOutput out{packed_recv_x_buffer, expert_ids, expert_scales};
    out.deep_ep_ll_output.reset(new DeepEPDispatchOutputLowLatency(dispatch_output));

    return out;
}

FfnLayerOutput CudaDevice::deepEpLLCombine(const MoeCombineParams& params) {
    if (params.overlapped) {
        overlap_hold_buffers_.clear();
    }
    FT_CHECK(params.deep_ep_ll_output != nullptr);

    cudaDeviceSynchronize();

    auto& expert_ids    = params.expert_ids;
    auto& expert_scales = params.expert_scales;

    torch::Tensor input_tensor = 
        Buffer2torchTensorWithDstType(*params.input, false, torch::kBFloat16);  // [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]
    torch::Tensor topk_idx_tensor = Buffer2torchTensorWithDstType(*expert_ids, false, c10::kLong);
    torch::Tensor topk_weights_tensor = Buffer2torchTensorWithDstType(*expert_scales, false, c10::kFloat);
    cudaDeviceSynchronize();

    auto combine_output = deepep_buffer_->lowLatencyCombine(input_tensor,
                                                            topk_idx_tensor,
                                                            topk_weights_tensor,
                                                            params.deep_ep_ll_output->handle,
                                                            false,
                                                            false);
    cudaDeviceSynchronize();

    BufferPtr all_output;
    if (params.output != nullptr) {
        all_output = torchTensor2BufferWithDstType(combine_output.combined_x, dataTypeToTorchType(params.output->type()));
    } else {
        all_output = torchTensor2BufferWithDstType(combine_output.combined_x, dataTypeToTorchType(params.input->type()));
    }
    return gatherCombineOutput(all_output, params, all_output);
}

FfnLayerOutput CudaDevice::deepEpLLMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) {
    const auto&       moe_conf = params.configs.moe_configs.value();
    const auto top_k = gate_output.expert_ids->shape()[1];
    MoeDispatchOutput dispatched_output =
        deepEpLLDispatch({params.input, *gate_output.expert_ids, *gate_output.expert_scales, moe_conf});

    cudaDeviceSynchronize();

    BufferPtr hidden_states = dispatched_output.hidden;  // [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]

    BufferPtr recv_count = torchTensor2Buffer(dispatched_output.deep_ep_ll_output->packed_recv_count);
    auto recv_count_cpu = clone(CloneParams(*recv_count, AllocationType::HOST, BufferHints(), init_params_.enable_comm_overlap));

    auto start_expert_id   = moe_conf.ep_rank * moe_conf.expert_num / moe_conf.ep_size;

    // 每个expert单独计算output hidden state
    // 准备expert 粒度的 context
    std::vector<DeepEPLowLatencyExpertContext> expert_contexts;
    for (int i = 0; i < recv_count_cpu->shape()[0]; i++) {
        auto token_num = recv_count_cpu->data<int32_t>()[i];
        if (token_num == 0) {
            expert_contexts.emplace_back(DeepEPLowLatencyExpertContext(i));
            continue;
        }

        // hidden states with valid token
        BufferPtr expert_all_token_hidden_states = hidden_states->index(i);
        BufferPtr expert_token_hidden_states = expert_all_token_hidden_states->slice(0, token_num);
        
        // mock expert ids and scales only for one expert, may be can change to topk=1, and cudaMemset once 
        BufferPtr expert_ids_cpu_buffer = allocateBuffer({DataType::TYPE_INT32, {(uint64_t)token_num, top_k}, AllocationType::HOST});
        BufferPtr expert_scales_cpu_buffer = allocateBuffer({DataType::TYPE_FP32, {(uint64_t)token_num, top_k}, AllocationType::HOST});

        // mock expert idx and scales like [[2,2,2,2]], [[0.05,0.05,0.05,0.85]]
        for (int token_idx = 0; token_idx < token_num; token_idx++) {
            auto token_expert_ids_cpu_buffer = expert_ids_cpu_buffer->index(token_idx);
            auto token_expert_scales_cpu_buffer = expert_scales_cpu_buffer->index(token_idx);
            float scale_weights = 0; // no more than 20 topk
            for (int topk_idx = 0; topk_idx < top_k; topk_idx++) {
                *(token_expert_ids_cpu_buffer->dataWithOffset<int32_t>(topk_idx)) = start_expert_id + i;
                if (topk_idx == top_k - 1) {
                    *(token_expert_scales_cpu_buffer->dataWithOffset<float>(topk_idx)) = 1 - scale_weights;
                } else {
                    *(token_expert_scales_cpu_buffer->dataWithOffset<float>(topk_idx)) = 0.05f;
                    scale_weights += 0.05f;
                }
            }
        }

        auto expert_ids_gpu_buffer = clone(CloneParams(*expert_ids_cpu_buffer, AllocationType::DEVICE, BufferHints(), init_params_.enable_comm_overlap));
        auto expert_scales_gpu_buffer = clone(CloneParams(*expert_scales_cpu_buffer, AllocationType::DEVICE, BufferHints(), init_params_.enable_comm_overlap));

        expert_contexts.emplace_back(DeepEPLowLatencyExpertContext(i, token_num, expert_all_token_hidden_states, expert_token_hidden_states, expert_ids_gpu_buffer, expert_scales_gpu_buffer, expert_ids_cpu_buffer, expert_scales_cpu_buffer));
    }
    cudaDeviceSynchronize();

    // call moeffn one by one
    for (auto& context : expert_contexts) {
        if (context.token_num == 0) {
            continue;
        }
        auto moe_ffn_params =
            FfnLayerParams(*context.hidden_states, params.configs, params.weights, params.residual, params.qscheme);
        context.out_hidden_states = moeFfn(moe_ffn_params, {context.expert_ids, context.expert_scales}).hidden_states;
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    // copy expert output buffer to output buffer
    auto out_hidden_states = allocateBufferLike(*hidden_states, AllocationType::DEVICE);
    for (auto& context : expert_contexts) {
        if (context.token_num == 0) {
            continue;
        }
        //FT_LOG_INFO("start copy expert %d, token num %d", start_expert_id + context.index, context.token_num);
        auto expert_out_hidden_states = out_hidden_states->index(context.index);
        auto expert_token_hidden_states = expert_out_hidden_states->slice(0, context.token_num);
        copy({*expert_token_hidden_states, *context.out_hidden_states, init_params_.enable_comm_overlap});
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    // combine with local token expert_ids and expert_scales
    MoeCombineParams combine_params{out_hidden_states, nullptr, params.output, {}, {}, moe_conf, params.input.shape()[0],  init_params_.enable_comm_overlap, nullptr, dispatched_output.deep_ep_ll_output, dispatched_output.expert_ids, dispatched_output.expert_scales};
    return deepEpLLCombine(combine_params);
}
}  // namespace fastertransformer
