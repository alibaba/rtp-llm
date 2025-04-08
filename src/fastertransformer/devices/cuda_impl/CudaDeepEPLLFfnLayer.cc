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

#ifdef ENABLE_DEEP_EP

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
    // cudaDeviceSynchronize();

    auto dispatch_output = deepep_buffer_->lowLatencyDispatch(input_tensor,
                                                              topk_idx_tensor,
                                                              ll_num_max_token_per_rank,
                                                              expert_num,
                                                              true /*use_fp8*/,
                                                              false /*async_finish*/,
                                                              params.overlapped /*return_recv_hook*/
    );
    // cudaDeviceSynchronize();

    BufferPtr packed_recv_x_buffer = torchTensor2BufferWithDstType(dispatch_output.packed_recv_x, torch::kHalf);

    // TODO: deep_ep_output might should be removed from output objects.
    MoeDispatchOutput out{packed_recv_x_buffer, expert_ids, expert_scales};
    out.deep_ep_ll_output.reset(new DeepEPDispatchOutputLowLatency(dispatch_output));

    if (params.overlapped) {
        RUNTIME_ASSERT_OP_ARG(dispatch_output.hook.has_value(), "recv hook is null when overlapped");
        std::vector<BufferPtr> hold_buffers;
        std::vector<torch::Tensor> hold_tensors;
        hold_tensors.push_back(dispatch_output.packed_recv_x);
        if (dispatch_output.packed_recv_x_scales.has_value()) {
            hold_tensors.push_back(dispatch_output.packed_recv_x_scales.value());
        }
        out.comm_barrier_hook = std::make_unique<DeepEPRecvHook>(
            dispatch_output.hook.value(), hold_buffers, hold_tensors);
    }

    return out;
}

MoeCombineOutput CudaDevice::deepEpLLCombine(const MoeCombineParams& params) {
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
    // cudaDeviceSynchronize();

    auto combine_output = deepep_buffer_->lowLatencyCombine(input_tensor,
                                                            topk_idx_tensor,
                                                            topk_weights_tensor,
                                                            params.deep_ep_ll_output->handle,
                                                            false,
                                                            params.overlapped);
    // cudaDeviceSynchronize();

    BufferPtr all_output;
    auto output_type = params.output ? params.output->type() : params.input->type();
    all_output = torchTensor2BufferWithDstType(combine_output.combined_x, dataTypeToTorchType(output_type));

    return MoeCombineOutput({all_output, all_output, params});
}

FfnLayerOutput CudaDevice::deepEpLLMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) {
    const auto&       moe_conf = params.configs.moe_configs.value();
    MoeDispatchOutput dispatched_output =
        deepEpLLDispatch({params.input, *gate_output.expert_ids, *gate_output.expert_scales, moe_conf});

    cudaDeviceSynchronize();

    auto moe_ffn_params =
        FfnLayerParams(*dispatched_output.hidden, params.configs, params.weights, params.residual, params.qscheme);
    auto out_hidden_states = deepEpFfnFp8(moe_ffn_params, dispatched_output).hidden_states;
    cudaDeviceSynchronize();

    // combine with local token expert_ids and expert_scales
    MoeCombineParams combine_params{out_hidden_states, nullptr, params.output, {}, {}, moe_conf, params.input.shape()[0],  init_params_.enable_comm_overlap, nullptr, dispatched_output.deep_ep_ll_output, dispatched_output.expert_ids, dispatched_output.expert_scales};
    auto combine_out = deepEpLLCombine(combine_params);
    return gatherCombineOutput(combine_out);
}

#else

MoeDispatchOutput CudaDevice::deepEpLLDispatch(const MoeDispatchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MoeCombineOutput CudaDevice::deepEpLLCombine(const MoeCombineParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput CudaDevice::deepEpLLMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

#endif

}  // namespace fastertransformer
