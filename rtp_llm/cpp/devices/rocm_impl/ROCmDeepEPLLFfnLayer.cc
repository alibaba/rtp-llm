#include "rtp_llm/cpp/kernels/rocm/experts_stats_kernels.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"

#include "quant.h"

using namespace std;

namespace rtp_llm {

#ifdef ENABLE_DEEP_EP

MoeDispatchOutput ROCmDevice::deepEpLLDispatch(const MoeDispatchParams& params) {
    const auto& moe_conf   = params.moe_configs;
    auto const  tp_size    = moe_conf.tp_size;
    auto const  expert_num = moe_conf.expert_num + moe_conf.extra_expert_num;
    auto const  use_fp8    = params.qscheme == QScheme::Qfp8PerTokenBlock;

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
    try {
        auto dispatch_output = deepep_buffer_->lowLatencyDispatch(input_tensor,
                                                                topk_idx_tensor,
                                                                ll_num_max_token_per_rank,
                                                                expert_num,
                                                                use_fp8 /*use_fp8*/,
                                                                false /*async_finish*/,
                                                                params.overlapped /*return_recv_hook*/
        );

        BufferPtr packed_recv_x_buffer = torchTensor2Buffer(dispatch_output.packed_recv_x);
        auto expert_stats = params.expert_stats;
        auto ep_rank = moe_conf.ep_rank;
        auto packed_recv_count = dispatch_output.packed_recv_count;

        auto stats_func = [this, expert_stats, packed_recv_count, ep_rank]() {
            if (expert_stats.has_value()) {
                auto& expert_stats_v = expert_stats.value();
                int*  gpu_loads    = expert_stats_v.getLayerGpuLoads();
                launch_update_gpu_loads_ll(packed_recv_count.data_ptr<int>(),
                                           gpu_loads,
                                           packed_recv_count.size(0),
                                           ep_rank,
                                           stream_);
            }
        };

        // TODO: deep_ep_output might should be removed from output objects.
        MoeDispatchOutput out{packed_recv_x_buffer, expert_ids, expert_scales};
        out.deep_ep_ll_output.reset(new DeepEPDispatchOutputLowLatency(dispatch_output));

        if (params.overlapped) {
            RUNTIME_ASSERT_OP_ARG(dispatch_output.hook.has_value(), "recv hook is null when overlapped");
            out.comm_barrier_hook = std::make_unique<DeepEPRecvHook>(
                dispatch_output.hook.value(), std::move(stats_func), std::vector<BufferPtr>(), std::vector<torch::Tensor>());
        } else {
            stats_func();
        }
        return out;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("DeepEP ll dispatch failed: %s", e.what());
        fflush(stdout);
        fflush(stderr);
        throw OpException({OpErrorType::ERROR_INTERNAL, "dispatch failed " + std::string(e.what())});
    }
}

MoeCombineOutput ROCmDevice::deepEpLLCombine(const MoeCombineParams& params) { 
    RTP_LLM_CHECK(params.deep_ep_ll_output != nullptr);

    auto& expert_ids    = params.expert_ids;
    auto& expert_scales = params.expert_scales;

    torch::Tensor input_tensor =
        Buffer2torchTensorWithDstType(*params.input, false, torch::kBFloat16);  // [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]
    torch::Tensor topk_idx_tensor = Buffer2torchTensorWithDstType(*expert_ids, false, c10::kLong);
    torch::Tensor topk_weights_tensor = Buffer2torchTensorWithDstType(*expert_scales, false, c10::kFloat);

    auto combine_output = deepep_buffer_->lowLatencyCombine(input_tensor,
                                                            topk_idx_tensor,
                                                            topk_weights_tensor,
                                                            params.deep_ep_ll_output->handle,
                                                            false,
                                                            params.overlapped);
                                                            // false);

    BufferPtr all_output;
    const auto output_type = params.output ? params.output->type() : params.input->type();
    const auto combined_type = torchDTypeToDataType(combine_output.combined_x.dtype());

    all_output = torchTensor2BufferWithDstType(combine_output.combined_x, dataTypeToTorchType(output_type));

    DeviceHookPtr comm_hook;
    if (params.overlapped) {
        RUNTIME_ASSERT_OP_ARG(combine_output.hook.has_value(), "recv hook is null when overlapped");
        RUNTIME_ASSERT_OP_ARG(combined_type == output_type, "combined output type %d not equal expected output type %d when overlapped", combined_type, output_type);
        auto empty_func = []() {};
        comm_hook = std::make_unique<DeepEPRecvHook>(
            combine_output.hook.value(), std::move(empty_func), std::vector<BufferPtr>(), std::vector<torch::Tensor>());
    }
    return MoeCombineOutput({all_output, all_output, params, move(comm_hook)});
}

#else

MoeDispatchOutput ROCmDevice::deepEpLLDispatch(const MoeDispatchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MoeCombineOutput ROCmDevice::deepEpLLCombine(const MoeCombineParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

#endif

}  // namespace rtp_llm
