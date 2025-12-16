#include "rtp_llm/cpp/kernels/eplb/experts_stats_kernels.h"
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

bool ROCmDevice::initDeepEPBuffer() {
    auto   nccl_param  = getNcclParam(ParallelMode::DP_AND_TP);
    size_t world_rank  = nccl_param.rank_;
    size_t world_size  = nccl_param.world_size_;
    int    num_experts = init_params_.num_experts + init_params_.extra_experts;

    // TODO: check if get right
    ll_num_max_token_per_rank =
        (init_params_.max_generate_batch_size + init_params_.tp_size - 1) / init_params_.tp_size;
    int64_t num_rdma_bytes   = 0;
    int     num_qps_per_rank = 1;
    if (init_params_.use_deepep_low_latency) {  // low-latency mode
        num_rdma_bytes = DeepEPBuffer::getLowLatencyRdmaSizeHint(
            ll_num_max_token_per_rank, init_params_.hidden_size, world_size, num_experts);
        num_qps_per_rank = num_experts / init_params_.ep_size;
    } else if (init_params_.use_deepep_internode) {  // normal-kernel internode
        num_rdma_bytes   = int(1e9);
        num_qps_per_rank = std::max(12, (int)(num_experts / init_params_.ep_size));
    } else {
        num_rdma_bytes   = 0;  // normal-kernel intranode
        num_qps_per_rank = 1;
    }

    try {
        RTP_LLM_LOG_INFO("deep ep init with num_rdma_bytes %ld, world_rank %ld, world_size %ld",
                         num_rdma_bytes,
                         world_rank,
                         world_size);
#if USE_ACCL_EP
        num_qps_per_rank = num_experts / init_params_.ep_size;
        deepep_buffer_.reset(new DeepEPBuffer(this,
                                              world_rank,
                                              world_size,
                                              int(1e9),
                                              num_rdma_bytes,
                                              init_params_.use_deepep_low_latency,
                                              num_qps_per_rank));
#else
        int64_t num_nvl_bytes = init_params_.use_deepep_low_latency ? 0 : 1e9;
        deepep_buffer_.reset(new DeepEPBuffer(this,
                                              world_rank,
                                              world_size,
                                              num_nvl_bytes,
                                              num_rdma_bytes,
                                              init_params_.use_deepep_low_latency || init_params_.use_deepep_internode,
                                              num_qps_per_rank));
#endif
        bool success = deepep_buffer_->init();
        if (!success) {
            RTP_LLM_LOG_ERROR("Failed to initialize DeepEPBuffer");
            return false;
        }
        auto deep_ep_num_sm = autil::EnvUtil::getEnv("DEEP_EP_NUM_SM", 0UL);
        if (deep_ep_num_sm) {
            RTP_LLM_LOG_INFO("Set DEEP_EP_NUM_SM to %ld", deep_ep_num_sm);
            deepep_buffer_->setNumSMs(deep_ep_num_sm);
        } else {
            RTP_LLM_LOG_INFO("Deep EP use default sm num");
        }
        return true;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Failed to create DeepEPBuffer: %s", e.what());
        return false;
    }
}

MoeDispatchOutput ROCmDevice::deepEpDispatch(const MoeDispatchParams& params) {
    const auto& moe_conf   = params.moe_configs;
    auto const  ep_size    = moe_conf.ep_size;
    auto const  tp_size    = moe_conf.tp_size;
    auto const  expert_num = moe_conf.expert_num + moe_conf.extra_expert_num;
    size_t      token_num  = params.expert_ids.shape()[0];

    RTP_LLM_CHECK(ep_size == tp_size * moe_conf.dp_size);

    // slice tokens for this rank process
    size_t    tp_token_size = (token_num + tp_size - 1) / tp_size;
    size_t    slice_begin   = std::min(tp_token_size * moe_conf.tp_rank, token_num);
    size_t    slice_size    = std::min(token_num - slice_begin, tp_token_size);
    BufferPtr hidden        = params.input.slice(slice_begin, slice_size);          // [tp_token_size, hidden_size]
    auto      expert_ids    = params.expert_ids.slice(slice_begin, slice_size);     // [tp_token_size, topk]
    auto      expert_scales = params.expert_scales.slice(slice_begin, slice_size);  // [tp_token_size, topk]

    // prepare tensors for call deepep dispatch layout
    torch::Tensor topk_idx_tensor;
    torch::Tensor topk_weights_tensor;

    if (expert_ids->shape()[0] == 0) {
        topk_idx_tensor =
            torch::empty({0, static_cast<long int>(expert_ids->shape()[1])},
                         torch::dtype(torch::kInt64).device(torch::Device(torch::kCUDA)));  //[num_tokens, top_k]
        topk_weights_tensor =
            torch::empty({0, static_cast<long int>(expert_ids->shape()[1])},
                         torch::dtype(torch::kFloat).device(torch::Device(torch::kCUDA)));  //[num_tokens, top_k]
    } else {
        topk_idx_tensor     = Buffer2torchTensor(expert_ids, false).toType(torch::kInt64);  //[num_tokens, top_k]
        topk_weights_tensor = Buffer2torchTensor(expert_scales, false);                     //[num_tokens, top_k]
    }
    RUNTIME_ASSERT_OP_ARG(hidden->type() == DataType::TYPE_BF16,
                          "hidden must be bf16 in deepEpDispatch, actual: %d",
                          int(hidden->type()));
    BufferPtr                    quantized_hidden;
    QBufferPtr                   q_hidden;
    torch::Tensor                x;
    std::optional<torch::Tensor> x_scales;
    BufferPtr                    hidden_quant, hidden_quant_scale;
    if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
        const size_t  model_dim     = hidden->shape()[1];
        const int     block_scale_k = 128;
        torch::Tensor hidden_tensor = Buffer2torchTensor(hidden, false);
        hidden_quant = allocateBuffer({DataType::TYPE_FP8_E4M3, {token_num, model_dim}}, {"rocm_hidden_quant"});
        hidden_quant_scale =
            allocateBuffer({DataType::TYPE_FP32, {token_num, model_dim / block_scale_k}}, {"rocm_hidden_quant_scale"});
        x = Buffer2torchTensorWithDstType(*hidden_quant, false, dataTypeToTorchType(hidden_quant->type()));
        x_scales =
            Buffer2torchTensorWithDstType(*hidden_quant_scale, false, dataTypeToTorchType(hidden_quant_scale->type()));

        hidden_tensor = hidden_tensor.view({(int)token_num, (int)model_dim / block_scale_k, block_scale_k});
        x             = x.view({(int)token_num, (int)model_dim / block_scale_k, block_scale_k});

        // invoke aiter quant kernel
        aiter::dynamic_per_token_scaled_quant(
            /*out=*/x,
            /*input=*/hidden_tensor,
            /*scales=*/x_scales.value(),
            /*scale_ub=*/nullopt);
    } else if (params.qscheme == QScheme::Qfp8PerToken) {
        q_hidden = std::dynamic_pointer_cast<QBuffer>(
            quantize(QuantizeParams(*hidden, DataType::TYPE_QFP8_E4M3, 1, params.qscheme, 128, 0)));
        x = Buffer2torchTensorWithDstType(q_hidden->kernel(), false, dataTypeToTorchType(q_hidden->kernel().type()));
        x_scales =
            Buffer2torchTensorWithDstType(q_hidden->scales(), false, dataTypeToTorchType(q_hidden->scales().type()));
    } else {
        x = Buffer2torchTensorWithDstType(
            hidden, false, dataTypeToTorchType(hidden->type()));  // [num_tokens, hidden_size]
    }

    const auto dispatch_begin_event = deepep_buffer_->capture();

    try {
        // call dispatch and force sync, maybe sync is not necessary
        // RTP_LLM_LOG_INFO("get dispatch layout expert num %ld, expert_ids: %s, expert_scales: %s, hidden: %s",
        //             expert_num,
        //             expert_ids->debugString().c_str(),
        //             expert_scales ? expert_scales->debugString().c_str() : "null",
        //             hidden->debugString().c_str());
        auto dispatch_layout_output = deepep_buffer_->getDispatchLayout(
            topk_idx_tensor, expert_num, dispatch_begin_event, true /*async*/, false /*allocate_on_comm_stream*/);

        deep_ep::Config dispatch_config = deepep_buffer_->getDispatchConfig();

        auto dispatch_output = deepep_buffer_->dispatch(x,
                                                        x_scales /*x_scales*/,
                                                        std::nullopt /*handle*/,
                                                        dispatch_layout_output.num_tokens_per_rank,
                                                        dispatch_layout_output.num_tokens_per_rdma_rank,
                                                        dispatch_layout_output.is_token_in_rank,
                                                        dispatch_layout_output.num_tokens_per_expert,
                                                        std::optional<torch::Tensor>(topk_idx_tensor),
                                                        std::optional<torch::Tensor>(topk_weights_tensor),
                                                        1 /*expert_alignment*/,
                                                        dispatch_config,
                                                        dispatch_layout_output.event_overlap,
                                                        true /*async_finish*/,
                                                        false /*allocate_on_comm_stream*/);

        DeviceHookPtr comm_hook;
        if (params.overlapped) {
            std::vector<BufferPtr> hold_buffers = {
                hidden,
                quantized_hidden,
                expert_ids,
                expert_scales,
            };
            std::vector<torch::Tensor> hold_tensors = {
                x,
                x_scales.value_or(torch::Tensor()),
                topk_idx_tensor,
                topk_weights_tensor,
                dispatch_layout_output.num_tokens_per_rank,
                dispatch_layout_output.num_tokens_per_rdma_rank.value_or(torch::Tensor()),
                dispatch_layout_output.num_tokens_per_expert,
                dispatch_layout_output.is_token_in_rank,
                dispatch_output.recv_x,
                dispatch_output.recv_x_scales.value_or(torch::Tensor()),
                dispatch_output.recv_topk_idx.value_or(torch::Tensor()),
                dispatch_output.recv_topk_weights.value_or(torch::Tensor()),
            };
            comm_hook = std::make_unique<DeepEPCudaEventHook>(*torch_default_stream_,
                                                              *(dispatch_output.event_overlap->event()),
                                                              hold_buffers,
                                                              hold_tensors,
                                                              dispatch_output.handle);
        } else {
            dispatch_output.event_overlap->currentStreamWait();
        }

        BufferPtr recv_topk_idx_buffer     = torchTensor2Buffer(dispatch_output.recv_topk_idx.value());
        BufferPtr recv_topk_weights_buffer = torchTensor2Buffer(dispatch_output.recv_topk_weights.value());
        BufferPtr recv_x_buffer;
        if (params.qscheme == QScheme::Qfp8PerTokenBlock || params.qscheme == QScheme::Qfp8PerToken) {
            recv_x_buffer.reset(new QBuffer(
                std::move(torchTensor2Buffer(dispatch_output.recv_x)),
                std::move(torchTensor2Buffer(dispatch_output.recv_x_scales.value())),
                std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
        } else {
            recv_x_buffer = torchTensor2Buffer(dispatch_output.recv_x);
        }
        MoeDispatchOutput out{recv_x_buffer, recv_topk_idx_buffer, recv_topk_weights_buffer};
        out.deep_ep_output.reset(new DeepEPDispatchOutput(std::move(dispatch_output)));
        out.comm_barrier_hook = std::move(comm_hook);

        if (params.expert_stats.has_value()) {
            auto& experts_stats = params.expert_stats.value();
            update_gpu_loads_deepep_kernel(recv_topk_idx_buffer->data<int64_t>(),
                                           experts_stats.getLayerGpuLoads(),
                                           recv_topk_idx_buffer->size(),
                                           moe_conf.ep_rank,
                                           stream_);
        }

        return out;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Failed to dispatch: %s", e.what());
        fflush(stdout);
        fflush(stderr);
        throw OpException({OpErrorType::ERROR_INTERNAL, "dispatch failed " + std::string(e.what())});
    }
}

MoeCombineOutput ROCmDevice::deepEpCombine(const MoeCombineParams& params) {
    RTP_LLM_CHECK(params.deep_ep_output != nullptr && params.deep_ep_output->handle.has_value());

    torch::Tensor input_tensor;
    if (params.input->shape()[0] == 0) {
        input_tensor = torch::empty({0, static_cast<long int>(params.input->shape()[1])},
                                    torch::dtype(torch::kBFloat16).device(torch::Device(torch::kCUDA)));
    } else {
        input_tensor = Buffer2torchTensor(params.input, false).toType(torch::kBFloat16);  //[num_tokens, hidden_size]
    }

    auto  combine_config  = deepep_buffer_->getCombineConfig();
    auto& dispatch_output = params.deep_ep_output;

    auto compute_event  = deepep_buffer_->capture();
    auto combine_output = deepep_buffer_->combine(input_tensor,
                                                  dispatch_output->handle.value(),
                                                  dispatch_output->recv_topk_weights,
                                                  combine_config,
                                                  compute_event,
                                                  true /*async_finish*/,
                                                  false /*allocate_on_comm_stream*/);
    syncAndCheck();
    RUNTIME_ASSERT_OP_ARG(combine_output.event_overlap, "combine overlap should always exist.");

    BufferPtr  all_output;
    const auto output_type   = params.output ? params.output->type() : params.input->type();
    const auto combined_type = torchDTypeToDataType(combine_output.recv_x.dtype());

    DeviceHookPtr comm_hook;

    if (params.overlapped) {
        RUNTIME_ASSERT_OP_ARG(combined_type == output_type,
                              "combined output type %d not equal expected output type %d when overlapped",
                              combined_type,
                              output_type);
        std::vector<BufferPtr> hold_buffers = {
            params.input,
        };
        std::vector<torch::Tensor> hold_tensors = {
            input_tensor,
            combine_output.recv_x,
        };
        if (dispatch_output->recv_topk_weights) {
            hold_tensors.push_back(dispatch_output->recv_topk_weights.value());
        }
        if (combine_output.recv_topk_weights) {
            hold_tensors.push_back(combine_output.recv_topk_weights.value());
        }
        comm_hook = std::make_unique<DeepEPCudaEventHook>(*torch_default_stream_,
                                                          *(combine_output.event_overlap->event()),
                                                          hold_buffers,
                                                          hold_tensors,
                                                          dispatch_output->handle);
    } else {
        torch_default_stream_->unwrap().wait(*(combine_output.event_overlap->event().value().event));
        combine_output.event_overlap->currentStreamWait();
    }

    all_output = torchTensor2BufferWithDstType(combine_output.recv_x, dataTypeToTorchType(output_type));
    printBufferData(*all_output, "all_output");
    return MoeCombineOutput({all_output, all_output, params, move(comm_hook)});
}

#endif

}  // namespace rtp_llm
