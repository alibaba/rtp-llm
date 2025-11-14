#include "rtp_llm/cpp/kernels/eplb/experts_stats_kernels.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/core/torch_utils/TorchEvent.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

#ifdef ENABLE_DEEP_EP

size_t CudaDevice::initDeepEPLLMaxTokenPerRank(const DeviceInitParams& params) {
    size_t max_token = (init_params_.max_generate_batch_size + init_params_.tp_size - 1) / init_params_.tp_size;

    // for speculative decoding, need consider gen_num_per_cycle
    if (init_params_.sp_config.type != SP_TYPE_NONE) {
        max_token = max_token * (init_params_.sp_config.gen_num_per_cycle + 1);
    }

    vector<int> matched_tokens = {16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};

    if (max_token > 128) {
        // padding to multiple of 128
        max_token = ((max_token + 127) / 128) * 128;
        return max_token;
    }

    for (auto t : matched_tokens) {
        if (max_token <= t) {
            max_token = t;
            return max_token;
        }
    }

    return 128;
}

bool CudaDevice::initDeepEPBuffer() {
    auto   nccl_param       = getNcclParam(ParallelMode::DP_AND_TP);
    size_t world_rank       = nccl_param.rank_;
    size_t world_size       = nccl_param.world_size_;
    size_t local_world_size = init_params_.parallelism_config.local_world_size;

    int num_experts    = init_params_.num_experts + init_params_.extra_experts;
    int deep_ep_num_sm = init_params_.moe_config.deep_ep_num_sm > 0 ? init_params_.moe_config.deep_ep_num_sm : 24;

    // TODO: check if get right
    ll_num_max_token_per_rank = initDeepEPLLMaxTokenPerRank(init_params_);

    int64_t num_nvl_bytes    = 0;
    int64_t num_rdma_bytes   = 0;
    int     num_qps_per_rank = 1;
    if (init_params_.use_deepep_low_latency) {  // low-latency mode
        num_rdma_bytes = DeepEPBuffer::getLowLatencyRdmaSizeHint(
            ll_num_max_token_per_rank, init_params_.hidden_size, world_size, num_experts);
        num_qps_per_rank = num_experts / init_params_.ep_size;
    } else if (init_params_.use_deepep_internode) {  // normal-kernel internode
        num_nvl_bytes  = int(2e9);
        num_rdma_bytes = int(1e9);
        // normal ibgda
        if (autil::EnvUtil::getEnv("ACCL_NORMAL_MODE", "IBRC") == "IBGDA") {
            setenv("ACCL_NORMAL_MODE", "IBGDA", 1);
            num_qps_per_rank = std::max(deep_ep_num_sm / 2, (int)(num_experts / init_params_.ep_size));
        }
        // normal ibrc
        else {
            setenv("ACCL_NORMAL_MODE", "IBRC", 1);
            num_qps_per_rank = deep_ep_num_sm / 2;
        }

    } else {
        num_nvl_bytes    = int(2e9);  // normal-kernel intranode
        num_qps_per_rank = 1;
    }

    try {
        RTP_LLM_LOG_INFO("deep ep init with num_rdma_bytes %ld, world_rank %ld, world_size %ld",
                         num_rdma_bytes,
                         world_rank,
                         world_size);
        deepep_buffer_.reset(new DeepEPBuffer(this,
                                              world_rank,
                                              local_world_size,
                                              world_size,
                                              num_nvl_bytes,
                                              num_rdma_bytes,
                                              init_params_.use_deepep_low_latency,
                                              num_qps_per_rank));
        bool success = deepep_buffer_->init();
        if (!success) {
            RTP_LLM_LOG_ERROR("Failed to initialize DeepEPBuffer");
            return false;
        }

        RTP_LLM_LOG_INFO("Set DEEP_EP_NUM_SM to %ld", deep_ep_num_sm);
        deepep_buffer_->setNumSMs(deep_ep_num_sm);
        return true;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Failed to create DeepEPBuffer: %s", e.what());
        return false;
    }
}

MoeDispatchOutput CudaDevice::deepEpDispatch(const MoeDispatchParams& params) {
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
    BufferPtr hidden        = params.input.isQBuffer() ?
                                  dynamic_cast<const QBuffer*>(&(params.input))->qslice(slice_begin, slice_size) :
                                  params.input.slice(slice_begin, slice_size);      // [tp_token_size, hidden_size]
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
        RUNTIME_ASSERT_OP_ARG(
            expert_ids->type() == DataType::TYPE_INT64, "expert_ids must be int64 but got %d", int(expert_ids->type()));
        topk_idx_tensor     = Buffer2torchTensor(expert_ids, false);     //[num_tokens, top_k]
        topk_weights_tensor = Buffer2torchTensor(expert_scales, false);  //[num_tokens, top_k]
    }
    RUNTIME_ASSERT_OP_ARG(
        hidden->type() == DataType::TYPE_BF16
            || (params.qscheme == QScheme::Qfp8PerTokenBlock && hidden->type() == DataType::TYPE_QFP8_E4M3),
        "hidden must be bf16 or fp8 in deepEpDispatch, actual: %d",
        int(hidden->type()));
    BufferPtr                    quantized_hidden;
    torch::Tensor                x;
    std::optional<torch::Tensor> x_scales;

    if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
        quantized_hidden =
            hidden->isQBuffer() ? hidden : quantize({*hidden, DataType::TYPE_QFP8_E4M3, 1, params.qscheme});
        auto kernel_ptr = reinterpret_cast<const QBuffer&>(*quantized_hidden).kernelPtr();
        auto scales_ptr = reinterpret_cast<const QBuffer&>(*quantized_hidden).scalesPtr();
        x        = Buffer2torchTensorWithDstType(kernel_ptr, false, TORCH_FP8_E4M3_TYPE);  // [num_tokens, hidden_size]
        x_scales = Buffer2torchTensorWithDstType(
            scales_ptr, false, dataTypeToTorchType(scales_ptr->type()));  // [num_tokens, hidden_size / 128]
    } else {
        x = Buffer2torchTensorWithDstType(
            hidden, false, dataTypeToTorchType(hidden->type()));  // [num_tokens, hidden_size]
    }

    std::shared_ptr<EventOverlap> dispatch_begin_event;
    if (params.overlapped && params.compute_stream_event) {
        auto casted_event    = dynamic_cast<TorchEvent*>(params.compute_stream_event.get());
        auto event_handle    = deep_ep::EventHandle();
        event_handle.event   = casted_event->event;
        dispatch_begin_event = std::make_shared<EventOverlap>(event_handle);
    } else {
        dispatch_begin_event = deepep_buffer_->capture();
    }

    try {
        auto dispatch_layout_output = deepep_buffer_->getDispatchLayout(
            topk_idx_tensor, expert_num, dispatch_begin_event, true /*async*/, true /*allocate_on_comm_stream*/);

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
                                                        true /*allocate_on_comm_stream*/);

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
        if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
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

MoeCombineOutput CudaDevice::deepEpCombine(const MoeCombineParams& params) {
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

    std::shared_ptr<EventOverlap> combine_begin_event;
    if (params.overlapped && params.compute_stream_event) {
        auto casted_event   = dynamic_cast<TorchEvent*>(params.compute_stream_event.get());
        auto event_handle   = deep_ep::EventHandle();
        event_handle.event  = casted_event->event;
        combine_begin_event = std::make_shared<EventOverlap>(event_handle);
    } else {
        combine_begin_event = deepep_buffer_->capture();
    }

    auto combine_output = deepep_buffer_->combine(input_tensor,
                                                  dispatch_output->handle.value(),
                                                  dispatch_output->recv_topk_weights,
                                                  combine_config,
                                                  move(combine_begin_event),
                                                  true /*async_finish*/,
                                                  true /*allocate_on_comm_stream*/);

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

#else

bool CudaDevice::initDeepEPBuffer() {
    return false;
}

MoeDispatchOutput CudaDevice::deepEpDispatch(const MoeDispatchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MoeCombineOutput CudaDevice::deepEpCombine(const MoeCombineParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

#endif

}  // namespace rtp_llm
