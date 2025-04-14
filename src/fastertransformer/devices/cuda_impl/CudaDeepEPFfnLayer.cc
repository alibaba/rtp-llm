#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cuda/Dispatch.h"

using namespace std;

namespace fastertransformer {

#ifdef ENABLE_DEEP_EP

bool CudaDevice::initDeepEPBuffer() {
    auto   nccl_param = getNcclParam(ParallelMode::EP);
    size_t world_rank = nccl_param.rank_;
    size_t world_size = nccl_param.world_size_;

    // TODO: check if get right
    ll_num_max_token_per_rank = (init_params_.max_generate_batch_size + init_params_.tp_size - 1) / init_params_.tp_size;
    int64_t num_rdma_bytes = 0;
    if(init_params_.use_deepep_low_latency) { // low-latency mode
        num_rdma_bytes = DeepEPBuffer::getLowLatencyRdmaSizeHint(
            ll_num_max_token_per_rank, init_params_.hidden_size, world_size, init_params_.num_experts);
    }
    else if (init_params_.use_deepep_internode) { // normal-kernel internode
        num_rdma_bytes = int(1e9);
    }
    else{
        num_rdma_bytes = 0; // normal-kernel intranode
    }
    deepep_buffer_.reset(new DeepEPBuffer(this,
                                          world_rank,
                                          world_size,
                                          int(1e9),
                                          num_rdma_bytes,
                                          init_params_.use_deepep_low_latency,
                                          init_params_.num_experts / init_params_.ep_size));
    return deepep_buffer_->init();
}

MoeDispatchOutput CudaDevice::deepEpDispatch(const MoeDispatchParams& params) {
    const auto& moe_conf   = params.moe_configs;
    auto const  ep_size    = moe_conf.ep_size;
    auto const  tp_size    = moe_conf.tp_size;
    auto const  expert_num = moe_conf.expert_num;
    size_t      token_num  = params.expert_ids.shape()[0];

    FT_CHECK(ep_size == tp_size * moe_conf.dp_size);

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
    RUNTIME_ASSERT_OP_ARG(hidden->type() == DataType::TYPE_BF16, "moe configs not set");
    BufferPtr quantized_hidden;
    torch::Tensor x;
    std::optional<torch::Tensor> x_scales;
    if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
        quantized_hidden = quantize({*hidden, DataType::TYPE_QFP8_E4M3, 1, params.qscheme});
        auto kernel_ptr = reinterpret_cast<const QBuffer&>(*quantized_hidden).kernelPtr();
        auto scales_ptr = reinterpret_cast<const QBuffer&>(*quantized_hidden).scalesPtr();
        x = Buffer2torchTensor(kernel_ptr, false); // [num_tokens, hidden_size]
        x_scales = Buffer2torchTensor(scales_ptr, false); // [num_tokens, hidden_size / 128]
    } else {
        x = Buffer2torchTensor(hidden, false);  // [num_tokens, hidden_size]
    }

    // call dispatch and force sync, maybe sync is not necessary
    auto dispatch_layout_output = deepep_buffer_->getDispatchLayout(
        topk_idx_tensor, expert_num, nullptr /*previous_event*/, false /*async*/, false /*allocate_on_comm_stream*/);

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
                                                    nullptr /*previous_event*/,
                                                    false /*async_finish*/,
                                                    false /*allocate_on_comm_stream*/);

    BufferPtr recv_topk_idx_buffer = torchTensor2BufferWithDstType(dispatch_output.recv_topk_idx.value(), torch::kInt32);
    BufferPtr recv_topk_weights_buffer = torchTensor2Buffer(dispatch_output.recv_topk_weights.value());
    const size_t num_experts_per_node = expert_num / moe_conf.ep_size;
    tensorrt_llm::kernels::genSourceRowRevert(recv_topk_idx_buffer->data<int>(), recv_topk_idx_buffer->shape()[0], recv_topk_idx_buffer->shape()[1], num_experts_per_node * moe_conf.ep_rank, stream_);
    BufferPtr recv_x_buffer;
    if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
        recv_x_buffer.reset(new QBuffer(std::move(torchTensor2Buffer(dispatch_output.recv_x)),
                                        std::move(torchTensor2Buffer(dispatch_output.recv_x_scales.value())),
                                        std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                                                       DataType::TYPE_INVALID,
                                                                       {0},
                                                                       nullptr)))));
    } else  {
        recv_x_buffer = torchTensor2Buffer(dispatch_output.recv_x);
    }
    MoeDispatchOutput out{recv_x_buffer, recv_topk_idx_buffer, recv_topk_weights_buffer, nullptr};
    out.deep_ep_output.reset(new DeepEPDispatchOutput(std::move(dispatch_output)));
    return out;
}

MoeCombineOutput CudaDevice::deepEpCombine(const MoeCombineParams& params) {
    FT_CHECK(params.deep_ep_output != nullptr && params.deep_ep_output->handle.has_value());

    torch::Tensor input_tensor;
    if (params.input->shape()[0] == 0) {
        input_tensor = torch::empty({0, static_cast<long int>(params.input->shape()[1])},
                                    torch::dtype(torch::kBFloat16).device(torch::Device(torch::kCUDA)));
    } else {
        input_tensor = Buffer2torchTensor(params.input, false).toType(torch::kBFloat16);  //[num_tokens, hidden_size]
    }

    auto  combine_config  = deepep_buffer_->getCombineConfig();
    auto& dispatch_output = params.deep_ep_output;

    auto combine_output = deepep_buffer_->combine(input_tensor,
                                                  dispatch_output->handle.value(),
                                                  dispatch_output->recv_topk_weights,
                                                  combine_config,
                                                  nullptr /*previous_event*/,
                                                  false /*async_finish*/,
                                                  false /*allocate_on_comm_stream*/);
    // wait combine kernel done, no need, will sync wait on next stream op
    BufferPtr all_output;
    if (params.output != nullptr) {
        all_output = torchTensor2BufferWithDstType(combine_output.recv_x, dataTypeToTorchType(params.output->type()));
    } else {
        all_output = torchTensor2BufferWithDstType(combine_output.recv_x, dataTypeToTorchType(params.input->type()));
    }
    printBufferData(*all_output, "all_output");
    return MoeCombineOutput({all_output, all_output, params});
}

FfnLayerOutput CudaDevice::deepEpMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) {
    const auto& moe_conf = params.configs.moe_configs.value();

    MoeDispatchOutput dispatched_output =
        deepEpDispatch({params.input, *gate_output.expert_ids, *gate_output.expert_scales, moe_conf, false, params.qscheme});
    auto moe_ffn_params = FfnLayerParams(
            {*dispatched_output.hidden, params.configs, params.weights, params.residual, params.qscheme});
    BufferPtr hidden_states =
        moeFfn(moe_ffn_params, {dispatched_output.expert_ids, dispatched_output.expert_scales}).hidden_states;
    auto combine_out = deepEpCombine({hidden_states,
                                      dispatched_output.indices,
                                      // nullptr,
                                      params.output,
                                      dispatched_output.input_split_sizes,
                                      dispatched_output.output_split_sizes,
                                      moe_conf,
                                      params.input.shape()[0],
                                      init_params_.enable_comm_overlap,
                                      dispatched_output.deep_ep_output});
    return gatherCombineOutput(combine_out);
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

FfnLayerOutput CudaDevice::deepEpMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

#endif

}  // namespace fastertransformer
