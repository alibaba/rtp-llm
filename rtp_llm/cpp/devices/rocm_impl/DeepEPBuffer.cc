#include "rtp_llm/cpp/devices/rocm_impl/DeepEPBuffer.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

using namespace std;

namespace rtp_llm {

/**
 * @brief Initialize the communication buffer.
 */
bool DeepEPBuffer::init() {

    try {
        buffer_ = std::make_unique<deep_ep::Buffer>(world_rank_, world_size_, num_nvl_bytes_, num_rdma_bytes_, low_latency_mode_);
    } catch (const std::bad_alloc& e) {
        RTP_LLM_LOG_ERROR("Failed to allocate memory for deep_ep::Buffer: ", e.what());
        throw std::runtime_error("DeepEPBuffer initialization failed: memory allocation failed");
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Failed to create deep_ep::Buffer: ", e.what());
        throw std::runtime_error(std::string("DeepEPBuffer initialization failed: ") + e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("Failed to create deep_ep::Buffer: unknown error");
        throw std::runtime_error("DeepEPBuffer initialization failed: unknown error");
    }

    int              local_device_id = buffer_->get_local_device_id();
    std::vector<int> device_ids      = allGatherDeviceIds(local_device_id);

    std::string              local_ipc_handle = buffer_->get_local_ipc_handle_string();
    std::vector<std::string> ipc_handles      = allGatherIpcHandles(local_ipc_handle);

    std::string root_unique_id;
    if (buffer_->get_num_rdma_ranks() > 1 || low_latency_mode_) {
        // low latency set env
#if USE_ACCL_EP
        if (low_latency_mode_) {
            setLowLatencyEnv();
        }
#else
        setLowLatencyEnv();
#endif

        root_unique_id = getRootUniqueId();
    }
    buffer_->sync_string(device_ids, ipc_handles, root_unique_id);
// #if USE_ACCL_EP
    if (buffer_->is_low_latency_optimize()) {
        RTP_LLM_LOG_INFO("aclcep low latency optimized, start get pxn handle");
        std::string              local_pxn_ipc_handle = buffer_->get_local_pxn_ipc_handle_string();
        std::vector<std::string> pxn_ipc_handles      = allGatherIpcHandles(local_pxn_ipc_handle);
        buffer_->sync_pxn_handles_string(device_ids, pxn_ipc_handles);
    }
// #endif
    return true;
}

void DeepEPBuffer::setLowLatencyEnv() {
    RTP_LLM_CHECK(num_qps_per_rank_ > 0);
    setenv("NVSHMEM_DISABLE_P2P", "1", 1);
    setenv("NVSHMEM_IB_ENABLE_IBGDA", "1", 1);
    setenv("NVSHMEM_IBGDA_NIC_HANDLER", "gpu", 1);
    std::string num_qps_per_rank_str = std::to_string(num_qps_per_rank_);
    setenv("NVSHMEM_IBGDA_NUM_RC_PER_PE", num_qps_per_rank_str.c_str(), 1);
    RTP_LLM_LOG_DEBUG("num_qps_per_rank is set to %s", num_qps_per_rank_str.c_str());
    //! Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
    setenv("NVSHMEM_QP_DEPTH", "1024", 1);
    //! NVSHMEM initialization requires at least 256 MiB
    std::string nvshmem_cumem_granularity_str = std::to_string(1 << 29);  // 2^29 = 536870912 (512 MiB)
    setenv("NVSHMEM_CUMEM_GRANULARITY", nvshmem_cumem_granularity_str.c_str(), 1);
}

std::vector<int> DeepEPBuffer::allGatherDeviceIds(const int local_device_id) {
    std::vector<int> device_ids(world_size_, 0);
    device_ids[world_rank_] = local_device_id;

    BufferPtr device_ids_buffer_cpu = vector2Buffer(device_ids);
    BufferPtr device_ids_buffer_gpu = device_->clone({*device_ids_buffer_cpu, AllocationType::DEVICE});
    device_->allGather({{device_ids_buffer_gpu}, ParallelMode::DP_AND_TP});
    device_->copy({*device_ids_buffer_cpu, *device_ids_buffer_gpu});

    device_ids = buffer2vector<int>(*device_ids_buffer_cpu);
    return device_ids;
}

std::vector<std::string> DeepEPBuffer::allGatherIpcHandles(const std::string& local_ipc_handle) {
    std::vector<std::string> ipc_handles(world_size_, "");
    assert(local_ipc_handle.size() == HIP_IPC_HANDLE_SIZE);

    // local ipc handle to buffer
    std::vector<int8_t> local_ipc_handle_vec(local_ipc_handle.begin(), local_ipc_handle.end());
    BufferPtr           local_ipc_handle_buffer_cpu = vector2Buffer(local_ipc_handle_vec);
    local_ipc_handle_buffer_cpu->reshape({1, HIP_IPC_HANDLE_SIZE});

    // init all_ipc_handle_buffer with local ip handle
    BufferPtr all_ipc_handle_buffer_gpu =
        device_->allocateBuffer({DataType::TYPE_INT8, {world_size_, HIP_IPC_HANDLE_SIZE}, AllocationType::DEVICE});
    device_->copy({all_ipc_handle_buffer_gpu->view(world_rank_, 1), *local_ipc_handle_buffer_cpu});

    // do all gather and result to cpu
    device_->allGather({{all_ipc_handle_buffer_gpu}, ParallelMode::DP_AND_TP});

    BufferPtr all_ipc_handles_buffer_cpu = device_->clone({*all_ipc_handle_buffer_gpu, AllocationType::HOST});

    // transfer to string
    for (int i = 0; i < world_size_; i++) {
        auto offset = i * HIP_IPC_HANDLE_SIZE;
        auto data   = all_ipc_handles_buffer_cpu->dataWithOffset<int8_t>(offset);
        ipc_handles[i].assign((char*)(data), HIP_IPC_HANDLE_SIZE);
    }
    return ipc_handles;
}

std::string DeepEPBuffer::getRootUniqueId() {
    std::string root_unique_id;
    auto        NVSHMEM_UNIQUE_ID_SIZE = sizeof(nvshmemx_uniqueid_t);
    BufferPtr   all_nvshmem_unique_ids_buffer_gpu =
        device_->allocateBuffer({DataType::TYPE_INT8, {world_size_, NVSHMEM_UNIQUE_ID_SIZE}, AllocationType::DEVICE});
    if ((low_latency_mode_ && world_rank_ == 0) || (!low_latency_mode_ && buffer_->get_rdma_rank() == 0)) {
        auto local_nvshmem_unique_id = buffer_->get_local_nvshmem_unique_id_string();
        RTP_LLM_CHECK(local_nvshmem_unique_id.size() == NVSHMEM_UNIQUE_ID_SIZE);
        std::vector<int8_t> local_nvshmem_unique_id_vec(local_nvshmem_unique_id.begin(), local_nvshmem_unique_id.end());
        BufferPtr           local_nvshmem_unique_id_buffer_cpu = vector2Buffer(local_nvshmem_unique_id_vec);
        local_nvshmem_unique_id_buffer_cpu->reshape({1, NVSHMEM_UNIQUE_ID_SIZE});
        device_->copy({all_nvshmem_unique_ids_buffer_gpu->view(world_rank_, 1), *local_nvshmem_unique_id_buffer_cpu});
    }

    device_->allGather({{all_nvshmem_unique_ids_buffer_gpu}, ParallelMode::DP_AND_TP});

    BufferPtr all_nvshmem_unique_ids_buffer_cpu =
        device_->clone({*all_nvshmem_unique_ids_buffer_gpu, AllocationType::HOST});
    if (low_latency_mode_) {
        auto data = all_nvshmem_unique_ids_buffer_cpu->dataWithOffset<int8_t>(0);
        root_unique_id.assign((char*)data, NVSHMEM_UNIQUE_ID_SIZE);
    } else {
        auto data = all_nvshmem_unique_ids_buffer_cpu->dataWithOffset<int8_t>(buffer_->get_root_rdma_rank(true)
                                                                              * NVSHMEM_UNIQUE_ID_SIZE);
        root_unique_id.assign((char*)(data), NVSHMEM_UNIQUE_ID_SIZE);
    }
    return root_unique_id;
}

void DeepEPBuffer::setNumSMs(size_t new_num_sms) {
    RTP_LLM_LOG_INFO("DeepEP set num sm = %ld", new_num_sms);
    RTP_LLM_CHECK(new_num_sms % 2 == 0);
    num_sms_ = new_num_sms;
}

std::shared_ptr<EventOverlap> DeepEPBuffer::capture() {
    return std::make_shared<EventOverlap>(deep_ep::EventHandle());
}

/**
 * @brief Get the low latency rdma size hint
 * @param num_max_dispatch_tokens_per_rank the maximum number of tokens to dispatch, all the ranks must hold the same
 * value.
 * @param hidden the hidden dimension of each token.
 * @param num_ranks the number of EP group ranks.
 * @param num_experts the number of all experts.
 * @return the low latency rdma size hint.
 */
size_t DeepEPBuffer::getLowLatencyRdmaSizeHint(int num_max_dispatch_tokens_per_rank,
                                               int hidden,
                                               int num_ranks,
                                               int num_experts) {
    return deep_ep::get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
}

/**
 * @brief Get a recommended dispatch config.
 * @param num_ranks: the number of ranks.
 * @return config: the recommended config.
 */
deep_ep::Config DeepEPBuffer::getDispatchConfig(int num_ranks) {
    if (num_ranks == 0) {
        num_ranks = world_size_;
    }
    switch (num_ranks) {
        case 2:
            return deep_ep::Config(num_sms_, 16, 256, 6, 128);
        case 4:
            return deep_ep::Config(num_sms_, 16, 256, 6, 128);
        case 8:
            return deep_ep::Config(num_sms_, 6, 256, 6, 128);
        case 16:
            return deep_ep::Config(num_sms_, 16, 288, 20, 128);
        case 24:
            return deep_ep::Config(num_sms_, 8, 288, 32, 128);
        case 32:
            return deep_ep::Config(num_sms_, 8, 288, 32, 128);
        case 64:
            return deep_ep::Config(num_sms_, 20, 288, 28, 128);
        case 128:
            return deep_ep::Config(num_sms_, 20, 560, 32, 128);
        case 144:
            return deep_ep::Config(num_sms_, 32, 720, 12, 128);
        case 160:
            return deep_ep::Config(num_sms_, 28, 720, 12, 128);
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "num_ranks is not supported");
    }
    return deep_ep::Config(num_sms_, 2, 288, 28, 128);
}

/**
 * @brief Get a recommended combine config.
 * @param num_ranks: the number of ranks.
 * @return config: the recommended config.
 */
deep_ep::Config DeepEPBuffer::getCombineConfig(int num_ranks) {
    if (num_ranks == 0) {
        num_ranks = world_size_;
    }
    switch (num_ranks) {
        case 2:
            return deep_ep::Config(num_sms_, 6, 256, 6, 128);
        case 4:
            return deep_ep::Config(num_sms_, 6, 256, 6, 128);
        case 8:
            return deep_ep::Config(num_sms_, 6, 256, 6, 128);
        case 16:
            return deep_ep::Config(num_sms_, 2, 288, 28, 128);
        case 24:
            return deep_ep::Config(num_sms_, 1, 288, 20, 128);
        case 32:
            return deep_ep::Config(num_sms_, 1, 288, 20, 128);
        case 64:
            return deep_ep::Config(num_sms_, 1, 288, 20, 128);
        case 128:
            return deep_ep::Config(num_sms_, 1, 560, 12, 128);
        case 144:
            return deep_ep::Config(num_sms_, 2, 720, 8, 128);
        case 160:
            return deep_ep::Config(num_sms_, 2, 720, 8, 128);
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "num_ranks is not supported");
    }
    return deep_ep::Config(num_sms_, 2, 288, 28, 128);
}

/**
 * @brief Calculate the layout required for later communication.
 * @param topk_idx Tensor of shape [num_tokens, num_topk] with scalar_type torch.int64. The expert indices selected by
 * each token (-1 means no selection).
 * @param num_experts The number of experts.
 * @param previous_event the event to wait before actually executing the kernel.
 * @param async_finish the current stream will not wait for the communication kernels to be finished if set.
 * @param allocate_on_comm_stream control whether all the allocated tensors' ownership to be on the communication
 * stream.
 * @return tuple containing:
 * - num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
 * - num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA rank
 * (with the same GPU index), return `None` for intranode settings.
 * - num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
 * - is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
 * - event: the event after executing the kernel (valid only if `async_finish` is set).
 */
DeepEPDispatchLayoutOutput DeepEPBuffer::getDispatchLayout(const torch::Tensor&                 topk_idx,
                                                           int                                  num_experts,
                                                           const std::shared_ptr<EventOverlap>& previous_event,
                                                           bool                                 async,
                                                           bool allocate_on_comm_stream) {
    RTP_LLM_CHECK(topk_idx.scalar_type() == c10::kLong);

    std::optional<deep_ep::EventHandle> event = previous_event != nullptr ? previous_event->event() : std::nullopt;

    auto [num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, recv_event] =
        buffer_->get_dispatch_layout(topk_idx, num_experts, event, async, allocate_on_comm_stream);

    return DeepEPDispatchLayoutOutput(num_tokens_per_rank,
                                      num_tokens_per_rdma_rank,
                                      num_tokens_per_expert,
                                      is_token_in_rank,
                                      make_shared<EventOverlap>(recv_event));
}

/**
 * @brief Dispatch tokens to different ranks, both intranode and internode settings are supported.
 * @details Intranode kernels require all the ranks should be visible via NVLink.
 *          Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same
 * GPU index should be visible via RDMA.
 *
 * @param x input hidden state
    if not has x_scales , the shape must be `[num_tokens, hidden]`, and type must be `torch.bfloat16`;
    else, must be shaped as `[num_tokens, hidden]` with type `torch.float8_e4m3fn`
 * @param x_scales if has must be `[num_tokens, hidden // 128]` (requiring divisible) with type `torch.float`.
 * @param handle an optional communication handle, if set, the CPU will reuse the layout information to save some time.
 * @param num_tokens_per_rank `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
 * @param num_tokens_per_rdma_rank [num_rdma_ranks] tensor (int) for RDMA-enabled ranks (None in intranode)
 * @param is_token_in_rank [num_tokens, num_ranks] bool tensor marking token distribution
 * @param num_tokens_per_expert [num_experts] tensor (int) showing tokens per expert
 * @param topk_idx [num_tokens, num_topk] tensor (int64) with expert indices (-1 indicates no selection)
 * @param topk_weights [num_tokens, num_topk] tensor (float) with expert weights
 * @param expert_alignment align the number of tokens received by each local expert to this variable.
 * @param config the performance tuning config.
 * @param previous_event the event to wait before actually executing the kernel.
 * @param async_finish the current stream will not wait for the communication kernels to be finished if set.
 * @param allocate_on_comm_stream control whether all the allocated tensors' ownership to be on the communication
 stream.
 *
 * @return tuple containing:
 *     - recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
 received token count.
 *     - recv_topk_idx: received expert indices.
 *     - recv_topk_weights: received expert weights.
 *     - num_recv_tokens_per_expert_list:  Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`.
 *     - handle: the returned communication handle.
 *     - event: the event after executing the kernel (valid only if `async_finish` is set).
 */
DeepEPDispatchOutput DeepEPBuffer::dispatch(const torch::Tensor&                       x,
                                            const std::optional<torch::Tensor>&        x_scales,
                                            const std::optional<DeepEPDispatchHandle>& handle,
                                            const std::optional<torch::Tensor>&        num_tokens_per_rank,
                                            const std::optional<torch::Tensor>&        num_tokens_per_rdma_rank,
                                            const std::optional<torch::Tensor>&        is_token_in_rank,
                                            const std::optional<torch::Tensor>&        num_tokens_per_expert,
                                            const std::optional<torch::Tensor>&        topk_idx,
                                            const std::optional<torch::Tensor>&        topk_weights,
                                            int                                        expert_alignment,
                                            std::optional<deep_ep::Config>             config,
                                            const std::shared_ptr<EventOverlap>&       previous_event,
                                            bool                                       async_finish,
                                            bool                                       allocate_on_comm_stream) {
    if (x_scales.has_value()) {
        RTP_LLM_CHECK(x.scalar_type() == c10::kFloat8_e4m3fn && x.sizes().size() == 2);  // [num_tokens, hidden // 128]
        RTP_LLM_CHECK(x_scales->scalar_type() == c10::kFloat);
    } else {
        RTP_LLM_CHECK(x.scalar_type() == c10::kBFloat16);  // [num_tokens, hidden]
    }
    RTP_LLM_CHECK(num_tokens_per_rank.has_value() == false || num_tokens_per_expert->scalar_type() == c10::kInt);
    RTP_LLM_CHECK(num_tokens_per_rdma_rank.has_value() == false || num_tokens_per_rdma_rank->scalar_type() == c10::kInt);
    RTP_LLM_CHECK(is_token_in_rank.has_value() == false || is_token_in_rank->scalar_type() == c10::kBool);
    RTP_LLM_CHECK(num_tokens_per_expert.has_value() == false || num_tokens_per_expert->scalar_type() == c10::kInt);
    RTP_LLM_CHECK(topk_idx.has_value() == false || topk_idx->scalar_type() == c10::kLong);
    RTP_LLM_CHECK(topk_weights.has_value() == false || topk_weights->scalar_type() == c10::kFloat);

    if (config.has_value() == false) {
        config = getDispatchConfig(world_size_);
    }

    if (buffer_->get_num_rdma_ranks() > 1) {
        std::optional<DeepEPDispatchHandleInter> inter_handle =
            handle.has_value() ? handle->inter_handle : std::nullopt;
        return internodeDispatch(x,
                                 x_scales,
                                 inter_handle,
                                 num_tokens_per_rank,
                                 num_tokens_per_rdma_rank,
                                 is_token_in_rank,
                                 num_tokens_per_expert,
                                 topk_idx,
                                 topk_weights,
                                 expert_alignment,
                                 config.value(),
                                 previous_event,
                                 async_finish,
                                 allocate_on_comm_stream);
    }
    std::optional<DeepEPDispatchHandleIntra> intra_handle = handle.has_value() ? handle->intra_handle : std::nullopt;
    return intranodeDispatch(x,
                             x_scales,
                             intra_handle,
                             num_tokens_per_rank,
                             is_token_in_rank,
                             num_tokens_per_expert,
                             topk_idx,
                             topk_weights,
                             expert_alignment,
                             config.value(),
                             previous_event,
                             async_finish,
                             allocate_on_comm_stream);
}

DeepEPDispatchOutput DeepEPBuffer::intranodeDispatch(const torch::Tensor&                            x,
                                                     const std::optional<torch::Tensor>&             x_scales,
                                                     const std::optional<DeepEPDispatchHandleIntra>& handle,
                                                     const std::optional<torch::Tensor>&  num_tokens_per_rank,
                                                     const std::optional<torch::Tensor>&  is_token_in_rank,
                                                     const std::optional<torch::Tensor>&  num_tokens_per_expert,
                                                     const std::optional<torch::Tensor>&  topk_idx,
                                                     const std::optional<torch::Tensor>&  topk_weights,
                                                     int                                  expert_alignment,
                                                     const deep_ep::Config&               config,
                                                     const std::shared_ptr<EventOverlap>& previous_event,
                                                     bool                                 async_finish,
                                                     bool                                 allocate_on_comm_stream) {
    std::optional<deep_ep::EventHandle> event = previous_event != nullptr ? previous_event->event() : std::nullopt;

    if (handle.has_value()) {
        // Launch the kernel with cached
        RTP_LLM_CHECK(topk_idx.has_value());
        RTP_LLM_CHECK(topk_weights.has_value());

        auto num_recv_tokens = handle->recv_src_idx.size(0);
        auto [recv_x,
              recv_x_scales,
              recv_topk_idx,
              recv_topk_weights,
              num_recv_tokens_per_expert_list,
              rank_prefix_matrix,
              channel_prefix_matrix,
              recv_channel_prefix_matrix,
              recv_src_idx,
              send_head,
              recv_event]    = buffer_->intranode_dispatch(x,
                                                        x_scales,
                                                        std::nullopt,
                                                        std::nullopt,
                                                        std::nullopt,
                                                        handle->is_token_in_rank,
                                                        std::nullopt,
                                                        num_recv_tokens,
                                                        handle->rank_prefix_matrix,
                                                        handle->channel_prefix_matrix,
                                                        expert_alignment,
                                                        config,
                                                        event,
                                                        async_finish,
                                                        allocate_on_comm_stream);
        return {recv_x,
                recv_x_scales,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                make_shared<EventOverlap>(recv_event)};
    } else {
        // launch the kernel without cached
        RTP_LLM_CHECK(num_tokens_per_rank.has_value());
        RTP_LLM_CHECK(num_tokens_per_expert.has_value());
        RTP_LLM_CHECK(is_token_in_rank.has_value());
        auto [recv_x,
              recv_x_scales,
              recv_topk_idx,
              recv_topk_weights,
              num_recv_tokens_per_expert_list,
              rank_prefix_matrix,
              channel_prefix_matrix,
              recv_channel_prefix_matrix,
              recv_src_idx,
              send_head,
              recv_event] = buffer_->intranode_dispatch(x,
                                                        x_scales,
                                                        topk_idx,
                                                        topk_weights,
                                                        num_tokens_per_rank,
                                                        is_token_in_rank.value(),
                                                        num_tokens_per_expert,
                                                        0,
                                                        std::nullopt,
                                                        std::nullopt,
                                                        expert_alignment,
                                                        config,
                                                        event,
                                                        async_finish,
                                                        allocate_on_comm_stream);
        DeepEPDispatchHandle handle;
        handle.intra_handle = DeepEPDispatchHandleIntra(rank_prefix_matrix,
                                                        channel_prefix_matrix,
                                                        recv_channel_prefix_matrix,
                                                        recv_src_idx,
                                                        is_token_in_rank.value(),
                                                        send_head);
        return {recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                make_shared<EventOverlap>(recv_event)};
    }
}

// Internode dispatch implementation, for more details, please refer to the `dispatch` docs. Normally, you should not
// directly call this function.
DeepEPDispatchOutput DeepEPBuffer::internodeDispatch(const torch::Tensor&                            x,
                                                     const std::optional<torch::Tensor>&             x_scales,
                                                     const std::optional<DeepEPDispatchHandleInter>& handle,
                                                     const std::optional<torch::Tensor>&  num_tokens_per_rank,
                                                     const std::optional<torch::Tensor>&  num_tokens_per_rdma_rank,
                                                     const std::optional<torch::Tensor>&  is_token_in_rank,
                                                     const std::optional<torch::Tensor>&  num_tokens_per_expert,
                                                     const std::optional<torch::Tensor>&  topk_idx,
                                                     const std::optional<torch::Tensor>&  topk_weights,
                                                     int                                  expert_alignment,
                                                     const deep_ep::Config&               config,
                                                     const std::shared_ptr<EventOverlap>& previous_event,
                                                     bool                                 async_finish,
                                                     bool                                 allocate_on_comm_stream) {
    std::optional<deep_ep::EventHandle> event = previous_event != nullptr ? previous_event->event() : std::nullopt;
    if (handle.has_value()) {
        RTP_LLM_CHECK(topk_idx.has_value());
        RTP_LLM_CHECK(topk_weights.has_value());
        RTP_LLM_CHECK(handle->recv_src_meta.has_value());
        RTP_LLM_CHECK(handle->send_nvl_head.has_value());

        auto num_recv_tokens      = handle->recv_src_meta->size(0);
        auto num_rdma_recv_tokens = handle->send_nvl_head->size(0);

        auto [recv_x,
              recv_x_scales,
              recv_topk_idx,
              recv_topk_weights,
              num_recv_tokens_per_expert_list,
              rdma_channel_prefix_matrix,
              gbl_channel_prefix_matrix,
              recv_rdma_channel_prefix_matrix,
              recv_rdma_rank_prefix_sum,
              recv_gbl_channel_prefix_matrix,
              recv_gbl_rank_prefix_sum,
              recv_src_meta,
              send_rdma_head,
              send_nvl_head,
              recv_event] = buffer_->internode_dispatch(x,
                                                        x_scales,
                                                        topk_idx,
                                                        topk_weights,
                                                        std::nullopt,  // num_tokens_per_rank
                                                        std::nullopt,  // num_tokens_per_rdma_rank
                                                        handle->is_token_in_rank,
                                                        std::nullopt,  // num_tokens_per_expert
                                                        num_recv_tokens,
                                                        num_rdma_recv_tokens,
                                                        handle->rdma_channel_prefix_matrix,
                                                        handle->recv_rdma_rank_prefix_sum,
                                                        handle->gbl_channel_prefix_matrix,
                                                        handle->recv_gbl_rank_prefix_sum,
                                                        expert_alignment,
                                                        config,
                                                        event,
                                                        async_finish,
                                                        allocate_on_comm_stream);
        return {recv_x,
                recv_x_scales,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                make_shared<EventOverlap>(recv_event)};
    } else {
        RTP_LLM_CHECK(num_tokens_per_rank.has_value());
        RTP_LLM_CHECK(num_tokens_per_expert.has_value());
        RTP_LLM_CHECK(is_token_in_rank.has_value());

        auto [recv_x,
              recv_x_scales,
              recv_topk_idx,
              recv_topk_weights,
              num_recv_tokens_per_expert_list,
              rdma_channel_prefix_matrix,
              gbl_channel_prefix_matrix,
              recv_rdma_channel_prefix_matrix,
              recv_rdma_rank_prefix_sum,
              recv_gbl_channel_prefix_matrix,
              recv_gbl_rank_prefix_sum,
              recv_src_meta,
              send_rdma_head,
              send_nvl_head,
              recv_event] = buffer_->internode_dispatch(x,
                                                        x_scales,
                                                        topk_idx,
                                                        topk_weights,
                                                        num_tokens_per_rank,
                                                        num_tokens_per_rdma_rank,
                                                        is_token_in_rank.value(),
                                                        num_tokens_per_expert,
                                                        0,
                                                        0,
                                                        std::nullopt /*rdma_channel_prefix_matrix*/,
                                                        std::nullopt /*recv_rdma_rank_prefix_sum*/,
                                                        std::nullopt /*gbl_channel_prefix_matrix*/,
                                                        std::nullopt /*recv_gbl_rank_prefix_sum*/,
                                                        expert_alignment,
                                                        config,
                                                        event,
                                                        async_finish,
                                                        allocate_on_comm_stream);

        DeepEPDispatchHandle handle;
        handle.inter_handle = DeepEPDispatchHandleInter(is_token_in_rank.value(),
                                                        rdma_channel_prefix_matrix,
                                                        gbl_channel_prefix_matrix,
                                                        recv_rdma_channel_prefix_matrix,
                                                        recv_rdma_rank_prefix_sum,
                                                        recv_gbl_channel_prefix_matrix,
                                                        recv_gbl_rank_prefix_sum,
                                                        recv_src_meta,
                                                        send_rdma_head,
                                                        send_nvl_head);

        return {recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                make_shared<EventOverlap>(recv_event)};
    }
}

/**
 * @brief Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
 * settings are supported.
 * @details
 * Intranode kernels require all ranks to be NVLink-visible.
 * Internode kernels require NVLink visibility within nodes and RDMA connectivity for same-GPU-index ranks across nodes.
 *
 * @param x `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
 * @param handle: a must-set communication handle, you can obtain this from the dispatch function.
 * @param topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its
 original ranks.
 * @param config: the performance tuning config.
 * @param previous_event the event to wait before actually executing the kernel.
 * @param async_finish the current stream will not wait for the communication kernels to be finished if set.
 * @param allocate_on_comm_stream control whether all the allocated tensors' ownership to be on the communication
 stream.
 *
 * @return tuple containing:
 * - recv_x: the reduced token from its dispatched ranks.
 * - recv_topk_weights: the reduced top-k weights from its dispatch ranks.
 * - event: the event after executing the kernel (valid only if `async_finish` is set).
 */
DeepEPCombineOutput DeepEPBuffer::combine(const torch::Tensor&                 x,
                                          const DeepEPDispatchHandle&          handle,
                                          const std::optional<torch::Tensor>&  topk_weights,
                                          std::optional<deep_ep::Config>       config,
                                          const std::shared_ptr<EventOverlap>& previous_event,
                                          bool                                 async_finish,
                                          bool                                 allocate_on_comm_stream) {
    RTP_LLM_CHECK(x.scalar_type() == c10::kBFloat16);
    RTP_LLM_CHECK(topk_weights.has_value() == false || topk_weights->scalar_type() == c10::kFloat);

    if (config.has_value() == false) {
        config = getCombineConfig(world_size_);
    }

    if (buffer_->get_num_rdma_ranks() > 1) {
        return internodeCombine(x,
                                handle.inter_handle,
                                topk_weights,
                                config.value(),
                                previous_event,
                                async_finish,
                                allocate_on_comm_stream);
    }
    return intranodeCombine(
        x, handle.intra_handle, topk_weights, config.value(), previous_event, async_finish, allocate_on_comm_stream);
}

DeepEPCombineOutput DeepEPBuffer::intranodeCombine(const torch::Tensor&                            x,
                                                   const std::optional<DeepEPDispatchHandleIntra>& handle,
                                                   const std::optional<torch::Tensor>&             topk_weights,
                                                   const deep_ep::Config&                          config,
                                                   const std::shared_ptr<EventOverlap>&            previous_event,
                                                   bool                                            async_finish,
                                                   bool allocate_on_comm_stream) {
    RTP_LLM_CHECK(handle.has_value());
    std::optional<deep_ep::EventHandle> event    = previous_event != nullptr ? previous_event->event() : std::nullopt;
    auto [recv_x, recv_topk_weights, recv_event] = buffer_->intranode_combine(x,
                                                                              topk_weights,
                                                                              handle->recv_src_idx,
                                                                              handle->rank_prefix_matrix,
                                                                              handle->recv_channel_prefix_matrix,
                                                                              handle->send_head,
                                                                              config,
                                                                              event,
                                                                              async_finish,
                                                                              allocate_on_comm_stream);
    return {recv_x, recv_topk_weights, make_shared<EventOverlap>(recv_event)};
}

// Internode combine implementation, for more details, please refer to the `combine` docs.  Normally, you should not
// directly call this function.
DeepEPCombineOutput DeepEPBuffer::internodeCombine(const torch::Tensor&                            x,
                                                   const std::optional<DeepEPDispatchHandleInter>& handle,
                                                   const std::optional<torch::Tensor>&             topk_weights,
                                                   const deep_ep::Config&                          config,
                                                   const std::shared_ptr<EventOverlap>&            previous_event,
                                                   bool                                            async_finish,
                                                   bool allocate_on_comm_stream) {
    RTP_LLM_CHECK(handle.has_value());

    std::optional<deep_ep::EventHandle> event = previous_event != nullptr ? previous_event->event() : std::nullopt;

    auto [combined_x, combined_topk_weights, recv_event] =
        buffer_->internode_combine(x,
                                   topk_weights,
                                   handle->recv_src_meta.value(),
                                   handle->is_token_in_rank,
                                   handle->recv_rdma_channel_prefix_matrix.value(),
                                   handle->recv_rdma_rank_prefix_sum,
                                   handle->recv_gbl_channel_prefix_matrix.value(),
                                   handle->send_rdma_head.value(),
                                   handle->send_nvl_head.value(),
                                   config,
                                   event,
                                   async_finish,
                                   allocate_on_comm_stream);
    return {combined_x, combined_topk_weights, make_shared<EventOverlap>(recv_event)};
}

/**
 * @brief A low-latency implementation for dispatching with IBGDA.
 * @details
      This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA (specifically,
 IBGDA must be enabled). Even for ranks in the same node, NVLink are fully disabled for simplicity.
 //! Warning: as there are only two buffers, and the returned tensors reuse the buffer, you can not hold more than 2
 //! low-latency kernels' result tensor at a single moment.
 * @param x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
 supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
 * @param topk_idx: `torch.Tensor` with `torch.int64`, shaped as `[num_tokens, num_topk]`, only several top-k shapes are
 supported. `-1` indices (not selecting any expert) are supported.
 * @param num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same
 value.
 * @param num_experts: the number of all experts.
 * @param use_fp8: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling
 factors.
 * @param async_finish: the current stream will not wait for the communication kernels to be finished if set.
 * @param return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues, but
 * without actually receiving the data**. You must call the received hook to make sure the data's arrival. If you not
 set this flag, the kernel will ensure the data's arrival.
 * @return tuple contains:
 * recv_x: a tensor or tuple with received tokens for each expert.
      With `use_fp8=True`:
           the first element is a `torch.Tensor` shaped as `[num_local_experts, num_max_dispatch_tokens_per_rank *
 num_ranks, hidden]` with `torch.float8_e4m3fn`.
      The second tensor is the corresponding scales for the first element with shape `[num_local_experts,
 num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`.
//! Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
      With `use_fp8=False`, the result would be a tensor shaped as `[num_local_experts,
 num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`.
 //! Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are, as we
 //! do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
 * recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each expert
receive. As mentioned before, all not tokens are valid in `recv_x`.
 * handle: the communication handle to be used in the `low_latency_combine` function.
 * event: the event after executing the kernel (valid only if `async_finish` is set).
 * hook: the receiving hook function (valid only if `return_recv_hook` is set).
*/
DeepEPDispatchOutputLowLatency DeepEPBuffer::lowLatencyDispatch(const torch::Tensor& x,
                                                                const torch::Tensor& topk_idx,
                                                                int                  num_max_dispatch_tokens_per_rank,
                                                                int                  num_experts,
                                                                bool                 use_fp8,
                                                                bool                 async_finish,
                                                                bool                 return_recv_hook) {
    // only several hidden shapes are supported 2560 / 5120 / 7168(r1)
    RTP_LLM_CHECK_WITH_INFO(x.scalar_type() == torch::kBFloat16 && x.size(0) <= num_max_dispatch_tokens_per_rank,
                       "x should be bf16, acutal: %d; num_tokens should <= %d, actual: %d in lowLatencyDispatch",
                       (int)x.scalar_type(),
                       num_max_dispatch_tokens_per_rank,
                       (int)x.size(0));

    // only several top-k shapes are supported
    RTP_LLM_CHECK(topk_idx.scalar_type() == torch::kLong);

    auto [packed_recv_x,
          packed_recv_x_scales,
          packed_recv_count,
          packed_recv_src_info,
          packed_recv_layout_range,
          event,
          hook] =
        buffer_->low_latency_dispatch(
            x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, use_fp8, async_finish, return_recv_hook);

    DeepEPDispatchHandleLowLatency handle(
        packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, num_experts);

    std::shared_ptr<EventOverlap> event_overlap;
    if (async_finish) {
        std::vector<std::optional<torch::Tensor>> tensors_to_record{x,
                                                                    topk_idx,
                                                                    packed_recv_x,
                                                                    packed_recv_x_scales,
                                                                    packed_recv_count,
                                                                    packed_recv_src_info,
                                                                    packed_recv_layout_range};
        event_overlap = make_shared<EventOverlap>(event, tensors_to_record);
    }

    if (use_fp8) {
        return {packed_recv_x, packed_recv_x_scales, packed_recv_count, handle, event_overlap, hook};
    }
    return {packed_recv_x, std::nullopt, packed_recv_count, handle, event_overlap, hook};
}

/**
 * @brief A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
 * @details
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA (specifically,
 IBGDA must be enabled). Even for ranks in the same node, NVLink are fully disabled for simplicity. Warning: as there
 are only two buffers, and the returned tensors reuse the buffer, you can not hold more than 2 low-latency kernels'
 result tensor at a single moment.
 * @param x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`, the
 local calculated tokens to be sent to this original rank and reduced.
 * @param topk_idx: `[num_combined_tokens, num_topk]` with `torch.int64`, the expert indices selected by the dispatched
 tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals to the number of
 dispatched tokens.
 * @param topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the
 dispatched tokens. The received tokens will be reduced with the weights in this tensor.
 * @param handle: the communication handle given by the `dispatch` function.
 * @param async_finish: the current stream will not wait for the communication kernels to be finished if set.
 * @param return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues, but
 without actually receiving the data**. You must call the received hook to make sure the data's arrival. If you not set
 this flag, the kernel will ensure the data's arrival.
 * @param out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it
 directly.

 * @return tuple contains:
 * combined_x: the reduced token tensor, with shape `[num_combined_tokens, num_topk]` and type `torch.bfloat16`.
 * event: the event after executing the kernel (valid only if `async_finish` is set).
 * hook: the receiving hook function (valid only if `return_recv_hook` is set).
 */
DeepEPCombineOutputLowLatency DeepEPBuffer::lowLatencyCombine(const torch::Tensor&                  x,
                                                              const torch::Tensor&                  topk_idx,
                                                              const torch::Tensor&                  topk_weights,
                                                              const DeepEPDispatchHandleLowLatency& handle,
                                                              bool                                  async_finish,
                                                              bool                                  return_recv_hook) {
    RTP_LLM_CHECK(x.scalar_type() == c10::kBFloat16);
    RTP_LLM_CHECK(topk_idx.scalar_type() == c10::kLong);
    RTP_LLM_CHECK(topk_weights.scalar_type() == c10::kFloat);

    auto [combined_x, event, hook] = buffer_->low_latency_combine(x,
                                                                  topk_idx,
                                                                  topk_weights,
                                                                  handle.packed_recv_src_info,
                                                                  handle.packed_recv_layout_range,
                                                                  handle.num_max_dispatch_tokens_per_rank,
                                                                  handle.num_experts,
                                                                  false /*zero_copy*/,
                                                                  async_finish,
                                                                  return_recv_hook);

    printTorchTensorData(combined_x, "combine combined_x");
    std::shared_ptr<EventOverlap> event_overlap;
    if (async_finish) {
        std::vector<std::optional<torch::Tensor>> tensors_to_record{
            x, topk_idx, topk_weights, handle.packed_recv_src_info, handle.packed_recv_layout_range, combined_x};
        event_overlap = make_shared<EventOverlap>(event, tensors_to_record);
    }
    return {combined_x, event_overlap, hook};
}
}  // namespace rtp_llm
