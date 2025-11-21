#pragma once

#include <torch/extension.h>
#include <torch/all.h>
#include "event_hip.hpp"

namespace rtp_llm {

/**
 * A wrapper class to manage CUDA events, also for better overlapping convenience.
 * Attributes:
 *   event: the CUDA event captured.
 *   extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
 */
// TODO(wangyin.yx): this event overlap should be removed.
class EventOverlap {

public:
    EventOverlap(const std::optional<deep_ep::EventHandle>& event): event_(event), extra_tensors_(std::nullopt) {}

    EventOverlap(const std::optional<deep_ep::EventHandle>&       event,
                 const std::vector<std::optional<torch::Tensor>>& extra_tensors):
        event_(event), extra_tensors_(extra_tensors) {}

    ~EventOverlap() {
    }

public:
    void currentStreamWait() {
        RTP_LLM_CHECK(event_.has_value());
        event_.value().current_stream_wait();
    }

    std::optional<deep_ep::EventHandle>& event() {
        return event_;
    }

private:
    std::optional<deep_ep::EventHandle>                      event_;
    std::optional<std::vector<std::optional<torch::Tensor>>> extra_tensors_;
};

struct DeepEPDispatchLayoutOutput {
    torch::Tensor                 num_tokens_per_rank;
    std::optional<torch::Tensor>  num_tokens_per_rdma_rank;
    torch::Tensor                 num_tokens_per_expert;
    torch::Tensor                 is_token_in_rank;
    std::shared_ptr<EventOverlap> event_overlap;

    DeepEPDispatchLayoutOutput(const torch::Tensor&                 num_tokens_per_rank,
                               const std::optional<torch::Tensor>&  num_tokens_per_rdma_rank,
                               const torch::Tensor&                 num_tokens_per_expert,
                               const torch::Tensor&                 is_token_in_rank,
                               const std::shared_ptr<EventOverlap>& event_overlap):
        num_tokens_per_rank(num_tokens_per_rank),
        num_tokens_per_rdma_rank(num_tokens_per_rdma_rank),
        num_tokens_per_expert(num_tokens_per_expert),
        is_token_in_rank(is_token_in_rank),
        event_overlap(event_overlap) {}
};

struct DeepEPDispatchHandleIntra {
    torch::Tensor rank_prefix_matrix;
    torch::Tensor channel_prefix_matrix;
    torch::Tensor recv_channel_prefix_matrix;
    torch::Tensor recv_src_idx;
    torch::Tensor is_token_in_rank;
    torch::Tensor send_head;

    DeepEPDispatchHandleIntra(const torch::Tensor& rank_prefix_matrix,
                              const torch::Tensor& channel_prefix_matrix,
                              const torch::Tensor& recv_channel_prefix_matrix,
                              const torch::Tensor& recv_src_idx,
                              const torch::Tensor& is_token_in_rank,
                              const torch::Tensor& send_head):
        rank_prefix_matrix(rank_prefix_matrix),
        channel_prefix_matrix(channel_prefix_matrix),
        recv_channel_prefix_matrix(recv_channel_prefix_matrix),
        recv_src_idx(recv_src_idx),
        is_token_in_rank(is_token_in_rank),
        send_head(send_head) {}
};

struct DeepEPDispatchHandleInter {
    torch::Tensor                is_token_in_rank;
    torch::Tensor                rdma_channel_prefix_matrix;
    torch::Tensor                gbl_channel_prefix_matrix;
    std::optional<torch::Tensor> recv_rdma_channel_prefix_matrix;
    torch::Tensor                recv_rdma_rank_prefix_sum;
    std::optional<torch::Tensor> recv_gbl_channel_prefix_matrix;
    torch::Tensor                recv_gbl_rank_prefix_sum;
    std::optional<torch::Tensor> recv_src_meta;
    std::optional<torch::Tensor> send_rdma_head;
    std::optional<torch::Tensor> send_nvl_head;
    DeepEPDispatchHandleInter(const torch::Tensor&                is_token_in_rank,
                              const torch::Tensor&                rdma_channel_prefix_matrix,
                              const torch::Tensor&                gbl_channel_prefix_matrix,
                              const std::optional<torch::Tensor>& recv_rdma_channel_prefix_matrix,
                              const torch::Tensor&                recv_rdma_rank_prefix_sum,
                              const std::optional<torch::Tensor>& recv_gbl_channel_prefix_matrix,
                              const torch::Tensor&                recv_gbl_rank_prefix_sum,
                              const std::optional<torch::Tensor>& recv_src_meta,
                              const std::optional<torch::Tensor>& send_rdma_head,
                              const std::optional<torch::Tensor>& send_nvl_head):
        is_token_in_rank(is_token_in_rank),
        rdma_channel_prefix_matrix(rdma_channel_prefix_matrix),
        gbl_channel_prefix_matrix(gbl_channel_prefix_matrix),
        recv_rdma_channel_prefix_matrix(recv_rdma_channel_prefix_matrix),
        recv_rdma_rank_prefix_sum(recv_rdma_rank_prefix_sum),
        recv_gbl_channel_prefix_matrix(recv_gbl_channel_prefix_matrix),
        recv_gbl_rank_prefix_sum(recv_gbl_rank_prefix_sum),
        recv_src_meta(recv_src_meta),
        send_rdma_head(send_rdma_head),
        send_nvl_head(send_nvl_head) {}
};

struct DeepEPDispatchHandle {
    std::optional<DeepEPDispatchHandleIntra> intra_handle;
    std::optional<DeepEPDispatchHandleInter> inter_handle;
};

struct DeepEPDispatchOutput {
    // received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the received token
    // count.
    torch::Tensor                recv_x;
    std::optional<torch::Tensor> recv_x_scales;
    // received expert indices.
    std::optional<torch::Tensor> recv_topk_idx;
    // received expert weights.
    std::optional<torch::Tensor> recv_topk_weights;
    // list shaped `[num_local_experts]`, the received token count by each local expert, aligned to the input
    // `expert_alignment`.
    std::optional<std::vector<int>> num_recv_tokens_per_expert_list;
    // the returned communication handle.
    std::optional<DeepEPDispatchHandle> handle;
    //  the event after executing the kernel (valid only if `async_finish` is set).
    std::shared_ptr<EventOverlap> event_overlap;

    DeepEPDispatchOutput(const torch::Tensor&                       recv_x,
                         const std::optional<torch::Tensor>&        recv_x_scales,
                         const std::optional<torch::Tensor>&        recv_topk_idx,
                         const std::optional<torch::Tensor>&        recv_topk_weights,
                         const std::optional<std::vector<int>>&     num_recv_tokens_per_expert_list,
                         const std::optional<DeepEPDispatchHandle>& handle,
                         const std::shared_ptr<EventOverlap>&       event_overlap):
        recv_x(recv_x),
        recv_x_scales(recv_x_scales),
        recv_topk_idx(recv_topk_idx),
        recv_topk_weights(recv_topk_weights),
        num_recv_tokens_per_expert_list(num_recv_tokens_per_expert_list),
        handle(handle),
        event_overlap(event_overlap) {}
};

struct DeepEPCombineOutput {
    // the reduced token from its dispatched ranks.
    torch::Tensor recv_x;
    // the reduced top-k weights from its dispatch ranks.
    std::optional<torch::Tensor> recv_topk_weights = std::nullopt;
    // the event after executing the kernel (valid only if `async_finish` is set).
    std::shared_ptr<EventOverlap> event_overlap;

    DeepEPCombineOutput(const torch::Tensor&                 recv_x,
                        const std::optional<torch::Tensor>&  recv_topk_weights,
                        const std::shared_ptr<EventOverlap>& event_overlap):
        recv_x(recv_x), recv_topk_weights(recv_topk_weights), event_overlap(event_overlap) {}
};

struct DeepEPDispatchHandleLowLatency {
    torch::Tensor packed_recv_src_info;
    torch::Tensor packed_recv_layout_range;
    int           num_max_dispatch_tokens_per_rank;
    int           num_experts;

    DeepEPDispatchHandleLowLatency(const torch::Tensor& packed_recv_src_info,
                                   const torch::Tensor& packed_recv_layout_range,
                                   int                  num_max_dispatch_tokens_per_rank,
                                   int                  num_experts):
        packed_recv_src_info(packed_recv_src_info),
        packed_recv_layout_range(packed_recv_layout_range),
        num_max_dispatch_tokens_per_rank(num_max_dispatch_tokens_per_rank),
        num_experts(num_experts) {}
};

struct DeepEPDispatchOutputLowLatency {
    // a tensor or tuple with received tokens for each expert.
    // if use_fp8 == true, shaped as `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with
    // `torch.float8_e4m3fn`. else shaped as `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`
    // with `torch.bfloat16`. ! Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank *
    // num_ranks` are, as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if
    // synced).
    // TODO: check which count will be ok
    torch::Tensor packed_recv_x;
    // the corresponding scales for the first element with shape
    // if use_fp8 == true, shaped as `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]`
    // with `torch.float`.
    std::optional<torch::Tensor> packed_recv_x_scales;
    // a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each expert receive. As
    // mentioned before, all not tokens are valid in `recv_x`.
    torch::Tensor packed_recv_count;
    // the communication handle to be used in the `low_latency_combine` function.
    DeepEPDispatchHandleLowLatency handle;
    // the event after executing the kernel (valid only if `async_finish` is set).
    std::shared_ptr<EventOverlap> event_overlap;
    // the receiving hook function (valid only if `return_recv_hook` is set).
    std::optional<std::function<void()>> hook;

    DeepEPDispatchOutputLowLatency(const torch::Tensor&                  packed_recv_x,
                                   const std::optional<torch::Tensor>&   packed_recv_x_scales,
                                   const torch::Tensor&                  packed_recv_count,
                                   const DeepEPDispatchHandleLowLatency& handle,
                                   const std::shared_ptr<EventOverlap>&  event_overlap,
                                   std::optional<std::function<void()>>  hook):
        packed_recv_x(packed_recv_x),
        packed_recv_x_scales(packed_recv_x_scales),
        packed_recv_count(packed_recv_count),
        handle(handle),
        event_overlap(event_overlap),
        hook(hook) {}
};

struct DeepEPCombineOutputLowLatency {
    // the reduced token tensor, with shape `[num_combined_tokens, num_topk]` and type `torch.bfloat16`.
    torch::Tensor combined_x;
    // the event after executing the kernel (valid only if `async_finish` is set).
    std::shared_ptr<EventOverlap> event_overlap;
    // the receiving hook function (valid only if `return_recv_hook` is set).
    std::optional<std::function<void()>> hook;

    DeepEPCombineOutputLowLatency(torch::Tensor                        combined_x,
                                  std::shared_ptr<EventOverlap>        event_overlap,
                                  std::optional<std::function<void()>> hook):
        combined_x(combined_x), event_overlap(event_overlap), hook(hook) {}
};

struct DeepEPLowLatencyExpertContext {
    uint32_t index{0};
    uint64_t token_num{0};
    BufferPtr all_hidden_states;
    BufferPtr hidden_states;
    BufferPtr expert_ids;
    BufferPtr expert_scales;
    BufferPtr expert_ids_cpu_buffer;
    BufferPtr expert_scales_cpu_buffer;
    BufferPtr out_hidden_states;
    DeepEPLowLatencyExpertContext(uint32_t index) : index(index) {}
    DeepEPLowLatencyExpertContext(uint32_t index, uint64_t token_num, const BufferPtr& all_hidden_states, const BufferPtr& hidden_states, const BufferPtr& expert_ids, const BufferPtr& expert_scales, const BufferPtr& expert_ids_cpu_buffer, const BufferPtr& expert_scales_cpu_buffer): index(index), token_num(token_num), all_hidden_states(all_hidden_states), hidden_states(hidden_states), expert_ids(expert_ids), expert_scales(expert_scales), expert_ids_cpu_buffer(expert_ids_cpu_buffer), expert_scales_cpu_buffer(expert_scales_cpu_buffer) {}
};

class DeepEPRecvHook : public DeviceHook {
public:
    DeepEPRecvHook(
        const std::function<void()>& hook,
        const std::function<void()>&& stats_hook,
        const std::vector<BufferPtr>& hold_buffers,
        const std::vector<torch::Tensor>& hold_tensors)
    : hook_(hook)
    , stats_hook_(stats_hook)
    , hold_buffers_(hold_buffers)
    , hold_tensors_(hold_tensors)
    , synchronized_(false)
    {};

    ~DeepEPRecvHook() override {
        RTP_LLM_CHECK(synchronized_);
    };

    void hook_sync() const override {
        RTP_LLM_CHECK(!synchronized_);
        if (hook_) {
            hook_();
            stats_hook_();
            synchronized_ = true;
        }
    }

private:
    std::function<void()> hook_;
    std::function<void()> stats_hook_;
    std::vector<BufferPtr> hold_buffers_;
    std::vector<torch::Tensor> hold_tensors_;
    mutable bool synchronized_;
};

class DeepEPCudaEventHook : public DeviceHook {
public:
    DeepEPCudaEventHook(
        at::hip::HIPStreamMasqueradingAsCUDA main_stream,
        deep_ep::EventHandle event_handle,
        const std::vector<BufferPtr>& hold_buffers = {},
        const std::vector<torch::Tensor>& hold_tensors = {},
        std::optional<DeepEPDispatchHandle> dispatch_handle = std::nullopt
    )
    : main_stream_(main_stream)
    , event_handle_(event_handle)
    , hold_buffers_(hold_buffers)
    , hold_tensors_(hold_tensors)
    , dispatch_handle_(dispatch_handle)
    , synchronized_(false)
    {
        RTP_LLM_CHECK(bool(event_handle_.event));
    };

    ~DeepEPCudaEventHook() override {
        RTP_LLM_CHECK(synchronized_);
    };

    void hook_sync() const override {
        RTP_LLM_CHECK(!synchronized_);
        main_stream_.unwrap().wait(*event_handle_.event);
        synchronized_ = true;
    }

private:
    at::hip::HIPStreamMasqueradingAsCUDA main_stream_;
    deep_ep::EventHandle event_handle_;
    std::vector<BufferPtr> hold_buffers_;
    std::vector<torch::Tensor> hold_tensors_;
    std::optional<DeepEPDispatchHandle> dispatch_handle_;
    mutable bool synchronized_;
};

}  // namespace rtp_llm
