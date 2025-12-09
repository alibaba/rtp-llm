#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/rocm_impl/DeepEPDefs.h"

#ifdef ENABLE_DEEP_EP
#include "config_hip.hpp"
#include "deep_ep_hip.hpp"

// c++ version of deepep buffer.py
namespace rtp_llm {

// The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
//   - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
//   - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
//   - low-latency all-to-all (dispatch and combine, using RDMA)
//
class DeepEPBuffer {
public:
    // Initialize the communication buffer.
    // device: device use to call nccl to initialize
    // world_rank: this rank's rank in the world
    // world_size: the world size
    // num_nvl_bytes: the buffer size for intranode NVLink communication.
    // num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
    // low_latency_mode: whether to use low-latency mode.
    // num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals to the number
    // of local experts.
    DeepEPBuffer(DeviceBase* device,
                 int         world_rank,
                 int         world_size,
                 int64_t     num_nvl_bytes    = 0,
                 int64_t     num_rdma_bytes   = 0,
                 bool        low_latency_mode = false,
                 int         num_qps_per_rank = 1):
        device_(device),
        world_rank_(world_rank),
        world_size_(world_size),
        num_nvl_bytes_(num_nvl_bytes),
        num_rdma_bytes_(num_rdma_bytes),
        low_latency_mode_(low_latency_mode),
        num_qps_per_rank_(num_qps_per_rank) {}
    ~DeepEPBuffer() = default;

public:
    bool init();

private:
    void setLowLatencyEnv();

    std::vector<int> allGatherDeviceIds(int local_device_id);

    std::vector<std::string> allGatherIpcHandles(const std::string& local_ipc_handle);

    std::string getRootUniqueId();

public:
    // Set the number of SMs to use in high-throughput kernels.
    void setNumSMs(size_t new_num_sms);

    // Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.
    std::shared_ptr<EventOverlap> capture();

    // Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.
    static size_t
    getLowLatencyRdmaSizeHint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts);

    // Get a recommended dispatch config.
    deep_ep::Config getDispatchConfig(int num_ranks = 0);

    // Get a recommended combine config.
    deep_ep::Config getCombineConfig(int num_ranks = 0);

    // Calculate the layout required for later communication.
    DeepEPDispatchLayoutOutput getDispatchLayout(const torch::Tensor&                 tokp_idx,
                                                 int                                  num_experts,
                                                 const std::shared_ptr<EventOverlap>& previous_event          = nullptr,
                                                 bool                                 async                   = false,
                                                 bool                                 allocate_on_comm_stream = false);

    // Dispatch tokens to different ranks, both intranode and internode settings are supported.
    // Intranode kernels require all the ranks should be visible via NVLink.
    // Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
    // index should be visible via RDMA.
    DeepEPDispatchOutput dispatch(const torch::Tensor&                       x,
                                  const std::optional<torch::Tensor>&        x_scales,
                                  const std::optional<DeepEPDispatchHandle>& handle                   = std::nullopt,
                                  const std::optional<torch::Tensor>&        num_tokens_per_rank      = std::nullopt,
                                  const std::optional<torch::Tensor>&        num_tokens_per_rdma_rank = std::nullopt,
                                  const std::optional<torch::Tensor>&        is_token_in_rank         = std::nullopt,
                                  const std::optional<torch::Tensor>&        num_tokens_per_expert    = std::nullopt,
                                  const std::optional<torch::Tensor>&        topk_idx                 = std::nullopt,
                                  const std::optional<torch::Tensor>&        topk_weights             = std::nullopt,
                                  int                                        expert_alignment         = 1,
                                  std::optional<deep_ep::Config>             config                   = std::nullopt,
                                  const std::shared_ptr<EventOverlap>&       previous_event           = nullptr,
                                  bool                                       async_finish             = false,
                                  bool                                       allocate_on_comm_stream  = false);

    /*
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
       settings are supported. Intranode kernels require all the ranks should be visible via NVLink. Internode kernels
       require the ranks in a node should be visible via NVLink, while the ranks with the same GPU index should be
       visible via RDMA.
    */
    DeepEPCombineOutput combine(const torch::Tensor&                 x,
                                const DeepEPDispatchHandle&          handle,
                                const std::optional<torch::Tensor>&  topk_weights            = std::nullopt,
                                std::optional<deep_ep::Config>       config                  = std::nullopt,
                                const std::shared_ptr<EventOverlap>& previous_event          = nullptr,
                                bool                                 async_finish            = false,
                                bool                                 allocate_on_comm_stream = false);
    /*
      A low-latency implementation for dispatching with IBGDA.
          This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
              (specifically, IBGDA must be enabled).
          Even for ranks in the same node, NVLink are fully disabled for simplicity.
          Warning: as there are only two buffers, and the returned tensors reuse the buffer, you can not hold more than
      2 low-latency kernels' result tensor at a single moment.
      */
    DeepEPDispatchOutputLowLatency lowLatencyDispatch(const torch::Tensor& x,
                                                      const torch::Tensor& topk_idx,
                                                      int                  num_max_dispatch_tokens_per_rank,
                                                      int                  num_experts,
                                                      bool                 use_fp8          = true,
                                                      bool                 async_finish     = false,
                                                      bool                 return_recv_hook = false);

    /*
    A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Even for ranks in the same node, NVLink are fully disabled for simplicity.
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you can not hold more than 2
            low-latency kernels' result tensor at a single moment.
    */
    DeepEPCombineOutputLowLatency lowLatencyCombine(const torch::Tensor&                  x,
                                                    const torch::Tensor&                  topk_idx,
                                                    const torch::Tensor&                  topk_weights,
                                                    const DeepEPDispatchHandleLowLatency& handle,
                                                    bool                                  async_finish,
                                                    bool                                  return_recv_hook);

private:
    DeepEPDispatchOutput intranodeDispatch(const torch::Tensor&                            x,
                                           const std::optional<torch::Tensor>&             x_scales,
                                           const std::optional<DeepEPDispatchHandleIntra>& handle,
                                           const std::optional<torch::Tensor>&             num_tokens_per_rank,
                                           const std::optional<torch::Tensor>&             is_token_in_rank,
                                           const std::optional<torch::Tensor>&             num_tokens_per_expert,
                                           const std::optional<torch::Tensor>&             topk_idx,
                                           const std::optional<torch::Tensor>&             topk_weights,
                                           int                                             expert_alignment,
                                           const deep_ep::Config&                          config,
                                           const std::shared_ptr<EventOverlap>&            previous_event,
                                           bool                                            async_finish,
                                           bool                                            allocate_on_comm_stream);
    DeepEPCombineOutput  intranodeCombine(const torch::Tensor&                            x,
                                          const std::optional<DeepEPDispatchHandleIntra>& handle,
                                          const std::optional<torch::Tensor>&             topk_weights,
                                          const deep_ep::Config&                          config,
                                          const std::shared_ptr<EventOverlap>&            previous_event,
                                          bool                                            async_finish,
                                          bool                                            allocate_on_comm_stream);

    /*
    Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
    */
    DeepEPDispatchOutput internodeDispatch(const torch::Tensor&                            x,
                                           const std::optional<torch::Tensor>&             x_scales,
                                           const std::optional<DeepEPDispatchHandleInter>& handle,
                                           const std::optional<torch::Tensor>&             num_tokens_per_rank,
                                           const std::optional<torch::Tensor>&             num_tokens_per_rdma_rank,
                                           const std::optional<torch::Tensor>&             is_token_in_rank,
                                           const std::optional<torch::Tensor>&             num_tokens_per_expert,
                                           const std::optional<torch::Tensor>&             topk_idx,
                                           const std::optional<torch::Tensor>&             topk_weights,
                                           int                                             expert_alignment,
                                           const deep_ep::Config&                          config,
                                           const std::shared_ptr<EventOverlap>&            previous_event,
                                           bool                                            async_finish,
                                           bool                                            allocate_on_comm);

    /*
     Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
    */
    DeepEPCombineOutput internodeCombine(const torch::Tensor&                            x,
                                         const std::optional<DeepEPDispatchHandleInter>& handle,
                                         const std::optional<torch::Tensor>&             topk_weights,
                                         const deep_ep::Config&                          config,
                                         const std::shared_ptr<EventOverlap>&            previous_event,
                                         bool                                            async_finish,
                                         bool                                            allocate_on_comm_stream);

private:
    DeviceBase* device_{nullptr};
    size_t      world_rank_;
    size_t      world_size_;
    int64_t     num_nvl_bytes_{0};
    int64_t     num_rdma_bytes_{0};
    bool        low_latency_mode_{false};
    int         num_qps_per_rank_{1};

    std::unique_ptr<deep_ep::Buffer> buffer_;

    size_t num_sms_{24};
};

}  // namespace rtp_llm

#endif  // ENABLE_DEEP_EP
