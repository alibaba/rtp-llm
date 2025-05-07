#pragma once

#include <torch/torch.h>
#include "maga_transformer/cpp/eplb/ExpertBalancerPythonWrapper.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/devices/DeviceBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/utils/EplbConfig.h"

namespace rtp_llm {


struct EplbPlanBuffers {
    int           layer_id = -1;
    rtp_llm::BufferPtr layer_id_buf;      // [1]
    rtp_llm::BufferPtr logic_expert_cnt;  // [log_exp_num]
    rtp_llm::BufferPtr logic_expert_cnt_host;
    rtp_llm::BufferPtr log2phy;           // [layer, log_exp_num, phy_exp_num - log_exp_num + 1]
    rtp_llm::BufferPtr log2phy_host;
    rtp_llm::BufferPtr phy2log;           // [layer, phy_exp_num]
    rtp_llm::BufferPtr moe_weight_1;      // w1 & w3
    rtp_llm::BufferPtr moe_weight_2;      // w2

    void init(size_t          log_exp_num,
              size_t          phy_exp_num,
              size_t          hidden_size,
              size_t          moe_size,
              size_t          ep_size,
              rtp_llm::DataType    dtype,
              rtp_llm::QuantAlgo   quant_algo,
              rtp_llm::DeviceBase* device);
};

struct BalanceStatsBuffers {
    torch::Tensor log_stats;
    torch::Tensor gpu_loads;

    // gpu buffer
    rtp_llm::BufferPtr log_stats_buf;
    rtp_llm::BufferPtr gpu_loads_buf;

    torch::Tensor log_stats_gpu;
    torch::Tensor gpu_loads_gpu;

    void init(int layer_num, int log_exp_num, int ep_size, rtp_llm::DeviceBase* device);
    void reset();
};

struct LoadFlags {
    rtp_llm::BufferPtr flag_gpu;
    rtp_llm::BufferPtr flag_sync;
    rtp_llm::BufferPtr flag_host;

    void init(rtp_llm::DeviceBase* device);

    void setReady(bool ready, rtp_llm::DeviceBase* device);
    bool isReady(rtp_llm::DeviceBase* device);
};

enum class EplbPlanStatus
{
    INIT,
    PREPARING,
    LOADING,
    LOADED
};

class ExpertBalancer {
public:
    ExpertBalancer() = default;
    ExpertBalancer(size_t          log_exp_num,
                   size_t          phy_exp_num,
                   size_t          num_layers,
                   size_t          moe_size,
                   size_t          hidden_size,
                   size_t          update_time,
                   size_t          ep_rank,
                   size_t          ep_size,
                   py::object      py_eplb,
                   rtp_llm::DataType    dtype,
                   rtp_llm::DeviceBase* device,
                   rtp_llm::EplbMode    eplb_mode,
                   rtp_llm::QuantAlgo   quant_algo,
                   kmonitor::MetricsReporterPtr metrics_reporter);
    ~ExpertBalancer();

    void stepForward(GptModel& model, RtpLLMExecutorMetricsCollector& executor_collector);

private:
    void reportStats(rtp_llm::OverallExpertStats& stats);
    void excuteEplbPlan(rtp_llm::OverallExpertStats& stats, GptModel& model);

    void setPlanStatus(EplbPlanStatus status);
    EplbPlanStatus getPlanStatus() const;

    void resetPlan(bool force_clean = false);
    void createPlan();
    void updateStats(rtp_llm::OverallExpertStats& stats);
    void loadPlanWeights();
    bool syncPlanWeightsLoadStatus();
    void processPlanWeights();
    void applyPlanWeights(GptModel& model);

    // helpful functions
    void copyFromTensor(torch::Tensor& tensor, rtp_llm::BufferPtr& buffer);
    void copyToTensor(rtp_llm::BufferPtr& buffer, torch::Tensor& tensor);

private:
    rtp_llm::DeviceBase* device_;

    size_t num_logic_experts_;
    size_t num_physic_experts_;

    EplbPlanStatus eplb_plan_status_ = EplbPlanStatus::INIT;

    size_t update_cnt_  = 0;
    size_t update_time_ = 0;

    size_t eplb_plan_cnt_ = 0;

    size_t ep_rank_     = 0;
    size_t ep_size_     = 1;

    BalanceStatsBuffers stats_;
    EplbPlanBuffers     eplb_plan_buffers_;
    EplbPlanTensors     eplb_plan_tensors_;
    LoadFlags           load_flags_;
    rtp_llm::EplbMode        eplb_mode_;

    RtpLLmEplbMetricsCollector   executor_collector_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    ExpertBalancerPythonWrapper  eplb_python_wrapper_;

    bool enable_stats_ = false;
    bool enable_eplb_ = false;

    mutable std::mutex eplb_plan_status_mutex_;
};

}  // namespace rtp_llm