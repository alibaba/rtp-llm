#pragma once

#include <torch/torch.h>
#include "maga_transformer/cpp/eplb/ExpertBalancerPythonWrapper.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "src/fastertransformer/utils/EplbConfig.h"

namespace rtp_llm {
namespace ft = fastertransformer;

struct EplbPlanBuffers {
    int           layer_id = -1;
    ft::BufferPtr layer_id_buf;      // [1]
    ft::BufferPtr logic_expert_cnt;  // [log_exp_num]
    ft::BufferPtr logic_expert_cnt_host;
    ft::BufferPtr log2phy;           // [layer, log_exp_num, phy_exp_num - log_exp_num + 1]
    ft::BufferPtr log2phy_host;
    ft::BufferPtr phy2log;           // [layer, phy_exp_num]
    ft::BufferPtr moe_weight_1;      // w1 & w3
    ft::BufferPtr moe_weight_2;      // w2

    void init(size_t          log_exp_num,
              size_t          phy_exp_num,
              size_t          hidden_size,
              size_t          moe_size,
              size_t          ep_size,
              ft::DataType    dtype,
              ft::QuantAlgo   quant_algo,
              ft::DeviceBase* device);
};

struct BalanceStatsBuffers {
    torch::Tensor log_stats;
    torch::Tensor gpu_loads;

    // gpu buffer
    ft::BufferPtr log_stats_buf;
    ft::BufferPtr gpu_loads_buf;

    torch::Tensor log_stats_gpu;
    torch::Tensor gpu_loads_gpu;

    void init(int layer_num, int log_exp_num, int ep_size, ft::DeviceBase* device);
    void reset();
};

struct LoadFlags {
    ft::BufferPtr flag_gpu;
    ft::BufferPtr flag_sync;
    ft::BufferPtr flag_host;

    void init(ft::DeviceBase* device);

    void setReady(bool ready, ft::DeviceBase* device);
    bool isReady(ft::DeviceBase* device);
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
                   ft::DataType    dtype,
                   ft::DeviceBase* device,
                   ft::EplbMode    eplb_mode,
                   ft::QuantAlgo   quant_algo,
                   kmonitor::MetricsReporterPtr metrics_reporter);
    ~ExpertBalancer();

    void stepForward(GptModel& model, RtpLLMExecutorMetricsCollector& executor_collector);

private:
    void reportStats(ft::OverallExpertStats& stats);
    void excuteEplbPlan(ft::OverallExpertStats& stats, GptModel& model);

    void setPlanStatus(EplbPlanStatus status);
    EplbPlanStatus getPlanStatus() const;

    void resetPlan(bool force_clean = false);
    void createPlan();
    void updateStats(ft::OverallExpertStats& stats);
    void loadPlanWeights();
    bool syncPlanWeightsLoadStatus();
    void processPlanWeights();
    void applyPlanWeights(GptModel& model);

    // helpful functions
    void copyFromTensor(torch::Tensor& tensor, ft::BufferPtr& buffer);
    void copyToTensor(ft::BufferPtr& buffer, torch::Tensor& tensor);

private:
    ft::DeviceBase* device_;

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
    ft::EplbMode        eplb_mode_;

    RtpLLmEplbMetricsCollector   executor_collector_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    ExpertBalancerPythonWrapper  eplb_python_wrapper_;

    bool enable_stats_ = false;
    bool enable_eplb_ = false;

    mutable std::mutex eplb_plan_status_mutex_;
};

}  // namespace rtp_llm