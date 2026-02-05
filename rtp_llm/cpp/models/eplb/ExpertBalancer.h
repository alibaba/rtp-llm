#pragma once

#include <torch/extension.h>
#include "rtp_llm/cpp/models/eplb/ExpertBalancerPythonWrapper.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/models/elastic_ep_manager/ElasticEPManager.h"

namespace rtp_llm {

// Forward declarations
class GptModel;

struct EplbPlanBuffers {
    int       layer_id = -1;
    BufferPtr layer_id_buf;      // [1]
    BufferPtr logic_expert_cnt;  // [log_exp_num]
    BufferPtr logic_expert_cnt_host;
    BufferPtr log2phy;  // [layer, log_exp_num, phy_exp_num - log_exp_num + 1]
    BufferPtr log2phy_host;
    BufferPtr phy2log;       // [layer, phy_exp_num]
    BufferPtr moe_weight_1;  // w1 & w3
    BufferPtr moe_weight_2;  // w2

    void init(size_t      log_exp_num,
              size_t      phy_exp_num,
              size_t      hidden_size,
              size_t      moe_size,
              size_t      ep_size,
              DataType    dtype,
              QuantAlgo   quant_algo,
              DeviceBase* device);
};

struct BalanceStatsBuffers {
    torch::Tensor log_stats;
    torch::Tensor gpu_loads;

    // gpu buffer
    BufferPtr log_stats_buf;
    BufferPtr gpu_loads_buf;

    torch::Tensor log_stats_gpu;
    torch::Tensor gpu_loads_gpu;

    void init(int layer_num, int log_exp_num, int ep_size, DeviceBase* device);
    void reset();
};

struct LoadFlags {
    BufferPtr flag_gpu;
    BufferPtr flag_sync;
    BufferPtr flag_host;

    void init(DeviceBase* device);

    void setReady(bool ready, DeviceBase* device);
    bool isReady(DeviceBase* device);
};

enum class EplbPlanStatus {
    INIT,
    PREPARING,
    LOADING,
    LOADED
};

class EplbController {
private:
    std::mutex eplb_control_mutex;
    EPLBConfig eplb_control_data;

    BufferPtr eplb_control_data_buf_host;
    BufferPtr eplb_control_data_buf_device;

    int control_step = 100;
    int cur_step     = 0;

public:
    void       init(const EPLBConfig& eplb_control_data, DeviceBase* device, const EPLBConfig& eplb_config);
    void       setData(const EPLBConfig& updated_control_data);
    bool       stepAndCheckSyncStep();
    EPLBConfig getAndSyncData(DeviceBase* device);
};

class ExpertBalancer {
public:
    __attribute__((visibility("default"))) ExpertBalancer(size_t                       log_exp_num,
                                                          size_t                       phy_exp_num,
                                                          size_t                       num_layers,
                                                          size_t                       moe_size,
                                                          size_t                       hidden_size,
                                                          size_t                       ep_rank,
                                                          size_t                       ep_size,
                                                          py::object                   py_eplb,
                                                          DataType                     dtype,
                                                          DeviceBase*                  device,
                                                          QuantAlgo                    quant_algo,
                                                          kmonitor::MetricsReporterPtr metrics_reporter,
                                                          const EPLBConfig&            eplb_config);
    ~ExpertBalancer();

    void
    stepForward(GptModel& model, RtpLLMExecutorMetricsCollector& executor_collector, const ElasticEPStats& ep_stats);

    bool updateEplbConfig(const EPLBConfig& config);

private:
    void syncController();
    void reportStats(OverallExpertStats& stats);
    bool checkDownScale(int active_ranks_num);
    void executeDownScale(GptModel& model, const torch::Tensor& active_ranks_tensor_cpu);
    void excuteEplbPlan(OverallExpertStats& stats, GptModel& model, const torch::Tensor& active_ranks_tensor_cpu);

    void           setPlanStatus(EplbPlanStatus status);
    EplbPlanStatus getPlanStatus() const;

    void resetPlan(bool force_clean = false);
    void createPlan(const torch::Tensor& active_ranks_tensor_cpu);
    void updateStats(OverallExpertStats& stats);
    void loadPlanWeights();
    bool syncPlanWeightsLoadStatus();
    void processPlanWeights();
    void applyPlanWeights(GptModel& model);
    void syncUpdateWeights();
    void updateBalanceWeight(EplbPlanTensors& eplb_plan, GptModel& model);

    // helpful functions
    void copyFromTensor(torch::Tensor& tensor, BufferPtr& buffer);
    void copyToTensor(BufferPtr& buffer, torch::Tensor& tensor);

private:
    DeviceBase* device_;

    size_t num_logic_experts_;
    size_t num_physic_experts_;

    EplbPlanStatus eplb_plan_status_ = EplbPlanStatus::INIT;

    size_t update_cnt_ = 0;

    size_t eplb_plan_cnt_ = 0;

    size_t ep_rank_ = 0;
    size_t ep_size_ = 1;

    size_t balance_layer_cnt_      = 0;
    size_t balance_layer_per_step_ = 1;

    BalanceStatsBuffers stats_;
    EplbPlanBuffers     eplb_plan_buffers_;
    EplbPlanTensors     eplb_plan_tensors_;
    LoadFlags           load_flags_;

    EplbController eplb_controller_;
    EPLBConfig     eplb_control_data_;

    RtpLLmEplbMetricsCollector   executor_collector_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    ExpertBalancerPythonWrapper  eplb_python_wrapper_;

    mutable std::mutex eplb_plan_status_mutex_;

    bool test_mode_ = false;
};

}  // namespace rtp_llm