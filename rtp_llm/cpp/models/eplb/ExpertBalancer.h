#pragma once

#include <torch/extension.h>
#include "rtp_llm/cpp/models/eplb/ExpertBalancerPythonWrapper.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/model_utils/QuantInfo.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"

namespace rtp_llm {

// Forward declarations
class ModelBase;

struct EplbPlanBuffers {
    int           layer_id = -1;
    torch::Tensor layer_id_buf;           // [1], INT32, GPU
    torch::Tensor logic_expert_cnt;       // [log_exp_num], INT32, GPU
    torch::Tensor logic_expert_cnt_host;  // [log_exp_num], INT32, CPU pinned
    torch::Tensor log2phy;                // [log_exp_num, phy-log+1], INT32, GPU
    torch::Tensor log2phy_host;           // [log_exp_num, phy-log+1], INT32, CPU pinned
    torch::Tensor phy2log;                // [phy_exp_num], INT32, GPU

    // MoE weight buffers (for quantized: kernel + scales stored separately)
    torch::Tensor moe_weight_1;  // w1 & w3 kernel
    torch::Tensor moe_scale_1;   // w1 & w3 scales (only for fp8)
    torch::Tensor moe_weight_2;  // w2 kernel
    torch::Tensor moe_scale_2;   // w2 scales (only for fp8)
    bool          is_quantized = false;

    void init(size_t    log_exp_num,
              size_t    phy_exp_num,
              size_t    hidden_size,
              size_t    moe_size,
              size_t    ep_size,
              DataType  dtype,
              QuantAlgo quant_algo);
};

struct BalanceStatsBuffers {
    torch::Tensor log_stats;      // host [layer, log_exp_num]
    torch::Tensor gpu_loads;      // host [layer, ep_size]
    torch::Tensor log_stats_gpu;  // device [layer, log_exp_num]
    torch::Tensor gpu_loads_gpu;  // device [layer, ep_size]

    void init(int layer_num, int log_exp_num, int ep_size);
    void reset();
};

struct LoadFlags {
    torch::Tensor flag_gpu;   // [1], INT32, GPU
    torch::Tensor flag_sync;  // [1], INT32, GPU
    torch::Tensor flag_host;  // [1], INT32, CPU pinned

    void init();

    void setReady(bool ready);
    bool isReady();
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

    torch::Tensor eplb_control_data_buf_host;    // INT32, CPU pinned
    torch::Tensor eplb_control_data_buf_device;  // INT32, GPU

    int control_step = 100;
    int cur_step     = 0;

public:
    void       init(const EPLBConfig& eplb_control_data, const EPLBConfig& eplb_config);
    void       setData(const EPLBConfig& updated_control_data);
    bool       stepAndCheckSyncStep();
    EPLBConfig getAndSyncData();
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
                                                          QuantAlgo                    quant_algo,
                                                          kmonitor::MetricsReporterPtr metrics_reporter,
                                                          const EPLBConfig&            eplb_config);
    ~ExpertBalancer();

    void stepForward(ModelBase& model, RtpLLMExecutorMetricsCollector& executor_collector);

    bool updateEplbConfig(const EPLBConfig& config);

private:
    void syncController();
    void reportStats(OverallExpertStats& stats);
    void excuteEplbPlan(OverallExpertStats& stats, ModelBase& model);

    void           setPlanStatus(EplbPlanStatus status);
    EplbPlanStatus getPlanStatus() const;

    void resetPlan(bool force_clean = false);
    void createPlan();
    void updateStats(OverallExpertStats& stats);
    void loadPlanWeights();
    bool syncPlanWeightsLoadStatus();
    void processPlanWeights();
    void applyPlanWeights(ModelBase& model);

    // helpful functions
    void copyFromTensor(const torch::Tensor& src, torch::Tensor& dst);
    void copyToTensor(const torch::Tensor& src, torch::Tensor& dst);

private:
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