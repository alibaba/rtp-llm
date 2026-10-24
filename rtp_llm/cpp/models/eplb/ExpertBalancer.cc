#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/eplb/ExpertLoadMonitor.h"
#include <algorithm>
#include <thread>
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

using namespace std;

namespace rtp_llm {

void EplbPlanBuffers::init(size_t    log_exp_num,
                           size_t    phy_exp_num,
                           size_t    hidden_size,
                           size_t    moe_size,
                           size_t    ep_size,
                           DataType  dtype,
                           QuantAlgo quant_algo) {
    auto gpu_i32 = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    auto cpu_i32 = torch::kInt32;

    layer_id_buf          = torch::zeros({1}, gpu_i32);
    logic_expert_cnt      = torch::zeros({(int64_t)log_exp_num}, gpu_i32);
    logic_expert_cnt_host = torch::zeros({(int64_t)log_exp_num}, cpu_i32).pin_memory();
    log2phy               = torch::zeros({(int64_t)log_exp_num, (int64_t)(phy_exp_num - log_exp_num + 1)}, gpu_i32);
    log2phy_host = torch::zeros({(int64_t)log_exp_num, (int64_t)(phy_exp_num - log_exp_num + 1)}, cpu_i32).pin_memory();
    phy2log      = torch::zeros({(int64_t)phy_exp_num}, gpu_i32);

    size_t expert_per_ep = phy_exp_num / ep_size;

    // note: only support fp8 per token quant
    if (quant_algo.isFp8()) {
        is_quantized    = true;
        int  group_size = quant_algo.getGroupSize();
        auto gpu_fp8    = torch::TensorOptions(torch::kFloat8_e4m3fn).device(torch::kCUDA);
        auto gpu_fp32   = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);

        moe_weight_1 = torch::zeros({(int64_t)expert_per_ep, (int64_t)(moe_size * 2), (int64_t)hidden_size}, gpu_fp8);
        moe_scale_1  = torch::zeros(
            {(int64_t)expert_per_ep, (int64_t)(moe_size * 2 / group_size), (int64_t)(hidden_size / group_size)},
            gpu_fp32);
        moe_weight_2 = torch::zeros({(int64_t)expert_per_ep, (int64_t)hidden_size, (int64_t)moe_size}, gpu_fp8);
        moe_scale_2  = torch::zeros(
            {(int64_t)expert_per_ep, (int64_t)(hidden_size / group_size), (int64_t)(moe_size / group_size)}, gpu_fp32);
    } else {
        is_quantized   = false;
        auto gpu_dtype = torch::TensorOptions(dataTypeToTorchType(dtype)).device(torch::kCUDA);
        moe_weight_1 = torch::zeros({(int64_t)expert_per_ep, (int64_t)(moe_size * 2), (int64_t)hidden_size}, gpu_dtype);
        moe_weight_2 = torch::zeros({(int64_t)expert_per_ep, (int64_t)hidden_size, (int64_t)moe_size}, gpu_dtype);
    }
}

void BalanceStatsBuffers::init(int layer_num, int log_exp_num, int ep_size) {
    // host tensors
    log_stats = torch::zeros({layer_num, log_exp_num}, torch::kInt32);
    gpu_loads = torch::zeros({layer_num, ep_size}, torch::kInt32);

    // gpu tensors
    log_stats_gpu = torch::zeros({layer_num, log_exp_num}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    gpu_loads_gpu = torch::zeros({layer_num, ep_size}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
}

void BalanceStatsBuffers::reset() {
    log_stats.zero_();
    gpu_loads.zero_();
    log_stats_gpu.zero_();
    gpu_loads_gpu.zero_();
}

void LoadFlags::init() {
    flag_gpu  = torch::zeros({1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    flag_sync = torch::zeros({1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    flag_host = torch::zeros({1}, torch::kInt32).pin_memory();
}

void LoadFlags::setReady(bool ready) {
    int value = ready ? 0 : -1;
    flag_gpu.fill_(value);
}

bool LoadFlags::isReady(size_t world_size) {
    if (world_size > 1) {
        // sync all ranks load_flag_tensor_
        flag_sync = execAllReduce({flag_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP, flag_sync}).buffer;
    } else {
        flag_sync.copy_(flag_gpu);
    }

    flag_host.copy_(flag_sync);
    // Repeatedly executing “x = ReduceSum(x)” with tp8 will cause an overflow and set the value of x to 0
    flag_gpu.fill_(-1);
    // if all load_flag_tensor_ is 0, return true
    return flag_host.item<int>() == 0;
}

void EplbController::init(const EPLBConfig& eplb_control_data, const EPLBConfig& eplb_config) {
    this->eplb_control_data = eplb_control_data;

    control_step = eplb_config.eplb_control_step;
    RTP_LLM_LOG_INFO("EPLB control step: %d", control_step);

    auto eplb_control_data_list  = eplb_control_data.toList();
    eplb_control_data_buf_host   = torch::zeros({(int64_t)eplb_control_data_list.size()}, torch::kInt32).pin_memory();
    eplb_control_data_buf_device = torch::zeros({(int64_t)eplb_control_data_list.size()},
                                                torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
}

void EplbController::setData(const EPLBConfig& updated_control_data) {
    // lock mutex
    lock_guard<mutex> lock(eplb_control_mutex);
    eplb_control_data = updated_control_data;
}

bool EplbController::stepAndCheckSyncStep() {
    // check if current step is sync step
    cur_step++;
    if (cur_step >= control_step) {
        cur_step = 0;
        return true;
    }
    return false;
}

EPLBConfig EplbController::getAndSyncData(size_t world_size) {
    // copy control data to host buffer
    EPLBConfig cur_data;
    {
        lock_guard<mutex> lock(eplb_control_mutex);
        cur_data = eplb_control_data;
    }
    auto eplb_control_data_list     = eplb_control_data.toList();
    int* eplb_control_data_host_ptr = eplb_control_data_buf_host.data_ptr<int>();
    for (size_t i = 0; i < eplb_control_data_list.size(); ++i) {
        eplb_control_data_host_ptr[i] = eplb_control_data_list[i];
    }

    // copy to device
    eplb_control_data_buf_device.copy_(eplb_control_data_buf_host, /*non_blocking=*/true);

    if (world_size > 1) {
        execBroadcast({{eplb_control_data_buf_device}, 0, ParallelMode::DP_AND_TP});
    }

    // copy to host
    eplb_control_data_buf_host.copy_(eplb_control_data_buf_device);

    // convert to EPLBConfig
    auto eplb_control_data = EPLBConfig::fromList(eplb_control_data_host_ptr);

    return eplb_control_data;
}

ExpertBalancer::ExpertBalancer(const std::optional<MoeConfigs>& moe,
                               const GptModelInitParams&        params,
                               Weights&                         weights,
                               OverallExpertStats&              expert_stats):
    weights_(weights),
    expert_stats_(expert_stats),
    ep_rank_(params.parallelism_config.ep_rank),
    ep_size_(params.parallelism_config.ep_size),
    world_size_(params.parallelism_config.world_size),
    metrics_reporter_(params.metrics_reporter) {
    c10::InferenceMode inference_guard(true);
    const auto&        config      = params.eplb_config;
    const auto         layer_count = params.weights.layers.size();
    RTP_LLM_CHECK_WITH_INFO(config.eplb_metrics_report_interval >= 0,
                            "EPLB_METRICS_REPORT_INTERVAL must be non-negative");

    if (config.enable_eplb() && moe) {
        RTP_LLM_CHECK_WITH_INFO(!params.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate,
                                "EPLB planner does not support FFN disaggregate/alignment-only steps");
        RTP_LLM_CHECK_WITH_INFO(params.py_eplb && !params.py_eplb.is_none(),
                                "EPLB planner requires a Python EPLB instance");
        const auto first_moe_layer =
            std::find_if(params.weights.layers.begin(), params.weights.layers.end(), [](const auto& layer) {
                return layer.ffn_weights.moe_gate_weight != nullptr;
            });
        RTP_LLM_CHECK_WITH_INFO(first_moe_layer != params.weights.layers.end(),
                                "EPLB requires at least one MoE layer with moe_gate_weight");
        const auto& kernel = first_moe_layer->ffn_weights.moe_gate_weight->kernel;
        const auto  moe_size =
            isGatedActivation(params.description.ffn_conf.activation_type) ? kernel.size(1) / 2 : kernel.size(1);
        num_logic_experts_  = moe->expert_num;
        num_physic_experts_ = moe->expert_num + moe->extra_expert_num;
        eplb_control_data_  = config;

        // Keep planner allocations in the warmup peak-memory budget; step() skips execution.
        stats_.init(layer_count, num_logic_experts_, ep_size_);
        eplb_plan_buffers_.init(num_logic_experts_,
                                num_physic_experts_,
                                params.hidden_size,
                                moe_size,
                                ep_size_,
                                torchDTypeToDataType(kernel.dtype()),
                                params.quant_algo);
        eplb_plan_tensors_.init(num_logic_experts_, num_physic_experts_);
        load_flags_.init();
        load_flags_.setReady(false);
        eplb_controller_.init(eplb_control_data_, config);
        test_mode_              = config.eplb_test_mode;
        balance_layer_per_step_ = config.eplb_balance_layer_per_step;
        resetPlan(true);
        planner_initialized_ = true;
    }

    if (config.eplb_metrics_report_interval > 0) {
        load_monitor_ = ExpertLoadMonitor::create(layer_count,
                                                            moe ? moe->expert_num + moe->extra_expert_num : 0,
                                                            params.parallelism_config,
                                                            config,
                                                            params.model_id,
                                                            params.metrics_reporter,
                                                            params.model_id == 0 ? "target" : "draft");
    }
    if (load_monitor_) {
        // EP ranks own distinct experts. Replicated TP experts only write on TP0;
        // the other ranks still participate in step() with zero counts.
        if (moe && (ep_size_ > 1 || params.parallelism_config.tp_rank == 0)) {
            expert_token_counts_ = load_monitor_->stepCountsBuffer();
        }
    }

    if (planner_initialized_) {
        py::gil_scoped_acquire acquire;
        eplb_python_wrapper_ = ExpertBalancerPythonWrapper(py::reinterpret_borrow<py::object>(params.py_eplb));
    }
}

ExpertBalancer::~ExpertBalancer() = default;

void ExpertBalancer::step(bool warmup) {
    if (load_monitor_) {
        if (!warmup) {
            load_monitor_->recordStep();
        } else {
            load_monitor_->clearRecordBuffer();
        }
    }
    if (warmup || !planner_initialized_) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("py_model.eplb_step");
    const auto start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    syncController();
    int64_t latency_us = 0;
    if (!eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::NONE)) {
        reportStats(expert_stats_);
        excuteEplbPlan(expert_stats_);
        latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_us;
    }
    // Keep the planner metric independent of monitor work and monitor-only models.
    if (metrics_reporter_) {
        RtpLLmEplbMetricsCollector collector;
        collector.step_latency_us = latency_us;
        metrics_reporter_->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(nullptr, &collector);
    }
}

bool ExpertBalancer::updateEplbConfig(const EPLBConfig& config) {
    if (!planner_initialized_) {
        return !load_monitor_;
    }
    eplb_controller_.setData(config);
    return true;
}

void ExpertBalancer::syncController() {
    // sync control data
    if (eplb_controller_.stepAndCheckSyncStep()) {
        auto eplb_control_data = eplb_controller_.getAndSyncData(world_size_);
        if (eplb_control_data.eplb_mode != eplb_control_data_.eplb_mode
            || eplb_control_data.eplb_update_time != eplb_control_data_.eplb_update_time) {
            eplb_control_data_ = eplb_control_data;
            RTP_LLM_LOG_INFO("EPLB config changed to %s", eplb_control_data_.to_string().c_str());
        }
    }
}

void ExpertBalancer::reportStats(OverallExpertStats& stats) {
    executor_collector_.ep_rank = ep_rank_;
    if (load_monitor_) {
        // The independent monitor owns load reporting and its window D2H.
        // Keep weight-update events without the legacy per-forward load copy.
        if (metrics_reporter_ && executor_collector_.update_weights_qps
            && eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::STATS, EplbMode::ALL)) {
            executor_collector_.gpu_loads.clear();
            metrics_reporter_->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(nullptr, &executor_collector_);
        }
        return;
    }
    if (metrics_reporter_
        && eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::STATS, EplbMode::ALL)) {
        int layer_num = stats.layer_num;
        executor_collector_.gpu_loads.resize(layer_num);

        auto gpu_loads_tensor = stats.stats_buf.gpu_loads_buf.cpu();
        int* gpu_loads        = gpu_loads_tensor.data_ptr<int>();

        for (int i = 0; i < layer_num; ++i) {
            executor_collector_.gpu_loads[i] = gpu_loads[i * ep_size_ + ep_rank_];
        }

        metrics_reporter_->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(nullptr, &executor_collector_);
    }
}

void ExpertBalancer::setPlanStatus(EplbPlanStatus status) {
    lock_guard<mutex> lock(eplb_plan_status_mutex_);
    eplb_plan_status_ = status;
}

EplbPlanStatus ExpertBalancer::getPlanStatus() const {
    lock_guard<mutex> lock(eplb_plan_status_mutex_);
    auto              status = eplb_plan_status_;
    return status;
}

void ExpertBalancer::excuteEplbPlan(OverallExpertStats& stats) {
    if (eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::EPLB, EplbMode::ALL)) {
        EplbPlanStatus status = getPlanStatus();
        switch (status) {
            case EplbPlanStatus::INIT:
                update_cnt_++;
                updateStats(stats);
                if (update_cnt_ >= eplb_control_data_.eplb_update_time) {
                    setPlanStatus(EplbPlanStatus::PREPARING);
                }
                break;
            case EplbPlanStatus::PREPARING: {
                createPlan();
                setPlanStatus(EplbPlanStatus::LOADING);
                thread load_thread([this]() {
                    loadPlanWeights();
                    setPlanStatus(EplbPlanStatus::LOADED);
                });
                if (test_mode_) {
                    load_thread.join();
                } else {
                    load_thread.detach();
                }
                break;
            }
            case EplbPlanStatus::LOADING:
                syncPlanWeightsLoadStatus();
                break;
            case EplbPlanStatus::LOADED:
                load_flags_.setReady(true);
                if (syncPlanWeightsLoadStatus()) {
                    processPlanWeights();
                    applyPlanWeights();
                    load_flags_.setReady(false);
                    balance_layer_cnt_++;

                    if (balance_layer_cnt_ >= balance_layer_per_step_) {
                        update_cnt_        = 0;
                        balance_layer_cnt_ = 0;
                    } else {
                        update_cnt_ = eplb_control_data_.eplb_update_time;  // quick update
                    }

                    setPlanStatus(EplbPlanStatus::INIT);

                    resetPlan();
                }
                break;
        }
    }
}

void ExpertBalancer::resetPlan(bool force_clean) {
    stats_.reset();
}

void ExpertBalancer::copyFromTensor(const torch::Tensor& src, torch::Tensor& dst) {
    dst.copy_(src, /*non_blocking=*/true);
}

void ExpertBalancer::copyToTensor(const torch::Tensor& src, torch::Tensor& dst) {
    dst.copy_(src, /*non_blocking=*/true);
}

void ExpertBalancer::createPlan() {
    // pre run
    if (world_size_ > 1) {
        execAllReduce({stats_.log_stats_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});
        execAllReduce({stats_.gpu_loads_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});
    }

    // copy stats gpu tensor to host tensor [implicit sync]
    stats_.log_stats.copy_(stats_.log_stats_gpu);
    stats_.gpu_loads.copy_(stats_.gpu_loads_gpu);

    if (ep_rank_ == 0) {
        eplb_python_wrapper_.createBalancePlan(stats_.log_stats, stats_.gpu_loads, eplb_plan_tensors_);

        // copy tensor(host) to gpu tensor
        // note: it's ok to use async copy, since the tensor host ptr will not be released
        copyFromTensor(eplb_plan_tensors_.layer_id_buf, eplb_plan_buffers_.layer_id_buf);
        copyFromTensor(eplb_plan_tensors_.logic_expert_cnt, eplb_plan_buffers_.logic_expert_cnt);
        copyFromTensor(eplb_plan_tensors_.log2phy, eplb_plan_buffers_.log2phy);
        copyFromTensor(eplb_plan_tensors_.phy2log, eplb_plan_buffers_.phy2log);
    }

    if (world_size_ > 1) {
        execBroadcast({{eplb_plan_buffers_.layer_id_buf,
                        eplb_plan_buffers_.logic_expert_cnt,
                        eplb_plan_buffers_.log2phy,
                        eplb_plan_buffers_.phy2log},
                       0,
                       ParallelMode::DP_AND_TP});
    }

    // copy plan gpu tensor to host tensor [implicit sync]
    copyToTensor(eplb_plan_buffers_.layer_id_buf, eplb_plan_tensors_.layer_id_buf);
    copyToTensor(eplb_plan_buffers_.logic_expert_cnt, eplb_plan_tensors_.logic_expert_cnt);
    copyToTensor(eplb_plan_buffers_.log2phy, eplb_plan_tensors_.log2phy);
    copyToTensor(eplb_plan_buffers_.phy2log, eplb_plan_tensors_.phy2log);
}

void ExpertBalancer::processPlanWeights() {
    eplb_plan_buffers_.layer_id = eplb_plan_tensors_.layer_id;
    if (eplb_plan_buffers_.is_quantized) {
        copyFromTensor(eplb_plan_tensors_.moe_weight_1, eplb_plan_buffers_.moe_weight_1);
        copyFromTensor(eplb_plan_tensors_.moe_scale_1, eplb_plan_buffers_.moe_scale_1);
        copyFromTensor(eplb_plan_tensors_.moe_weight_2, eplb_plan_buffers_.moe_weight_2);
        copyFromTensor(eplb_plan_tensors_.moe_scale_2, eplb_plan_buffers_.moe_scale_2);
    } else {
        copyFromTensor(eplb_plan_tensors_.moe_weight_1, eplb_plan_buffers_.moe_weight_1);
        copyFromTensor(eplb_plan_tensors_.moe_weight_2, eplb_plan_buffers_.moe_weight_2);
    }

    executor_collector_.update_weights_qps = true;
    executor_collector_.update_layer_id    = eplb_plan_buffers_.layer_id;
}

void ExpertBalancer::loadPlanWeights() {
    int64_t start_time_ms = autil::TimeUtility::currentTimeInMilliSeconds();

    eplb_python_wrapper_.loadBalanceWeight(ep_rank_, ep_size_, eplb_plan_tensors_);

    executor_collector_.update_weights_latency_ms = autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms;
}

void ExpertBalancer::applyPlanWeights() {
    // applyPlanWeights is reached after the existing all-rank readiness check.
    if (load_monitor_) {
        load_monitor_->beforeMappingChange();
    }
    auto& result         = eplb_plan_buffers_;
    auto& balanced_layer = weights_.layers[result.layer_id].ffn_weights;

    // Exchange DenseWeights kernel/scales/zeros with eplb plan tensors.
    auto exchange_dense_weight = [&result](DenseWeights& dense, torch::Tensor& plan_weight, torch::Tensor& plan_scale) {
        auto old_kernel = dense.kernel;
        auto old_scales = dense.scales;
        auto old_zeros  = dense.zeros;

        // Set model weights from plan tensors
        dense.kernel = plan_weight;
        if (result.is_quantized) {
            dense.scales = plan_scale;
            dense.zeros  = torch::Tensor();
        } else {
            dense.scales = torch::Tensor();
            dense.zeros  = torch::Tensor();
        }

        // Store old model weights back into plan tensors for next cycle
        plan_weight = old_kernel;
        if (result.is_quantized && old_scales.defined()) {
            plan_scale = old_scales;
        }
    };

    // Exchange torch::Tensor directly (for non-quantized fields)
    auto exchange_tensors = [](torch::Tensor& model_tensor, torch::Tensor& plan_tensor) {
        std::swap(model_tensor, plan_tensor);
    };

    exchange_dense_weight(*balanced_layer.moe_gate_weight, result.moe_weight_1, result.moe_scale_1);
    exchange_dense_weight(*balanced_layer.moe_down_weight, result.moe_weight_2, result.moe_scale_2);
    exchange_tensors(balanced_layer.log2phy, result.log2phy);
    exchange_tensors(balanced_layer.logic_expert_cnt, result.logic_expert_cnt);
}

bool ExpertBalancer::syncPlanWeightsLoadStatus() {
    return load_flags_.isReady(world_size_);
}

void ExpertBalancer::updateStats(OverallExpertStats& stats) {
    stats_.log_stats_gpu.add_(stats.stats_buf.log_stats_buf);
    stats_.gpu_loads_gpu.add_(stats.stats_buf.gpu_loads_buf);
}

}  // namespace rtp_llm
