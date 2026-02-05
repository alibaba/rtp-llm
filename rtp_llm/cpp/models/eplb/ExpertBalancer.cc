#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include <thread>
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

void EplbPlanBuffers::init(size_t      log_exp_num,
                           size_t      phy_exp_num,
                           size_t      hidden_size,
                           size_t      moe_size,
                           size_t      ep_size,
                           DataType    dtype,
                           QuantAlgo   quant_algo,
                           DeviceBase* device) {
    layer_id_buf     = device->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::DEVICE}, {"eplb_layer_id"});
    logic_expert_cnt = device->allocateBuffer({DataType::TYPE_INT32, {log_exp_num}, AllocationType::DEVICE},
                                              {"eplb_logic_expert_cnt"});
    logic_expert_cnt_host = device->allocateBuffer({DataType::TYPE_INT32, {log_exp_num}, AllocationType::HOST},
                                                   {"eplb_logic_expert_cnt_host"});
    log2phy               = device->allocateBuffer(
        {DataType::TYPE_INT32, {log_exp_num, phy_exp_num - log_exp_num + 1}, AllocationType::DEVICE}, {"eplb_log2phy"});
    log2phy_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {log_exp_num, phy_exp_num - log_exp_num + 1}, AllocationType::HOST},
        {"eplb_log2phy_host"});
    phy2log = device->allocateBuffer({DataType::TYPE_INT32, {phy_exp_num}, AllocationType::DEVICE}, {"eplb_phy2log"});
    size_t expert_per_ep = phy_exp_num / ep_size;

    // note: only support fp8 per token quant
    if (quant_algo.isFp8()) {
        int  group_size = quant_algo.getGroupSize();
        auto create_moe_qbuffer =
            [&device](size_t expert_num, size_t dim1, size_t dim2, size_t group_size, const string& postfix) {
                auto w =
                    device->allocateBuffer({DataType::TYPE_FP8_E4M3, {expert_num, dim1, dim2}, AllocationType::DEVICE},
                                           {"eplb_moe_w_" + postfix});
                auto s = device->allocateBuffer(
                    {DataType::TYPE_FP32, {expert_num, dim1 / group_size, dim2 / group_size}, AllocationType::DEVICE},
                    {"eplb_moe_s_" + postfix});
                auto z = BufferPtr(new Buffer(s->where(), s->type(), {0}, nullptr));
                return BufferPtr(new QBuffer(std::move(w), std::move(s), std::move(z)));
            };

        moe_weight_1 = create_moe_qbuffer(expert_per_ep, moe_size * 2, hidden_size, group_size, "1");
        moe_weight_2 = create_moe_qbuffer(expert_per_ep, hidden_size, moe_size, group_size, "2");
    } else {
        moe_weight_1 = device->allocateBuffer(
            {dtype, {expert_per_ep, moe_size * 2, hidden_size}, AllocationType::DEVICE}, {"eplb_moe_w_1"});
        moe_weight_2 = device->allocateBuffer({dtype, {expert_per_ep, hidden_size, moe_size}, AllocationType::DEVICE},
                                              {"eplb_moe_w_2"});
    }
}

void BalanceStatsBuffers::init(int layer_num, int log_exp_num, int ep_size, DeviceBase* device) {
    // device tensor
    log_stats = torch::zeros({layer_num, log_exp_num}, torch::kInt32);
    gpu_loads = torch::zeros({layer_num, ep_size}, torch::kInt32);

    log_stats_buf = device->allocateBuffer(
        {DataType::TYPE_INT32, {(size_t)layer_num, (size_t)log_exp_num}, AllocationType::DEVICE}, {"eplb_log_stats"});
    gpu_loads_buf = device->allocateBuffer(
        {DataType::TYPE_INT32, {(size_t)layer_num, (size_t)ep_size}, AllocationType::DEVICE}, {"eplb_gpu_loads"});

    log_stats_gpu = Buffer2torchTensor(log_stats_buf, false);
    gpu_loads_gpu = Buffer2torchTensor(gpu_loads_buf, false);
}

void BalanceStatsBuffers::reset() {
    log_stats.zero_();
    gpu_loads.zero_();
    log_stats_gpu.zero_();
    gpu_loads_gpu.zero_();
}

void LoadFlags::init(DeviceBase* device) {
    flag_gpu  = device->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::DEVICE}, {"flag_gpu"});
    flag_sync = device->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::DEVICE}, {"flag_sync"});
    flag_host = device->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::HOST}, {"flag_host"});
}

void LoadFlags::setReady(bool ready, DeviceBase* device) {
    int value = ready ? 0 : -1;
    device->bufMemset(*flag_gpu, value);
}

bool LoadFlags::isReady(DeviceBase* device) {
    // sync all ranks load_flag_tensor_
    flag_sync = device->allReduce({flag_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP, flag_sync}).buffer;

    device->copy({*flag_host, *flag_sync});
    // Repeatedly executing “x = ReduceSum(x)” with tp8 will cause an overflow and set the value of x to 0
    device->bufMemset(*flag_gpu, -1);
    // if all load_flag_tensor_ is 0, return true
    return *flag_host->data<int>() == 0;
}

void EplbController::init(const EPLBConfig& eplb_control_data, DeviceBase* device, const EPLBConfig& eplb_config) {
    this->eplb_control_data = eplb_control_data;

    control_step = eplb_config.eplb_control_step;
    RTP_LLM_LOG_INFO("EPLB control step: %d", control_step);

    auto eplb_control_data_list = eplb_control_data.toList();
    eplb_control_data_buf_host  = device->allocateBuffer(
        {DataType::TYPE_INT32, {eplb_control_data_list.size()}, AllocationType::HOST}, {"eplb_control_data_host"});
    eplb_control_data_buf_device = device->allocateBuffer(
        {DataType::TYPE_INT32, {eplb_control_data_list.size()}, AllocationType::DEVICE}, {"eplb_control_data_device"});
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

EPLBConfig EplbController::getAndSyncData(DeviceBase* device) {
    // copy control data to host buffer
    EPLBConfig cur_data;
    {
        lock_guard<mutex> lock(eplb_control_mutex);
        cur_data = eplb_control_data;
    }
    auto eplb_control_data_list     = eplb_control_data.toList();
    int* eplb_control_data_host_ptr = eplb_control_data_buf_host->data<int>();
    for (int i = 0; i < eplb_control_data_list.size(); ++i) {
        eplb_control_data_host_ptr[i] = eplb_control_data_list[i];
    }

    // copy to device
    device->copy({*eplb_control_data_buf_device, *eplb_control_data_buf_host, false});

    // broadcast to all ranks
    device->broadcast({{eplb_control_data_buf_device}, 0, ParallelMode::DP_AND_TP});

    // copy to host
    device->copy({*eplb_control_data_buf_host, *eplb_control_data_buf_device});

    // convert to EPLBConfig
    auto eplb_control_data = EPLBConfig::fromList(eplb_control_data_host_ptr);

    return eplb_control_data;
}

ExpertBalancer::ExpertBalancer(size_t                       log_exp_num,
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
                               const EPLBConfig&            eplb_config):

    device_(device),
    num_logic_experts_(log_exp_num),
    num_physic_experts_(phy_exp_num),
    ep_rank_(ep_rank),
    ep_size_(ep_size),
    metrics_reporter_(metrics_reporter),
    eplb_python_wrapper_(py_eplb) {
    eplb_control_data_ = eplb_config;

    // init memory
    stats_.init(num_layers, log_exp_num, ep_size_, device_);
    eplb_plan_buffers_.init(log_exp_num, phy_exp_num, hidden_size, moe_size, ep_size, dtype, quant_algo, device_);
    eplb_plan_tensors_.init(log_exp_num, phy_exp_num);
    load_flags_.init(device_);
    load_flags_.setReady(false, device_);
    eplb_controller_.init(eplb_control_data_, device, eplb_config);

    test_mode_ = eplb_config.eplb_test_mode;

    balance_layer_per_step_ = eplb_config.eplb_balance_layer_per_step;

    resetPlan(true);
}

ExpertBalancer::~ExpertBalancer() {}

bool ExpertBalancer::checkDownScale(int active_ranks_num) {
    if (num_physic_experts_ / ep_size_ * active_ranks_num >= num_logic_experts_) {
        RTP_LLM_LOG_INFO(
            "physical expert num support downscale, num_physic_experts_(%d) / ep_size_(%d) * active_ranks_num_(%d) = %d, num_logic_experts_(%d)",
            num_physic_experts_,
            ep_size_,
            active_ranks_num,
            num_physic_experts_ / ep_size_ * active_ranks_num,
            num_logic_experts_);
        return true;
    }
    RTP_LLM_LOG_INFO(
        "physical expert num not support downscale, num_physic_experts_(%d) / ep_size_(%d) * active_ranks_num_(%d) = %d, num_logic_experts_(%d)",
        num_physic_experts_,
        ep_size_,
        active_ranks_num,
        num_physic_experts_ / ep_size_ * active_ranks_num,
        num_logic_experts_);
    return false;
}

void ExpertBalancer::executeDownScale(GptModel& model, const torch::Tensor& active_ranks_tensor_cpu) {
    size_t                       num_layers = stats_.log_stats.size(0);
    std::vector<EplbPlanTensors> eplb_plan_all_layer_tensors(num_layers);

    try {
        // 1. create downscale plan
        RTP_LLM_LOG_INFO("Start create downscale plan, num_layers = %ld", num_layers);
        eplb_python_wrapper_.createDownScalePlan(
            stats_.log_stats, eplb_plan_all_layer_tensors, active_ranks_tensor_cpu);
        // 2. load downscale weight
        RTP_LLM_LOG_INFO("Finish create downscale plan, start load downscale weight");
        for (size_t i = 0; i < num_layers; i++) {
            try {
                // 2.1 load weight disk to cpu
                eplb_python_wrapper_.loadBalanceWeight(ep_rank_, ep_size_, eplb_plan_all_layer_tensors[i]);
                // 2.2 update weight cpu to gpu
                updateBalanceWeight(eplb_plan_all_layer_tensors[i], model);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("ExpertBalancer: Failed to load balance weight for layer %zu: %s", i, e.what());
                throw;
            }
        }
        // 3. active ranks synchronize to avoid timeout
        RTP_LLM_LOG_INFO("Finish load downscale weight, start active ranks synchronize");
        syncUpdateWeights();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("ExpertBalancer: Downscale plan execution failed: %s", e.what());
        throw;
    }
}

void ExpertBalancer::stepForward(GptModel&                       model,
                                 RtpLLMExecutorMetricsCollector& executor_collector,
                                 const ElasticEPStats&           ep_stats) {
    const bool          is_downscale            = ep_stats.is_downscale_;
    const int           active_ranks_num        = ep_stats.active_ranks_num_;
    const torch::Tensor active_ranks_tensor_cpu = ep_stats.active_ranks_tensor_cpu_;
    if (is_downscale && checkDownScale(active_ranks_num)) {
        executeDownScale(model, active_ranks_tensor_cpu);
        return;
    }

    if (eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::NONE)) {
        return;
    }

    syncController();

    OverallExpertStats& stats = model.overall_expert_stats_;

    // report stats
    reportStats(stats);

    // eplb plan
    excuteEplbPlan(stats, model, active_ranks_tensor_cpu);
}

bool ExpertBalancer::updateEplbConfig(const EPLBConfig& config) {
    eplb_controller_.setData(config);
    return true;
}

void ExpertBalancer::syncController() {
    // sync control data
    if (eplb_controller_.stepAndCheckSyncStep()) {
        auto eplb_control_data = eplb_controller_.getAndSyncData(device_);
        if (eplb_control_data.eplb_mode != eplb_control_data_.eplb_mode
            || eplb_control_data.eplb_update_time != eplb_control_data_.eplb_update_time) {
            eplb_control_data_ = eplb_control_data;
            RTP_LLM_LOG_INFO("EPLB config changed to %s", eplb_control_data_.to_string().c_str());
        }
    }
}

void ExpertBalancer::reportStats(OverallExpertStats& stats) {
    if (metrics_reporter_
        && eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::STATS, EplbMode::ALL)) {
        int layer_num = stats.layer_num;
        executor_collector_.gpu_loads.resize(layer_num);
        executor_collector_.ep_rank = ep_rank_;

        BufferPtr gpu_loads_cpu = device_->clone({*stats.stats_buf.gpu_loads_buf, AllocationType::HOST});
        int*      gpu_loads     = gpu_loads_cpu->data<int>();

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

void ExpertBalancer::excuteEplbPlan(OverallExpertStats&  stats,
                                    GptModel&            model,
                                    const torch::Tensor& active_ranks_tensor_cpu) {
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
                createPlan(active_ranks_tensor_cpu);
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
                load_flags_.setReady(true, device_);
                if (syncPlanWeightsLoadStatus()) {
                    processPlanWeights();
                    applyPlanWeights(model);
                    // TODO: pymodel update balance weight in EPLB
                    // updateBalanceWeight(eplb_plan_tensors_, model);
                    load_flags_.setReady(false, device_);
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

void ExpertBalancer::copyFromTensor(torch::Tensor& tensor, BufferPtr& buffer) {
    auto tensor_buf = torchTensor2Buffer(tensor);
    device_->copy({*buffer, *tensor_buf, false});
}

void ExpertBalancer::copyToTensor(BufferPtr& buffer, torch::Tensor& tensor) {
    auto tensor_buf = torchTensor2Buffer(tensor);
    device_->copy({*tensor_buf, *buffer, false});
}

void ExpertBalancer::createPlan(const torch::Tensor& active_ranks_tensor_cpu) {
    // pre run
    device_->allReduce({stats_.log_stats_buf, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});
    device_->allReduce({stats_.gpu_loads_buf, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});

    // copy stats buffer(device) to tensor(host) [implicit sync]
    copyToTensor(stats_.log_stats_buf, stats_.log_stats);
    copyToTensor(stats_.gpu_loads_buf, stats_.gpu_loads);

    if (ep_rank_ == 0) {
        eplb_python_wrapper_.createBalancePlan(
            stats_.log_stats, stats_.gpu_loads, eplb_plan_tensors_, active_ranks_tensor_cpu);

        // copy tensor(host) to buffer(device)
        // note: it's ok to use async copy, since the tensor host ptr will not be released
        copyFromTensor(eplb_plan_tensors_.layer_id_buf, eplb_plan_buffers_.layer_id_buf);
        copyFromTensor(eplb_plan_tensors_.logic_expert_cnt, eplb_plan_buffers_.logic_expert_cnt);
        copyFromTensor(eplb_plan_tensors_.log2phy, eplb_plan_buffers_.log2phy);
        copyFromTensor(eplb_plan_tensors_.phy2log, eplb_plan_buffers_.phy2log);
    }

    device_->broadcast({{eplb_plan_buffers_.layer_id_buf,
                         eplb_plan_buffers_.logic_expert_cnt,
                         eplb_plan_buffers_.log2phy,
                         eplb_plan_buffers_.phy2log},
                        0,
                        ParallelMode::DP_AND_TP});

    // copy plan buffer(device) to tensor(host) [inplict sync]
    copyToTensor(eplb_plan_buffers_.layer_id_buf, eplb_plan_tensors_.layer_id_buf);
    copyToTensor(eplb_plan_buffers_.logic_expert_cnt, eplb_plan_tensors_.logic_expert_cnt);
    copyToTensor(eplb_plan_buffers_.log2phy, eplb_plan_tensors_.log2phy);
    copyToTensor(eplb_plan_buffers_.phy2log, eplb_plan_tensors_.phy2log);
}

void ExpertBalancer::processPlanWeights() {
    eplb_plan_buffers_.layer_id = eplb_plan_tensors_.layer_id;
    if (eplb_plan_buffers_.moe_weight_1->isQBuffer()) {
        auto moe_buf_1 = static_cast<QBuffer*>(eplb_plan_buffers_.moe_weight_1.get());
        auto moe_buf_2 = static_cast<QBuffer*>(eplb_plan_buffers_.moe_weight_2.get());
        auto moe_w1    = moe_buf_1->kernelPtr();
        auto moe_w2    = moe_buf_2->kernelPtr();
        auto moe_s1    = moe_buf_1->scalesPtr();
        auto moe_s2    = moe_buf_2->scalesPtr();
        copyFromTensor(eplb_plan_tensors_.moe_weight_1, moe_w1);
        copyFromTensor(eplb_plan_tensors_.moe_scale_1, moe_s1);
        copyFromTensor(eplb_plan_tensors_.moe_weight_2, moe_w2);
        copyFromTensor(eplb_plan_tensors_.moe_scale_2, moe_s2);
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

void ExpertBalancer::applyPlanWeights(GptModel& model) {
    auto& result          = eplb_plan_buffers_;
    auto& balanced_layer  = model.weights_.layers[result.layer_id].ffn_weights;
    auto  exchange_weight = [&](ConstBufferPtr& old_weight, BufferPtr& new_weight) {
        new_weight = const_pointer_cast<Buffer>(exchange(old_weight, new_weight));
    };

    exchange_weight(balanced_layer.moe_gate_weight->kernel, result.moe_weight_1);
    exchange_weight(balanced_layer.moe_down_weight->kernel, result.moe_weight_2);
    exchange_weight(balanced_layer.log2phy, result.log2phy);
    exchange_weight(balanced_layer.logic_expert_cnt, result.logic_expert_cnt);
}

bool ExpertBalancer::syncPlanWeightsLoadStatus() {
    return load_flags_.isReady(device_);
}

void ExpertBalancer::updateStats(OverallExpertStats& stats) {
    auto log_stats = Buffer2torchTensor(stats.stats_buf.log_stats_buf, false);
    auto gpu_loads = Buffer2torchTensor(stats.stats_buf.gpu_loads_buf, false);
    stats_.log_stats_gpu.add_(log_stats);
    stats_.gpu_loads_gpu.add_(gpu_loads);
}

void ExpertBalancer::syncUpdateWeights() {
    // Create a tensor with size ep_size_ to store allgather results
    // Initialize with current rank id at the corresponding position
    py::gil_scoped_acquire acquire;
    try {
        py::module deepep         = py::module::import("rtp_llm.models_py.distributed.deepep_wrapper");
        py::object deepep_wrapper = deepep.attr("get_deepep_wrapper_if_initialized")();
        if (deepep_wrapper.is_none()) {
            RTP_LLM_LOG_WARNING("ExpertBalancer: DeepEP wrapper is not initialized");
            return;
        }
        torch::Tensor allgather_tensor = torch::zeros({static_cast<int64_t>(ep_size_)}, torch::kInt32).cuda();

        allgather_tensor[ep_rank_] = 1;
        deepep_wrapper.attr("buffer").attr("low_latency_allgather")(allgather_tensor, false, false, false);

        torch::cuda::synchronize();

        // print allgather result
        torch::Tensor result_cpu   = allgather_tensor.cpu();
        std::string   rank_ids_str = "[";
        for (int64_t i = 0; i < ep_size_; ++i) {
            int rank_id_value = result_cpu[i].item<int>();
            rank_ids_str += std::to_string(rank_id_value);
            if (i < ep_size_ - 1) {
                rank_ids_str += ", ";
            }
        }
        rank_ids_str += "]";
        RTP_LLM_LOG_INFO("ExpertBalancer: allgather completed successfully, received all rank ids: %s",
                         rank_ids_str.c_str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("ExpertBalancer: syncUpdateWeights failed with exception: %s", e.what());
    }
}

void ExpertBalancer::updateBalanceWeight(EplbPlanTensors& eplb_plan, GptModel& model) {
    py::gil_scoped_acquire acquire;
    // Try to get Python model from PyWrappedModel
    PyWrappedModel* py_model = dynamic_cast<PyWrappedModel*>(&model);
    if (py_model == nullptr) {
        // Not a PyWrappedModel, skip Python weight update
        RTP_LLM_LOG_ERROR("ExpertBalancer: Model is not PyWrappedModel, failed to update balance weight");
        return;
    }

    try {
        // Update weights using updateLayerWeight method
        // Note: weight names should match W.moe_w1, W.moe_w2, etc.
        py_model->updateLayerWeight(
            eplb_plan.layer_id, "partial_moe_weights.intermediate_weight.kernel", eplb_plan.moe_weight_1, true);
        py_model->updateLayerWeight(
            eplb_plan.layer_id, "partial_moe_weights.intermediate_weight2.kernel", eplb_plan.moe_weight_2, true);

        // Update scales if they exist and are not empty
        if (eplb_plan.moe_scale_1.defined() && eplb_plan.moe_scale_1.numel() > 0) {
            py_model->updateLayerWeight(eplb_plan.layer_id,
                                        "partial_moe_weights.intermediate_weight.weight_only_quant_scale",
                                        eplb_plan.moe_scale_1,
                                        true);
        }
        if (eplb_plan.moe_scale_2.defined() && eplb_plan.moe_scale_2.numel() > 0) {
            py_model->updateLayerWeight(eplb_plan.layer_id,
                                        "partial_moe_weights.intermediate_weight2.weight_only_quant_scale",
                                        eplb_plan.moe_scale_2,
                                        true);
        }

        // Update EPLB mapping weights
        if (eplb_plan.log2phy.defined() && eplb_plan.log2phy.numel() > 0) {
            py_model->updateLayerWeight(eplb_plan.layer_id, "moe_eplb.log2phy", eplb_plan.log2phy, true);
        }
        if (eplb_plan.logic_expert_cnt.defined() && eplb_plan.logic_expert_cnt.numel() > 0) {
            py_model->updateLayerWeight(
                eplb_plan.layer_id, "moe_eplb.logic_expert_cnt", eplb_plan.logic_expert_cnt, true);
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR(
            "ExpertBalancer: Failed to update balance weight for layer %d: %s", eplb_plan.layer_id, e.what());
        throw;
    }
}
}  // namespace rtp_llm
