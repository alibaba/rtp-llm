#include "maga_transformer/cpp/eplb/ExpertBalancer.h"
#include <thread>
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

void EplbPlanBuffers::init(size_t          log_exp_num,
                           size_t          phy_exp_num,
                           size_t          hidden_size,
                           size_t          moe_size,
                           size_t          ep_size,
                           ft::DataType    dtype,
                           ft::QuantAlgo   quant_algo,
                           ft::DeviceBase* device) {
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

void BalanceStatsBuffers::init(int layer_num, int log_exp_num, int ep_size, ft::DeviceBase* device) {
    // device tensor
    log_stats = torch::zeros({layer_num, log_exp_num}, torch::kInt32);
    gpu_loads = torch::zeros({layer_num, ep_size}, torch::kInt32);

    log_stats_buf = device->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)layer_num, (size_t)log_exp_num}, ft::AllocationType::DEVICE},
        {"eplb_log_stats"});
    gpu_loads_buf = device->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)layer_num, (size_t)ep_size}, ft::AllocationType::DEVICE},
        {"eplb_gpu_loads"});

    log_stats_gpu = Buffer2torchTensor(log_stats_buf, false);
    gpu_loads_gpu = Buffer2torchTensor(gpu_loads_buf, false);
}

void BalanceStatsBuffers::reset() {
    log_stats.zero_();
    gpu_loads.zero_();
    log_stats_gpu.zero_();
    gpu_loads_gpu.zero_();
}

void LoadFlags::init(ft::DeviceBase* device) {
    flag_gpu  = device->allocateBuffer({ft::DataType::TYPE_INT32, {1}, ft::AllocationType::DEVICE}, {"flag_gpu"});
    flag_sync = device->allocateBuffer({ft::DataType::TYPE_INT32, {1}, ft::AllocationType::DEVICE}, {"flag_sync"});
    flag_host = device->allocateBuffer({ft::DataType::TYPE_INT32, {1}, ft::AllocationType::HOST}, {"flag_host"});
}

void LoadFlags::setReady(bool ready, ft::DeviceBase* device) {
    int value = ready ? 0 : -1;
    device->bufMemset(*flag_gpu, value);
}

bool LoadFlags::isReady(ft::DeviceBase* device) {
    // sync all ranks load_flag_tensor_
    flag_sync = device->allReduce({flag_gpu, ft::ReduceOp::Sum, false, ft::ParallelMode::DP_AND_TP, flag_sync}).buffer;

    device->copy({*flag_host, *flag_sync});

    // if all load_flag_tensor_ is 0, return true
    return *flag_host->data<int>() == 0;
}

ExpertBalancer::ExpertBalancer(size_t                       log_exp_num,
                               size_t                       phy_exp_num,
                               size_t                       num_layers,
                               size_t                       moe_size,
                               size_t                       hidden_size,
                               size_t                       update_time,
                               size_t                       stats_update_time,
                               size_t                       ep_rank,
                               size_t                       ep_size,
                               py::object                   py_eplb,
                               ft::DataType                 dtype,
                               ft::DeviceBase*              device,
                               ft::EplbMode                 eplb_mode,
                               ft::QuantAlgo                quant_algo,
                               kmonitor::MetricsReporterPtr metrics_reporter):
    device_(device),
    num_logic_experts_(log_exp_num),
    num_physic_experts_(phy_exp_num),
    update_time_(update_time),
    ep_rank_(ep_rank),
    ep_size_(ep_size),
    eplb_mode_(eplb_mode),
    metrics_reporter_(metrics_reporter),
    eplb_python_wrapper_(py_eplb) {
    switch (eplb_mode_) {
        case EplbMode::NONE:
            return;  // no need to init
        case EplbMode::EPLB:
            enable_eplb_ = true;
            break;
        case EplbMode::STATS:
            enable_stats_ = true;
            break;
        case EplbMode::ALL:
            enable_eplb_  = true;
            enable_stats_ = true;
            break;
    }

    // init memory
    stats_.init(num_layers, log_exp_num, ep_size_, device_);
    eplb_plan_buffers_.init(log_exp_num, phy_exp_num, hidden_size, moe_size, ep_size, dtype, quant_algo, device_);
    eplb_plan_tensors_.init(log_exp_num, phy_exp_num);
    load_flags_.init(device_);
    load_flags_.setReady(false, device_);
    clean_stats_step_ = (stats_update_time + update_time - 1) / update_time;
    resetPlan(true);
}

ExpertBalancer::~ExpertBalancer() {}

void ExpertBalancer::stepForward(GptModel& model, RtpLLMExecutorMetricsCollector& executor_collector) {
    if (eplb_mode_ == EplbMode::NONE) {
        return;
    }

    ft::OverallExpertStats& stats = model.overall_expert_stats_;

    // report stats
    reportStats(stats);

    // eplb plan
    excuteEplbPlan(stats, model);
}

void ExpertBalancer::reportStats(ft::OverallExpertStats& stats) {
    if (metrics_reporter_ && enable_stats_) {
        int layer_num = stats.layer_num;
        executor_collector_.gpu_loads.resize(layer_num);
        executor_collector_.ep_rank = ep_rank_;

        BufferPtr gpu_loads_cpu = device_->clone({*stats.stats_buf.gpu_loads_buf, ft::AllocationType::HOST});
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

void ExpertBalancer::excuteEplbPlan(ft::OverallExpertStats& stats, GptModel& model) {
    if (enable_eplb_) {
        EplbPlanStatus status = getPlanStatus();
        switch (status) {
            case EplbPlanStatus::INIT:
                update_cnt_++;
                updateStats(stats);
                if (update_cnt_ >= update_time_) {
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
                load_thread.detach();
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
                    load_flags_.setReady(false, device_);
                    setPlanStatus(EplbPlanStatus::INIT);
                    resetPlan();
                }
                break;
        }
    }
}

void ExpertBalancer::resetPlan(bool force_clean) {
    stats_.reset();
    update_cnt_ = 0;
}

void ExpertBalancer::copyFromTensor(torch::Tensor& tensor, ft::BufferPtr& buffer) {
    auto tensor_buf = ft::torchTensor2Buffer(tensor);
    device_->copy({*buffer, *tensor_buf, false});
}

void ExpertBalancer::copyToTensor(ft::BufferPtr& buffer, torch::Tensor& tensor) {
    auto tensor_buf = ft::torchTensor2Buffer(tensor);
    device_->copy({*tensor_buf, *buffer, false});
}

void ExpertBalancer::createPlan() {
    // pre run
    device_->allReduce({stats_.log_stats_buf, ReduceOp::Sum, false, ft::ParallelMode::DP_AND_TP});
    device_->allReduce({stats_.gpu_loads_buf, ReduceOp::Sum, false, ft::ParallelMode::DP_AND_TP});

    // copy stats buffer(device) to tensor(host) [implicit sync]
    copyToTensor(stats_.log_stats_buf, stats_.log_stats);
    copyToTensor(stats_.gpu_loads_buf, stats_.gpu_loads);

    if (ep_rank_ == 0) {
        eplb_python_wrapper_.createBalancePlan(stats_.log_stats, stats_.gpu_loads, eplb_plan_tensors_);

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
                        ft::ParallelMode::DP_AND_TP});

    // copy plan buffer(device) to tensor(host) [inplict sync]
    copyToTensor(eplb_plan_buffers_.layer_id_buf, eplb_plan_tensors_.layer_id_buf);
    copyToTensor(eplb_plan_buffers_.logic_expert_cnt, eplb_plan_tensors_.logic_expert_cnt);
    copyToTensor(eplb_plan_buffers_.log2phy, eplb_plan_tensors_.log2phy);
    copyToTensor(eplb_plan_buffers_.phy2log, eplb_plan_tensors_.phy2log);
}

void ExpertBalancer::processPlanWeights() {
    eplb_plan_buffers_.layer_id = eplb_plan_tensors_.layer_id;
    if (eplb_plan_buffers_.moe_weight_1->isQBuffer()) {
        auto moe_buf_1 = static_cast<ft::QBuffer*>(eplb_plan_buffers_.moe_weight_1.get());
        auto moe_buf_2 = static_cast<ft::QBuffer*>(eplb_plan_buffers_.moe_weight_2.get());
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

void ExpertBalancer::updateStats(ft::OverallExpertStats& stats) {
    auto log_stats = Buffer2torchTensor(stats.stats_buf.log_stats_buf, false);
    auto gpu_loads = Buffer2torchTensor(stats.stats_buf.gpu_loads_buf, false);
    stats_.log_stats_gpu.add_(log_stats);
    stats_.gpu_loads_gpu.add_(gpu_loads);
}

}  // namespace rtp_llm
