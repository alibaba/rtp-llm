#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "src/fastertransformer/utils/logger.h"

namespace rtp_llm {

// default 1 minute and 1000
const size_t StepRecorder::STEP_RECORDS_TIME_RANGE =
    autil::EnvUtil::getEnv("STEP_RECORDS_TIME_RANGE", 60l * 1000 * 1000);
const size_t StepRecorder::STEP_RECORDS_MAX_SIZE = autil::EnvUtil::getEnv("STEP_RECORDS_MAX_SIZE", 1000);

PIController::PIController(float kp, float ki): kp_(kp), ki_(ki) {}

float PIController::getCurrent() {
    return current_;
}

void PIController::addTarget(float target) {
    float diff = target - current_;
    sum_diffs += diff;
    float diff_next = kp_ * diff + ki_ * sum_diffs;
    if (target > current_) {
        current_ += std::min(diff, diff_next);
    } else {
        current_ += std::max(diff, diff_next);
    }
    if (current_ < lower_limit_) {
        current_  = lower_limit_;
        sum_diffs = 0;
    }
}

void PIController::reset() {
    current_  = 0;
    sum_diffs = 0;
}

size_t StepRecorder::getStepPerMin() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (step_time_records_.size() < 2) {
        return STEP_RECORDS_TIME_RANGE / min_step_latency_;
    }
    const auto range = step_time_records_.back() - step_time_records_.front();
    return step_time_records_.size() * STEP_RECORDS_TIME_RANGE / range;
}

size_t StepRecorder::getStepLatency() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (step_time_records_.size() < 2) {
        return min_step_latency_;
    }
    return (step_time_records_.back() - step_time_records_.front()) / step_time_records_.size();
}

size_t StepRecorder::getStepCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return step_count_controller_.getCurrent();
}

void StepRecorder::addStepCount(size_t step_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    step_count_controller_.addTarget(step_count);
}

void StepRecorder::addStepTime(size_t step_time_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!step_time_records_.empty()) {
        if (step_time_us < step_time_records_.back()) {
            FT_LOG_ERROR("step time not in order");
            return;
        }
        min_step_latency_ = std::min(min_step_latency_, step_time_us - step_time_records_.back());
    }
    step_time_records_.push(step_time_us);
    while (step_time_records_.size() > STEP_RECORDS_MAX_SIZE
           || step_time_us - step_time_records_.front() > STEP_RECORDS_TIME_RANGE) {
        step_time_records_.pop();
    }
    if (step_time_records_.size() > 1) {
        avg_latency_controller_.addTarget((step_time_records_.back() - step_time_records_.front()) * 1.0
                                          / step_time_records_.size());
    }
}

void StepRecorder::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!step_time_records_.empty()) {
        step_time_records_.pop();
    }
    avg_latency_controller_.reset();
}

void registerLoadBalanceInfo(const py::module& m) {
    pybind11::class_<LoadBalanceInfo>(m, "LoadBalanceInfo")
        .def(pybind11::init<>())
        .def_readwrite("step_latency_us", &LoadBalanceInfo::step_latency_us)
        .def_readwrite("step_per_minute", &LoadBalanceInfo::step_per_minute)
        .def_readwrite("iterate_count", &LoadBalanceInfo::iterate_count)
        .def_readwrite("available_kv_cache", &LoadBalanceInfo::available_kv_cache)
        .def_readwrite("total_kv_cache", &LoadBalanceInfo::total_kv_cache);
}

}  // namespace rtp_llm
