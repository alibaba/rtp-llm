#include "rtp_llm/cpp/engine_base/LoadBalance.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

size_t StepRecorder::STEP_RECORDS_TIME_RANGE = 60 * 1000 * 1000;
size_t StepRecorder::STEP_RECORDS_MAX_SIZE   = 1000;

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
    if (step_records_.size() < 2) {
        return min_step_latency_ ? (STEP_RECORDS_TIME_RANGE / min_step_latency_) : 0;
    }
    const auto interval = getIntervalPerStepLatency();
    if (interval == 0) {
        return 0;
    }
    return STEP_RECORDS_TIME_RANGE / getIntervalPerStepLatency();
}

size_t StepRecorder::getStepLatency() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (step_records_.size() < 2) {
        return min_step_latency_;
    }
    return getIntervalPerStepLatency();
}

size_t StepRecorder::getStepCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return step_count_controller_.getCurrent();
}

void StepRecorder::addStepCount(size_t step_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    step_count_controller_.addTarget(step_count);
}

void StepRecorder::registerStep(size_t step_time_us, size_t batch_avg_gen_num) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!step_records_.empty()) {
        if (step_time_us < step_records_.back().time_us) {
            RTP_LLM_LOG_ERROR("step time not in order");
            return;
        }
        min_step_latency_ = std::min(min_step_latency_,
                                     (size_t)((step_time_us - step_records_.back().time_us) * 1.0
                                              / ((batch_avg_gen_num + step_records_.back().batch_avg_gen_num) / 2.0)));
    }
    step_records_.push({step_time_us, batch_avg_gen_num});
    queue_total_gen_num_ += batch_avg_gen_num;
    while (step_records_.size() > STEP_RECORDS_MAX_SIZE
           || step_time_us - step_records_.front().time_us > STEP_RECORDS_TIME_RANGE) {
        queue_total_gen_num_ -= step_records_.front().batch_avg_gen_num;
        step_records_.pop();
    }
    if (step_records_.size() > 1) {
        avg_latency_controller_.addTarget(getIntervalPerStepLatency());
    }
}

void StepRecorder::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!step_records_.empty()) {
        step_records_.pop();
    }
    queue_total_gen_num_ = 0;
    avg_latency_controller_.reset();
}

bool StepRecorder::empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return step_records_.empty();
}

}  // namespace rtp_llm
