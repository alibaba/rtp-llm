#pragma once

#include <mutex>
#include <queue>
#include <vector>
#include "autil/EnvUtil.h"
#include "src/fastertransformer/utils/py_utils/pybind_utils.h"

namespace rtp_llm {

class PIController {
public:
    PIController(float kp = 0.0, float ki = 0.1);

    float getCurrent();

    void addTarget(float target);

    void reset();

private:
    float current_     = 1.0;
    float sum_diffs    = 0;
    float kp_          = 0.0;
    float ki_          = 0.1;
    float lower_limit_ = 1.0;
};

class StepRecorder {
public:
    size_t getStepLatency();

    size_t getStepCount();

    size_t getStepPerMin();

    void addStepCount(size_t step_count);

    void addStepTime(size_t step_time_us);

    void reset();

private:
    // all time is us
    const static size_t STEP_RECORDS_MAX_SIZE;
    const static size_t STEP_RECORDS_TIME_RANGE;

    PIController avg_latency_controller_;
    PIController step_count_controller_;

    std::queue<int64_t> step_time_records_;
    size_t              min_step_latency_ = 10 * 1000 * 1000;  // 10s
    std::mutex          mutex_;
};

struct LoadBalanceInfo {
    int64_t step_latency_us;
    int64_t iterate_count;
    int64_t step_per_minute;
    int64_t available_kv_cache;
    int64_t total_kv_cache;
};

void registerLoadBalanceInfo(const pybind11::module& m);

}  // namespace rtp_llm
