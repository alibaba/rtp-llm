#pragma once
#include <cstdint>
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/estimator/EstimatorConfig.h"

namespace rtp_llm {
namespace rtp_llm_master {

struct TaskInfo {
    int prefix_length;
    int input_length;
};

// simple version impl, use input length as time
class TimeEstimatorBase {
public:
    virtual absl::StatusOr<int64_t> estimate(const std::string& machine_info, const TaskInfo& task_info) const = 0;
};

// just for ut
class SimpleTimeEstimator: public TimeEstimatorBase {
public:
    SimpleTimeEstimator() = default;
    absl::StatusOr<int64_t> estimate(const std::string& machine_info, const TaskInfo& task_info) const override;
};

std::shared_ptr<TimeEstimatorBase> createPrefillTimeEstimator(const EstimatorConfig& config);

}  // namespace rtp_llm_master
}  // namespace rtp_llm