#pragma once
#include <cstdint>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/PrefillTimeEstimator.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/LookupMapImpl.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/EstimatorConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace rtp_llm_master {

class LookupPrefillEstimator: public TimeEstimatorBase {
public:
    LookupPrefillEstimator() = default;
    bool                    init(const std::vector<LookupConfig>& configs);
    absl::StatusOr<int64_t> estimate(const std::string& machine_info, const TaskInfo& task_info) const override;

private:
    std::unordered_map<std::string, LookupMapImpl> impl_map_;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm