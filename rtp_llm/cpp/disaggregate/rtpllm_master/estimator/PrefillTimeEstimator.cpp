#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/PrefillTimeEstimator.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/LookupPrefillEstimator.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {
namespace rtp_llm_master {
absl::StatusOr<int64_t> SimpleTimeEstimator::estimate(const std::string& machine_info,
                                                      const TaskInfo&    task_info) const {
    return std::max(task_info.input_length, 16);
}

std::shared_ptr<TimeEstimatorBase> createPrefillTimeEstimator(const EstimatorConfig& config) {
    if (config.estimator_type == "local") {
        return std::make_shared<SimpleTimeEstimator>();
    } else if (config.estimator_type == "lookup") {
        auto estimator = std::make_shared<LookupPrefillEstimator>();
        if (!estimator->init(config.lookup_configs)) {
            return nullptr;
        }
        return estimator;
    }
    RTP_LLM_LOG_WARNING("unknown estimator type: [%s]", config.estimator_type.c_str());
    return nullptr;
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm