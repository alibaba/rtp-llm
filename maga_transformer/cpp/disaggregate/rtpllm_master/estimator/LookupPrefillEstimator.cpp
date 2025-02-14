#include "autil/StringUtil.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/estimator/LookupPrefillEstimator.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {
namespace rtp_llm_master {
    bool LookupPrefillEstimator::init(const std::vector<LookupConfig>& configs) {
        for (auto& config: configs) {
            impl_map_.emplace(config.machine_info, LookupMapImpl());
            if (!impl_map_[config.machine_info].init(config.config_path)) {
                FT_LOG_ERROR("failed to init LookupMapImpl with machine:[%s], config:[%s], init LookupPrefillEstimator failed");
                return false;
            }
        }
        return true;
    }

    absl::StatusOr<int64_t> LookupPrefillEstimator::estimate(const std::string& machine_info, const TaskInfo& task_info) const {
        auto lookup_map_iter = impl_map_.find(machine_info);
        if (lookup_map_iter == impl_map_.end()) {
            static std::string error_msg_format = "failed to find %s in impl_map_";
            return absl::InternalError(autil::StringUtil::formatString(error_msg_format, machine_info.c_str()));
        }
        return lookup_map_iter->second.estimate(task_info);
    }
}
}  // namespace rtp_llm