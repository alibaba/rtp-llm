#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "autil/StringUtil.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/LookupMapImpl.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {
namespace rtp_llm_master {

int floorOfValueWithStep(int x, int step) {
    return x / step * step;
}

int ceilOfValueWithStep(int x, int step) {
    return x == 0 ? 0 : ((x - 1) / step + 1) * step;
}

float calNodeDistance(const std::pair<int, int>& x, const std::pair<int, int>& y) {
    return std::sqrt(std::pow(x.first - y.first, 2) + std::pow(x.second - y.second, 2));
}

#define HELPER_FUNC(input)                                                                                             \
    if (s == #input)                                                                                                   \
        return EstimateFunction::input;
EstimateFunction get_estimate_func_from_string(const std::string& s) {
    HELPER_FUNC(MEAN);
    HELPER_FUNC(FLOOR);
    HELPER_FUNC(CEIL);
    RTP_LLM_LOG_WARNING("unrecognized estimate function: [%s], using EstimateFunction::CEIL as default", s.c_str());
    return EstimateFunction::CEIL;
}
#undef HELPER_FUNC

bool LookupMapImpl::init(const std::string& filePath) {
    std::ifstream fileStream(filePath);
    if (!fileStream.is_open()) {
        RTP_LLM_LOG_ERROR("failed to open file at [%s], LookupMapImpl init failed", filePath.c_str());
        return false;
    }
    std::stringstream buffer;
    buffer << fileStream.rdbuf();
    std::string fileContent = buffer.str();
    fileStream.close();
    LookupMapJson lookup_map;
    try {
        autil::legacy::FromJsonString(lookup_map, fileContent);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("response deserialize failed, error: %s", e.what());
        return false;
    } catch (...) {
        RTP_LLM_LOG_WARNING("failed to parse file [%s] to LookupMapJson, please check", filePath.c_str());
        return false;
    }
    return initFromLookupMapJson(lookup_map);
}

bool LookupMapImpl::checkConfigValid(int                 input_lower_bound,
                                     int                 prefix_lower_bound,
                                     const SingleConfig& input_config,
                                     const SingleConfig& prefix_config) const {
    // prefix_config of different scope support euqal now
    if (input_lower_bound != input_config.lower_bound
        || (prefix_lower_bound != prefix_config.lower_bound && prefix_lower_bound != prefix_config.lower_bound + 1)) {
        RTP_LLM_LOG_WARNING("input or prefix lower bound match failed, expect: [%d:%d], actual: [%d:%d]",
                            input_lower_bound,
                            prefix_lower_bound,
                            input_config.lower_bound,
                            prefix_config.lower_bound);
        return false;
    }
    if ((input_config.lower_bound % input_config.step_size != 0)
        || (input_config.upper_bound % input_config.step_size != 0)) {
        RTP_LLM_LOG_WARNING("input config step length error, range: (%d, %d], step: %d",
                            input_config.lower_bound,
                            input_config.upper_bound,
                            input_config.step_size);
        return false;
    }
    if ((prefix_config.lower_bound % prefix_config.step_size != 0)
        || (prefix_config.upper_bound % prefix_config.step_size != 0)) {
        RTP_LLM_LOG_WARNING("prefix config step length error, range: (%d, %d], step: %d",
                            prefix_config.lower_bound,
                            prefix_config.upper_bound,
                            prefix_config.step_size);
        return false;
    }
    return true;
}

absl::Status LookupMapImpl::checkTaskBoundary(const TaskInfo& task_info) const {
    static std::string error_msg = "illegal node: (%d, %d) with boundary: (%d, %d) - [%d, %d]";
    if (task_info.prefix_length < 0 || task_info.input_length <= 0 || task_info.prefix_length > max_prefix_length_
        || task_info.input_length > max_input_length_) {
        return absl::OutOfRangeError(autil::StringUtil::formatString(
            error_msg, task_info.prefix_length, task_info.input_length, 0, 0, max_prefix_length_, max_input_length_));
    }
    return absl::OkStatus();
}

absl::StatusOr<int64_t> LookupMapImpl::getItemFromMap(const std::pair<int, int>& key) const {
    auto iter = map_.find(key);
    if (iter == map_.end()) {
        static std::string error_msg = "InteralError: failed to find node (%d, %d) from map";
        return absl::InternalError(autil::StringUtil::formatString(error_msg, key.first, key.second));
    }
    return iter->second;
}

absl::StatusOr<int64_t> LookupMapImpl::estimateInternal(const TaskInfo& task_info, const LookupMapScope& scope) const {
    int prefix_step = scope.prefix_config.step_size;
    int input_step  = scope.input_config.step_size;
    if (task_info.prefix_length % prefix_step == 0 && task_info.input_length % input_step == 0) {
        return getItemFromMap({task_info.prefix_length, task_info.input_length});
    }
    std::pair<int, int> lower(floorOfValueWithStep(task_info.prefix_length, prefix_step),
                              floorOfValueWithStep(task_info.input_length, input_step));
    std::pair<int, int> upper(ceilOfValueWithStep(task_info.prefix_length, prefix_step),
                              ceilOfValueWithStep(task_info.input_length, input_step));
    int64_t             floor_value, ceil_value;
    CHECK_AND_ASSIGN(floor_value, getItemFromMap(lower));
    CHECK_AND_ASSIGN(ceil_value, getItemFromMap(upper));
    switch (scope.estimate_func) {
        case EstimateFunction::CEIL:
            return ceil_value;
        case EstimateFunction::FLOOR:
            return floor_value;
        case EstimateFunction::MEAN:
            return floor_value
                   + int64_t((ceil_value - floor_value) * 1.0
                             * (calNodeDistance({task_info.prefix_length, task_info.input_length}, lower)
                                / calNodeDistance(lower, upper)));
        default:
            return absl::InternalError("unknown EstimateFunction");
    }
}

absl::StatusOr<int64_t> LookupMapImpl::estimate(const TaskInfo& task_info) const {
    RETURN_IF_STATUS_ERROR(checkTaskBoundary(task_info));
    for (int i = 0; i < scopes_.size(); i++) {
        if (scopes_[i].prefix_config.upper_bound >= task_info.prefix_length
            && scopes_[i].input_config.upper_bound >= task_info.input_length) {
            return estimateInternal(task_info, scopes_[i]);
        }
    }

    return absl::InternalError("internal error can't find scope for task");
}

bool LookupMapImpl::initFromLookupMapJson(LookupMapJson& map_json) {
    max_input_length_  = 0;
    max_prefix_length_ = -1;
    if (map_json.items.size() == 0) {
        RTP_LLM_LOG_WARNING("map json items size == 0, init LookupMapImpl failed");
        return false;
    }
    for (int i = 0; i < map_json.items.size(); i++) {
        auto& scope = map_json.items[i];
        RETURN_IF_NOT_SUCCESS(
            checkConfigValid(max_input_length_, max_prefix_length_, scope.input_config, scope.prefix_config));
        RETURN_IF_NOT_SUCCESS(checkInitScopeMap(scope));
        max_input_length_  = scope.input_config.upper_bound;
        max_prefix_length_ = scope.prefix_config.upper_bound;
    }
    map_[{0, 0}] = 0;
    std::swap(scopes_, map_json.items);
    if (max_input_length_ <= 0 || max_prefix_length_ < 0) {
        RTP_LLM_LOG_WARNING(
            "get max_input_length == %d and max_prefix_length == %d not valid, init LookupMapImpl failed",
            max_input_length_,
            max_prefix_length_);
        return false;
    }
    return true;
}

bool LookupMapImpl::checkInitScopeMap(const LookupMapScope& scope) {
    // insert
    for (auto& item : scope.map_items) {
        if (map_.find({item.prefix_length, item.input_length}) != map_.end()) {
            RTP_LLM_LOG_WARNING(
                "find repeat entry: {%d, %d} in map, init scope map failed", item.prefix_length, item.input_length);
            return false;
        }
        map_[{item.prefix_length, item.input_length}] = item.cost_time_ms;
    }
    // validate
    for (int i = scope.prefix_config.step_size; i <= scope.prefix_config.upper_bound;
         i += scope.prefix_config.step_size) {
        for (int j = scope.input_config.lower_bound + scope.input_config.step_size; j <= scope.input_config.upper_bound;
             j += scope.input_config.step_size) {
            if (map_.find({i, j}) == map_.end()) {
                RTP_LLM_LOG_WARNING("failed to find entry: {%d, %d} in map, init scope map failed", i, j);
                return false;
            }
        }
    }
    for (int i = scope.prefix_config.lower_bound + scope.prefix_config.step_size; i <= scope.prefix_config.upper_bound;
         i += scope.prefix_config.step_size) {
        for (int j = scope.input_config.step_size; j <= scope.input_config.upper_bound;
             j += scope.input_config.step_size) {
            if (map_.find({i, j}) == map_.end()) {
                RTP_LLM_LOG_WARNING("failed to find entry: {%d, %d} in map, init scope map failed", i, j);
                return false;
            }
        }
    }
    return true;
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm