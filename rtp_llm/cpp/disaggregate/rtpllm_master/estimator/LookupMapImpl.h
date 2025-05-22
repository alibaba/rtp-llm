#pragma once
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/PrefillTimeEstimator.h"
#include "rtp_llm/cpp/utils/PairUnorderedMap.h"
#include "absl/status/statusor.h"
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {
namespace rtp_llm_master {

enum class EstimateFunction {
    MEAN,
    FLOOR,
    CEIL
};

// [lower_bound, upper_bound)
// lower_bound % step_size == 0 && upper_bound % step_size == 0
EstimateFunction get_estimate_func_from_string(const std::string& s);

class SingleConfig: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("lower_bound", lower_bound);
        json.Jsonize("upper_bound", upper_bound);
        json.Jsonize("step_size", step_size);
    }

public:
    int lower_bound;
    int upper_bound;
    int step_size;
};

class MapItem: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("prefix_length", prefix_length);
        json.Jsonize("input_length", input_length);
        json.Jsonize("cost_time_ms", cost_time_ms);
    }

public:
    int     prefix_length;
    int     input_length;
    int64_t cost_time_ms;
};

class LookupMapScope: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("prefix_config", prefix_config);
        json.Jsonize("input_config", input_config);
        json.Jsonize("map_items", map_items);
        json.Jsonize("estimate_func_str", estimate_func_str);
        estimate_func = get_estimate_func_from_string(estimate_func_str);
    }

public:
    SingleConfig         prefix_config;
    SingleConfig         input_config;
    std::string          estimate_func_str;
    EstimateFunction     estimate_func = EstimateFunction::CEIL;
    std::vector<MapItem> map_items;
};

class LookupMapJson: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("items", items);
    }

public:
    std::vector<LookupMapScope> items;
};

class LookupMapImpl {
public:
    LookupMapImpl() = default;
    bool                    init(const std::string& filePath);
    absl::StatusOr<int64_t> estimate(const TaskInfo& task_info) const;

protected:
    bool                    initFromLookupMapJson(LookupMapJson& map_json);
    bool                    checkConfigValid(int                 input_lower_bound,
                                             int                 prefix_lower_bound,
                                             const SingleConfig& input_config,
                                             const SingleConfig& prefix_config) const;
    bool                    checkInitScopeMap(const LookupMapScope& scope);
    absl::Status            checkTaskBoundary(const TaskInfo& task_info) const;
    absl::StatusOr<int64_t> getItemFromMap(const std::pair<int, int>& key) const;
    bool                    checkUpperBoundary() const;
    absl::StatusOr<int64_t> estimateInternal(const TaskInfo& task_info, const LookupMapScope& scope) const;

private:
    std::vector<LookupMapScope> scopes_;
    // key: {prefix_length, input_length}
    // value: cost_ms
    PairUnorderedMap<int, int, int64_t> map_;
    int                                 max_input_length_;
    int                                 max_prefix_length_;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm