#pragma once
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {
namespace rtp_llm_master {

class LookupConfig: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("machine_info", machine_info);
        json.Jsonize("config_path", config_path);        
    }
public:
    std::string machine_info;
    std::string config_path;
};

class EstimatorConfig: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("lookup_configs", lookup_configs, {});
        json.Jsonize("estimator_type", estimator_type);
    }

public:
    std::string               estimator_type;
    std::vector<LookupConfig> lookup_configs;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm