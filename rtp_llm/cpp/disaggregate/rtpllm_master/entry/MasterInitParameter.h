#include <string>
#include <unordered_map>
#include <torch/extension.h>

#pragma once
namespace rtp_llm {
namespace rtp_llm_master {

enum class PySubscribeConfigType {
    CM2,
    LOCAL
};

class PySubscribeConfig {
public:
    PySubscribeConfigType type;
    // for SubscribeConfigType::CM2
    std::string cluster_name;
    std::string zk_host;
    std::string zk_path;
    uint32_t    zk_timeout_ms{10 * 1000};

    // for SubscribeConfigType::LOCAL
    std::string local_ip;
    int local_http_port;
    int local_rpc_port;
};

class PyLoadbalanceConfig {
public:
    PySubscribeConfig subscribe_config;
    int64_t           update_interval_ms{500};
    int64_t           sync_status_interval_ms{500};
};

class PyEstimatorConfig {
public:
    std::unordered_map<std::string, std::string> estimator_config_map;
};

class MasterInitParameter {
public:
    PyLoadbalanceConfig load_balance_config;
    PyEstimatorConfig   estimator_config;
    int                 port;
};

void registerMasterInitParameter(py::module m);

}  // namespace rtp_llm_master
}  // namespace rtp_llm
