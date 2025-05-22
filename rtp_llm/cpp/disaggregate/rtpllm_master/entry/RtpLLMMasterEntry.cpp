#include <exception>
#include "rtp_llm/cpp/disaggregate/rtpllm_master/entry/RtpLLMMasterEntry.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {
namespace rtp_llm_master {

const std::string UNITTEST_DEFAULT_LOG_CONF = R"conf(
alog.rootLogger=INFO, unittestAppender
alog.max_msg_len=4096
alog.appender.unittestAppender=ConsoleAppender
alog.appender.unittestAppender.flush=true
alog.appender.unittestAppender.layout=PatternLayout
alog.appender.unittestAppender.layout.LogPattern=[%%d] [%%l] [%%t,%%F -- %%f():%%n] [%%m]
alog.logger.arpc=WARN
)conf";

bool RtpLLMMasterEntry::init(const MasterInitParameter& param) {
    param_ = param;
    RETURN_IF_NOT_SUCCESS(initLoadBalancer(param));
    RETURN_IF_NOT_SUCCESS(initTokenizeService());
    RETURN_IF_NOT_SUCCESS(initHttpServer());
    return true;
}

RtpLLMMasterEntry::RtpLLMMasterEntry() {
    AUTIL_LOG_CONFIG_FROM_STRING(UNITTEST_DEFAULT_LOG_CONF.c_str());
}

LoadBalancerInitParams RtpLLMMasterEntry::createLoadBalancerInitParams(const MasterInitParameter& param) {
    LoadBalancerInitParams load_balance_params;
    load_balance_params.sync_status_interval_ms = param.load_balance_config.sync_status_interval_ms;
    load_balance_params.update_interval_ms      = param.load_balance_config.update_interval_ms;
        const PySubscribeConfig& py_config = param.load_balance_config.subscribe_config;
        if (py_config.type == PySubscribeConfigType::CM2) {
            CM2SubscribeServiceConfig config;
            config.zk_host       = py_config.zk_host;
            config.zk_path       = py_config.zk_path;
            config.zk_timeout_ms = py_config.zk_timeout_ms;
            config.clusters      = {py_config.cluster_name};
            load_balance_params.subscribe_config.cm2_configs.push_back(config);
            biz_name_ = py_config.cluster_name;
        } else if (py_config.type == PySubscribeConfigType::LOCAL) {
            LocalNodeJsonize node("local", py_config.local_ip, py_config.local_rpc_port, py_config.local_http_port);
            LocalSubscribeServiceConfig local_config;
            local_config.nodes.push_back(node);
            load_balance_params.subscribe_config.local_configs.push_back(local_config);
            biz_name_ = "local";
        } else {
            RTP_LLM_LOG_ERROR("unsupported subscribe config type %d", py_config.type);
        }
    return load_balance_params;
}

EstimatorConfig RtpLLMMasterEntry::createEstimatorConfig(const PyEstimatorConfig& py_config) {
    EstimatorConfig config;
    config.estimator_type = "lookup";
    for (const auto& [key, value] : py_config.estimator_config_map) {
        LookupConfig lookup_config;
        lookup_config.machine_info = key;
        lookup_config.config_path  = value;
        config.lookup_configs.push_back(lookup_config);
    }
    return config;
}

bool RtpLLMMasterEntry::initLoadBalancer(const MasterInitParameter& param) {
    load_balancer_ = std::make_shared<PrefillLoadBalancer>();
    LoadBalancerInitParams load_balance_params;
    EstimatorConfig        estimator_config;
    try {
        load_balance_params = createLoadBalancerInitParams(param);
        estimator_config    = createEstimatorConfig(param.estimator_config);
    } catch (std::exception& e) {
        RTP_LLM_LOG_ERROR("create subscribe_config failed, error %s", e.what());
        return false;
    }
    return load_balancer_->initWithEstimator(load_balance_params, estimator_config);
}

bool RtpLLMMasterEntry::initTokenizeService() {
    tokenize_service_ = std::make_shared<RemoteTokenizeModule>();
    return tokenize_service_->init(load_balancer_);
}

bool RtpLLMMasterEntry::initHttpServer() {
    http_server_ = std::make_shared<MasterHttpServer>(tokenize_service_, load_balancer_, biz_name_, param_.port);
    return http_server_->start();
}

RtpLLMMasterEntry::~RtpLLMMasterEntry() {
    http_server_.reset();
    tokenize_service_.reset();
    load_balancer_.reset();
}

void registerRtpLLMMasterEntry(py::module m) {
    pybind11::class_<RtpLLMMasterEntry>(m, "RtpLLMMasterEntry")
        .def(pybind11::init<>())
        .def("init", &RtpLLMMasterEntry::init, py::arg("param"));
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm