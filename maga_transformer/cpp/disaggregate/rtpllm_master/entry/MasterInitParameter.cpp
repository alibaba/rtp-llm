#include "maga_transformer/cpp/disaggregate/rtpllm_master/entry/MasterInitParameter.h"
namespace rtp_llm {
namespace rtp_llm_master {

void registerMasterInitParameter(py::module m) {
    py::enum_<rtp_llm::rtp_llm_master::PySubscribeConfigType>(m, "PySubscribeConfigType")
        .value("CM2", rtp_llm::rtp_llm_master::PySubscribeConfigType::CM2)
        .value("LOCAL", rtp_llm::rtp_llm_master::PySubscribeConfigType::LOCAL);

    py::class_<rtp_llm::rtp_llm_master::PySubscribeConfig>(m, "PySubscribeConfig")
        .def(py::init<>())
        .def_readwrite("type", &rtp_llm::rtp_llm_master::PySubscribeConfig::type)
        .def_readwrite("cluster_name", &rtp_llm::rtp_llm_master::PySubscribeConfig::cluster_name)
        .def_readwrite("zk_host", &rtp_llm::rtp_llm_master::PySubscribeConfig::zk_host)
        .def_readwrite("zk_path", &rtp_llm::rtp_llm_master::PySubscribeConfig::zk_path)
        .def_readwrite("zk_timeout_ms", &rtp_llm::rtp_llm_master::PySubscribeConfig::zk_timeout_ms)
        .def_readwrite("local_ip", &rtp_llm::rtp_llm_master::PySubscribeConfig::local_ip)
        .def_readwrite("local_http_port", &rtp_llm::rtp_llm_master::PySubscribeConfig::local_http_port)
        .def_readwrite("local_rpc_port", &rtp_llm::rtp_llm_master::PySubscribeConfig::local_rpc_port);

    py::class_<rtp_llm::rtp_llm_master::PyLoadbalanceConfig>(m, "PyLoadbalanceConfig")
        .def(py::init<>())
        .def_readwrite("subscribe_config", &rtp_llm::rtp_llm_master::PyLoadbalanceConfig::subscribe_config)
        .def_readwrite("update_interval_ms", &rtp_llm::rtp_llm_master::PyLoadbalanceConfig::update_interval_ms)
        .def_readwrite("sync_status_interval_ms",
                       &rtp_llm::rtp_llm_master::PyLoadbalanceConfig::sync_status_interval_ms);

    py::class_<rtp_llm::rtp_llm_master::PyEstimatorConfig>(m, "PyEstimatorConfig")
        .def(py::init<>())
        .def_readwrite("estimator_config_map", &rtp_llm::rtp_llm_master::PyEstimatorConfig::estimator_config_map);

    py::class_<rtp_llm::rtp_llm_master::MasterInitParameter>(m, "MasterInitParameter")
        .def(py::init<>())
        .def_readwrite("load_balance_config", &rtp_llm::rtp_llm_master::MasterInitParameter::load_balance_config)
        .def_readwrite("estimator_config", &rtp_llm::rtp_llm_master::MasterInitParameter::estimator_config)
        .def_readwrite("port", &rtp_llm::rtp_llm_master::MasterInitParameter::port);
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm