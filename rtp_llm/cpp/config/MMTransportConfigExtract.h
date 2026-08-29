#pragma once

#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace py = pybind11;

namespace rtp_llm {

inline RdmaConfig extractRdmaConfig(const py::object& rdma_config) {
    RdmaConfig cfg;
    cfg.bind_ip            = rdma_config.attr("bind_ip").cast<std::string>();
    cfg.port               = rdma_config.attr("port").cast<int>();
    cfg.connect_timeout_ms = rdma_config.attr("connect_timeout_ms").cast<int>();
    cfg.read_timeout_ms    = rdma_config.attr("read_timeout_ms").cast<int64_t>();
    cfg.qp_count           = rdma_config.attr("qp_count").cast<uint32_t>();
    cfg.slot_gc_timeout_ms = rdma_config.attr("slot_gc_timeout_ms").cast<int64_t>();
    cfg.max_slot_bytes     = rdma_config.attr("max_slot_bytes").cast<int64_t>();
    return cfg;
}

inline MMControlConfig extractMMControlConfig(const py::object& control_config) {
    MMControlConfig cfg;
    cfg.release_timeout_ms = control_config.attr("release_timeout_ms").cast<int64_t>();
    return cfg;
}

// Missing Python attributes are configuration errors.
inline MMTransportConfig extractMMTransportConfig(const py::object& transport_config) {
    MMTransportConfig cfg;
    cfg.mode    = transport_config.attr("mode").cast<std::string>();
    cfg.control = extractMMControlConfig(transport_config.attr("control"));
    cfg.rdma    = extractRdmaConfig(transport_config.attr("rdma"));
    return cfg;
}

}  // namespace rtp_llm
