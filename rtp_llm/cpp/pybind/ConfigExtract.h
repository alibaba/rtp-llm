#pragma once

#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/MMTransportMode.h"

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
    cfg.max_receipt_bytes  = rdma_config.attr("max_receipt_bytes").cast<int64_t>();
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
    cfg.mode    = validateMMTransportMode(transport_config.attr("mode").cast<std::string>());
    cfg.control = extractMMControlConfig(transport_config.attr("control"));
    cfg.rdma    = extractRdmaConfig(transport_config.attr("rdma"));
    return cfg;
}

inline VitConfig extractVitConfig(const py::object& vit_config) {
    VitConfig cfg;
    if (vit_config.is_none()) {
        return cfg;
    }
    cfg.vit_separation                  = static_cast<VitSeparation>(vit_config.attr("vit_separation").cast<int>());
    cfg.output_transport                = extractMMTransportConfig(vit_config.attr("output_transport"));
    const int64_t configured_timeout_ms = vit_config.attr("mm_timeout_ms").cast<int64_t>();
    const int64_t worker_timeout_ms     = configured_timeout_ms > 0 ? configured_timeout_ms : 120 * 1000;
    cfg.output_transport.default_rpc_timeout_ms = worker_timeout_ms + cfg.output_transport.rpc_timeout_margin_ms;
    return cfg;
}

}  // namespace rtp_llm
