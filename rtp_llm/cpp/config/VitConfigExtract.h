#pragma once

#include "rtp_llm/cpp/config/MMTransportConfigExtract.h"

namespace rtp_llm {

inline VitConfig extractVitConfig(const py::object& vit_config) {
    VitConfig cfg;
    if (vit_config.is_none()) {
        return cfg;
    }
    cfg.vit_separation                 = static_cast<VitSeparation>(vit_config.attr("vit_separation").cast<int>());
    cfg.output_transport               = extractMMTransportConfig(vit_config.attr("output_transport"));
    const int64_t configured_timeout_ms = vit_config.attr("mm_timeout_ms").cast<int64_t>();
    const int64_t worker_timeout_ms     = configured_timeout_ms > 0 ? configured_timeout_ms : 120 * 1000;
    cfg.output_transport.default_rpc_timeout_ms = worker_timeout_ms + cfg.output_transport.rpc_timeout_margin_ms;
    return cfg;
}

}  // namespace rtp_llm
