#pragma once

#include "rtp_llm/cpp/config/MMTransportConfigExtract.h"

namespace rtp_llm {

inline VitConfig extractVitConfig(const py::object& vit_config) {
    VitConfig cfg;
    if (vit_config.is_none()) {
        return cfg;
    }
    cfg.vit_separation = static_cast<VitSeparation>(vit_config.attr("vit_separation").cast<int>());
    cfg.output_transport = extractMMTransportConfig(vit_config.attr("output_transport"));
    return cfg;
}

}  // namespace rtp_llm
