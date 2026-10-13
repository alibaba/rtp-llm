#pragma once

#include <memory>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"

namespace rtp_llm {

// Reader priority follows registration order.
std::unique_ptr<MMRemoteOutputTransport>
createMMRemoteOutputTransport(const MMTransportConfig&     transport_config,
                              kmonitor::MetricsReporterPtr reporter,
                              int                          device_id = -1);

}  // namespace rtp_llm
