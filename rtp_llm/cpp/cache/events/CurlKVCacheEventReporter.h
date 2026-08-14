#pragma once

#include <memory>
#include <string>

#include "rtp_llm/cpp/cache/events/KVCacheEventReporter.h"

namespace rtp_llm::detail {

// Production transport factory kept behind the reporter seam. KVCMPublisher
// supplies a validated, normalized endpoint and owns protocol/recovery
// semantics; this component owns only bounded, cancellable HTTP I/O and
// process-wide libcurl initialization.
std::shared_ptr<KVCacheEventReporter> makeCurlKVCacheEventReporter(std::string endpoint, int request_timeout_ms);

}  // namespace rtp_llm::detail
