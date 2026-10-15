#pragma once

#include <string>
#include <cstdint>

namespace rtp_llm::detail {

struct KVCMReportFeedback {
    bool valid = false;
    bool ok = false;
    bool snapshot_required = false;
    bool registration_required = true;
    uint64_t retry_after_ms = 0;
};

std::string normalizeKVCacheEventEndpoint(std::string endpoint);
bool        kvcmResponseIsOk(const std::string& response);
KVCMReportFeedback parseKVCMReportFeedback(const std::string& response);

}  // namespace rtp_llm::detail
