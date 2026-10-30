#pragma once

#include <string>
#include <optional>

namespace rtp_llm::detail {

struct KVCMReportFeedback {
    bool valid = false;
    bool ok = false;
    bool snapshot_required = false;
    bool registration_required = false;
    std::optional<bool> remote_cache_enabled;
};

std::string normalizeKVCacheEventEndpoint(std::string endpoint);
bool        kvcmResponseIsOk(const std::string& response);
KVCMReportFeedback parseKVCMReportFeedback(const std::string& response);

}  // namespace rtp_llm::detail
