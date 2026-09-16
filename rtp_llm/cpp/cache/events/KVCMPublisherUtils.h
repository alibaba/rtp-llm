#pragma once

#include <string>

namespace rtp_llm::detail {

std::string normalizeKVCacheEventEndpoint(std::string endpoint);
bool        kvcmResponseIsOk(const std::string& response);

}  // namespace rtp_llm::detail
