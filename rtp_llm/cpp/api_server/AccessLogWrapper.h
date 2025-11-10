#pragma once

#include <optional>

#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

class AccessLogWrapper {
public:
    static void logQueryAccess(const std::string& raw_request, int64_t request_id, std::optional<bool> private_request);

    static void logSuccessAccess(const std::string&                raw_request,
                                 int64_t                           request_id,
                                 int64_t                           start_time_ms,
                                 const std::optional<std::string>& logable_response,
                                 std::optional<bool>               private_request);

    static void logSuccessAccess(const std::string&              raw_request,
                                 int64_t                         request_id,
                                 const std::vector<std::string>& complete_response,
                                 std::optional<bool>             private_request);

    static void logExceptionAccess(const std::string& raw_request, int64_t request_id, const std::string& exception);

    static bool default_private_request;
};

}  // namespace rtp_llm
