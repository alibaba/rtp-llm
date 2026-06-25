#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

class AccessLogWrapper {
public:
    static void logQueryAccess(const std::string& raw_request, int64_t request_id, bool private_request);

    static void logQueryAccess(const std::string&                       raw_request,
                               int64_t                                  request_id,
                               const std::vector<std::vector<int32_t>>& input_token_ids,
                               bool                                     private_request);

    static void logSuccessAccess(const std::string&                raw_request,
                                 int64_t                           request_id,
                                 int64_t                           start_time_ms,
                                 const std::optional<std::string>& logable_response,
                                 bool                              private_request);

    static void logSuccessAccess(const std::string&              raw_request,
                                 int64_t                         request_id,
                                 const std::vector<std::string>& complete_response,
                                 bool                            private_request);

    static void logSuccessAccess(const std::string&                       raw_request,
                                 int64_t                                  request_id,
                                 const std::vector<std::string>&          complete_response,
                                 const std::vector<std::vector<int32_t>>& input_token_ids,
                                 const std::vector<std::vector<int32_t>>& output_token_ids,
                                 bool                                     private_request);

    static void logExceptionAccess(const std::string& raw_request, int64_t request_id, const std::string& exception);
};

}  // namespace rtp_llm
