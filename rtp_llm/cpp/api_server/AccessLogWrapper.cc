#include <sstream>
#include "rtp_llm/cpp/api_server/AccessLogWrapper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "autil/TimeUtility.h"
using namespace autil::legacy::json;
using namespace autil::legacy;

namespace rtp_llm {

std::string logFormatTimeInMilliseconds(const int64_t& now_time_us) {
    auto               now_time = autil::TimeUtility::usFormat(now_time_us, "%Y-%m-%d %H:%M:%S");
    auto               ms_part  = now_time_us / 1000 % 1000;
    std::ostringstream ms_str;
    ms_str << std::setfill('0') << std::setw(3) << ms_part;
    return now_time + "." + ms_str.str();
}

class RequestLogInfo: public autil::legacy::Jsonizable {
public:
    RequestLogInfo(const std::string& request): request_(request) {}

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("request_json", request_, request_);
    }

    void clearRequest() {
        request_.clear();
    }

private:
    std::string request_;
};

class ResponseLogInfo: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("responses", response_list_, response_list_);
        json.Jsonize("exception_traceback", exception_traceback_, exception_traceback_);
    }

    void addResponse(const std::string& response) {
        response_list_.push_back(response);
    }

    void setResponseList(const std::vector<std::string>& response_list) {
        response_list_ = response_list;
    }

    void setExceptionTraceback(const std::string& exception_traceback) {
        exception_traceback_ = exception_traceback;
    }

private:
    std::vector<std::string> response_list_;
    std::string              exception_traceback_;
};

class AccessLogInfoEmbedding: public autil::legacy::Jsonizable {
public:
    AccessLogInfoEmbedding(const RequestLogInfo&  request,
                           const ResponseLogInfo& response,
                           int64_t                request_id,
                           int64_t                start_time_us):
        request_(request),
        response_(response),
        request_id_(request_id),
        log_time_(logFormatTimeInMilliseconds(autil::TimeUtility::currentTimeInMicroSeconds())),
        query_time_(logFormatTimeInMilliseconds(start_time_us)) {}

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("request", request_, request_);
        json.Jsonize("response", response_, response_);
        json.Jsonize("id", request_id_, request_id_);
        json.Jsonize("log_time", log_time_, log_time_);
        json.Jsonize("query_time", query_time_, query_time_);
    }

private:
    RequestLogInfo  request_;
    ResponseLogInfo response_;
    int64_t         request_id_;
    std::string     log_time_;
    std::string     query_time_;
};

class AccessLogInfo: public autil::legacy::Jsonizable {
public:
    AccessLogInfo(const RequestLogInfo& request, const ResponseLogInfo& response, int64_t request_id):
        request_(request),
        response_(response),
        request_id_(request_id),
        log_time_(logFormatTimeInMilliseconds(autil::TimeUtility::currentTimeInMicroSeconds())) {}

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("request", request_, request_);
        json.Jsonize("response", response_, response_);
        json.Jsonize("id", request_id_, request_id_);
        json.Jsonize("log_time", log_time_, log_time_);
    }

private:
    RequestLogInfo  request_;
    ResponseLogInfo response_;
    int64_t         request_id_;
    std::string     log_time_;
};

bool AccessLogWrapper::default_private_request = false;

void AccessLogWrapper::logQueryAccess(const std::string&  raw_request,
                                      int64_t             request_id,
                                      std::optional<bool> private_request) {
    if (private_request.value_or(default_private_request)) {
        return;
    }

    RequestLogInfo  request(raw_request);
    ResponseLogInfo response;
    AccessLogInfo   access_log_info(request, response, request_id);

    try {
        std::string access_log_info_str = autil::legacy::ToJsonString(access_log_info, /*isCompact=*/true);
        RTP_LLM_QUERY_ACCESS_LOG_INFO("%s", access_log_info_str.c_str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("AccessLogWrapper logQueryAccess ToJsonString failed, error: %s", e.what());
    }
}

std::string decodeUnicode(const std::string& input) {
    std::string result;
    size_t      i = 0;
    while (i < input.size()) {
        if (input[i] == '\\' && i + 5 < input.size() && input[i + 1] == 'u') {
            // 提取 \uXXXX 中的 XXXX
            std::string       hexStr = input.substr(i + 2, 4);
            unsigned int      codePoint;
            std::stringstream ss;
            ss << std::hex << hexStr;
            ss >> codePoint;

            // 转换为 UTF-8 编码
            if (codePoint <= 0x7F) {
                result += static_cast<char>(codePoint);
            } else if (codePoint <= 0x7FF) {
                result += static_cast<char>((codePoint >> 6) | 0xC0);
                result += static_cast<char>((codePoint & 0x3F) | 0x80);
            } else if (codePoint <= 0xFFFF) {
                result += static_cast<char>((codePoint >> 12) | 0xE0);
                result += static_cast<char>(((codePoint >> 6) & 0x3F) | 0x80);
                result += static_cast<char>((codePoint & 0x3F) | 0x80);
            }
            i += 6;  // 跳过 \uXXXX
        } else {
            result += input[i];
            ++i;
        }
    }
    return result;
}

std::string removeEscapedQuotes(const std::string& jsonString) {
    std::string result       = jsonString;
    std::string escapedQuote = "\\\"";
    std::string quote        = "\"";
    // Remove escaped quotes
    size_t pos = 0;
    while ((pos = result.find(escapedQuote, pos)) != std::string::npos) {
        result.replace(pos, escapedQuote.length(), quote);
        pos += quote.length();
    }
    return result;
}

std::string removeQuotesAroundBraces(const std::string& input) {
    std::string result;
    for (size_t i = 0; i < input.length(); ++i) {
        if (input[i] == '"') {
            if (i > 0 && input[i - 1] == '}') {
                // Skip this quote as it's right after '}'
                continue;
            }
            if (i < input.length() - 1 && input[i + 1] == '{') {
                // Skip this quote as it's right before '{'
                continue;
            }
        }
        result += input[i];
    }
    return result;
}

// for embedding model
void AccessLogWrapper::logSuccessAccess(const std::string&                raw_request,
                                        int64_t                           request_id,
                                        int64_t                           start_time_us,
                                        const std::optional<std::string>& logable_response,
                                        std::optional<bool>               private_request) {
    if (private_request.value_or(default_private_request)) {
        return;
    }
    if (logable_response.has_value() == false) {
        return;
    }

    try {
        RequestLogInfo  request(decodeUnicode(raw_request));
        ResponseLogInfo response;
        response.addResponse(logable_response.value());
        AccessLogInfoEmbedding access_log_info(request, response, request_id, start_time_us);

        std::string access_log_info_str = autil::legacy::ToJsonString(access_log_info, /*isCompact=*/true);
        RTP_LLM_ACCESS_LOG_INFO("%s", removeQuotesAroundBraces(removeEscapedQuotes(access_log_info_str)).c_str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("AccessLogWrapper logSuccessAccess failed, error: %s", e.what());
    }
}

// for normal model
void AccessLogWrapper::logSuccessAccess(const std::string&              raw_request,
                                        int64_t                         request_id,
                                        const std::vector<std::string>& complete_response,
                                        std::optional<bool>             private_request) {
    if (private_request.value_or(default_private_request)) {
        return;
    }

    RequestLogInfo  request(raw_request);
    ResponseLogInfo response;
    response.setResponseList(complete_response);
    AccessLogInfo access_log_info(request, response, request_id);

    try {
        std::string access_log_info_str = autil::legacy::ToJsonString(access_log_info, /*isCompact=*/true);
        RTP_LLM_ACCESS_LOG_INFO("%s", access_log_info_str.c_str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("AccessLogWrapper logSuccessAccess ToJsonString failed, error: %s", e.what());
    }
}

static bool isPrivate(const std::string& raw_request) {
    bool private_request = AccessLogWrapper::default_private_request;
    try {
        auto body    = ParseJson(raw_request);
        auto bodyMap = AnyCast<JsonMap>(body);
        auto it      = bodyMap.find("private_request");
        if (it == bodyMap.end()) {
            return private_request;
        }
        FromJson(private_request, it->second);
        return private_request;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("AccessLogWrapper private_request field failed, error: %s", e.what());
    }
    return private_request;
}

void AccessLogWrapper::logExceptionAccess(const std::string& raw_request,
                                          int64_t            request_id,
                                          const std::string& exception) {
    try {
        RequestLogInfo request(raw_request);
        if (isPrivate(raw_request)) {
            request.clearRequest();
        }
        ResponseLogInfo response;
        response.setExceptionTraceback(exception);
        AccessLogInfo access_log_info(request, response, request_id);

        std::string access_log_info_str = autil::legacy::ToJsonString(access_log_info, /*isCompact=*/true);
        RTP_LLM_ACCESS_LOG_INFO("%s", access_log_info_str.c_str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("AccessLogWrapper logExceptionAccess failed, error: %s", e.what());
    }
}

}  // namespace rtp_llm
