#include "maga_transformer/cpp/api_server/AccessLogWrapper.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "autil/TimeUtility.h"

using namespace autil::legacy::json;
using namespace autil::legacy;

namespace rtp_llm {

class RequestLogInfo: public autil::legacy::Jsonizable {
public:
    RequestLogInfo(const std::string& request): request_(request) {}

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("request_str", request_, request_);
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
        json.Jsonize("response", response_list_, response_list_);
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

class AccessLogInfo: public autil::legacy::Jsonizable {
public:
    AccessLogInfo(const RequestLogInfo& request, const ResponseLogInfo& response, int64_t request_id):
        request_(request),
        response_(response),
        request_id_(request_id),
        log_time_(autil::TimeUtility::currentTimeString("%Y-%m-%d %H:%M:%S")) {}

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("request", request_, request_);
        json.Jsonize("response", response_, response_);
        json.Jsonize("request_id", request_id_, request_id_);
        json.Jsonize("log_time", log_time_, log_time_);
    }

private:
    RequestLogInfo  request_;
    ResponseLogInfo response_;
    int64_t         request_id_;
    std::string     log_time_;
};

void AccessLogWrapper::logQueryAccess(const std::string& raw_request,
                                      int64_t            request_id,
                                      bool               private_request) {
    if (private_request) {
        return;
    }

    RequestLogInfo  request(raw_request);
    ResponseLogInfo response;
    AccessLogInfo   access_log_info(request, response, request_id);

    try {
        std::string access_log_info_str = autil::legacy::ToJsonString(access_log_info, /*isCompact=*/true);
        FT_QUERY_ACCESS_LOG_INFO("%s", access_log_info_str.c_str());
    } catch (const std::exception& e) {
        FT_LOG_ERROR("AccessLogWrapper logQueryAccess ToJsonString failed, error: %s", e.what());
    }
}

void AccessLogWrapper::logSuccessAccess(const std::string&                raw_request,
                                        int64_t                           request_id,
                                        const std::optional<std::string>& logable_response,
                                        bool                              private_request) {
    if (private_request) {
        return;
    }

    try {
        RequestLogInfo  request(raw_request);
        ResponseLogInfo response;
        if (logable_response.has_value()) {
            response.addResponse(logable_response.value());
        }
        AccessLogInfo access_log_info(request, response, request_id);

        std::string access_log_info_str = autil::legacy::ToJsonString(access_log_info, /*isCompact=*/true);
        FT_ACCESS_LOG_INFO("%s", access_log_info_str.c_str());
    } catch (const std::exception& e) {
        FT_LOG_ERROR("AccessLogWrapper logSuccessAccess failed, error: %s", e.what());
    }
}

void AccessLogWrapper::logSuccessAccess(const std::string&              raw_request,
                                        int64_t                         request_id,
                                        const std::vector<std::string>& complete_response,
                                        bool                            private_request) {
    if (private_request) {
        return;
    }

    RequestLogInfo  request(raw_request);
    ResponseLogInfo response;
    response.setResponseList(complete_response);
    AccessLogInfo access_log_info(request, response, request_id);

    try {
        std::string access_log_info_str = autil::legacy::ToJsonString(access_log_info, /*isCompact=*/true);
        FT_ACCESS_LOG_INFO("%s", access_log_info_str.c_str());
    } catch (const std::exception& e) {
        FT_LOG_ERROR("AccessLogWrapper logSuccessAccess ToJsonString failed, error: %s", e.what());
    }
}

bool isPrivate(const std::string& raw_request) {
    bool private_request = false;
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
        FT_LOG_ERROR("AccessLogWrapper private_request field failed, error: %s", e.what());
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
        FT_ACCESS_LOG_INFO("%s", access_log_info_str.c_str());
    } catch (const std::exception& e) {
        FT_LOG_ERROR("AccessLogWrapper logExceptionAccess failed, error: %s", e.what());
    }
}

}  // namespace rtp_llm
