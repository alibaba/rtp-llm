#include "rtp_llm/cpp/utils/RpcAccessLogWrapper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "autil/AtomicCounter.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <unordered_map>
#include <memory>

namespace rtp_llm {

// 全局计数器，按请求类型分别计数
static std::unordered_map<std::string, std::shared_ptr<autil::AtomicCounter>> request_counters_;
static std::mutex                                                             counters_mutex_;

std::string RpcAccessLogWrapper::serializeMessagePlaintext(const google::protobuf::Message& message) {
    std::string text_format;
    google::protobuf::TextFormat::PrintToString(message, &text_format);
    return text_format;
}

std::string RpcAccessLogWrapper::serializeMessageWithCompress(const google::protobuf::Message& message) {
    std::string json_string;
    auto        status = google::protobuf::util::MessageToJsonString(message, &json_string);
    if (!status.ok()) {
        RTP_LLM_LOG_ERROR("Failed to serialize message to JSON: %s", status.ToString().c_str());
        return message.ShortDebugString();
    }
    return json_string;
}

bool RpcAccessLogWrapper::shouldLog(const std::string& requestType) {
    return true;
}

void RpcAccessLogWrapper::incrementCounter(const std::string& requestType) {
    std::lock_guard<std::mutex> lock(counters_mutex_);
    if (request_counters_.find(requestType) == request_counters_.end()) {
        request_counters_[requestType] = std::make_shared<autil::AtomicCounter>();
    }
    request_counters_[requestType]->incAndReturn();
}

void RpcAccessLogWrapper::logRpcRequest(const RpcAccessLogConfig&        config,
                                        const std::string&               requestType,
                                        const google::protobuf::Message& request,
                                        const google::protobuf::Message& output,
                                        const std::string&               request_key) {

    if (!config.enable_rpc_access_log || config.access_log_interval <= 0) {
        return;
    }

    // 获取或创建计数器
    std::lock_guard<std::mutex> lock(counters_mutex_);
    if (request_counters_.find(requestType) == request_counters_.end()) {
        request_counters_[requestType] = std::make_shared<autil::AtomicCounter>();
    }
    auto& counter = request_counters_[requestType];

    int counter_value = counter->incAndReturn();
    if (counter_value % config.access_log_interval == 0) {
        // 序列化请求和响应
        std::string request_str, output_str;

        if (config.log_plaintext) {
            request_str = serializeMessagePlaintext(request);
            output_str  = serializeMessagePlaintext(output);
        } else {
            request_str = serializeMessageWithCompress(request);
            output_str  = serializeMessageWithCompress(output);
        }

        // 记录日志
        std::string log_key = request_key.empty() ? requestType : request_key;
        RTP_LLM_ACCESS_LOG_INFO("%s: {\"request\": \"%s\", \"response\": \"%s\"}",
                                log_key.c_str(),
                                request_str.c_str(),
                                output_str.c_str());

        // 重置计数器
        counter->setValue(0);
    }
}

}  // namespace rtp_llm