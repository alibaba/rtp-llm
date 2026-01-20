#pragma once

#include <string>
#include <memory>
#include <vector>
#include <google/protobuf/message.h>
#include "alog/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

// Helper macros for logging based on level
#define LOG_QUERY_BY_LEVEL(level, format, ...)                                                                         \
    do {                                                                                                               \
        switch (level) {                                                                                               \
            case alog::LOG_LEVEL_DEBUG:                                                                                \
                RTP_LLM_QUERY_LOG_DEBUG(format, ##__VA_ARGS__);                                                        \
                break;                                                                                                 \
            case alog::LOG_LEVEL_INFO:                                                                                 \
                RTP_LLM_QUERY_LOG_INFO(format, ##__VA_ARGS__);                                                         \
                break;                                                                                                 \
            case alog::LOG_LEVEL_WARN:                                                                                 \
                RTP_LLM_QUERY_LOG_WARNING(format, ##__VA_ARGS__);                                                      \
                break;                                                                                                 \
            case alog::LOG_LEVEL_ERROR:                                                                                \
                RTP_LLM_QUERY_LOG_ERROR(format, ##__VA_ARGS__);                                                        \
                break;                                                                                                 \
            default:                                                                                                   \
                RTP_LLM_QUERY_LOG_INFO(format, ##__VA_ARGS__);                                                         \
                break;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define LOG_ACCESS_BY_LEVEL(level, format, ...)                                                                        \
    do {                                                                                                               \
        switch (level) {                                                                                               \
            case alog::LOG_LEVEL_DEBUG:                                                                                \
                RTP_LLM_ACCESS_LOG_DEBUG(format, ##__VA_ARGS__);                                                       \
                break;                                                                                                 \
            case alog::LOG_LEVEL_INFO:                                                                                 \
                RTP_LLM_ACCESS_LOG_INFO(format, ##__VA_ARGS__);                                                        \
                break;                                                                                                 \
            case alog::LOG_LEVEL_WARN:                                                                                 \
                RTP_LLM_ACCESS_LOG_WARNING(format, ##__VA_ARGS__);                                                     \
                break;                                                                                                 \
            case alog::LOG_LEVEL_ERROR:                                                                                \
                RTP_LLM_ACCESS_LOG_ERROR(format, ##__VA_ARGS__);                                                       \
                break;                                                                                                 \
            default:                                                                                                   \
                RTP_LLM_ACCESS_LOG_INFO(format, ##__VA_ARGS__);                                                        \
                break;                                                                                                 \
        }                                                                                                              \
    } while (0)

class RpcAccessLogWrapper {
public:
    // 带日志级别的日志记录函数
    static void logQuery(const RpcAccessLogConfig&        config,
                         const std::string&               requestType,
                         const google::protobuf::Message* request,
                         uint32_t                         logLevel = alog::LOG_LEVEL_INFO);

    static void logAccess(const RpcAccessLogConfig&        config,
                          const std::string&               requestType,
                          const google::protobuf::Message* request,
                          const google::protobuf::Message* output,
                          const std::string&               errorMsg = "",
                          uint32_t                         logLevel = alog::LOG_LEVEL_INFO);

    static std::string serializeTensorPBPlaintext(const TensorPB& tensor_pb);
    static std::string serializeMessageBinary(const google::protobuf::Message* message);
    static std::string serializeMessagePlaintext(const google::protobuf::Message* message);

private:
};

// 宏定义来自动生成方法名
#define RPC_METHOD_NAME(method) #method
#define RPC_METHOD_NAME_QUERY(method) #method "_Query"

// 特定日志级别的宏定义
#define LOG_RPC_QUERY_DEBUG(config, method, request)                                                                   \
    RpcAccessLogWrapper::logQuery(config, RPC_METHOD_NAME_QUERY(method), request, alog::LOG_LEVEL_DEBUG)

#define LOG_RPC_QUERY_INFO(config, method, request)                                                                    \
    RpcAccessLogWrapper::logQuery(config, RPC_METHOD_NAME_QUERY(method), request, alog::LOG_LEVEL_INFO)

#define LOG_RPC_QUERY_WARN(config, method, request)                                                                    \
    RpcAccessLogWrapper::logQuery(config, RPC_METHOD_NAME_QUERY(method), request, alog::LOG_LEVEL_WARN)

#define LOG_RPC_QUERY_ERROR(config, method, request)                                                                   \
    RpcAccessLogWrapper::logQuery(config, RPC_METHOD_NAME_QUERY(method), request, alog::LOG_LEVEL_ERROR)

#define LOG_RPC_ACCESS_DEBUG(config, method, request, response, status)                                                \
    LOG_RPC_ACCESS_WITH_STATUS(config, RPC_METHOD_NAME(method), request, response, status, alog::LOG_LEVEL_DEBUG)

#define LOG_RPC_ACCESS_INFO(config, method, request, response, status)                                                 \
    LOG_RPC_ACCESS_WITH_STATUS(config, RPC_METHOD_NAME(method), request, response, status, alog::LOG_LEVEL_INFO)

#define LOG_RPC_ACCESS_WARN(config, method, request, response, status)                                                 \
    LOG_RPC_ACCESS_WITH_STATUS(config, RPC_METHOD_NAME(method), request, response, status, alog::LOG_LEVEL_WARN)

#define LOG_RPC_ACCESS_ERROR(config, method, request, response, status)                                                \
    LOG_RPC_ACCESS_WITH_STATUS(config, RPC_METHOD_NAME(method), request, response, status, alog::LOG_LEVEL_ERROR)

#define LOG_RPC_ACCESS_WITH_STATUS(config, method, request, response, status, level)                                   \
    do {                                                                                                               \
        if (status.ok()) {                                                                                             \
            RpcAccessLogWrapper::logAccess(config, method, request, response, "", level);                              \
        } else {                                                                                                       \
            RpcAccessLogWrapper::logAccess(config, method, request, nullptr, status.error_message(), level);           \
        }                                                                                                              \
    } while (0)

}  // namespace rtp_llm