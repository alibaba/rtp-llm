#pragma once

#include <string>
#include <google/protobuf/message.h>
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class RpcAccessLogWrapper {
public:
    static void logQuery(const RpcAccessLogConfig&        config,
                         const std::string&               requestType,
                         const google::protobuf::Message& request,
                         const std::string&               request_key = "");
    static void logAccess(const RpcAccessLogConfig&        config,
                          const std::string&               requestType,
                          const google::protobuf::Message& request,
                          const google::protobuf::Message& output,
                          const std::string&               request_key = "");

private:
    static std::string serializeMessagePlaintext(const google::protobuf::Message& message);
    static std::string serializeMessageWithCompress(const google::protobuf::Message& message);
    static void        incrementCounter(const std::string& requestType);
    static bool        shouldLog(const std::string& requestType);
};

}  // namespace rtp_llm