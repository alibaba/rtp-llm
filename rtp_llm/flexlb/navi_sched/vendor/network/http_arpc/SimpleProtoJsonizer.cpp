#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/SimpleProtoJsonizer.h"

#include "autil/Log.h"
#include "google/protobuf/util/json_util.h"

namespace http_arpc {

AUTIL_DECLARE_AND_SETUP_LOGGER(http_arpc, SimpleProtoJsonizer);

using google::protobuf::util::JsonParseOptions;
using google::protobuf::util::JsonPrintOptions;
using google::protobuf::util::JsonStringToMessage;
using google::protobuf::util::MessageToJsonString;

bool SimpleProtoJsonizer::fromJson(const std::string &jsonStr, ::google::protobuf::Message *message) {
    JsonParseOptions options;
    options.ignore_unknown_fields = true;
    const auto &status = JsonStringToMessage(jsonStr, message, options);
    if (!status.ok()) {
        AUTIL_LOG(WARN,
                  "failed to parse json to message:[%s] with error:[%s]",
                  message->GetTypeName().c_str(),
                  status.ToString().c_str());
        return false;
    }
    return true;
}

std::string SimpleProtoJsonizer::toJson(const ::google::protobuf::Message &message) {
    std::string output;
    const auto &status = MessageToJsonString(message, &output);
    if (!status.ok()) {
        AUTIL_LOG(WARN,
                  "failed to print message:[%s] as json with error:[%s]",
                  message.GetTypeName().c_str(),
                  status.ToString().c_str());
        return "";
    }
    return output;
}

} // namespace http_arpc
