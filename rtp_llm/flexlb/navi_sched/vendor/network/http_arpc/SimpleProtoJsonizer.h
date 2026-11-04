#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/ProtoJsonizer.h"
#include "google/protobuf/message.h"

namespace http_arpc {

class SimpleProtoJsonizer : public http_arpc::ProtoJsonizer {
public:
    SimpleProtoJsonizer() = default;
    ~SimpleProtoJsonizer() = default;
    SimpleProtoJsonizer(const SimpleProtoJsonizer &) = delete;
    SimpleProtoJsonizer &operator=(const SimpleProtoJsonizer &) = delete;

public:
    bool fromJson(const std::string &jsonStr, ::google::protobuf::Message *message) override;
    std::string toJson(const ::google::protobuf::Message &message) override;
};

} // namespace http_arpc
