#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/ProtoJsonizer.h"

namespace navi {

class GraphVisDef;

class NaviDebugProtoJsonizer : public http_arpc::ProtoJsonizer
{
public:
    NaviDebugProtoJsonizer(const std::shared_ptr<GraphVisDef> &visDef,
                           const http_arpc::ProtoJsonizerPtr &realProtoJsonizer);
    ~NaviDebugProtoJsonizer();
    NaviDebugProtoJsonizer(const NaviDebugProtoJsonizer &) = delete;
    NaviDebugProtoJsonizer &operator=(const NaviDebugProtoJsonizer &) = delete;
public:
    bool fromJson(const std::string &jsonStr,
                  google::protobuf::Message *message) override;
    std::string toJson(const google::protobuf::Message &message) override;
private:
    std::shared_ptr<GraphVisDef> _visDef;
    http_arpc::ProtoJsonizerPtr _realProtoJsonizer;
};

}

