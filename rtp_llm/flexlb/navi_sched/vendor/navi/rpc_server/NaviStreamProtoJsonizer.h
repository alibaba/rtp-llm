#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/ProtoJsonizer.h"
#include "navi/engine/NaviUserResult.h"

namespace navi {

class NaviStreamProtoJsonizer : public http_arpc::ProtoJsonizer
{
public:
    NaviStreamProtoJsonizer() {
    }
    ~NaviStreamProtoJsonizer() {
    }
    NaviStreamProtoJsonizer(const NaviStreamProtoJsonizer &) = delete;
    NaviStreamProtoJsonizer &operator=(const NaviStreamProtoJsonizer &) = delete;
public:
    virtual bool collect(google::protobuf::Message *response,
                         const NaviUserData &userData, bool eof,
                         google::protobuf::Message *streamMsg) = 0;
};

}

