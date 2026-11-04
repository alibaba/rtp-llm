#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc/GigClosure.h"

namespace multi_call {

class CommonClosure : public GigClosure
{
public:
    CommonClosure(google::protobuf::Closure *closure) : _closure(closure) {
    }
    ~CommonClosure() {
        _closure = NULL;
    }

private:
    CommonClosure(const CommonClosure &);
    CommonClosure &operator=(const CommonClosure &);

public:
    google::protobuf::Closure *getClosure() {
        return _closure;
    }
    void Run() override;
    ProtocolType getProtocolType() override {
        return MC_PROTOCOL_UNKNOWN;
    }

private:
    google::protobuf::Closure *_closure;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(CommonClosure);

} // namespace multi_call

