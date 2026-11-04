#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc/GigClosure.h"

namespace arpc {
class RPCServerClosure;
}

namespace multi_call {

class ArpcClosure : public GigClosure
{
public:
    ArpcClosure(arpc::RPCServerClosure *closure, const QueryInfoPtr &queryInfo,
                const CompatibleFieldInfo *info)
        : _closure(closure) {
        assert(queryInfo);
        assert(_closure);
        setQueryInfo(queryInfo);
        setNeedFinishQueryInfo(true);
        setCompatibleFieldInfo(info);
    }
    ~ArpcClosure() {
        _closure = NULL;
    }

private:
    ArpcClosure(const ArpcClosure &);
    ArpcClosure &operator=(const ArpcClosure &);

public:
    int64_t getStartTime() const override;
    void Run() override;
    ProtocolType getProtocolType() override;
    const std::shared_ptr<google::protobuf::Arena> &getProtobufArena() const;
private:
    arpc::RPCServerClosure *_closure;
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ArpcClosure);

} // namespace multi_call

