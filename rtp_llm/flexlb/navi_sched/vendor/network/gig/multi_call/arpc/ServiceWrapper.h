#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/arpc/ArpcClosure.h"
#include "google/protobuf/service.h"

namespace multi_call {

class GigRpcServer;
class GigAgent;

class ServiceWrapper : public google::protobuf::Service
{
public:
    ServiceWrapper(GigRpcServer &owner, google::protobuf::Service *rpcService,
                   const CompatibleFieldInfo &compatibleInfo);
    ~ServiceWrapper();

private:
    ServiceWrapper(const ServiceWrapper &);
    ServiceWrapper &operator=(const ServiceWrapper &);

public:
    const google::protobuf::ServiceDescriptor *GetDescriptor() override;
    void CallMethod(const google::protobuf::MethodDescriptor *method,
                    google::protobuf::RpcController *controller,
                    const google::protobuf::Message *request, google::protobuf::Message *response,
                    google::protobuf::Closure *done) override;
    const google::protobuf::Message &
    GetRequestPrototype(const google::protobuf::MethodDescriptor *method) const override;
    const google::protobuf::Message &
    GetResponsePrototype(const google::protobuf::MethodDescriptor *method) const override;

public:
    void setRpcServicePtr(const std::shared_ptr<google::protobuf::Service> &rpcServicePtr) {
        _rpcServicePtr = rpcServicePtr;
    }
    const std::shared_ptr<google::protobuf::Service> &getRpcServicePtr() const {
        return _rpcServicePtr;
    }

private:
    std::string getGigRequestMeta(arpc::Tracer *trace, const google::protobuf::Message *request);


private:
    GigRpcServer &_owner;
    std::shared_ptr<GigAgent> _agent;
    google::protobuf::Service *_rpcService;
    CompatibleFieldInfo _compatibleInfo;
    std::shared_ptr<google::protobuf::Service> _rpcServicePtr;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ServiceWrapper);

} // namespace multi_call

