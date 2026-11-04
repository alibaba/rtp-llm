#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/arpc/ServiceWrapper.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerClosure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/GigAgent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/arpc/CommonClosure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/QuerySession.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc/GigRpcServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/ProtobufUtil.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCServerClosure.h"
#include "autil/legacy/base64.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, ServiceWrapper);

ServiceWrapper::ServiceWrapper(GigRpcServer &owner, google::protobuf::Service *rpcService,
                               const CompatibleFieldInfo &compatibleInfo)
    : _owner(owner)
    , _agent(owner.getAgent())
    , _rpcService(rpcService)
    , _compatibleInfo(compatibleInfo) {
}

ServiceWrapper::~ServiceWrapper() {
}

const google::protobuf::ServiceDescriptor *ServiceWrapper::GetDescriptor() {
    return _rpcService->GetDescriptor();
}

void ServiceWrapper::CallMethod(const google::protobuf::MethodDescriptor *method,
                                google::protobuf::RpcController *controller,
                                const google::protobuf::Message *request,
                                google::protobuf::Message *response,
                                google::protobuf::Closure *done) {
    GigClosure *gigClosure = nullptr;
    QuerySessionPtr session;
    auto searchService = _owner.getSearchService();

    auto arpcClosure = dynamic_cast<arpc::RPCServerClosure *>(done);
    QueryInfoPtr queryInfo;
    if (arpcClosure) {
        auto tracer = arpcClosure->GetTracer();
        assert(tracer);
        auto gigMeta = getGigRequestMeta(tracer, request);
        queryInfo = _agent->getQueryInfo(gigMeta);
        gigClosure = new ArpcClosure(arpcClosure, queryInfo, &_compatibleInfo);
        session.reset(new QuerySession(searchService, queryInfo));
    } else { // http_arpc mode
        auto httpClosure = dynamic_cast<http_arpc::HTTPRPCServerClosure *>(done);
        if (httpClosure) {
            const http_arpc::EagleInfo &eagleInfo = httpClosure->getEagleInfo();
            if (!eagleInfo.gigdata.empty()) {
                std::string gigData = autil::legacy::Base64DecodeFast(eagleInfo.gigdata);
                queryInfo = _agent->getQueryInfo(gigData);
            }
            session.reset(new QuerySession(searchService, queryInfo));
            
        } else {
            session.reset(new QuerySession(searchService));
        }
        gigClosure = new CommonClosure(done);
    }

    

    gigClosure->getController().setRpcController(controller);
    gigClosure->setQueryInfo(queryInfo);
    gigClosure->setQuerySession(session);

    _rpcService->CallMethod(method, &gigClosure->getController(), request, response, gigClosure);
}

const google::protobuf::Message &
ServiceWrapper::GetRequestPrototype(const google::protobuf::MethodDescriptor *method) const {
    return _rpcService->GetRequestPrototype(method);
}

const google::protobuf::Message &
ServiceWrapper::GetResponsePrototype(const google::protobuf::MethodDescriptor *method) const {
    return _rpcService->GetResponsePrototype(method);
}

std::string ServiceWrapper::getGigRequestMeta(arpc::Tracer *tracer,
                                              const google::protobuf::Message *request) {
    auto meta = tracer->getUserPayload();
    if (!meta.empty()) {
        return meta;
    }
    const std::string &field = _compatibleInfo.requestInfoField;
    ProtobufCompatibleUtil::getGigMetaField(request, field, meta);
    return meta;
}



} // namespace multi_call
