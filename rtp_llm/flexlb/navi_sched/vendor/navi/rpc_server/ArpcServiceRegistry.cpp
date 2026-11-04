#include "navi/rpc_server/ArpcServiceRegistry.h"
#include "navi/engine/NaviSnapshot.h"
#include "navi/log/NaviLogger.h"
#include "navi/util/CommonUtil.h"
#include "navi/rpc_server/ArpcGraphClosure.h"
#include "navi/rpc_server/NaviArpcRequestData.h"
#include <autil/StringUtil.h>
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerClosure.h"
#include <multi_call/arpc/CommonClosure.h>
#include <multi_call/arpc/ArpcClosure.h>

namespace navi {

void RegistryCallerInfo::init(Resource *caller) {
    addr = caller;
    if (caller) {
        const auto &resource = *caller;
        name = caller->getName();
        type = CommonUtil::parseSymbol(typeid(resource).name());
    }
}

ArpcServiceRegistry::ArpcServiceRegistry(NaviSnapshot *snapshot,
                                         const std::shared_ptr<google::protobuf::Service> &service)
    : _snapshot(snapshot)
    , _service(service)
{
}

ArpcServiceRegistry::~ArpcServiceRegistry() {
}

bool ArpcServiceRegistry::init(const ArpcRegistryParam &param, Resource *caller) {
    if (!caller) {
        NAVI_KERNEL_LOG(ERROR, "caller is null, must provide call resource");
        return false;
    }
    auto desc = _service->GetDescriptor();
    if (!desc) {
        NAVI_KERNEL_LOG(ERROR, "get service descriptor failed");
        return false;
    }
    const auto &methodName = param.arpcParam.method;
    if (methodName.empty()) {
        NAVI_KERNEL_LOG(ERROR, "empty method name");
        return false;
    }
    const auto &graphParam = param.graphParam;
    const auto &requestDataName = graphParam.requestDataName;
    if (requestDataName.empty()) {
        NAVI_KERNEL_LOG(ERROR, "empty request data name, method [%s]",
                        methodName.c_str());
        return false;
    }
    const auto &graphDef = graphParam.graph;
    auto subCount = graphDef.sub_graphs_size();
    if (subCount == 0) {
        NAVI_KERNEL_LOG(ERROR, "invalid graph, no sub graph, method [%s]",
                        methodName.c_str());
        return false;
    }
    for (int i = 0; i < subCount; i++) {
        auto graphId = graphDef.sub_graphs(i).graph_id();
        if (graphId >= 0) {
            _graphIds.push_back(graphId);
        }
    }
    auto methodDesc = desc->FindMethodByName(methodName);
    if (!methodDesc) {
        NAVI_KERNEL_LOG(ERROR, "method [%s] not exist in service [%s] def",
                        methodName.c_str(), desc->full_name().c_str());
        return false;
    }
    _param = std::make_shared<ArpcRegistryParam>(param);
    initHttpAliasMap();
    _callerInfo.init(caller);
    return true;
}

void ArpcServiceRegistry::initHttpAliasMap() {
    const auto &arpcParam = _param->arpcParam;
    auto httpPath = "/" + GetDescriptor()->name() + "/" + arpcParam.method;
    for (const auto &alias : arpcParam.httpAliasVec) {
        _httpAliasMap[alias] = httpPath;
    }
}

const google::protobuf::ServiceDescriptor *ArpcServiceRegistry::GetDescriptor() {
    return _service->GetDescriptor();
}

void ArpcServiceRegistry::CallMethod(
    const google::protobuf::MethodDescriptor *method,
    google::protobuf::RpcController *controller,
    const google::protobuf::Message *request,
    google::protobuf::Message *response,
    google::protobuf::Closure *done)
{
    if (unlikely(!_param)) {
        controller->SetFailed("rpc registry param is null, method: " + method->full_name() + "\n");
        done->Run();
        return;
    }
    const auto &methodName = _param->arpcParam.method;
    if (method->name() != methodName) {
        controller->SetFailed("rpc method not supported: " + method->full_name() + ", expect: " + methodName + "\n");
        done->Run();
        return;
    }
    runGraph(controller, request, response, done);
}

const google::protobuf::Message &ArpcServiceRegistry::GetRequestPrototype(
    const google::protobuf::MethodDescriptor *method) const
{
    return _service->GetRequestPrototype(method);
}

const google::protobuf::Message &ArpcServiceRegistry::GetResponsePrototype(
    const google::protobuf::MethodDescriptor *method) const
{
    return _service->GetResponsePrototype(method);
}

void ArpcServiceRegistry::runGraph(
    google::protobuf::RpcController *controller,
    const google::protobuf::Message *request,
    google::protobuf::Message *response,
    google::protobuf::Closure *done)
{
    const auto &methodName = _param->arpcParam.method;
    const auto &graphParam = _param->graphParam;
    std::string clientIp;
    auto gigController =
        dynamic_cast<multi_call::GigRpcController *>(controller);
    if (gigController) {
        clientIp = gigController->getClientIp();
    } else {
        clientIp = "unknown";
    }
    auto httpClosure = getHttpClosure(done);
    auto requestData = createRequestData(request, httpClosure, done, methodName, clientIp);
    RunGraphParams runGraphParams = graphParam.params;
    auto gigClosure = dynamic_cast<multi_call::GigClosure *>(done);
    if (gigClosure) {
        runGraphParams.setQuerySession(gigClosure->getQuerySession());
    }
    NamedData namedRequest;
    namedRequest.name = graphParam.requestDataName;
    namedRequest.data = requestData;
    for (auto graphId : _graphIds) {
        namedRequest.graphId = graphId;
        auto ret = runGraphParams.addNamedData(namedRequest);
        (void)ret;
        assert(ret);
    }
    auto graphDef = new GraphDef();
    graphDef->CopyFrom(graphParam.graph);
    auto closure = new ArpcGraphClosure(_param, controller, requestData, response, done, httpClosure);
    if (httpClosure) {
        const auto &debugParamStr = httpClosure->getAiosDebugType();
        if (!debugParamStr.empty()) {
            httpClosure->addResponseHeader("Access-Control-Allow-Origin", "*");
            if (debugParamStr != "navi_web_query") {
                runGraphParams.setDebugParamStr(debugParamStr);
                closure->setNeedDebug(true);
                closure->waitDestruct = true;
                requestData->setAiosDebug(true);
            }
        }
    }
    requestData->setStopping(_snapshot->isStopping());
    _snapshot->runGraphAsync(graphDef, runGraphParams, nullptr, closure);
}

http_arpc::HTTPRPCServerClosure *ArpcServiceRegistry::getHttpClosure(google::protobuf::Closure *done) {
    auto commonClosure = dynamic_cast<multi_call::CommonClosure *>(done);
    if (!commonClosure) {
        return nullptr;
    }
    return dynamic_cast<http_arpc::HTTPRPCServerClosure *>(commonClosure->getClosure());
}

std::shared_ptr<NaviArpcRequestData>
ArpcServiceRegistry::createRequestData(const google::protobuf::Message *request,
                                       http_arpc::HTTPRPCServerClosure *httpClosure,
                                       google::protobuf::Closure *done,
                                       const std::string &methodName,
                                       const std::string &clientIp) const
{
    if (httpClosure) {
        return std::make_shared<NaviArpcRequestData>(
            nullptr, nullptr, httpClosure->stealRequest(), methodName, clientIp, RPT_HTTP);
    }
    auto arpcClosure = dynamic_cast<multi_call::ArpcClosure *>(done);
    if (arpcClosure) {
        const auto &arena = arpcClosure->getProtobufArena();
        return std::make_shared<NaviArpcRequestData>(request, arena, nullptr, methodName, clientIp, RPT_ARPC);
    }
    // should not reach
    return std::make_shared<NaviArpcRequestData>(request, nullptr, nullptr, methodName, clientIp, RPT_UNKNOWN);
}

void ArpcServiceRegistry::fillProto(RpcRegistryDef &def) {
    def.set_service_name(getServiceName());
    {
        const auto &arpcParam = _param->arpcParam;
        auto &paramDef = *def.mutable_arpc_param();
        paramDef.set_method(arpcParam.method);
        paramDef.set_log_query(arpcParam.logQuery);
        {
            const auto &compatibleInfo = arpcParam.compatibleInfo;
            auto &compatibleInfoDef = *paramDef.mutable_compatible_info();
            compatibleInfoDef.set_request_info_field(compatibleInfo.requestInfoField);
            compatibleInfoDef.set_ec_field(compatibleInfo.ecField);
            compatibleInfoDef.set_response_info_field(compatibleInfo.responseInfoField);
        }
        {
            const auto &threadPoolDescriptor = arpcParam.threadPoolDescriptor;
            auto &threadPoolDescriptorDef = *paramDef.mutable_thread_pool_descriptor();
            threadPoolDescriptorDef.set_name(threadPoolDescriptor.threadPoolName);
            threadPoolDescriptorDef.set_thread_num(threadPoolDescriptor.threadNum);
            threadPoolDescriptorDef.set_queue_size(threadPoolDescriptor.queueSize);
            threadPoolDescriptorDef.set_factory((uint64_t)threadPoolDescriptor.factory.get());
            threadPoolDescriptorDef.set_allow_shared_pool_override(threadPoolDescriptor._allowSharedPoolOverride);
        }
        {
            auto &protoJsonizerDef = *paramDef.mutable_proto_jsonizer();
            protoJsonizerDef.set_addr((uint64_t)arpcParam.protoJsonizer.get());
            if (arpcParam.protoJsonizer) {
                const auto &jsonizer = *arpcParam.protoJsonizer;
                protoJsonizerDef.set_class_name(CommonUtil::parseSymbol(typeid(jsonizer).name()));
            }
        }
        {
            const auto &httpAliasVec = arpcParam.httpAliasVec;
            auto httpPath = "/" + GetDescriptor()->name() + "/" + arpcParam.method;
            auto &aliasDef = *paramDef.mutable_alias();
            aliasDef.set_path(httpPath);
            for (const auto &alias : httpAliasVec) {
                aliasDef.add_alias(alias);
            }
        }
    }
    {
        const auto &graphParam = _param->graphParam;
        auto &graphParamDef = *def.mutable_graph_param();
        graphParamDef.set_request_data_name(graphParam.requestDataName);
        graphParamDef.mutable_graph()->CopyFrom(graphParam.graph);
    }
    {
        auto &callerDef  = *def.mutable_caller();
        callerDef.set_name(_callerInfo.name);
        callerDef.set_type(_callerInfo.type);
        callerDef.set_addr((uint64_t)_callerInfo.addr);
    }
    {
        auto desc = _service->GetDescriptor();
        if (desc) {
            auto methodDesc = desc->FindMethodByName(_param->arpcParam.method);
            if (methodDesc) {
                auto serviceId = (uint32_t)(desc->options().GetExtension(arpc::global_service_id));
                auto methodId = (uint32_t)(methodDesc->options().GetExtension(arpc::local_method_id));
                auto &messageTypeDef = *def.mutable_message_type();
                messageTypeDef.set_global_service_id(serviceId);
                messageTypeDef.set_local_method_id(methodId);
                {
                    std::string depend;
                    const auto &requestMessage = GetRequestPrototype(methodDesc);
                    messageTypeDef.set_request_proto(getMessageProto(requestMessage, depend));
                    messageTypeDef.set_request_depend_proto(depend);
                }
                {
                    std::string depend;
                    const auto &responseMessage = GetResponsePrototype(methodDesc);
                    messageTypeDef.set_response_proto(getMessageProto(responseMessage, depend));
                    messageTypeDef.set_response_depend_proto(depend);
                }
            }
        }
    }
}

std::string ArpcServiceRegistry::getMessageProto(const google::protobuf::Message &message, std::string &dependProto) {
    auto descriptor = message.GetDescriptor();
    NaviUnorderedSet<std::string> generatedSet;
    return getMessageProtoRecur(descriptor, generatedSet, dependProto);
}

std::string ArpcServiceRegistry::getMessageProtoRecur(const google::protobuf::Descriptor *descriptor,
                                                      NaviUnorderedSet<std::string> &generatedSet,
                                                      std::string &dependProto)
{
    const auto &name = descriptor->full_name();
    generatedSet.insert(name);
    auto fieldCount = descriptor->field_count();
    for (int i = 0; i < fieldCount; i++) {
        auto field = descriptor->field(i);
        auto fieldDescriptor = field->message_type();
        if (fieldDescriptor) {
            const auto &fieldName = fieldDescriptor->full_name();
            if (generatedSet.end() == generatedSet.find(fieldName)) {
                std::string depend;
                auto fieldProto = getMessageProtoRecur(fieldDescriptor, generatedSet, depend);
                dependProto += fieldProto + depend;
            }
        } else {
            auto enumDescriptor = field->enum_type();
            if (enumDescriptor) {
                const auto &fieldName = enumDescriptor->full_name();
                if (generatedSet.end() == generatedSet.find(fieldName)) {
                    dependProto += enumDescriptor->DebugString();
                }
            }
        }
    }
    return descriptor->DebugString();
}

}
