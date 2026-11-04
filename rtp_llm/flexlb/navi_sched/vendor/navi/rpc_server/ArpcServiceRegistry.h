#pragma once

#include "autil/Lock.h"
#include "navi/rpc_server/ArpcRegistryParam.h"
#include <google/protobuf/service.h>

namespace http_arpc {
class HTTPRPCServerClosure;
}

namespace navi {

class NaviArpcRequestData;
class NaviSnapshot;
class RpcRegistryDef;

struct RegistryCallerInfo {
public:
    void init(Resource *caller);
public:
    std::string name;
    void *addr = nullptr;
    std::string type;
};

class ArpcServiceRegistry: public google::protobuf::Service
{
public:
    ArpcServiceRegistry(NaviSnapshot *snapshot,
                        const std::shared_ptr<google::protobuf::Service> &service);
    ~ArpcServiceRegistry();
    ArpcServiceRegistry(const ArpcServiceRegistry &) = delete;
    ArpcServiceRegistry &operator=(const ArpcServiceRegistry &) = delete;
public:
    bool init(const ArpcRegistryParam &param, Resource *caller);
    const ArpcRegistryParamPtr &getParam() const {
        return _param;
    }
    const std::string &getServiceName() const {
        return _service->GetDescriptor()->full_name();
    }
    const std::map<std::string, std::string> &getHttpAliasMap() const {
        return _httpAliasMap;
    }
    void fillProto(RpcRegistryDef &def);
private:
    const google::protobuf::ServiceDescriptor* GetDescriptor() override;
    void CallMethod(const google::protobuf::MethodDescriptor *method,
                    google::protobuf::RpcController *controller,
                    const google::protobuf::Message *request,
                    google::protobuf::Message *response,
                    google::protobuf::Closure *done) override;
    const google::protobuf::Message &GetRequestPrototype(
        const google::protobuf::MethodDescriptor *method) const override;
    const google::protobuf::Message &GetResponsePrototype(
        const google::protobuf::MethodDescriptor *method) const override;
private:
    void initHttpAliasMap();
    void runGraph(google::protobuf::RpcController *controller,
                  const google::protobuf::Message *request,
                  google::protobuf::Message *response,
                  google::protobuf::Closure *done);
    http_arpc::HTTPRPCServerClosure *getHttpClosure(google::protobuf::Closure *done);
    std::shared_ptr<NaviArpcRequestData> createRequestData(const google::protobuf::Message *request,
                                                           http_arpc::HTTPRPCServerClosure *httpClosure,
                                                           google::protobuf::Closure *done,
                                                           const std::string &methodName,
                                                           const std::string &clientIp) const;
private:
    static std::string getMessageProto(const google::protobuf::Message &message, std::string &dependProto);
    static std::string getMessageProtoRecur(const google::protobuf::Descriptor *descriptor,
                                            NaviUnorderedSet<std::string> &generatedSet,
                                            std::string &dependProto);
private:
    NaviSnapshot *_snapshot;
    std::shared_ptr<google::protobuf::Service> _service;
    ArpcRegistryParamPtr _param;
    std::vector<GraphId> _graphIds;
    std::map<std::string, std::string> _httpAliasMap;
    RegistryCallerInfo _callerInfo;
};

NAVI_TYPEDEF_PTR(ArpcServiceRegistry);

}

