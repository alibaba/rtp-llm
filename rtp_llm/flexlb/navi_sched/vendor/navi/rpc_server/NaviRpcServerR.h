#pragma once

#include "navi/engine/Resource.h"
#include "navi/rpc_server/ArpcRegistryParam.h"
#include "multi_call/rpc/GigRpcServer.h"

namespace navi {

class NaviSnapshot;
class ArpcServiceRegistry;
class RpcInfoDef;

class NaviRpcServerR : public navi::ExternalResource
{
public:
    NaviRpcServerR();
    NaviRpcServerR(NaviSnapshot *snapshot, multi_call::GigRpcServer *gigRpcServer);
    ~NaviRpcServerR();
    NaviRpcServerR(const NaviRpcServerR &) = delete;
    NaviRpcServerR &operator=(const NaviRpcServerR &) = delete;
public:
    template <typename RpcService>
    std::shared_ptr<google::protobuf::Service> registerArpcService(const ArpcRegistryParam &param,
                                                                   /* caller is required to show in navi_web */
                                                                   Resource *caller) const;
    bool unRegisterArpcService(const std::shared_ptr<google::protobuf::Service> &service);
public:
    multi_call::GigRpcServer *getGigRpcServer() const {
        return _gigRpcServer;
    }
private:
    std::shared_ptr<google::protobuf::Service>
    doRegisterArpcService(const std::shared_ptr<google::protobuf::Service> &service, const ArpcRegistryParam &param, Resource *caller) const;
    bool validate(const google::protobuf::ServiceDescriptor *desc,
                  const ArpcRegistryParam &param) const;
    bool flushRegistry();
    bool flushOneRegistry(const std::shared_ptr<ArpcServiceRegistry> &registry);
    void stop();
    void fillProto(RpcInfoDef &rpcInfoDef) const;
public:
    static const std::string RESOURCE_ID;
private:
    friend class NaviSnapshot;
private:
    NaviSnapshot *_snapshot;
    multi_call::GigRpcServer *_gigRpcServer;
    mutable NaviUnorderedMap<std::string, NaviUnorderedMap<std::string, std::shared_ptr<ArpcServiceRegistry>>> _registryMap;
    std::vector<std::shared_ptr<ArpcServiceRegistry>> _flushedRegistry;
    bool _flushed = false;
    mutable autil::ThreadMutex _registerLock;
};

template <typename RpcService>
std::shared_ptr<google::protobuf::Service> NaviRpcServerR::registerArpcService(const ArpcRegistryParam &param, Resource *caller) const {
    autil::ScopedLock lock(_registerLock);
    auto service = std::make_shared<typename RpcService::Stub>(nullptr);
    return doRegisterArpcService(service, param, caller);
}

NAVI_TYPEDEF_PTR(NaviRpcServerR);

}

