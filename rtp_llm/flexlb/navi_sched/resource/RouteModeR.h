#pragma once

#include "navi/engine/Resource.h"
#include "navi/rpc_server/NaviRpcServerR.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>

namespace navi_sched {

class RouteModeR : public navi::Resource {
public:
    struct Snapshot {
        std::string masterAddress;
        uint64_t generation = 1;
    };

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    Snapshot get() const;
    bool update(const std::string &masterAddress, Snapshot &snapshot);
    Snapshot waitAfter(uint64_t generation, int64_t timeoutMs) const;

public:
    static const std::string RESOURCE_ID;
    static const std::string RPC_REQUEST_NAME;

private:
    bool buildGraph(navi::GraphDef &graphDef) const;
    static bool validMasterAddress(const std::string &address);

private:
    mutable std::mutex _mutex;
    mutable std::condition_variable _condition;
    Snapshot _snapshot;
    std::string _bizName;
    navi::NaviRpcServerR *_naviRpcServerR = nullptr;
};

NAVI_TYPEDEF_PTR(RouteModeR);

}
