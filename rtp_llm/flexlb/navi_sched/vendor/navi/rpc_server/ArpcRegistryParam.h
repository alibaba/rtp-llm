#pragma once

#include "arpc/RPCServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/ProtoJsonizer.h"
#include "multi_call/common/common.h"
#include "navi/engine/RunGraphParams.h"
#include "navi/proto/GraphDef.pb.h"

namespace navi {

struct RegistryArpcParam {
    std::string method;
    multi_call::CompatibleFieldInfo compatibleInfo;
    arpc::ThreadPoolDescriptor threadPoolDescriptor;
    http_arpc::ProtoJsonizerPtr protoJsonizer;
    std::vector<std::string> httpAliasVec;
    bool logQuery = false;
};

struct RegistryGraphParam {
    std::string requestDataName;
    RunGraphParams params;
    GraphDef graph;
};

struct ArpcRegistryParam {
    RegistryArpcParam arpcParam;
    RegistryGraphParam graphParam;
};

NAVI_TYPEDEF_PTR(ArpcRegistryParam);
}

