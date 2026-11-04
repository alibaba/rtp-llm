#include "navi/resource/NaviStateRpcR.h"
#include "navi/rpc_server/NaviRpcServerR.h"
#include "navi/proto/NaviState.pb.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/ProtoJsonizer.h"

namespace navi {

class NaviStateProtoJsonizer : public http_arpc::ProtoJsonizer
{
public:
    NaviStateProtoJsonizer() {
    }
    ~NaviStateProtoJsonizer() {
    }
    NaviStateProtoJsonizer(const NaviStateProtoJsonizer &) = delete;
    NaviStateProtoJsonizer &operator=(const NaviStateProtoJsonizer &) = delete;
public:
    std::string toJson(const google::protobuf::Message &message) override {
        auto response = dynamic_cast<const NaviStateResponseDef *>(&message);
        if (!response) {
            return "error response type";
        } else {
            return response->response();
        }
    }
};

const std::string NaviStateRpcR::RESOURCE_ID = "navi.state_rpc_r";
const std::string NaviStateRpcR::REQUEST_NAME = "navi.state.rpc_request";

NaviStateRpcR::NaviStateRpcR() {
}

NaviStateRpcR::~NaviStateRpcR() {
}

void NaviStateRpcR::def(ResourceDefBuilder &builder) const {
    builder.name(RESOURCE_ID, RS_SNAPSHOT);
}

bool NaviStateRpcR::config(ResourceConfigContext &ctx) {
    return true;
}

ErrorCode NaviStateRpcR::init(ResourceInitContext &ctx) {
    ArpcRegistryParam param;
    param.arpcParam.method = "navi";
    param.arpcParam.httpAliasVec.push_back("/navi");
    param.arpcParam.protoJsonizer.reset(new NaviStateProtoJsonizer());
    param.graphParam.requestDataName = REQUEST_NAME;
    GraphBuilder builder(&param.graphParam.graph);
    if (!buildGraph(builder)) {
        return EC_ABORT;
    }
    auto service = _naviRpcServerR->registerArpcService<NaviStateService>(param, this);
    if (!service) {
        NAVI_KERNEL_LOG(ERROR, "register /sql arpc service failed");
        return EC_ABORT;
    }
    return EC_NONE;
}

bool NaviStateRpcR::buildGraph(GraphBuilder &builder) const {
    builder.newSubGraph(NAVI_BUILDIN_BIZ);
    auto naviState = builder.node("navi_state").kernel(NAVI_STATE_KERNEL_NAME);
    naviState.out(NAVI_STATE_KERNEL_OUTPUT_PORT).asGraphOutput("output");
    return builder.ok();
}

REGISTER_BUILDIN_RESOURCE(NaviStateRpcR);

}

