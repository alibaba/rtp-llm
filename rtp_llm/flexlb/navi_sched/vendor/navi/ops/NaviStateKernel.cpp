#include "navi/ops/NaviStateKernel.h"
#include "navi/engine/Navi.h"
#include "navi/engine/NaviSnapshot.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/engine/NaviHostInfo.h"
#include "navi/proto/NaviState.pb.h"
#include "navi/rpc_server/NaviArpcResponseData.h"

namespace navi {

NaviStateKernel::NaviStateKernel() {
}

NaviStateKernel::~NaviStateKernel() {
}

void NaviStateKernel::def(KernelDefBuilder &builder) const {
    builder.name(NAVI_STATE_KERNEL_NAME)
        .output(NAVI_STATE_KERNEL_OUTPUT_PORT, "");
}

bool NaviStateKernel::config(KernelConfigContext &ctx) {
    return true;
}

ErrorCode NaviStateKernel::init(KernelInitContext &ctx) {
    return EC_NONE;
}

ErrorCode NaviStateKernel::compute(KernelComputeContext &ctx) {
    auto requestPb = _requestData->getRequest<NaviStateRequestDef>();
    if (!requestPb) {
        NAVI_KERNEL_LOG(ERROR, "invalid request pb data type");
        return EC_ABORT;
    }
    auto responsePb = new NaviStateResponseDef();
    auto navi = _naviStateR->_navi;
    fillVisData(navi, responsePb);
    auto outputData = std::make_shared<NaviArpcResponseData>(responsePb);
    if (!ctx.setOutput(0, outputData, true)) {
        return EC_ABORT;
    }
    return EC_NONE;
}

void NaviStateKernel::fillVisData(Navi *navi, NaviStateResponseDef *responsePb) {
    if (!navi) {
        responsePb->set_response("navi is null");
        return;
    }
    auto hostInfo = navi->getHostInfo();
    auto visProto = hostInfo ? hostInfo->getInitVisProto(false) : nullptr;
    if (visProto) {
        if (visProto->nodes_size() > 0) {
            auto &nodeDef = *visProto->mutable_nodes(0);
            if (1 == nodeDef.sub_results_size()) {
                auto &subResults = *nodeDef.mutable_sub_results();
                auto &childNodeDef = subResults.begin()->second;
                if (childNodeDef.has_navi_summary()) {
                    auto realNavi = navi->getRealNavi();
                    if (realNavi) {
                        auto snapshot = realNavi->getSnapshot();
                        if (snapshot) {
                            snapshot->updateProto(*childNodeDef.mutable_navi_summary());
                        }
                    }
                }
            }
        }
        responsePb->set_response(visProto->SerializeAsString());
    }
}

REGISTER_BUILDIN_KERNEL(NaviStateKernel);

}
