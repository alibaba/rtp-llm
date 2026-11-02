#include "aios/navi_sched/kernel/RouteModeKernel.h"

#include "aios/navi_sched/proto/RouteMode.pb.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/rpc_server/NaviArpcResponseData.h"

namespace navi_sched {

const std::string RouteModeKernel::KERNEL_ID = "navi_sched.route_mode_k";
const std::string RouteModeKernel::OUTPUT_PORT = "navi_sched.route_mode_k.out";

void RouteModeKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID).output(OUTPUT_PORT, "");
}

navi::ErrorCode RouteModeKernel::compute(navi::KernelComputeContext &ctx)
{
    if (_requestData->isStopping()) {
        return navi::EC_ABORT;
    }
    auto request = dynamic_cast<const RouteModeRequest *>(_requestData->getRequest());
    if (!request) {
        NAVI_KERNEL_LOG(ERROR, "invalid route mode request type");
        return navi::EC_ABORT;
    }
    RouteModeR::Snapshot snapshot;
    bool accepted = true;
    if (request->update()) {
        accepted = _routeModeR->update(request->master_address(), snapshot);
    } else if (request->watch()) {
        snapshot = _routeModeR->waitAfter(request->generation(), request->timeout_ms());
    } else {
        snapshot = _routeModeR->get();
    }
    auto response = new RouteModeResponse();
    response->set_master_address(snapshot.masterAddress);
    response->set_generation(snapshot.generation);
    response->set_accepted(accepted);
    if (!accepted) {
        response->set_error("master_address must be empty or host:port");
    }
    ctx.setOutput(0, navi::DataPtr(new navi::NaviArpcResponseData(response)), true);
    return navi::EC_NONE;
}

REGISTER_KERNEL(RouteModeKernel);

}
