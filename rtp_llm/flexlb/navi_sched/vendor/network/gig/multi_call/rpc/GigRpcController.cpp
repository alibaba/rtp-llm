#include "multi_call/rpc/GigRpcController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCControllerBase.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GigRpcController);

std::string GigRpcController::getClientIp() const {
    auto protocolController = getProtocolController();
    auto anetRpcController = dynamic_cast<arpc::ANetRPCController *>(protocolController);
    if (anetRpcController) {
        return "arpc " + anetRpcController->GetClientAddress();
    } else {
        auto httpController = dynamic_cast<http_arpc::HTTPRPCControllerBase *>(protocolController);
        if (httpController) {
            return "http " + httpController->GetAddr();
        }
    }
    return "unknown";
}

}
