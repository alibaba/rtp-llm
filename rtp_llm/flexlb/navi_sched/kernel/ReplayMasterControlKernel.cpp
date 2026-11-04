#include "aios/navi_sched/kernel/ReplayMasterControlKernel.h"

#include "navi/engine/NaviRegisterMacro.h"

namespace navi_sched {

const std::string ReplayMasterControlKernel::KERNEL_ID =
    "navi_sched.replay_master_control_k";

ReplayMasterControlKernel::~ReplayMasterControlKernel()
{
    if (_replayControlR && _commandPipe && !_clientId.empty()) {
        _replayControlR->unsubscribe(_clientId, _commandPipe,
                                     _historyQueryPipe);
    }
}

void ReplayMasterControlKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("capability", ReplayClientCapabilityData::TYPE_ID)
        .output("command", ReplayControlCommandData::TYPE_ID)
        .output("history_query", ReplayHistoryQueryData::TYPE_ID);
}

navi::ErrorCode ReplayMasterControlKernel::init(navi::KernelInitContext &ctx)
{
    _commandPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _historyQueryPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_commandPipe || !_historyQueryPipe) {
        NAVI_KERNEL_LOG(ERROR, "create replay master command pipe failed");
        return navi::EC_INIT_SUB_GRAPH;
    }
    return navi::EC_NONE;
}

navi::ErrorCode ReplayMasterControlKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr capabilityData;
    bool capabilityEof = false;
    if (ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), capabilityData,
                    capabilityEof) && capabilityData) {
        auto capability = std::dynamic_pointer_cast<ReplayClientCapabilityData>(capabilityData);
        if (!capability || capability->clientId.empty()) {
            NAVI_KERNEL_LOG(ERROR, "invalid replay client capability");
            return navi::EC_ABORT;
        }
        if (capability->clientId != _clientId) {
            if (!_clientId.empty()) {
                _replayControlR->unsubscribe(_clientId, _commandPipe,
                                             _historyQueryPipe);
            }
            _clientId = capability->clientId;
            if (!_replayControlR->subscribe(_clientId, _commandPipe,
                                            _historyQueryPipe)) {
                NAVI_KERNEL_LOG(ERROR,
                                "subscribe replay client command failed, client [%s]",
                                _clientId.c_str());
                return navi::EC_ABORT;
            }
        }
        _replayControlR->updateHistory(capability->summary);
    }
    navi::DataPtr commandData;
    bool commandEof = false;
    if (_commandPipe->getData(commandData, commandEof)) {
        if (commandEof) {
            if (!ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr,
                               true)) {
                return navi::EC_ABORT;
            }
        } else {
            auto command =
                std::dynamic_pointer_cast<ReplayControlCommandData>(commandData);
            if (!command ||
                !ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), command,
                               false)) {
                return navi::EC_ABORT;
            }
        }
    }
    navi::DataPtr historyQueryData;
    bool historyQueryEof = false;
    if (_historyQueryPipe->getData(historyQueryData, historyQueryEof)) {
        if (historyQueryEof) {
            if (!ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), nullptr,
                               true)) {
                return navi::EC_ABORT;
            }
        } else {
            auto query =
                std::dynamic_pointer_cast<ReplayHistoryQueryData>(historyQueryData);
            if (!query ||
                !ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), query,
                               false)) {
                return navi::EC_ABORT;
            }
            NAVI_KERNEL_LOG(INFO,
                            "sent replay history query [%lu], client [%s]",
                            query->queryId, _clientId.c_str());
        }
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(ReplayMasterControlKernel);

}
