#include "rtp_llm/flexlb/navi_sched/data/AgentResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/ClientRequestToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/resource/ClientResponseR.h"
#include "navi/engine/Kernel.h"
#include "navi/log/NaviLogger.h"

namespace navi_sched {

class ClientResponseKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_NAMED_DATA(ClientRequestToMasterData, _request,
                      CLIENT_REQUEST_TO_MASTER_DATA_NAME);
    KERNEL_DEPEND_ON(ClientResponseR, _clientResponseR);
    bool _decisionEof = false;
    bool _replayEof = false;
    bool _responseEof = false;
    bool _finished = false;
};

void ClientResponseKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.client_response_k")
        .input("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID, navi::IT_OPTIONAL)
        .input("replay", MASTER_RESPONSE_TO_CLIENT_TYPE_ID, navi::IT_OPTIONAL, true)
        .input("response", AGENT_RESPONSE_TO_CLIENT_TYPE_ID, navi::IT_OPTIONAL)
        .output("finished", AGENT_RESPONSE_TO_CLIENT_TYPE_ID);
}

navi::ErrorCode ClientResponseKernel::compute(navi::KernelComputeContext &ctx)
{
    if (_finished || !_request || !_clientResponseR) {
        return _finished ? navi::EC_NONE : navi::EC_ABORT;
    }

    navi::DataPtr data;
    bool eof = false;
    if (!_decisionEof &&
        ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        if (data) {
            auto decision = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
            if (!decision ||
                !_clientResponseR->publishDecision(_request->requestId, decision)) {
                NAVI_KERNEL_LOG(ERROR, "failed to publish client decision, requestId [%lu]",
                                _request->requestId);
                return navi::EC_ABORT;
            }
        }
        _decisionEof = eof;
    }

    data.reset();
    eof = false;
    if (!_replayEof &&
        ctx.getInput(navi::PortIndex(1, navi::INVALID_INDEX), data, eof)) {
        if (data) {
            auto replay = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
            if (!replay || !_clientResponseR->publishReplay(_request->requestId, replay)) {
                NAVI_KERNEL_LOG(ERROR, "failed to publish client replay, requestId [%lu]",
                                _request->requestId);
                return navi::EC_ABORT;
            }
        }
        _replayEof = eof;
    } else if (!_replayEof && !_request->collectReplay) {
        _replayEof = true;
    }

    data.reset();
    eof = false;
    if (!_responseEof &&
        ctx.getInput(navi::PortIndex(2, navi::INVALID_INDEX), data, eof)) {
        if (data) {
            auto response = std::dynamic_pointer_cast<AgentResponseToClientData>(data);
            if (!response || !_clientResponseR->publishResponse(
                                 _request->requestId, response)) {
                NAVI_KERNEL_LOG(ERROR, "failed to publish client response, requestId [%lu]",
                                _request->requestId);
                return navi::EC_ABORT;
            }
        }
        _responseEof = eof;
    }

    if (_decisionEof && _replayEof && _responseEof) {
        _finished = true;
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(ClientResponseKernel);

} // namespace navi_sched
