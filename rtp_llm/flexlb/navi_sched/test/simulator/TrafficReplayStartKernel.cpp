#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayR.h"
#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayStartData.h"
#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayStartR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched::test {

class TrafficReplayStartKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_NAMED_DATA(TrafficReplayStartRequestData, _request,
                      TRAFFIC_REPLAY_START_REQUEST_DATA_NAME);
    KERNEL_DEPEND_ON(TrafficReplayStartR, _trafficReplayStartR);
    navi::AsyncPipePtr _responsePipe;
};

void TrafficReplayStartKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.master_traffic_replay_start_k")
        .output("start", TRAFFIC_REPLAY_START_RESPONSE_TYPE_ID);
}

navi::ErrorCode TrafficReplayStartKernel::init(navi::KernelInitContext &ctx)
{
    _responsePipe = ctx.createAsyncPipe();
    if (!_responsePipe || !_trafficReplayStartR->join(*_request, _responsePipe)) {
        return navi::EC_INIT_SUB_GRAPH;
    }
    NAVI_KERNEL_LOG(INFO, "traffic replay start kernel initialized");
    return navi::EC_NONE;
}

navi::ErrorCode TrafficReplayStartKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!_responsePipe->getData(data, eof)) {
        return navi::EC_NONE;
    }
    auto response = std::dynamic_pointer_cast<TrafficReplayStartResponseData>(data);
    if (!response) {
        return navi::EC_ABORT;
    }
    NAVI_KERNEL_LOG(INFO, "traffic replay start response received");
    ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), response, true);
    return navi::EC_NONE;
}

class TrafficReplayStartResultKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(TrafficReplayBridgeR, _trafficReplayBridgeR);
    std::shared_ptr<TrafficReplayR> _trafficReplayR;
};

void TrafficReplayStartResultKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.client_simulate_traffic_replay_start_result_k")
        .input("start", TRAFFIC_REPLAY_START_RESPONSE_TYPE_ID)
        .output("finished", "");
}

navi::ErrorCode TrafficReplayStartResultKernel::init(navi::KernelInitContext &)
{
    _trafficReplayR = _trafficReplayBridgeR->waitTrafficReplayR(0);
    return _trafficReplayR ? navi::EC_NONE : navi::EC_INIT_SUB_GRAPH;
}

navi::ErrorCode TrafficReplayStartResultKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        return navi::EC_NONE;
    }
    auto response = std::dynamic_pointer_cast<TrafficReplayStartResponseData>(data);
    if (!response ||
        !_trafficReplayR->start(response->startTimeMs, response->firstRequestTimeMs)) {
        return navi::EC_ABORT;
    }
    NAVI_KERNEL_LOG(INFO, "traffic replay start result received");
    ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
    return navi::EC_NONE;
}

REGISTER_KERNEL(TrafficReplayStartKernel);
REGISTER_KERNEL(TrafficReplayStartResultKernel);

} // namespace navi_sched::test
