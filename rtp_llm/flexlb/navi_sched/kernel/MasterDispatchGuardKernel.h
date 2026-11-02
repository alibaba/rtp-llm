#pragma once

#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class MasterDispatchGuardKernel : public navi::Kernel {
public:
    ~MasterDispatchGuardKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

protected:
    bool acceptDecision(const MasterResponseToClientDataPtr &decision);
    void consumeAdmissionTimeout(navi::KernelComputeContext &ctx);
    bool consumeDispatchAck();
    navi::ErrorCode consumeForwardDecision(navi::KernelComputeContext &ctx);
    void stopAdmissionTimeout(navi::KernelComputeContext &ctx);
    void releaseUnacknowledgedReservation();
    navi::ErrorCode finish(navi::KernelComputeContext &ctx);

protected:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(MasterStateR, _masterStateR);
    KERNEL_DEPEND_ON_FALSE(SchedulerConfigR, _schedulerConfigR);
    MasterResponseToClientDataPtr _decision;
    navi::AsyncPipePtr _dispatchAckPipe;
    navi::AsyncPipePtr _admissionTimeoutPipe;
    bool _registered = false;
    bool _acknowledged = false;
    bool _admissionTimedOut = false;
    bool _admissionTimeoutStopped = false;
    bool _forwardDecisionReceived = false;
    bool _forwardDecisionEof = false;
    bool _finished = false;
    int64_t _admissionTimeoutMs = 0;
    navi::IndexType _forwardDecisionPort = 1;
    navi::IndexType _finishedPort = 0;
};

}
