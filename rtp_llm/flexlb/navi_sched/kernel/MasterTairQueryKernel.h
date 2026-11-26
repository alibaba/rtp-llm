#pragma once
#include "aios/navi_sched/resource/MasterTairQueryR.h"
#include "navi/engine/Kernel.h"
#include <deque>

namespace navi_sched {
class MasterTairQueryKernel : public navi::Kernel {
public:
    static const std::string KERNEL_ID;
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;
private:
    void startBatch(MasterScheduleContextDataPtr data, bool eof, navi::KernelComputeContext &ctx);
    void applyResults();
    navi::ErrorCode emitBatch(navi::KernelComputeContext &ctx);
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(MasterTairQueryR, _queryR);
    navi::AsyncPipePtr _completionPipe;
    navi::AsyncPipePtr _deadlinePipe;
    bool _deadlineActive = false;
    MasterScheduleContextDataPtr _pending;
    MasterTairQueryR::BatchPtr _batch;
    bool _pendingEof = false;
    uint64_t _round = 0;
    std::deque<std::pair<MasterScheduleContextDataPtr, bool>> _queued;
};
} // namespace navi_sched
