#include "aios/navi_sched/scheduler/SchedulerBaseKernel.h"

#include "aios/navi_sched/scheduler/MasterScheduleContextData.h"

namespace navi_sched {

void SchedulerBaseKernel::def(navi::KernelDefBuilder &builder,
                              const std::string &name) const
{
    builder.name(name)
        .input("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID)
        .output("result", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
}

navi::ErrorCode SchedulerBaseKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        return navi::EC_NONE;
    }
    if (!data) {
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, eof);
        return navi::EC_NONE;
    }
    auto context = MasterScheduleContextData::getContext(data);
    if (validateScheduleInput(context) && !schedule(*context)) {
        context->fail("scheduler_error");
    }
    ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof);
    return navi::EC_NONE;
}

bool SchedulerBaseKernel::validateScheduleInput(
    const NaviSchedScheduleContextPtr &context)
{
    if (!context) {
        NAVI_KERNEL_LOG(ERROR, "schedule context is null");
        return false;
    }
    if (0 == context->requestCount()) {
        NAVI_KERNEL_LOG(ERROR, "schedule context has no request");
        context->fail("scheduler_error");
        return false;
    }
    if (!context->valid()) {
        NAVI_KERNEL_LOG(ERROR, "schedule context is invalid");
        context->fail("scheduler_error");
        return false;
    }
    if (0 == context->nodeCount()) {
        NAVI_KERNEL_LOG(WARN, "schedule context has no available node");
        context->fail("no_available_node");
        return false;
    }
    return true;
}

bool SchedulerBaseKernel::schedule(NaviSchedScheduleContext &) const
{
    NAVI_KERNEL_LOG(ERROR, "scheduler has no synchronous implementation");
    return false;
}

}
