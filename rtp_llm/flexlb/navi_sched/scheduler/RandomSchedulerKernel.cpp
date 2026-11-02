#include "aios/navi_sched/scheduler/RandomSchedulerKernel.h"

namespace navi_sched {

const std::string RandomSchedulerKernel::KERNEL_ID = "navi_sched.random_scheduler_k";

void RandomSchedulerKernel::def(navi::KernelDefBuilder &builder) const
{
    SchedulerBaseKernel::def(builder, KERNEL_ID);
}

bool RandomSchedulerKernel::schedule(NaviSchedScheduleContext &context) const
{
    const auto requestCount = context.requestCount();
    const auto nodeCount = context.nodeCount();
    for (size_t requestIndex = 0; requestIndex < requestCount; ++requestIndex) {
        auto nodeIndex = nextRandom() % nodeCount;
        if (!context.selectNode(requestIndex, nodeIndex, "random_selected")) {
            return false;
        }
    }
    return true;
}

uint64_t RandomSchedulerKernel::nextRandom() const
{
    _state ^= _state >> 12;
    _state ^= _state << 25;
    _state ^= _state >> 27;
    return _state * 0x2545f4914f6cdd1dULL;
}

REGISTER_KERNEL(RandomSchedulerKernel);

}
