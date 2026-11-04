#include "navi/engine/KernelWorkItem.h"
#include "navi/engine/Node.h"

namespace navi {

KernelWorkItem::KernelWorkItem(NaviWorkerBase *worker, const CollectFlags &collectFlags, Node *node)
    : NaviWorkerItem(worker, collectFlags)
    , _node(node)
{
}

KernelWorkItem::~KernelWorkItem() {
}

void KernelWorkItem::process() {
    if (!_worker->isStopped() || _collectFlags.collectMalloc()) {
        PerfEnableScope perfEnableScope(_worker->getNaviPerf(), _collectFlags);
        {
            NaviLoggerScope scope(_worker->getLogger());
            auto perfThread = perfEnableScope.getPerfThread();
            auto computeScope = _node->makeComputeScope(_schedInfo, perfThread);
            _node->compute(computeScope);
        }
    } else if (!_worker->isFinished()) {
        _worker->drop();
    }
}

void KernelWorkItem::destroy() {
    auto worker = _worker;
    worker->dec();
}

void KernelWorkItem::drop() {
}

}

