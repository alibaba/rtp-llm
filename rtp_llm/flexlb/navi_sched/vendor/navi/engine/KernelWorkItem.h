#pragma once

#include "navi/common.h"
#include "navi/engine/NaviWorkerBase.h"

namespace navi {

class Node;

class KernelWorkItem : public NaviWorkerItem
{
public:
    KernelWorkItem(NaviWorkerBase *worker, const CollectFlags &collectFlags, Node *node);
    ~KernelWorkItem();
public:
    void process() override;
    void destroy() override;
    void drop() override;
    void doProcess(NaviPerfThread *perfThread) override {
        assert(false);
    }
private:
    Node *_node;
};

}
