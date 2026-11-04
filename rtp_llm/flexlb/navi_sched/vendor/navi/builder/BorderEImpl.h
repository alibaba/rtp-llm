#pragma once

#include "navi/builder/EImpl.h"

namespace navi {

class BorderEdgeDef;

class BorderEImpl : public EImpl
{
public:
    BorderEImpl();
    ~BorderEImpl();
    BorderEImpl(const BorderEImpl &) = delete;
    BorderEImpl &operator=(const BorderEImpl &) = delete;
public:
    SingleE *addDownStream(P to, EImpl *realEdge) override;
    N split(const std::string &splitKernel) override;
    N merge(const std::string &mergeKernel) override;
    void backPressure(int64_t queueLow, int64_t queueHigh) override;
private:
    SingleE *addWithBorder(P to);
    SingleE *addWithoutBorder(P to);
    void addSplit();
    SingleE *addMerge(P to);
private:
    friend std::ostream &operator<<(std::ostream &os, const BorderEImpl &edge);
private:
    BorderEdgeDef *_fromBorder;
    BorderEdgeDef *_toBorder;
    BorderEdgeDef *_bpFromBorder;
    BorderEdgeDef *_bpToBorder;
};

}

