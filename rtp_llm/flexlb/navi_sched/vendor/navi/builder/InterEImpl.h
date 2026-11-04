#pragma once

#include "navi/builder/EImpl.h"

namespace navi {

class InterEImpl : public EImpl
{
public:
    InterEImpl();
    ~InterEImpl();
    InterEImpl(const InterEImpl &) = delete;
    InterEImpl &operator=(const InterEImpl &) = delete;
public:
    SingleE *addDownStream(P to, EImpl *realEdge) override;
    N split(const std::string &splitKernel) override;
    N merge(const std::string &mergeKernel) override;
private:
    void fillScopeBorder(P to) const;
private:
    static uint64_t getEdgeKey(P to) {
        return (((uint64_t)to._index) << 48) | ((uint64_t)to._impl);
    }
private:
    NaviUnorderedMap<uint64_t, SingleE *> _edgeMap;
};

}

