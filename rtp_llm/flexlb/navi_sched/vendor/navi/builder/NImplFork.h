#pragma once

#include "navi/builder/NImplFake.h"

namespace navi {

class NImplFork : public NImplFake
{
public:
    NImplFork();
    ~NImplFork();
    NImplFork(const NImplFork &) = delete;
    NImplFork &operator=(const NImplFork &) = delete;
public:
    P in(const std::string &port) override;
    P out(const std::string &port) override;
private:
    std::vector<std::pair<P, E>> getInputUpstreams(const std::string &portName) const override;
private:
    NaviUnorderedMap<std::string, PImpl *> _ins;
    NaviUnorderedMap<std::string, PImpl *> _outs;
};

}

