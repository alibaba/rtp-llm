#pragma once

#include "navi/common.h"
#include "navi/proto/GraphVis.pb.h"

namespace navi {

class StackTable {
public:
    StackTable(StackTableDef *stackTableDef);
public:
    void setStackDef(StackTableDef *def,
                     const std::vector<std::pair<uint64_t, uint32_t>> &addrs);
    std::vector<std::pair<uint64_t, uint32_t>> toAddrVec();
    void addStack(uint64_t hash, int32_t count, uint64_t *addrs);
    uint64_t addStack(int32_t count, uint64_t *addrs);
    static uint64_t getStackHash(int32_t count, uint64_t *addrs);
public:
    const NaviUnorderedMap<uint64_t, uint32_t> &getAddrMap() const {
        return _addrMap;
    }
private:
    uint32_t getAddrId(uint64_t ip);
private:
    StackTableDef *_stackTableDef = nullptr;
    NaviUnorderedMap<uint64_t, uint32_t> _addrMap;
};

}
