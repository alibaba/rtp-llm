#include "navi/engine/StackTable.h"
#include "navi/engine/CollectFlags.h"
#include "autil/HashAlgorithm.h"

namespace navi {

StackTable::StackTable(StackTableDef *stackTableDef)
    : _stackTableDef(stackTableDef)
{
}

void StackTable::addStack(uint64_t hash, int32_t count, uint64_t *addrs) {
    auto stackMap = _stackTableDef->mutable_stacks();
    auto it = stackMap->find(hash);
    if (stackMap->end() == it) {
        auto stackDef = &(*stackMap)[hash];
        stackDef->set_stack_hash(hash);
        for (int32_t i = 0; i < count; i++) {
            stackDef->add_addr(getAddrId(addrs[i]));
        }
    }
}

uint64_t StackTable::addStack(int32_t count, uint64_t *addrs) {
    auto hash = getStackHash(count, addrs);
    addStack(hash, count, addrs);
    return hash;
}

uint64_t StackTable::getStackHash(int32_t count, uint64_t *addrs) {
    if (count == 0) {
        return 0;
    }
    return autil::HashAlgorithm::hashString64((char *)addrs, count * 8);
}

uint32_t StackTable::getAddrId(uint64_t ip) {
    uint32_t index = 0;
    auto it = _addrMap.find(ip);
    if (_addrMap.end() == it) {
        index = _addrMap.size();
        _addrMap.emplace(ip, index);
    } else {
        index = it->second;
    }
    return index;
}

void StackTable::setStackDef(
    StackTableDef *def,
    const std::vector<std::pair<uint64_t, uint32_t>> &addrs) {
    _stackTableDef = def;
    for (const auto &pair : addrs) {
        _addrMap.emplace(pair);
    }
}

std::vector<std::pair<uint64_t, uint32_t>> StackTable::toAddrVec() {
    std::vector<std::pair<uint64_t, uint32_t>> addrs;
    addrs.reserve(_addrMap.size());
    for (const auto &pair : _addrMap) {
        addrs.emplace_back(std::make_pair(pair.first, pair.second));
    }
    _addrMap.clear();
    return addrs;
}

}
