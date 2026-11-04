#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {
struct HashNode {
    HashNode() : key(0), offset(0), label(0) {
    }
    uint64_t key;
    uint16_t offset;
    uint8_t label;
    bool operator<(const HashNode &rha) const;
} __attribute__((packed));

class ConsistentHash
{
public:
    typedef HashNode *Iterator;

public:
    ConsistentHash();
    ~ConsistentHash();

private:
    ConsistentHash(const ConsistentHash &);
    ConsistentHash &operator=(const ConsistentHash &);

public:
    void init(size_t rowCount, size_t virtualNodeCount);
    void add(size_t offset, uint64_t key);
    // you should call construct before use!
    void construct();

    Iterator get(uint64_t key) const;
    Iterator begin() const;
    Iterator end() const;

private:
    size_t _virtualNodeCount;
    size_t _hashCircleSize;
    HashNode *_hashCircle;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call

