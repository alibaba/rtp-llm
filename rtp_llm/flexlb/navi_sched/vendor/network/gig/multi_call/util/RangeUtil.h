#pragma once

#include "autil/Log.h"

namespace multi_call {
struct Range {
public:
    Range(uint32_t from, uint32_t to) : _from(from), _to(to) {
    }

public:
    uint32_t _from;
    uint32_t _to;
};

class RangeUtil
{
public:
    RangeUtil(uint32_t partCount, uint32_t rangeFrom = 0, uint32_t rangeTo = 65535);
    ~RangeUtil();

private:
    RangeUtil(const RangeUtil &);
    RangeUtil &operator=(const RangeUtil &);

public:
    static std::vector<Range> splitRange(uint32_t partitionCount, uint32_t rangeFrom = 0,
                                         uint32_t rangeTo = 65535);

public:
    const std::vector<Range> &getRanges();
    bool getRange(size_t pos, Range &range);
    int32_t getPartId(uint32_t hashId);

private:
    std::vector<Range> _ranges;
};

} // namespace multi_call

