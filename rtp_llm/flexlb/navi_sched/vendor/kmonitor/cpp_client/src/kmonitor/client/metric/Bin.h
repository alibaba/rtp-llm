#pragma once

#include <cstdint>

#include "kmonitor/client/common/Common.h"
BEGIN_KMONITOR_NAMESPACE(kmonitor);

class Bin {
public:
    Bin() = default;
    ~Bin() = default;

    Bin(int32_t index, int64_t count) : index_(index), count_(count) {}

    int32_t getIndex() const { return index_; }

    int64_t getCount() const { return count_; }

private:
    int32_t index_;
    int64_t count_;
};

class BinIterator {
public:
    BinIterator(int64_t *counts, int32_t offset, int32_t maxIndex, int32_t minIndex)
        : counts_(counts), offset_(offset), maxIndex_(maxIndex), minIndex_(minIndex) {}

    bool hasNext();
    Bin next();

protected:
    int64_t *counts_;
    int32_t offset_;
    int32_t maxIndex_;
    int32_t minIndex_;
    int32_t index_;
};

class BinAscendIterator : BinIterator {
public:
    BinAscendIterator(int64_t *counts, int32_t offset, int32_t maxIndex, int32_t minIndex)
        : BinIterator(counts, offset, maxIndex, minIndex) {
        index_ = minIndex_;
    };

    bool hasNext() { return index_ <= maxIndex_; }

    Bin next() {
        Bin bin(index_, counts_[index_ - offset_]);
        index_++;
        return bin;
    }
};

class BinDescendIterator : BinIterator {
public:
    BinDescendIterator(int64_t *counts, int32_t offset, int32_t maxIndex, int32_t minIndex)
        : BinIterator(counts, offset, maxIndex, minIndex) {
        index_ = maxIndex_;
    };

    bool hasNext() { return index_ >= minIndex_; }

    Bin next() {
        Bin bin(index_, counts_[index_ - offset_]);
        index_--;
        return bin;
    }
};

END_KMONITOR_NAMESPACE(kmonitor);
