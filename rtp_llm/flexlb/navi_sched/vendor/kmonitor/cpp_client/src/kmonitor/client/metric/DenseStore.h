#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2020-01-03 10:47
 * Author Name: yixuan.ly
 * Author Email: yixuan.ly@alibaba-inc.com
 */

#include <limits.h>
#include <string>
#include <vector>

#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/metric/Bin.h"
#include "kmonitor/protocol/DDSketchPB.pb.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);
static const int32_t DEFAULT_ARRAY_LENGTH_GROWTH_INCREMENT = 64;
static const double DEFAULT_ARRAY_LENGTH_OVERHEAD_RATIO = 0.1;

class DenseStore {
public:
    DenseStore(int32_t arrayLengthGrowthIncrement = DEFAULT_ARRAY_LENGTH_GROWTH_INCREMENT,
               int32_t arrayLengthOverhead = int32_t(DEFAULT_ARRAY_LENGTH_GROWTH_INCREMENT *
                                                     DEFAULT_ARRAY_LENGTH_OVERHEAD_RATIO));
    ~DenseStore();

    void add(int32_t index, int64_t count = 1);
    void merge(const DenseStore &other);
    int32_t normalize(int32_t index);
    int64_t getTotalCount(int32_t fromIndex, int32_t toIndex) const;
    void extendRange(int32_t newMinIndex, int32_t newMaxIndex);
    void centerCounts(int32_t newMinIndex, int32_t newMaxIndex);
    void shiftCounts(int32_t shift);
    void getCounts(std::vector<int64_t> &vec);
    void adjust(int newMinIndex, int newMaxIndex);

    int32_t getMinIndex() {
        if (isEmpty()) {
            return -1;
        }

        return minIndex_;
    }

    int32_t getMaxIndex() {
        if (isEmpty()) {
            return -1;
        }

        return maxIndex_;
    }

    int32_t getArrayLengthGrowthIncrement() { return arrayLengthGrowthIncrement_; }

    int32_t getArrayLengthOverhead() { return arrayLengthOverhead_; }

    int32_t getOffset() { return offset_; }

    bool isEmpty() const { return maxIndex_ < minIndex_; }

    int32_t getNewLength(int32_t desiredLength) {
        return ((desiredLength + arrayLengthOverhead_ - 1) / arrayLengthGrowthIncrement_ + 1) *
               arrayLengthGrowthIncrement_;
    }

    int64_t getTotalCount() const { return getTotalCount(minIndex_, maxIndex_); }

    void extendRange(int32_t index) { extendRange(index, index); }

    void toProto(kmonitor::protocol::DenseStore *const denseStore);

    bool fromProto(const kmonitor::protocol::DenseStore &denseStore);

    void swap(DenseStore &other) noexcept;

    BinAscendIterator getAscendingIterator();

    BinDescendIterator getDescendingIterator();

protected:
    int32_t arrayLengthGrowthIncrement_;
    int32_t arrayLengthOverhead_;
    int64_t *counts_;
    int32_t countsLength_;
    int32_t offset_;
    int32_t minIndex_;
    int32_t maxIndex_;
};

END_KMONITOR_NAMESPACE(kmonitor);
