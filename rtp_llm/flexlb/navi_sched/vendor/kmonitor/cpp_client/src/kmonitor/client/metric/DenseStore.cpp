/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2020-01-03 10:47
 * Author Name: yixuan.ly
 * Author Email: yixuan.ly@alibaba-inc.com
 */

#include "kmonitor/client/metric/DenseStore.h"

#include <limits.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/metric/Bin.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

namespace {
const int32_t MAX_DESERIALIZED_COUNTS_SIZE = 100000;
}

DenseStore::DenseStore(int32_t arrayLengthGrowthIncrement, int32_t arrayLengthOverhead)
    : arrayLengthGrowthIncrement_(arrayLengthGrowthIncrement), arrayLengthOverhead_(arrayLengthOverhead) {
    counts_ = NULL;
    countsLength_ = 0;
    offset_ = 0;
    minIndex_ = INT_MAX; // for empty
    maxIndex_ = INT_MIN;
}

DenseStore::~DenseStore() { delete[] counts_; }

void DenseStore::add(int32_t index, int64_t count) {
    if (count <= 0) {
        return;
    }

    int32_t arrayIndex = normalize(index);
    counts_[arrayIndex] += count;
}

void DenseStore::merge(const DenseStore &other) {
    if (other.isEmpty()) {
        return;
    }

    if (other.minIndex_ < minIndex_ || other.maxIndex_ > maxIndex_) {
        extendRange(other.minIndex_, other.maxIndex_);
    }

    for (int index = other.minIndex_; index <= other.maxIndex_; index++) {
        counts_[index - offset_] += other.counts_[index - other.offset_];
    }
}

int64_t DenseStore::getTotalCount(int32_t fromIndex, int32_t toIndex) const {
    if (isEmpty()) {
        return 0;
    }

    int32_t fromArrayIndex = (fromIndex > offset_) ? (fromIndex - offset_) : 0;
    int32_t toArrayIndex = ((toIndex - offset_) < (countsLength_ - 1)) ? (toIndex - offset_) : (countsLength_ - 1);

    int64_t totalCount = 0;
    for (int32_t arrayIndex = fromArrayIndex; arrayIndex <= toArrayIndex; arrayIndex++) {
        totalCount += (int64_t)counts_[arrayIndex];
    }

    return totalCount;
}

void DenseStore::extendRange(int32_t newMinIndex, int32_t newMaxIndex) {
    newMinIndex = (newMinIndex < minIndex_) ? newMinIndex : minIndex_;
    newMaxIndex = (newMaxIndex > maxIndex_) ? newMaxIndex : maxIndex_;
    if (isEmpty()) {
        countsLength_ = getNewLength(newMaxIndex - newMinIndex + 1);
        counts_ = new int64_t[countsLength_](); // add () to init 0 in counts_
        offset_ = newMinIndex;
        minIndex_ = newMinIndex;

        // java就是取的newMinIndex，而且因为newMinIndex==newMaxIndex，所以暂保持一致
        maxIndex_ = newMinIndex;
        adjust(newMinIndex, newMaxIndex);
    } else if ((newMinIndex >= offset_ && newMaxIndex < offset_ + countsLength_)) {
        minIndex_ = newMinIndex;
        maxIndex_ = newMaxIndex;
    } else {
        int32_t desiredLength = newMaxIndex - newMinIndex + 1;
        int32_t newLength = getNewLength(desiredLength);
        if (newLength > countsLength_) {
            int64_t *newCounts = new int64_t[newLength]();
            memcpy((void *)newCounts, (void *)counts_, countsLength_ * sizeof(int64_t));
            countsLength_ = newLength;
            delete[] counts_;
            counts_ = newCounts;
        }

        adjust(newMinIndex, newMaxIndex);
    }
}

void DenseStore::centerCounts(int32_t newMinIndex, int32_t newMaxIndex) {
    int32_t middleIndex = newMinIndex + (newMaxIndex - newMinIndex + 1) / 2;
    shiftCounts(offset_ + countsLength_ / 2 - middleIndex);

    minIndex_ = newMinIndex;
    maxIndex_ = newMaxIndex;
}

void DenseStore::getCounts(std::vector<int64_t> &vec) { vec.assign(counts_, counts_ + countsLength_); }

void DenseStore::shiftCounts(int32_t shift) {
    int32_t minArrayIndex = minIndex_ - offset_;
    int32_t maxArrayIndex = maxIndex_ - offset_;

    memmove((void *)(counts_ + (minArrayIndex + shift)),
            (void *)(counts_ + minArrayIndex),
            (maxArrayIndex - minArrayIndex + 1) * sizeof(int64_t));

    if (shift > 0) {
        memset((void *)(counts_ + minArrayIndex), 0, shift * sizeof(int64_t));
    } else {
        memset((void *)(counts_ + (maxArrayIndex + shift + 1)), 0, (-shift) * sizeof(int64_t));
    }

    // 初始及每次新申请空间时offset_ = 0 放的index即为midindex_value，这样就尽可能多存数据
    //  offset_ - (offset_ + countsLength_ / 2 - middleIndex) = middleIndex - countsLength_ / 2
    offset_ -= shift;
}

int32_t DenseStore::normalize(int32_t index) {
    if (index < minIndex_ || index > maxIndex_) {
        extendRange(index);
    }

    return index - offset_;
}

void DenseStore::adjust(int32_t newMinIndex, int32_t newMaxIndex) { centerCounts(newMinIndex, newMaxIndex); }

void DenseStore::toProto(kmonitor::protocol::DenseStore *const denseStore) {
    if (denseStore == nullptr) {
        return;
    }

    denseStore->set_arraylengthgrowthincrement(arrayLengthGrowthIncrement_);
    denseStore->set_arraylengthoverhead(arrayLengthOverhead_);
    denseStore->set_offset(offset_);
    denseStore->set_minindex(minIndex_);
    denseStore->set_maxindex(maxIndex_);
    denseStore->set_densestoretype(kmonitor::protocol::DenseStore::UNBOUNDED_SIZE);

    auto countsField = denseStore->mutable_counts();
    countsField->Clear();                // 清除现有数据
    countsField->Reserve(countsLength_); // 预先分配足够的空间
    for (int i = 0; i < countsLength_; i++) {
        countsField->Add(counts_[i]);
    }
}

bool DenseStore::fromProto(const kmonitor::protocol::DenseStore &denseStore) {
    if (denseStore.densestoretype() != kmonitor::protocol::DenseStore::UNBOUNDED_SIZE) {
        return false;
    }

    const int countsSize = denseStore.counts_size();
    if (countsSize > MAX_DESERIALIZED_COUNTS_SIZE) {
        return false;
    }
    if (denseStore.arraylengthgrowthincrement() <= 0 || denseStore.arraylengthoverhead() < 0) {
        return false;
    }

    const int32_t minIndex = denseStore.minindex();
    const int32_t maxIndex = denseStore.maxindex();
    const int32_t offset = denseStore.offset();
    const bool emptyStore = countsSize == 0 && maxIndex < minIndex;
    if (countsSize == 0 && !emptyStore) {
        return false;
    }
    int64_t minArrayIndex = 0;
    int64_t maxArrayIndex = -1;
    if (countsSize > 0) {
        if (minIndex > maxIndex) {
            return false;
        }
        minArrayIndex = static_cast<int64_t>(minIndex) - offset;
        maxArrayIndex = static_cast<int64_t>(maxIndex) - offset;
        if (minArrayIndex < 0 || maxArrayIndex < minArrayIndex || maxArrayIndex >= countsSize) {
            return false;
        }
    }

    int64_t *newCounts = nullptr;
    if (countsSize > 0) {
        newCounts = new int64_t[countsSize]();
        int64_t totalCount = 0;
        for (int i = 0; i < countsSize; ++i) {
            const int64_t count = denseStore.counts(i);
            if (count < 0 || count > LLONG_MAX - totalCount) {
                delete[] newCounts;
                return false;
            }
            if (count > 0 && (i < minArrayIndex || i > maxArrayIndex)) {
                delete[] newCounts;
                return false;
            }
            totalCount += count;
            newCounts[i] = count;
        }
    }

    delete[] counts_;
    counts_ = newCounts;
    countsLength_ = countsSize;
    arrayLengthGrowthIncrement_ = denseStore.arraylengthgrowthincrement();
    arrayLengthOverhead_ = denseStore.arraylengthoverhead();
    offset_ = offset;
    minIndex_ = minIndex;
    maxIndex_ = maxIndex;
    return true;
}

void DenseStore::swap(DenseStore &other) noexcept {
    using std::swap;
    swap(arrayLengthGrowthIncrement_, other.arrayLengthGrowthIncrement_);
    swap(arrayLengthOverhead_, other.arrayLengthOverhead_);
    swap(counts_, other.counts_);
    swap(countsLength_, other.countsLength_);
    swap(offset_, other.offset_);
    swap(minIndex_, other.minIndex_);
    swap(maxIndex_, other.maxIndex_);
}

BinAscendIterator DenseStore::getAscendingIterator() {
    return BinAscendIterator(counts_, offset_, maxIndex_, minIndex_);
}

BinDescendIterator DenseStore::getDescendingIterator() {
    return BinDescendIterator(counts_, offset_, maxIndex_, minIndex_);
}

END_KMONITOR_NAMESPACE(kmonitor);
