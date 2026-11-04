/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-24 10:47
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 */

#include "kmonitor/client/metric/DDSketch.h"

#include <limits.h>
#include <map>
#include <math.h>
#include <set>
#include <string>
#include <unordered_map>

#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/metric/DenseStore.h"
#include "kmonitor/client/metric/LogarithmicMapping.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);
using namespace std;

const size_t DDSketch::DDSKETCH_BATCH_SIZE = 100;

DDSketch::DDSketch(double relativeAccuracy, double minIndexedValue) : indexMapping_(relativeAccuracy) {
    minIndexedValue_ =
        (minIndexedValue > indexMapping_.minIndexableValue()) ? minIndexedValue : indexMapping_.minIndexableValue();
    maxIndexedValue_ = indexMapping_.maxIndexableValue();
    zeroCount_ = 0;
    indexCountMap_.reserve(DDSKETCH_BATCH_SIZE);
}

void DDSketch::accept(double value) {
    if (value < 0 || value > maxIndexedValue_) {
        return;
    }
    if (value < minIndexedValue_) {
        zeroCount_++;
    } else {
        indexCountMap_[indexMapping_.index(value)]++;
        if (indexCountMap_.size() >= DDSKETCH_BATCH_SIZE) {
            Flush();
        }
    }
}

void DDSketch::merge(const DDSketch &other) {
    if (indexMapping_ == other.indexMapping_) {
        store_.merge(other.store_);
        zeroCount_ += other.zeroCount_;
    }
}

double DDSketch::getValueAtQuantile(double quantile) { return getValueAtQuantile(quantile, getCount()); }

double DDSketch::getValueAtQuantile(double quantile, int64_t count) {
    if (!isfinite(quantile) || quantile < 0 || quantile > 1) {
        return 0;
    }

    if (count == 0) {
        return 0;
    }

    int64_t rank = 0;
    if (quantile >= 1) {
        rank = count - 1;
    } else if (quantile > 0) {
        const double rawRank = quantile * static_cast<double>(count - 1);
        rank = rawRank >= static_cast<double>(count - 1) ? count - 1 : static_cast<int64_t>(rawRank);
    }
    if (rank < zeroCount_) {
        return 0;
    }

    Bin bin;
    if (quantile <= 0.5) {
        auto binIterator = store_.getAscendingIterator();
        int64_t n = zeroCount_;
        do {
            bin = binIterator.next();
            n += bin.getCount();
        } while (n <= rank && binIterator.hasNext());

    } else {
        auto binIterator = store_.getDescendingIterator();
        int64_t n = count;
        do {
            bin = binIterator.next();
            n -= bin.getCount();
        } while (n > rank && binIterator.hasNext());
    }

    return indexMapping_.value(bin.getIndex());
}

int64_t DDSketch::getCount() {
    this->Flush();
    return zeroCount_ + store_.getTotalCount();
}

void DDSketch::Flush() {
    for (auto indexCount : indexCountMap_) {
        store_.add(indexCount.first, indexCount.second);
    }
    indexCountMap_.clear();
}

// use autil::legacy::ToJsonString(DDSketch, true) to get
void DDSketch::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("minIndexedValue", minIndexedValue_, minIndexedValue_);
    json.Jsonize("maxIndexedValue", maxIndexedValue_, maxIndexedValue_);
    json.Jsonize("zeroCount", zeroCount_, zeroCount_);
    int32_t arrayLengthGrowthIncrement = store_.getArrayLengthGrowthIncrement();
    json.Jsonize("arrayLengthGrowthIncrement", arrayLengthGrowthIncrement, arrayLengthGrowthIncrement);
    int32_t arrayLengthOverhead = store_.getArrayLengthOverhead();
    json.Jsonize("arrayLengthOverhead", arrayLengthOverhead, arrayLengthOverhead);
    int32_t offset = store_.getOffset();
    json.Jsonize("offset", offset, offset);
    int32_t minIndex = store_.getMinIndex();
    json.Jsonize("minIndex", minIndex, minIndex);
    int32_t maxIndex = store_.getMaxIndex();
    json.Jsonize("maxIndex", maxIndex, maxIndex);
    std::vector<int64_t> counts;
    store_.getCounts(counts);
    json.Jsonize("counts", counts, counts);
    double relativeAccuracy = indexMapping_.relativeAccuracy();
    json.Jsonize("relativeAccuracy", relativeAccuracy, relativeAccuracy);
}

void DDSketch::toProto(kmonitor::protocol::DDSketch *const ddSketchPB) {
    if (ddSketchPB == nullptr) {
        return;
    }

    this->Flush();

    kmonitor::protocol::IndexMapping *indexMappingPB = ddSketchPB->mutable_indexmapping();
    indexMapping_.toProto(indexMappingPB);

    kmonitor::protocol::DenseStore *denseStorePB = ddSketchPB->mutable_store()->mutable_densestore();
    store_.toProto(denseStorePB);

    ddSketchPB->set_minindexedvalue(minIndexedValue_);
    ddSketchPB->set_maxindexedvalue(maxIndexedValue_);
    ddSketchPB->set_zerocount(zeroCount_);
}

void DDSketch::serializeToString(std::string &str) {

    kmonitor::protocol::DDSketch ddSketchPB;

    this->toProto(&ddSketchPB);

    ddSketchPB.SerializeToString(&str);
}

bool DDSketch::fromProto(const kmonitor::protocol::DDSketch &ddSketchPB) {
    if (!ddSketchPB.has_indexmapping() || !ddSketchPB.has_store() || !ddSketchPB.store().has_densestore()) {
        return false;
    }
    if (ddSketchPB.indexmapping().interpolation() != kmonitor::protocol::IndexMapping::NONE) {
        return false;
    }

    const double relativeAccuracy = ddSketchPB.indexmapping().relativeaccuracy();
    if (relativeAccuracy <= 0 || relativeAccuracy >= 1 || !::isfinite(relativeAccuracy)) {
        return false;
    }
    if (ddSketchPB.zerocount() < 0 || ddSketchPB.zerocount() > INT_MAX || !::isfinite(ddSketchPB.minindexedvalue()) ||
        !::isfinite(ddSketchPB.maxindexedvalue()) || ddSketchPB.minindexedvalue() > ddSketchPB.maxindexedvalue()) {
        return false;
    }

    LogarithmicMapping newIndexMapping(relativeAccuracy);
    if (ddSketchPB.minindexedvalue() < newIndexMapping.minIndexableValue() ||
        ddSketchPB.maxindexedvalue() > newIndexMapping.maxIndexableValue()) {
        return false;
    }

    DenseStore newStore;
    if (!newStore.fromProto(ddSketchPB.store().densestore())) {
        return false;
    }
    if (newStore.getTotalCount() > LLONG_MAX - ddSketchPB.zerocount()) {
        return false;
    }

    indexMapping_ = newIndexMapping;
    store_.swap(newStore);
    indexCountMap_.clear();
    minIndexedValue_ = ddSketchPB.minindexedvalue();
    maxIndexedValue_ = ddSketchPB.maxindexedvalue();
    zeroCount_ = static_cast<int32_t>(ddSketchPB.zerocount());
    return true;
}

bool DDSketch::deserializeFromString(const std::string &str) {
    kmonitor::protocol::DDSketch ddSketchPB;
    if (!ddSketchPB.ParseFromString(str)) {
        return false;
    }
    return fromProto(ddSketchPB);
}

END_KMONITOR_NAMESPACE(kmonitor);
