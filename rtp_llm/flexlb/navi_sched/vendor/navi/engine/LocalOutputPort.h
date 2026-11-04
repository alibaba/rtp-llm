#pragma once

#include "navi/engine/LocalPortBase.h"

namespace navi {

struct BPOutputState {
    atomic64_t updateCount;
    volatile int64_t dataCount = 0;
    volatile int64_t maxDataCount = -1;
    volatile bool eof = false;
};

class BPOutputStates {
public:
    BPOutputStates();
    ~BPOutputStates();
private:
    BPOutputStates(const BPOutputStates &);
    BPOutputStates &operator=(const BPOutputStates &);
public:
    void init(NaviPartId usedCount, bool localMode, const BackPressureQueueDef &queueDef, bool backPressureEofStop);
    NaviPartId size() const {
        return _states.size();
    }
    void updateTotalCount(NaviPartId index, int64_t totalCount);
    void updateMaxDataCount(NaviPartId index, int64_t maxDataId, bool eof);
    bool isReady(NaviPartId index) const;
    bool isFinish(NaviPartId index) const;
    int64_t getTotalDataCount(NaviPartId index) const;
    int64_t getMaxDataCount(NaviPartId index) const;
    void fillState(const PartInfo &partInfo, KernelOutputPortStateDef *state) const;
private:
    bool _backPressureEofStop = false;
    std::vector<BPOutputState> _states;
};

class LocalOutputPort : public LocalPortBase
{
public:
    LocalOutputPort(const EdgeDef &def, SubGraphBorder *border);
    ~LocalOutputPort();
private:
    LocalOutputPort(const LocalOutputPort &);
    LocalOutputPort &operator=(const LocalOutputPort &);
public:
    void setCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) override;
    void getCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) override;
    void fillInputState(KernelOutputPortStateDef *state) const override;
public:
    void initBPNotify();
    void receiveBackPressure(NaviPartId partId, const DataPtr &data, bool eof);
private:
    BPOutputStates *_bpOutputStates = nullptr;
};

}
