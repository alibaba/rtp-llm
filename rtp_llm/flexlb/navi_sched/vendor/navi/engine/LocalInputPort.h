#pragma once

#include "navi/engine/LocalPortBase.h"

namespace navi {

struct BPInputState {
    atomic64_t updateCount;
    volatile int64_t notifyMaxCount = -1;
};

class BPInputStates {
public:
    BPInputStates();
    ~BPInputStates();
public:
    void init(NaviPartId usedCount, bool localMode);
    int64_t updateMaxCount(NaviPartId index, int64_t totalDataCount, int64_t queueSize, const BackPressureQueueDef &queueDef);
    NaviPartId size() const {
        return _states.size();
    }
    void fillState(const PartInfo &partInfo, KernelInputPortStateDef *state) const;
private:
    bool _localMode = false;
    std::vector<BPInputState> _states;
};

class LocalOutputPort;

class LocalInputPort : public LocalPortBase
{
public:
    LocalInputPort(const EdgeDef &def, SubGraphBorder *border);
    ~LocalInputPort();
private:
    LocalInputPort(const LocalInputPort &);
    LocalInputPort &operator=(const LocalInputPort &);
public:
    void setCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) override;
    void getCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) override;
    void fillOutputState(KernelInputPortStateDef *state) const override;
private:
    bool setBPPort(Port *port) override;
    bool setBPNotifyPort(Port *port) override;
private:
    void doCallBack(bool isSet, NaviPartId partId, bool eof);
    void sendBackPressure(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize);
    void notifyBP(NaviPartId partId);
private:
    Port *_bpPort = nullptr;
    LocalOutputPort *_bpNotifyPort = nullptr;
    BPInputStates *_bpInputStates = nullptr;
};

}
