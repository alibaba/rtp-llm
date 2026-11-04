#pragma once

#include "navi/common.h"
#include "navi/engine/Edge.h"
#include "navi/proto/KernelDef.pb.h"

namespace navi {

class ReadyBitMap;
class Node;
class PartInfo;
class LocalPortBase;

struct InputPortGroupInternal {
public:
    InputPortGroupInternal(InputTypeDef channelType, InputTypeDef groupType);
    ~InputPortGroupInternal() {}
public:
    IndexType addInput(IndexType index, EdgeOutputInfo &info);
    InputTypeDef getGroupType() const {
        return _groupType;
    }
    const NaviVector<std::pair<IndexType, EdgeOutputInfo>, 4> &getInputVec() const {
        return _inputs;
    }
    void setValidPortGroup(bool valid) {
        _validPortGroup = valid;
    }
    bool isValid() const {
        return !_inputs.empty() || _validPortGroup;
    }
    InlineReadyBitMap createReadyBitMap(const PartInfo &partInfo);
    bool postInit();
    IndexType getInputIndexStart() const;
    void setInputIndexStart(IndexType start);
    size_t getInputCount() const;
    EdgeOutputInfo getEdgeOutputInfo(IndexType index) const;
    void startUpStreamSchedule(LocalSubGraph *graph,
                               const Node *callNode,
                               bool graphStart) const;
    bool startUpStreamSchedule(LocalSubGraph *graph,
                               IndexType index,
                               const Node *callNode,
                               bool graphStart) const;
    void setEof(LocalSubGraph *graph,
                const Node *callNode) const;
    void setEof(LocalSubGraph *graph,
                IndexType index,
                const Node *callNode) const;
    bool isOk(LocalPortBase *port, LocalSubGraph *graph) const;
    bool isEofOrFinish(LocalPortBase *port, LocalSubGraph *graph, IndexType index) const;
    bool isEofOrFinish(LocalPortBase *port, LocalSubGraph *graph) const;
    bool hasReady(LocalPortBase *port, LocalSubGraph *graph) const;
    void finishOutputWork(LocalSubGraph *graph, const Node *callNode);
    void fillState(LocalPortBase *port,
                   LocalSubGraph *graph,
                   KernelInputGroupStateDef *state) const;

    static bool compare(const std::pair<IndexType, EdgeOutputInfo> &left,
                        const std::pair<IndexType, EdgeOutputInfo> &right);
private:
    InputTypeDef _type;
    bool _autoFillIndex;
    bool _validPortGroup;
    InputTypeDef _groupType;
    NaviVector<std::pair<IndexType, EdgeOutputInfo>, 4> _inputs;
    IndexType _inputIndexStart;
};

extern std::ostream &operator<<(std::ostream &os, const InputPortGroupInternal &inputGroup);
extern std::ostream &operator<<(std::ostream &os, const NaviVector<InputPortGroupInternal *, 1> &x);
}
