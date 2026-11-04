#pragma once

#include "navi/engine/Port.h"

namespace navi {

class ReadyBitMap;
class Node;

class LocalPortBase : public Port
{
public:
    LocalPortBase(const NodePortDef &nodePort,
                  const NodePortDef &peerPort,
                  IoType ioType,
                  SubGraphBorder *border);
    ~LocalPortBase();
private:
    LocalPortBase(const LocalPortBase &);
    LocalPortBase &operator=(const LocalPortBase &);
public:
    bool isOk() const {
        return _nodeReadyMap && _nodeReadyMap.isOk();
    }
    bool isFinish() const {
        return _nodeReadyMap && _nodeReadyMap.isFinish();
    }
    bool isFinish(IndexType index) const {
        return _nodeReadyMap && _nodeReadyMap.isFinish(index);
    }
    bool hasReady() const {
        return _nodeReadyMap && _nodeReadyMap.hasReady();
    }
    void setNode(Node *node, const InlineReadyBitMap &readyBitMap);
    virtual bool setBPPort(Port *port) {
        return true;
    }
    virtual bool setBPNotifyPort(Port *port) {
        return true;
    }
protected:
    void scheduleNode() const;
protected:
    autil::ThreadMutex _lock;
    Node *_node;
    InlineReadyBitMap _nodeReadyMap;
};

}
