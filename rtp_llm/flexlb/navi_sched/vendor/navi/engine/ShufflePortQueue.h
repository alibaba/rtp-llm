#pragma once

#include "navi/engine/PortDataQueue.h"

namespace navi {

class ShufflePortQueue : public PortDataQueues
{
public:
    ShufflePortQueue();
    ~ShufflePortQueue();
private:
    ShufflePortQueue(const ShufflePortQueue &);
    ShufflePortQueue &operator=(const ShufflePortQueue &);
public:
    ErrorCode setEof(NaviPartId fromPartId) override;
    ErrorCode push(NaviPartId toPartId, NaviPartId fromPartId,
                   const DataPtr &data, bool eof, bool &retEof) override;
    ErrorCode pop(NaviPartId toPartId, DataPtr &data, bool &eof) override;
};
}
