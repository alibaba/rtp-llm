#pragma once

#include "navi/engine/PortDataQueue.h"

namespace navi {

class NormalPortQueue : public PortDataQueues
{
public:
    NormalPortQueue();
    ~NormalPortQueue();
private:
    NormalPortQueue(const NormalPortQueue &);
    NormalPortQueue &operator=(const NormalPortQueue &);
public:
    ErrorCode setEof(NaviPartId fromPartId) override;
    ErrorCode push(NaviPartId toPartId, NaviPartId fromPartId,
                   const DataPtr &data, bool eof, bool &retEof) override;
    ErrorCode pop(NaviPartId toPartId, DataPtr &data, bool &eof) override;
};

}
