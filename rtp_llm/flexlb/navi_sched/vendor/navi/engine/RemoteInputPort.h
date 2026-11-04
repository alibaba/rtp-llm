#pragma once

#include "navi/engine/Port.h"

namespace navi {

class GraphDomain;

class RemoteInputPort : public Port
{
public:
    RemoteInputPort(const EdgeDef &def,
                    SubGraphBorder *border);
    ~RemoteInputPort();
private:
    RemoteInputPort(const RemoteInputPort &);
    RemoteInputPort &operator=(const RemoteInputPort &);
public:
    void setCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) override;
};

}
