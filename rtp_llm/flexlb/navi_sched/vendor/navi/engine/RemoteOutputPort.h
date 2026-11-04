#pragma once

#include "navi/engine/Port.h"

namespace navi {

class RemoteOutputPort : public Port
{
public:
    RemoteOutputPort(const EdgeDef &def,
                     SubGraphBorder *border);
    ~RemoteOutputPort();
private:
    RemoteOutputPort(const RemoteOutputPort &);
    RemoteOutputPort &operator=(const RemoteOutputPort &);
};

}
