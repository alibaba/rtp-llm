#pragma once

#include "navi/engine/Data.h"

namespace navi {

class ControlData : public Data {
public:
    ControlData();
    ~ControlData();
private:
    ControlData(const ControlData &);
    ControlData &operator=(const ControlData &);
};

}
