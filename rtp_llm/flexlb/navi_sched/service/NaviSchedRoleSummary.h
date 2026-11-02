#pragma once

#include "autil/legacy/jsonizable.h"
#include "navi/common.h"
#include <cstdint>
#include <string>

namespace navi_sched {

struct NaviSchedRoleSummary : public autil::legacy::Jsonizable {
public:
    void Jsonize(JsonWrapper &json) override;
    std::string format() const;

public:
    std::string name;
    std::string role;
    bool started = false;
    int32_t grpcPort = -1;
};

NAVI_TYPEDEF_PTR(NaviSchedRoleSummary);

}
