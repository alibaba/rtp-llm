#pragma once

#include <memory>

namespace navi {
class Navi;
}

namespace navi_sched {

class NaviSchedRole;
struct NaviSchedConfig;

class NaviSchedRoleRuntimeAccess {
public:
    static const std::shared_ptr<navi::Navi> &getNavi(const NaviSchedRole &role);
    static const NaviSchedConfig &getConfig(const NaviSchedRole &role);
};

}
