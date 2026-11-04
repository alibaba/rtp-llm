#pragma once

#include "navi/engine/NaviUserResult.h"

namespace navi {

class NaviFuncClosure : public NaviUserResultClosure {
public:
    NaviFuncClosure(std::function<void(NaviUserResultPtr)> func);
public:
    void run(NaviUserResultPtr userResult,
             const std::shared_ptr<ComputeScope> &computeScope) override;

private:
    std::function<void(NaviUserResultPtr)> _func;
};

}
