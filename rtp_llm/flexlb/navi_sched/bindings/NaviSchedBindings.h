#pragma once

#include <pybind11/pybind11.h>

namespace navi_sched {

void bindNaviSched(pybind11::module_ &module);
void bindNaviSchedModule(pybind11::module_ &module, const char *description);

} // namespace navi_sched
