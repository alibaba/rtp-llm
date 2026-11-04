#include "rtp_llm/flexlb/navi_sched/bindings/NaviSchedBindings.h"

PYBIND11_MODULE(_navi_sched_core, module)
{
    navi_sched::bindNaviSchedModule(module, "navi_sched");
}
