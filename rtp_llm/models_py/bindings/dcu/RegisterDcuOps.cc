#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/dcu/RegisterBaseBindings.hpp"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    registerBaseDcuBindings(rtp_ops_m);
}

}
