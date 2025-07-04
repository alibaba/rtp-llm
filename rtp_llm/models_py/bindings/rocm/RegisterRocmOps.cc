#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/rocm/RegisterBaseBindings.hpp"

using namespace rtp_llm;

namespace torch_ext {

void registerPyModuleOps(py::module &rtp_ops_m) {
    registerBaseRocmBindings(rtp_ops_m);
}

}
