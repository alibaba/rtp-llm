#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/rocm/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/rocm/RegisterAttnOpBindings.hpp"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    registerBaseRocmBindings(rtp_ops_m);
    registerAttnOpBindings(rtp_ops_m);
}

}
