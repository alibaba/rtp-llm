#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/xpu/RegisterXpuBaseBindings.hpp"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    torch_ext::registerBaseXpuBindings(rtp_ops_m);
}

}  // namespace rtp_llm
