#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <stdexcept>

namespace rtp_llm {

void execAllGather(const AllGatherParams&) {
    throw std::logic_error("single-rank monitor tests must not call execAllGather");
}

}  // namespace rtp_llm
