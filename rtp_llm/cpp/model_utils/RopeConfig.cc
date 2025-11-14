#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include <sstream>

namespace rtp_llm {

std::string RopeConfig::DebugRopeConfigStr() const {
    std::ostringstream oss;
    oss << "  style: " << static_cast<int>(style) << std::endl;
    oss << "  dim: " << dim << std::endl;
    oss << "  base: " << base << std::endl;
    oss << "  scale: " << scale << std::endl;
    oss << "  factor1: " << factor1 << std::endl;
    oss << "  factor2: " << factor2 << std::endl;
    oss << "  max_pos: " << max_pos << std::endl;
    oss << "  extrapolation_factor: " << extrapolation_factor << std::endl;
    oss << "  mscale: " << mscale << std::endl;
    oss << "  offset: " << offset << std::endl;
    oss << "  index_factor: " << index_factor << std::endl;
    oss << "  mrope_dim1: " << mrope_dim1 << std::endl;
    oss << "  mrope_dim2: " << mrope_dim2 << std::endl;
    oss << "  mrope_dim3: " << mrope_dim3 << std::endl;
    return oss.str();
}

}  // namespace rtp_llm

