#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {

int         StaticConfig::user_deep_gemm_num_sm          = -1;
bool        StaticConfig::user_arm_gemm_use_kai          = false;
bool        StaticConfig::user_ft_core_dump_on_exception = false;
bool        StaticConfig::user_disable_pdl               = true;
std::string StaticConfig::user_torch_cuda_profiler_dir   = "";
bool        StaticConfig::use_aiter_pa                   = true;

}  // namespace rtp_llm
