#pragma once
#include <string>

namespace rtp_llm {

// these configs are used in static or global method.
struct StaticConfig {
    static int         user_deep_gemm_num_sm;
    static bool        user_arm_gemm_use_kai;
    static bool        user_ft_core_dump_on_exception;
    static bool        user_disable_pdl;
    static bool        use_aiter_pa;
    static std::string user_torch_cuda_profiler_dir;
};

}  // namespace rtp_llm
