
#pragma once
namespace CompileConfig {

#ifdef ENABLE_BF16
static constexpr bool enable_bf16 = true;
#else
static constexpr bool enable_bf16 = false;
#endif


#ifdef USE_WEIGHT_ONLY
static constexpr bool use_weight_only = true;
#else
static constexpr bool use_weight_only = false;
#endif

}  // namespace CompileConfig
