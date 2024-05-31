
#pragma once
namespace CompileConfig {

#ifdef ENABLE_BF16
static constexpr bool enable_bf16 = true;
#else
static constexpr bool enable_bf16 = false;
#endif

#ifdef CUDART_VERSION
static constexpr int cudart_version = CUDART_VERSION;
#else
static constexpr int cudart_version = 0;
#endif

#ifdef USE_OLD_TRT_FMHA
static constexpr bool use_old_trt_fmha = true;
#else
static constexpr bool use_old_trt_fmha = false;
#endif

}  // namespace CompileConfig
