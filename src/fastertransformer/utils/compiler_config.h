
#pragma once
namespace CompileConfig {

#ifdef ENABLE_BF16
static constexpr bool enable_bf16 = true;
#else
static constexpr bool enable_bf16 = false;
#endif

}  // namespace CompileConfig
