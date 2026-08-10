#pragma once

#include <ATen/Generator.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <cstdint>
#include <string>

namespace rtp_llm {

inline constexpr uint32_t kLegacySamplerGeneratorStateVersion  = 0;
inline constexpr uint32_t kCurrentSamplerGeneratorStateVersion = 1;

absl::StatusOr<std::string> captureSamplerGeneratorState(bool has_explicit_seed, const at::Generator& generator);

absl::Status restoreSamplerGeneratorState(uint32_t            wire_version,
                                          bool                has_explicit_seed,
                                          at::Generator       generator,
                                          const std::string& serialized_state);

}  // namespace rtp_llm
