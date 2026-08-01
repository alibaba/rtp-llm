#pragma once

#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/Sampler.h"

namespace rtp_llm::normal_device_state {

// Restore the device-side token/length owner for a normal request received by
// a PD decode worker. Both NormalExecutor and DSpark's shared-target fallback
// must call this before gathering model inputs.
void prepareGrpc(RoleType role_type, const StreamGroups& stream_groups);

// Publish the sampled token and next sequence length before host dispatch.
// This keeps async-capable gatherers and GenerateStateMachine on one owner even
// when DSpark temporarily executes a request through the normal target path.
void publish(const StreamGroups& stream_groups, const SamplerOutput& sampler_output);

}  // namespace rtp_llm::normal_device_state
