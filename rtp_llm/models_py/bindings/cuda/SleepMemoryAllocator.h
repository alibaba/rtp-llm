#pragma once

namespace rtp_llm {
// Capability check for the private runtime-scratch pool, called at pool creation.
bool sleepMemoryAllocatorAvailable();
}  // namespace rtp_llm
