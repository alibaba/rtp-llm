#pragma once

namespace rtp_llm {

// Environment configuration is immutable after process startup. Cache the
// value on first use so every RemoteCache component observes the same setting.
bool remoteCacheGdrEnabled();

}  // namespace rtp_llm
