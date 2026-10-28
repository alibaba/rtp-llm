#pragma once

namespace rtp_llm::kvcm {

// The pinned 79404320 API has no named failed-but-buffer-safe result. An SDK
// error number alone cannot prove that accepted backend work has stopped.
template<typename Code>
constexpr bool sdkCodeProvesBufferSafety(Code code) noexcept {
    return code == Code::ER_OK;
}

}  // namespace rtp_llm::kvcm
