#pragma once

#include <type_traits>

namespace rtp_llm::kvcm {
namespace detail {
template<typename Code, typename = void>
struct HasExplicitBufferSafeCodes: std::false_type {};

template<typename Code>
struct HasExplicitBufferSafeCodes<
    Code,
    std::void_t<decltype(Code::ER_SDK_TIMEOUT_BUFFER_SAFE), decltype(Code::ER_SDK_ERROR_BUFFER_SAFE)>>:
    std::true_type {};
}  // namespace detail

// Older SDK headers have no explicit failed-but-buffer-safe contract.
template<typename Code>
bool sdkCodeProvesBufferSafety(Code code) {
    if (code == Code::ER_OK)
        return true;
    if constexpr (detail::HasExplicitBufferSafeCodes<Code>::value) {
        return code == Code::ER_SDK_TIMEOUT_BUFFER_SAFE || code == Code::ER_SDK_ERROR_BUFFER_SAFE;
    }
    return false;
}
}  // namespace rtp_llm::kvcm
