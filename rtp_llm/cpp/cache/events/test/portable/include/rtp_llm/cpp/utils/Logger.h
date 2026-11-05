#pragma once

namespace rtp_llm::portable_test {
template<class... Args>
void ignoreLog(const char*, const Args&...) {}
}  // namespace rtp_llm::portable_test

#define RTP_LLM_LOG_DEBUG(...) ::rtp_llm::portable_test::ignoreLog(__VA_ARGS__)
#define RTP_LLM_LOG_WARNING(...) ::rtp_llm::portable_test::ignoreLog(__VA_ARGS__)
