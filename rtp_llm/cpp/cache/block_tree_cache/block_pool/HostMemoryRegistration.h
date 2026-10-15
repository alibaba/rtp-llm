#pragma once

#include <cstddef>
#include <string>

namespace rtp_llm {

bool hostMemoryRegistrationSupported();
bool registerHostMemory(void* ptr, size_t size_bytes, std::string& error_message);
bool unregisterHostMemory(void* ptr, std::string& error_message);

}  // namespace rtp_llm
