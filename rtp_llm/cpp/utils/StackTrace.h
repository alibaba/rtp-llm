#pragma once

#include <string>

namespace rtp_llm {
std::string getStackTrace();
void        printStackTrace();
std::string getPythonStackTrace();
}  // namespace rtp_llm
