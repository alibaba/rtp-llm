#pragma once

#include <exception>
#include <string>

namespace rtp_llm {

bool        isTorchCudaOom(const std::exception& exception) noexcept;
std::string dumpTorchCudaOomDiagnostics(int detail_device) noexcept;

}  // namespace rtp_llm
