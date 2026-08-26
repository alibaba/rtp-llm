#pragma once

#include <exception>

namespace rtp_llm {

bool isTorchCudaOom(const std::exception& exception) noexcept;
void dumpTorchCudaOomDiagnostics(const char* tag, const std::exception& exception) noexcept;

}  // namespace rtp_llm
