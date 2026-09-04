#pragma once

#include <exception>
#include <string>
#include <utility>

namespace rtp_llm {

// Kept under the historical CUDA name for source compatibility; this recognizes
// both CUDA and HIP allocator failures.
bool isTorchCudaOom(const std::exception& exception) noexcept;

std::string dumpTorchCudaOomDiagnostics(int detail_device) noexcept;
void dumpFatalTorchCudaOomDiagnostics(int detail_device, const std::exception& exception) noexcept;

template<typename Operation, typename BeforeRetry>
void retryOnceOnTorchCudaOom(Operation&& operation, BeforeRetry&& before_retry) {
    try {
        std::forward<Operation>(operation)();
    } catch (const std::exception& exception) {
        if (!isTorchCudaOom(exception)) {
            throw;
        }
        std::forward<BeforeRetry>(before_retry)(exception);
        // Deliberately outside another catch/loop: a second failure propagates.
        std::forward<Operation>(operation)();
    }
}

}  // namespace rtp_llm
