#include "rtp_llm/cpp/utils/TorchCudaOom.h"

#include <algorithm>
#include <cctype>
#include <string>

#include <c10/util/Exception.h>
#include <pybind11/embed.h>

#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace py = pybind11;

namespace rtp_llm {

bool isTorchCudaOom(const std::exception& exception) noexcept {
    if (dynamic_cast<const c10::OutOfMemoryError*>(&exception) != nullptr) {
        return true;
    }

    try {
        std::string message = exception.what();
        std::transform(
            message.begin(), message.end(), message.begin(), [](unsigned char c) { return std::tolower(c); });
        return message.find("out of memory") != std::string::npos
               || message.find("cudaerrormemoryallocation") != std::string::npos
               || message.find("hiperroroutofmemory") != std::string::npos;
    } catch (...) {
        return false;
    }
}

std::string dumpTorchCudaOomDiagnostics(int detail_device) noexcept {
#if USING_CUDA
    try {
        constexpr size_t kMiB                  = 1024 * 1024;
        const auto [device_free, device_total] = c10::cuda::CUDACachingAllocator::get()->getMemoryInfo(detail_device);
        const auto stats                       = c10::cuda::CUDACachingAllocator::getDeviceStats(detail_device);
        const auto torch_reserved              = static_cast<size_t>(stats.reserved_bytes[0].current);
        const auto torch_allocated             = static_cast<size_t>(stats.allocated_bytes[0].current);
        const auto torch_active                = static_cast<size_t>(stats.active_bytes[0].current);
        const auto torch_cached_free           = torch_reserved > torch_active ? torch_reserved - torch_active : 0;
        RTP_LLM_LOG_ERROR("[Torch CUDA Diagnostics] device=%d GPU: used=%zu MiB free=%zu MiB total=%zu MiB | "
                          "Torch allocator: reserved=%zu MiB allocated=%zu MiB active=%zu MiB cached_free=%zu MiB",
                          detail_device,
                          (device_total - device_free) / kMiB,
                          device_free / kMiB,
                          device_total / kMiB,
                          torch_reserved / kMiB,
                          torch_allocated / kMiB,
                          torch_active / kMiB,
                          torch_cached_free / kMiB);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING(
            "[Torch CUDA Diagnostics] failed to log device=%d memory summary: %s", detail_device, error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("[Torch CUDA Diagnostics] failed to log device=%d memory summary: <unknown>",
                            detail_device);
    }
#endif

    try {
        py::gil_scoped_acquire gil;
        auto                   output_path = py::module_::import("rtp_llm.utils.oom_diag")
                               .attr("dump_oom_diagnostics")(py::arg("device") = detail_device);
        if (output_path.is_none()) {
            return {};
        }
        auto path = output_path.cast<std::string>();
        RTP_LLM_LOG_ERROR("[Torch CUDA Diagnostics] allocator diagnostics written to %s", path.c_str());
        return path;
    } catch (const std::exception& diagnostic_error) {
        RTP_LLM_LOG_WARNING("[Torch CUDA Diagnostics] failed to dump allocator diagnostics: %s",
                            diagnostic_error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("[Torch CUDA Diagnostics] failed to dump allocator diagnostics: <unknown>");
    }
    return {};
}

}  // namespace rtp_llm
