#include "rtp_llm/cpp/utils/TorchCudaOom.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>

#include <c10/util/Exception.h>
#include <pybind11/embed.h>

#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {
namespace {

std::string lowerCase(const char* message) {
    std::string normalized = message ? message : "";
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return normalized;
}

bool hasGpuOomMarker(const std::string& message) {
    constexpr std::array<const char*, 6> markers = {
        "cuda out of memory",
        "cuda error: out of memory",
        "cudaerrormemoryallocation",
        "hip out of memory",
        "hip error: out of memory",
        "hiperroroutofmemory",
    };
    return std::any_of(markers.begin(), markers.end(), [&message](const char* marker) {
        return message.find(marker) != std::string::npos;
    });
}

std::string exceptionBacktrace(const std::exception& exception) {
    const auto* c10_error = dynamic_cast<const c10::Error*>(&exception);
    if (c10_error == nullptr) {
        return {};
    }
    const auto& backtrace = c10_error->backtrace();
    return backtrace ? backtrace->get() : std::string{};
}

void logAllocatorSummary(int detail_device) noexcept {
#if USING_CUDA
    try {
        constexpr size_t kMiB        = 1024 * 1024;
        const auto stats             = c10::cuda::CUDACachingAllocator::getDeviceStats(detail_device);
        const auto torch_reserved    = static_cast<size_t>(stats.reserved_bytes[0].current);
        const auto torch_allocated   = static_cast<size_t>(stats.allocated_bytes[0].current);
        const auto torch_active      = static_cast<size_t>(stats.active_bytes[0].current);
        const auto torch_cached_free = torch_reserved > torch_active ? torch_reserved - torch_active : 0;
        RTP_LLM_LOG_ERROR("[Torch GPU Diagnostics] device=%d allocator: reserved=%zu MiB allocated=%zu MiB "
                          "active=%zu MiB cached_free=%zu MiB",
                          detail_device,
                          torch_reserved / kMiB,
                          torch_allocated / kMiB,
                          torch_active / kMiB,
                          torch_cached_free / kMiB);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING(
            "[Torch GPU Diagnostics] failed to log device=%d memory summary: %s", detail_device, error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING(
            "[Torch GPU Diagnostics] failed to log device=%d memory summary: <unknown>", detail_device);
    }
#else
    RTP_LLM_LOG_WARNING("[Torch GPU Diagnostics] torch allocator summary unavailable in this build (device=%d)",
                        detail_device);
#endif
}

std::string dumpDiagnostics(int                   detail_device,
                            const char*           tag,
                            const std::exception* exception,
                            bool                  reuse_observer_dump) noexcept {
    logAllocatorSummary(detail_device);
    try {
        const std::string exception_message = exception ? exception->what() : "";
        const std::string cpp_backtrace     = exception ? exceptionBacktrace(*exception) : "";
        if (exception != nullptr) {
            RTP_LLM_LOG_ERROR(
                "[Torch GPU OOM] device=%d original_exception=%s", detail_device, exception_message.c_str());
            RTP_LLM_LOG_ERROR("[Torch GPU OOM] original_cpp_backtrace:\n%s", cpp_backtrace.c_str());
        }

        py::gil_scoped_acquire gil;
        auto output_path = py::module_::import("rtp_llm.utils.oom_diag")
                               .attr("dump_oom_diagnostics")(py::arg("tag")                 = tag,
                                                             py::arg("device")              = detail_device,
                                                             py::arg("exception")           = exception_message,
                                                             py::arg("cpp_backtrace")       = cpp_backtrace,
                                                             py::arg("reuse_observer_dump") = reuse_observer_dump);
        if (output_path.is_none()) {
            return {};
        }
        auto path = output_path.cast<std::string>();
        RTP_LLM_LOG_ERROR("[Torch GPU Diagnostics] allocator diagnostics written to %s", path.c_str());
        return path;
    } catch (const std::exception& diagnostic_error) {
        RTP_LLM_LOG_WARNING("[Torch GPU Diagnostics] failed to dump allocator diagnostics: %s",
                            diagnostic_error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("[Torch GPU Diagnostics] failed to dump allocator diagnostics: <unknown>");
    }
    return {};
}

}  // namespace

bool isTorchCudaOom(const std::exception& exception) noexcept {
    if (dynamic_cast<const c10::OutOfMemoryError*>(&exception) != nullptr) {
        return true;
    }

    try {
        return hasGpuOomMarker(lowerCase(exception.what()));
    } catch (...) {
        return false;
    }
}

std::string dumpTorchCudaOomDiagnostics(int detail_device) noexcept {
    return dumpDiagnostics(detail_device, "allocator_dump", nullptr, false);
}

void dumpFatalTorchCudaOomDiagnostics(int detail_device, const std::exception& exception) noexcept {
    (void)dumpDiagnostics(detail_device, "fatal_gpu_oom", &exception, true);
}

}  // namespace rtp_llm
