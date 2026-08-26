#include "rtp_llm/cpp/utils/TorchCudaOom.h"

#include <algorithm>
#include <cctype>
#include <string>

#include <c10/util/Exception.h>
#include <pybind11/embed.h>

#include "rtp_llm/cpp/utils/Logger.h"

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

void dumpTorchCudaOomDiagnostics(const char* tag, const std::exception& exception) noexcept {
    try {
        RTP_LLM_LOG_ERROR("[Torch CUDA OOM] tag=%s original_exception=%s", tag, exception.what());
        std::string cpp_backtrace;
        if (const auto* c10_error = dynamic_cast<const c10::Error*>(&exception)) {
            if (const auto& backtrace = c10_error->backtrace()) {
                cpp_backtrace = backtrace->get();
            }
        }
        RTP_LLM_LOG_ERROR("[Torch CUDA OOM] original_cpp_backtrace:\n%s", cpp_backtrace.c_str());

        py::gil_scoped_acquire gil;
        py::module_::import("rtp_llm.utils.oom_diag")
            .attr("dump_oom_diagnostics")(py::arg("tag")           = tag,
                                          py::arg("exception")     = exception.what(),
                                          py::arg("cpp_backtrace") = cpp_backtrace);
    } catch (const std::exception& diagnostic_error) {
        RTP_LLM_LOG_WARNING("[Torch CUDA OOM] failed to dump allocator diagnostics: %s", diagnostic_error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("[Torch CUDA OOM] failed to dump allocator diagnostics: <unknown>");
    }
}

}  // namespace rtp_llm
