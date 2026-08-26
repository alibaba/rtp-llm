#include "rtp_llm/cpp/utils/TorchCudaOom.h"

#include <stdexcept>
#include <string>
#include <utility>

#include <c10/util/Exception.h>
#include <pybind11/embed.h>
#include "gtest/gtest.h"

namespace py = pybind11;

namespace rtp_llm {
namespace {

static_assert(noexcept(dumpTorchCudaOomDiagnostics("test", std::declval<const std::exception&>())),
              "OOM diagnostics must never replace the original exception");

TEST(CudaGraphReplayRetryTest, DetectsTorchAndDriverOomErrors) {
    try {
        C10_THROW_ERROR(OutOfMemoryError, "allocator marker");
    } catch (const std::exception& exception) {
        EXPECT_TRUE(isTorchCudaOom(exception));
    }

    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("CUDA out of memory")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("cudaErrorMemoryAllocation")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("hipErrorOutOfMemory")));
    EXPECT_FALSE(isTorchCudaOom(std::runtime_error("illegal memory access")));
}

TEST(CudaGraphReplayRetryTest, CppBridgeCallsPythonAndPreservesOriginalException) {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    std::string captured_tag;
    std::string captured_exception;
    std::string captured_backtrace;
    py::object  module;
    {
        py::gil_scoped_acquire gil;
        module                              = py::module_::import("types").attr("ModuleType")("rtp_llm.utils.oom_diag");
        module.attr("dump_oom_diagnostics") = py::cpp_function(
            [&](const std::string& tag, const std::string& exception, const std::string& cpp_backtrace) {
                captured_tag       = tag;
                captured_exception = exception;
                captured_backtrace = cpp_backtrace;
            },
            py::arg("tag"),
            py::arg("exception"),
            py::arg("cpp_backtrace"));
        py::dict modules                  = py::module_::import("sys").attr("modules");
        modules["rtp_llm.utils.oom_diag"] = module;
    }

    const std::exception* inner_exception = nullptr;
    const std::exception* outer_exception = nullptr;
    std::string           original_backtrace;
    try {
        try {
            C10_THROW_ERROR(OutOfMemoryError, "C++ to Python OOM bridge marker");
        } catch (const std::exception& exception) {
            inner_exception       = &exception;
            const auto* c10_error = dynamic_cast<const c10::Error*>(&exception);
            ASSERT_NE(c10_error, nullptr);
            if (const auto& backtrace = c10_error->backtrace()) {
                original_backtrace = backtrace->get();
            }
            dumpTorchCudaOomDiagnostics("bridge_test", exception);
            throw;
        }
    } catch (const std::exception& exception) {
        outer_exception = &exception;
    }

    EXPECT_EQ(outer_exception, inner_exception);
    EXPECT_EQ(captured_tag, "bridge_test");
    EXPECT_NE(captured_exception.find("C++ to Python OOM bridge marker"), std::string::npos);
    EXPECT_EQ(captured_backtrace, original_backtrace);

    {
        py::gil_scoped_acquire gil;
        module.attr("dump_oom_diagnostics") = py::cpp_function(
            [](const std::string&, const std::string&, const std::string&) {
                throw std::runtime_error("injected Python diagnostic failure");
            },
            py::arg("tag"),
            py::arg("exception"),
            py::arg("cpp_backtrace"));
    }

    inner_exception = nullptr;
    outer_exception = nullptr;
    try {
        try {
            C10_THROW_ERROR(OutOfMemoryError, "original OOM survives diagnostic failure");
        } catch (const std::exception& exception) {
            inner_exception = &exception;
            dumpTorchCudaOomDiagnostics("bridge_failure_test", exception);
            throw;
        }
    } catch (const std::exception& exception) {
        outer_exception = &exception;
        EXPECT_NE(std::string(exception.what()).find("original OOM survives diagnostic failure"), std::string::npos);
    }
    EXPECT_EQ(outer_exception, inner_exception);
}

}  // namespace
}  // namespace rtp_llm
