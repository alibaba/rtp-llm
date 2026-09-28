#include "rtp_llm/cpp/cuda_graph/ScopedPythonEnv.h"
#include "rtp_llm/cpp/utils/TorchCudaOom.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <c10/util/Exception.h>
#include <pybind11/embed.h>
#include "gtest/gtest.h"

namespace py = pybind11;

namespace rtp_llm {
namespace {

static_assert(noexcept(dumpTorchCudaOomDiagnostics(0)), "OOM diagnostics must never replace the original exception");
static_assert(std::is_nothrow_destructible<ScopedPythonEnvFlag>::value,
              "Environment restoration must never replace the original exception");

TEST(CudaGraphReplayRetryTest, ScopedPythonEnvFlagIsVisibleAndRestoresUnsetVariable) {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    py::gil_scoped_acquire gil;
    auto                   environ = py::module_::import("os").attr("environ");
    constexpr auto         name    = "RTP_LLM_TEST_SCOPED_PYTHON_ENV_UNSET";
    environ.attr("pop")(name, py::none());
    {
        ScopedPythonEnvFlag flag(name, "1");
        EXPECT_EQ(environ.attr("get")(name, "0").cast<std::string>(), "1");
        ASSERT_NE(std::getenv(name), nullptr);
        EXPECT_STREQ(std::getenv(name), "1");
    }
    EXPECT_TRUE(environ.attr("get")(name).is_none());
    EXPECT_EQ(std::getenv(name), nullptr);
}

TEST(CudaGraphReplayRetryTest, ScopedPythonEnvFlagRestoresExistingValueAndNestedScope) {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    py::gil_scoped_acquire gil;
    auto                   environ = py::module_::import("os").attr("environ");
    constexpr auto         name    = "RTP_LLM_TEST_SCOPED_PYTHON_ENV_EXISTING";
    ScopedPythonEnvFlag    original(name, "original");
    {
        ScopedPythonEnvFlag flag(name, "1");
        {
            ScopedPythonEnvFlag nested(name, "nested");
            EXPECT_EQ(environ.attr("get")(name).cast<std::string>(), "nested");
        }
        EXPECT_EQ(environ.attr("get")(name).cast<std::string>(), "1");
    }
    EXPECT_EQ(environ.attr("get")(name).cast<std::string>(), "original");
    EXPECT_STREQ(std::getenv(name), "original");
}

TEST(CudaGraphReplayRetryTest, ScopedPythonEnvFlagRestoresDuringExceptionUnwind) {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    py::gil_scoped_acquire gil;
    auto                   environ = py::module_::import("os").attr("environ");
    constexpr auto         name    = "RTP_LLM_TEST_SCOPED_PYTHON_ENV_EXCEPTION";
    environ.attr("pop")(name, py::none());
    for (bool originally_set : {false, true}) {
        if (originally_set) {
            environ[py::str(name)] = "original";
        }
        try {
            ScopedPythonEnvFlag flag(name, "1");
            EXPECT_EQ(environ.attr("get")(name).cast<std::string>(), "1");
            throw std::runtime_error("warmup failed");
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "warmup failed");
        }
        if (originally_set) {
            EXPECT_EQ(environ.attr("get")(name).cast<std::string>(), "original");
            EXPECT_STREQ(std::getenv(name), "original");
        } else {
            EXPECT_TRUE(environ.attr("get")(name).is_none());
            EXPECT_EQ(std::getenv(name), nullptr);
        }
    }
    environ.attr("pop")(name, py::none());
}

TEST(CudaGraphReplayRetryTest, ScopedPythonEnvFlagPreservesExistingLibcOnlyValue) {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    py::gil_scoped_acquire gil;
    auto                   environ = py::module_::import("os").attr("environ");
    constexpr auto         name    = "RTP_LLM_TEST_SCOPED_PYTHON_ENV_LIBC";
    environ.attr("pop")(name, py::none());
    ASSERT_EQ(setenv(name, "libc-only", 1), 0);
    EXPECT_TRUE(environ.attr("get")(name).is_none());
    {
        ScopedPythonEnvFlag flag(name, "1");
        EXPECT_EQ(environ.attr("get")(name).cast<std::string>(), "1");
        EXPECT_STREQ(std::getenv(name), "1");
    }
    EXPECT_TRUE(environ.attr("get")(name).is_none());
    EXPECT_STREQ(std::getenv(name), "libc-only");
    EXPECT_EQ(unsetenv(name), 0);
}

TEST(CudaGraphReplayRetryTest, ScopedPythonEnvFlagRestoreFailurePreservesOriginalException) {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    py::gil_scoped_acquire gil;
    auto                   os      = py::module_::import("os");
    auto                   environ = os.attr("environ");
    constexpr auto         name    = "RTP_LLM_TEST_SCOPED_PYTHON_ENV_RESTORE_ERROR";
    environ.attr("pop")(name, py::none());
    py::dict globals;
    py::exec("class BrokenEnvironment(dict):\n"
             "    def pop(self, *args):\n"
             "        raise RuntimeError('injected environment restoration failure')\n",
             globals);
    try {
        ScopedPythonEnvFlag flag(name, "1");
        os.attr("environ") = globals["BrokenEnvironment"]();
        throw std::runtime_error("original warmup failure");
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "original warmup failure");
    }
    os.attr("environ") = environ;
    environ.attr("pop")(name, py::none());
    EXPECT_FALSE(PyErr_Occurred());
    EXPECT_EQ(std::getenv(name), nullptr);
}

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

    py::object module;
    {
        py::gil_scoped_acquire gil;
        module = py::module_::import("types").attr("ModuleType")("rtp_llm.utils.oom_diag");
        module.attr("dump_oom_diagnostics") =
            py::cpp_function([](int) { return std::string("/tmp/allocator_dump.log"); }, py::arg("device"));
        py::dict modules                  = py::module_::import("sys").attr("modules");
        modules["rtp_llm.utils.oom_diag"] = module;
    }

    const std::exception* inner_exception = nullptr;
    const std::exception* outer_exception = nullptr;
    try {
        try {
            C10_THROW_ERROR(OutOfMemoryError, "C++ to Python OOM bridge marker");
        } catch (const std::exception& exception) {
            inner_exception = &exception;
            EXPECT_EQ(dumpTorchCudaOomDiagnostics(0), "/tmp/allocator_dump.log");
            throw;
        }
    } catch (const std::exception& exception) {
        outer_exception = &exception;
    }

    EXPECT_EQ(outer_exception, inner_exception);

    {
        py::gil_scoped_acquire gil;
        module.attr("dump_oom_diagnostics") =
            py::cpp_function([](int) -> std::string { throw std::runtime_error("injected Python diagnostic failure"); },
                             py::arg("device"));
    }

    inner_exception = nullptr;
    outer_exception = nullptr;
    try {
        try {
            C10_THROW_ERROR(OutOfMemoryError, "original OOM survives diagnostic failure");
        } catch (const std::exception& exception) {
            inner_exception = &exception;
            EXPECT_TRUE(dumpTorchCudaOomDiagnostics(0).empty());
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
