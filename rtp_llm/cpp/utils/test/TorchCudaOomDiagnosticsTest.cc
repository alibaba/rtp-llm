#include "rtp_llm/cpp/utils/TorchCudaOom.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include <c10/util/Exception.h>
#include <pybind11/embed.h>
#include "gtest/gtest.h"

namespace py = pybind11;

namespace rtp_llm {
namespace {

class TorchCudaOomDiagnosticsTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    void SetUp() override {
        auto     module_type                 = py::module_::import("types").attr("ModuleType");
        auto     package                     = module_type("rtp_llm");
        auto     utils                       = module_type("rtp_llm.utils");
        py::dict modules                     = py::module_::import("sys").attr("modules");
        package.attr("__path__")             = py::list();
        utils.attr("__path__")               = py::list();
        module_                              = module_type("rtp_llm.utils.oom_diag");
        package.attr("utils")                = utils;
        utils.attr("oom_diag")               = module_;
        modules["rtp_llm"]                   = package;
        modules["rtp_llm.utils"]             = utils;
        modules["rtp_llm.utils.oom_diag"]    = module_;
        module_.attr("dump_oom_diagnostics") = py::cpp_function([this](py::kwargs kwargs) {
            arguments_ = kwargs;
            ++calls_;
            if (fail_diagnostics_) {
                throw std::runtime_error("injected Python diagnostic failure");
            }
            return std::string("/tmp/allocator-diagnostics-test.log");
        });
    }

    void TearDown() override {
        // The stub callback captures this fixture; remove its modules before
        // the fixture is destroyed while retaining the interpreter for reuse.
        py::dict modules = py::module_::import("sys").attr("modules");
        modules.attr("pop")("rtp_llm.utils.oom_diag", py::none());
        modules.attr("pop")("rtp_llm.utils", py::none());
        modules.attr("pop")("rtp_llm", py::none());
    }

    // Linked pybind11 code caches interpreter-specific TLS across calls.
    // Finalize only after every per-test Python object has been destroyed.
    inline static std::unique_ptr<py::scoped_interpreter> interpreter_;
    py::object                                            module_;
    py::dict                                              arguments_;
    int                                                   calls_            = 0;
    bool                                                  fail_diagnostics_ = false;
};

TEST_F(TorchCudaOomDiagnosticsTest, ManualDumpPassesCorrelationAndReturnsPythonPath) {
    EXPECT_EQ(dumpTorchCudaOomDiagnostics(0, "manual-dump-123"), "/tmp/allocator-diagnostics-test.log");

    ASSERT_EQ(calls_, 1);
    EXPECT_EQ(arguments_.size(), 6u);
    EXPECT_EQ(arguments_["tag"].cast<std::string>(), "allocator_dump");
    EXPECT_EQ(arguments_["device"].cast<int>(), 0);
    EXPECT_EQ(arguments_["exception"].cast<std::string>(), "");
    EXPECT_EQ(arguments_["cpp_backtrace"].cast<std::string>(), "");
    EXPECT_FALSE(arguments_["reuse_observer_dump"].cast<bool>());
    EXPECT_EQ(arguments_["dump_correlation_id"].cast<std::string>(), "manual-dump-123");
}

TEST_F(TorchCudaOomDiagnosticsTest, FatalDumpPassesOriginalExceptionAndPreservesRethrow) {
    std::exception_ptr original;
    try {
        try {
            C10_THROW_ERROR(OutOfMemoryError, "original GPU OOM bridge marker");
        } catch (const std::exception& exception) {
            original = std::current_exception();
            dumpFatalTorchCudaOomDiagnostics(0, exception);

            ASSERT_EQ(calls_, 1);
            EXPECT_EQ(arguments_["tag"].cast<std::string>(), "fatal_gpu_oom");
            EXPECT_EQ(arguments_["exception"].cast<std::string>(), exception.what());
            EXPECT_TRUE(arguments_["reuse_observer_dump"].cast<bool>());
            EXPECT_EQ(arguments_["dump_correlation_id"].cast<std::string>(), "");
            const auto& backtrace = dynamic_cast<const c10::Error&>(exception).backtrace();
            EXPECT_EQ(arguments_["cpp_backtrace"].cast<std::string>(), backtrace ? backtrace->get() : "");
            throw;
        }
    } catch (const c10::OutOfMemoryError&) {
        EXPECT_EQ(std::current_exception(), original);
        return;
    }
    FAIL() << "the original GPU OOM was not rethrown";
}

TEST_F(TorchCudaOomDiagnosticsTest, PythonDiagnosticFailureDoesNotReplaceOriginalOom) {
    fail_diagnostics_ = true;
    EXPECT_TRUE(dumpTorchCudaOomDiagnostics(0, "failed-dump").empty());
    ASSERT_EQ(calls_, 1);

    std::exception_ptr original;
    try {
        try {
            C10_THROW_ERROR(OutOfMemoryError, "original OOM survives diagnostic failure");
        } catch (const std::exception& exception) {
            original = std::current_exception();
            dumpFatalTorchCudaOomDiagnostics(0, exception);
            ASSERT_EQ(calls_, 2);
            throw;
        }
    } catch (const c10::OutOfMemoryError& exception) {
        EXPECT_EQ(std::current_exception(), original);
        EXPECT_NE(std::string(exception.what()).find("original OOM survives diagnostic failure"), std::string::npos);
        return;
    }
    FAIL() << "the original GPU OOM was not rethrown";
}

}  // namespace
}  // namespace rtp_llm
