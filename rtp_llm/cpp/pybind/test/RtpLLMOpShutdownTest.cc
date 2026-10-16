#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"

#include <gtest/gtest.h>
#include <memory>
#include <pybind11/embed.h>
#include <stdexcept>

namespace rtp_llm {
namespace {

struct ShutdownCall {
    const char* name;
    void (*invoke)(RtpLLMOp&);
};

class RtpLLMOpShutdownTest: public ::testing::TestWithParam<ShutdownCall> {
protected:
    static void SetUpTestSuite() {
        // RtpLLMOp is a Python binding: stop() releases the GIL around
        // blocking teardown even when the backend has not been initialized.
        interpreter_ = std::make_unique<py::scoped_interpreter>();
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    void expectUnavailable(RtpLLMOp& op) {
        try {
            GetParam().invoke(op);
            FAIL() << "uninitialized lifecycle control must fail before accessing the engine";
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "backend lifecycle control is not initialized");
        }
    }
    inline static std::unique_ptr<py::scoped_interpreter> interpreter_;
};

TEST_P(RtpLLMOpShutdownTest, RejectsBeforeInitialization) {
    RtpLLMOp op;
    expectUnavailable(op);
}

TEST_P(RtpLLMOpShutdownTest, RejectsAfterStop) {
    RtpLLMOp op;
    op.stop();
    expectUnavailable(op);
    op.stop();  // The defensive check does not change stop's idempotency.
}

INSTANTIATE_TEST_SUITE_P(
    LifecycleControl,
    RtpLLMOpShutdownTest,
    ::testing::Values(ShutdownCall{"Status", [](RtpLLMOp& op) { (void)op.shutdownStatus(); }},
                      ShutdownCall{"Begin", [](RtpLLMOp& op) { op.beginShutdown(); }},
                      ShutdownCall{"Drain", [](RtpLLMOp& op) { op.drainShutdown(1000, false); }},
                      ShutdownCall{"SealAndDrain", [](RtpLLMOp& op) { op.drainShutdown(1000, true); }},
                      ShutdownCall{"Freeze", [](RtpLLMOp& op) { op.freezeShutdown(); }},
                      ShutdownCall{"Quiesce", [](RtpLLMOp& op) { op.quiesceShutdown("shutdown/test", 1000); }},
                      ShutdownCall{"Terminate", [](RtpLLMOp& op) { op.terminateShutdown(); }}),
    [](const ::testing::TestParamInfo<ShutdownCall>& info) { return info.param.name; });

}  // namespace
}  // namespace rtp_llm
