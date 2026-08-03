#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

#if USING_ROCM
#include <pybind11/embed.h>

#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

namespace rtp_llm::cuda_graph {

#if USING_ROCM
namespace {

std::vector<std::string> handshake_events;
uint64_t                 expected_owner_token{17};
uint64_t                 expected_generation{23};
bool                     malformed_acquire_result{false};
bool                     fail_python_enter{false};
bool                     fail_python_exit{false};

void ensurePythonInterpreter() {
    // Intentionally retain the interpreter for the process lifetime because
    // getGraphCaptureModule() retains the imported module in a function-local
    // static py::module_.
    static py::scoped_interpreter* interpreter = Py_IsInitialized() ? nullptr : new py::scoped_interpreter();
    (void)interpreter;
}

void installFakeLifecycleModule() {
    py::gil_scoped_acquire gil;
    auto module = py::module_::import("types").attr("ModuleType")("rtp_llm.models_py.distributed.rocm_rccl");
    module.attr("acquire_graph_owner")     = py::cpp_function([](uintptr_t owner_id) -> py::tuple {
        handshake_events.emplace_back("acquire:" + std::to_string(owner_id));
        if (malformed_acquire_result) {
            return py::make_tuple(expected_owner_token);
        }
        return py::make_tuple(expected_owner_token, expected_generation);
    });
    module.attr("begin_capture_planning")  = py::cpp_function([](uint64_t owner_token, uint64_t generation) {
        EXPECT_EQ(owner_token, expected_owner_token);
        EXPECT_EQ(generation, expected_generation);
        handshake_events.emplace_back("begin-planning");
    });
    module.attr("cancel_capture_planning") = py::cpp_function([](uint64_t owner_token, uint64_t generation) {
        EXPECT_EQ(owner_token, expected_owner_token);
        EXPECT_EQ(generation, expected_generation);
        handshake_events.emplace_back("cancel-planning");
    });
    module.attr("prepare_capture_arena")   = py::cpp_function([](uint64_t owner_token, uint64_t generation) {
        EXPECT_EQ(owner_token, expected_owner_token);
        EXPECT_EQ(generation, expected_generation);
        handshake_events.emplace_back("prepare-arena");
    });
    module.attr("release_graph_owner")     = py::cpp_function([](uint64_t owner_token, uint64_t generation) {
        if (owner_token == 0) {
            return;
        }
        EXPECT_EQ(owner_token, expected_owner_token);
        EXPECT_EQ(generation, expected_generation);
        handshake_events.emplace_back("release");
    });
    module.attr("release_graph_owner_after_acquire_failure") = py::cpp_function([](uintptr_t owner_id) {
        EXPECT_EQ(owner_id, 91);
        handshake_events.emplace_back("release-after-acquire-failure:" + std::to_string(owner_id));
    });
    module.attr("enter_graph_capture_mode")        = py::cpp_function([](uint64_t owner_token, uint64_t generation) {
        EXPECT_EQ(owner_token, expected_owner_token);
        EXPECT_EQ(generation, expected_generation);
        EXPECT_FALSE(rocm::isHipGraphCaptureEnabled());
        handshake_events.emplace_back("python-enter");
        if (fail_python_enter) {
            throw py::value_error("injected Python enter failure");
        }
    });
    module.attr("exit_graph_capture_mode")         = py::cpp_function([](uint64_t owner_token, uint64_t generation) {
        EXPECT_EQ(owner_token, expected_owner_token);
        EXPECT_EQ(generation, expected_generation);
        EXPECT_FALSE(rocm::isHipGraphCaptureEnabled());
        handshake_events.emplace_back("python-exit");
        if (fail_python_exit) {
            throw py::value_error("injected Python exit failure");
        }
    });
    module.attr("finish_hipgraph_capture_session") = py::cpp_function([](uint64_t owner_token, uint64_t generation) {
        if (owner_token == 0) {
            return;
        }
        EXPECT_EQ(owner_token, expected_owner_token);
        EXPECT_EQ(generation, expected_generation);
        handshake_events.emplace_back("finish-session");
    });
    py::module_::import("sys").attr("modules")["rtp_llm.models_py.distributed.rocm_rccl"] = module;
}

class HipGraphCaptureHandshakeTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ensurePythonInterpreter();
        installFakeLifecycleModule();
    }

    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        handshake_events.clear();
        expected_owner_token     = 17;
        expected_generation      = 23;
        malformed_acquire_result = false;
        fail_python_enter        = false;
        fail_python_exit         = false;
        rocm::setHipGraphCaptureEnabled(false);
    }

    void TearDown() override {
        rocm::setHipGraphCaptureEnabled(false);
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }

private:
    bool old_core_dump_on_exception_{false};
};

TEST_F(HipGraphCaptureHandshakeTest, RealShimPublishesAndClearsNativeFlagInOrder) {
    const GraphLifecycleContext context{17, 23};

    EXPECT_FALSE(rocm::isHipGraphCaptureEnabled());
    enter_graph_capture(&context);
    EXPECT_TRUE(rocm::isHipGraphCaptureEnabled());
    ASSERT_EQ(handshake_events, std::vector<std::string>({"python-enter"}));

    fail_python_exit = true;
    EXPECT_THROW(exit_graph_capture(&context), py::error_already_set);
    EXPECT_FALSE(rocm::isHipGraphCaptureEnabled());
    EXPECT_EQ(handshake_events, std::vector<std::string>({"python-enter", "python-exit"}));
}

TEST_F(HipGraphCaptureHandshakeTest, LifecycleWrappersForwardStableTokenAndGeneration) {
    GraphLifecycleContext context = acquire_graph_owner(91);
    EXPECT_EQ(context.owner_token, expected_owner_token);
    EXPECT_EQ(context.generation, expected_generation);

    begin_capture_planning(context);
    begin_capture_planning(context);
    cancel_capture_planning(context);
    prepare_capture_arena(context);
    finish_capture_session(context);
    release_graph_owner(context);

    EXPECT_EQ(handshake_events,
              std::vector<std::string>({"acquire:91",
                                        "begin-planning",
                                        "begin-planning",
                                        "cancel-planning",
                                        "prepare-arena",
                                        "finish-session",
                                        "release"}));
}

TEST_F(HipGraphCaptureHandshakeTest, MalformedAcquireResultIsRejected) {
    malformed_acquire_result = true;
    EXPECT_ANY_THROW(acquire_graph_owner(91));
    EXPECT_EQ(handshake_events, std::vector<std::string>({"acquire:91", "release-after-acquire-failure:91"}));
}

TEST_F(HipGraphCaptureHandshakeTest, NullOrFailedEnterNeverPublishesNativeFlag) {
    const GraphLifecycleContext context{17, 23};

    EXPECT_ANY_THROW(enter_graph_capture(nullptr));
    EXPECT_FALSE(rocm::isHipGraphCaptureEnabled());

    fail_python_enter = true;
    EXPECT_THROW(enter_graph_capture(&context), py::error_already_set);
    EXPECT_FALSE(rocm::isHipGraphCaptureEnabled());
    EXPECT_EQ(handshake_events, std::vector<std::string>({"python-enter"}));
}

TEST_F(HipGraphCaptureHandshakeTest, DegenerateTpCapturesWithoutPythonCommunicationHandshake) {
    const GraphLifecycleContext inactive_context{};

    begin_capture_planning(inactive_context);
    cancel_capture_planning(inactive_context);
    prepare_capture_arena(inactive_context);
    enter_graph_capture(&inactive_context);
    EXPECT_TRUE(rocm::isHipGraphCaptureEnabled());
    EXPECT_TRUE(handshake_events.empty());

    exit_graph_capture(&inactive_context);
    finish_capture_session(inactive_context);
    release_graph_owner(inactive_context);
    EXPECT_FALSE(rocm::isHipGraphCaptureEnabled());
    EXPECT_TRUE(handshake_events.empty());
}

TEST_F(HipGraphCaptureHandshakeTest, NullExitStillClearsNativeFlag) {
    rocm::setHipGraphCaptureEnabled(true);
    EXPECT_ANY_THROW(exit_graph_capture(nullptr));
    EXPECT_FALSE(rocm::isHipGraphCaptureEnabled());
    EXPECT_TRUE(handshake_events.empty());
}

}  // namespace
#else
TEST(HipGraphCaptureHandshakeTest, RequiresRocm) {
    GTEST_SKIP() << "HIPGraph capture handshake is ROCm-only";
}
#endif

}  // namespace rtp_llm::cuda_graph
