#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <pybind11/embed.h>

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

class StartupRpcServer: public LocalRpcServer {
public:
    void startServices() override {
        ++starts;
        if (on_start) {
            on_start();
        }
    }
    size_t onflightRequestNum() override { return 0; }
    int starts = 0;
    std::function<void()> on_start;
};

class RpcStartupTest: public DeviceTestBase {
public:
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
    }
    static void TearDownTestSuite() { interpreter_.reset(); }

protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        op_ = std::make_unique<RtpLLMOp>();
        op_->init_params_ = std::make_unique<EngineInitParams>();
        op_->init_params_->server_config = py::module_::import("types").attr("SimpleNamespace")(
            py::arg("rpc_server_port") = -1);
        server_ = std::make_shared<StartupRpcServer>();
        op_->model_rpc_service_ = std::make_unique<LocalRpcServiceImpl>();
        op_->model_rpc_service_->local_server_ = server_;
    }
    void TearDown() override {
        op_->stop();
        op_.reset();
        DeviceTestBase::TearDown();
    }
    static std::unique_ptr<py::scoped_interpreter> interpreter_;
    std::unique_ptr<RtpLLMOp> op_;
    std::shared_ptr<StartupRpcServer> server_;
};
std::unique_ptr<py::scoped_interpreter> RpcStartupTest::interpreter_;

TEST_F(RpcStartupTest, StartsRankWithoutListenerOnceAndRejectsLateEndpointUpdates) {
    ASSERT_NO_THROW(op_->startRPCServer());
    ASSERT_NO_THROW(op_->startRPCServer());
    EXPECT_EQ(server_->starts, 1);
    EXPECT_ANY_THROW(op_->updateRuntimeEndpoints(py::none()));
    op_->stop();
    EXPECT_ANY_THROW(op_->startRPCServer());
}

TEST_F(RpcStartupTest, PropagatesStartupFailure) {
    server_->on_start = [] { throw std::runtime_error("injected transport failure"); };
    try {
        op_->startRPCServer();
        FAIL() << "startup failure was lost";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("injected transport failure"), std::string::npos);
    }
}

TEST_F(RpcStartupTest, StopBeforeListenerPublicationJoinsTheStartupThread) {
    op_->init_params_->server_config.attr("rpc_server_port") = 0;
    std::promise<void> entered;
    auto starting = entered.get_future();
    server_->on_start = [&] {
        entered.set_value();
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (!op_->stop_requested_ && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(1ms);
        }
    };
    std::thread stopper([&] {
        if (starting.wait_for(5s) == std::future_status::ready) {
            py::gil_scoped_acquire acquire;
            op_->stop();
        }
    });
    try {
        op_->startRPCServer();
        ADD_FAILURE() << "startup ignored the stop request";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("stopped during startup"), std::string::npos);
    }
    {
        py::gil_scoped_release release;
        stopper.join();
    }
    EXPECT_FALSE(op_->grpc_server_thread_.joinable());
    EXPECT_EQ(op_->grpc_server_, nullptr);
    EXPECT_TRUE(op_->is_server_ready_.load());
    EXPECT_TRUE(op_->server_start_error_.empty());
}
}  // namespace
}  // namespace rtp_llm
