#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <pybind11/embed.h>
#include "torch/all.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

#define private public
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#undef private
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

struct StopState {
    std::atomic<bool> worker_finished{false};
    bool              finished_before_stop = false;
    bool              drain_holds_gil      = true;
    bool              stop_holds_gil       = true;
    bool              destructor_holds_gil = false;
};

class PythonDrainService: public RpcServiceImpl {
public:
    explicit PythonDrainService(StopState& state): state_(state) {}

    ~PythonDrainService() override {
        state_.destructor_holds_gil = PyGILState_Check();
    }

    void beginShutdown() override {
        worker_ = std::thread([this]() {
            pybind11::gil_scoped_acquire acquire;
            state_.worker_finished = true;
        });
    }

    size_t onflightRequestNum() override {
        state_.drain_holds_gil = PyGILState_Check();
        return state_.worker_finished ? 0 : 1;
    }

    void stop() override {
        state_.finished_before_stop = state_.worker_finished;
        state_.stop_holds_gil       = PyGILState_Check();
        worker_.join();
    }

private:
    StopState&  state_;
    std::thread worker_;
};

class CancellationService: public RpcServiceImpl {
public:
    explicit CancellationService(std::atomic<bool>& cancelled): cancelled_(cancelled) {}

    grpc::Status CheckHealth(grpc::ServerContext* context, const EmptyPB*, CheckHealthResponsePB*) override {
        active_ = true;
        entered.set_value();
        while (!context->IsCancelled()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        cancelled_ = true;
        active_    = false;
        return grpc::Status(grpc::StatusCode::CANCELLED, "server shutdown");
    }

    size_t onflightRequestNum() override {
        return active_ ? 1 : 0;
    }

    void stop() override {}

    std::promise<void> entered;

private:
    std::atomic<bool>  active_{false};
    std::atomic<bool>& cancelled_;
};

class RtpLLMOpStopTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<pybind11::scoped_interpreter>();
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    void SetUp() override {
        if (const auto* value = std::getenv("RTP_LLM_STOP_TIMEOUT_MS")) {
            previous_timeout_ = value;
        }
        setenv("RTP_LLM_STOP_TIMEOUT_MS", "100", 1);
    }

    void TearDown() override {
        if (previous_timeout_) {
            setenv("RTP_LLM_STOP_TIMEOUT_MS", previous_timeout_->c_str(), 1);
        } else {
            unsetenv("RTP_LLM_STOP_TIMEOUT_MS");
        }
    }

    int start(RtpLLMOp& op, std::unique_ptr<RpcServiceImpl> service) {
        int port              = 0;
        op.model_rpc_service_ = std::move(service);
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(op.model_rpc_service_.get());
        op.grpc_server_ = builder.BuildAndStart();
        return op.grpc_server_ ? port : 0;
    }

    inline static std::unique_ptr<pybind11::scoped_interpreter> interpreter_;
    std::optional<std::string>                                  previous_timeout_;
};

TEST_F(RtpLLMOpStopTest, PythonWorkFinishesBeforeEngineStop) {
    StopState state;
    RtpLLMOp  op;
    ASSERT_NE(start(op, std::make_unique<PythonDrainService>(state)), 0);

    op.stop();

    EXPECT_TRUE(state.finished_before_stop);
    EXPECT_FALSE(state.drain_holds_gil);
    EXPECT_FALSE(state.stop_holds_gil);
    EXPECT_TRUE(state.destructor_holds_gil);
    EXPECT_TRUE(PyGILState_Check());
    EXPECT_TRUE(op.is_server_shutdown_);
    EXPECT_EQ(op.grpc_server_, nullptr);
    EXPECT_EQ(op.model_rpc_service_, nullptr);
    op.stop();
}

TEST_F(RtpLLMOpStopTest, DrainTimeoutCancelsUnfinishedRpc) {
    std::atomic<bool> cancelled{false};
    RtpLLMOp          op;
    auto              service = std::make_unique<CancellationService>(cancelled);
    auto              entered = service->entered.get_future();
    const int         port    = start(op, std::move(service));
    ASSERT_NE(port, 0);
    auto stub = RpcService::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
    grpc::ClientContext context;
    // Bounds the failure path of a server that never cancels the RPC.
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    auto response = std::async(std::launch::async, [&]() {
        EmptyPB               request;
        CheckHealthResponsePB output;
        return stub->CheckHealth(&context, request, &output);
    });
    EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    const auto begin = std::chrono::steady_clock::now();
    op.stop();

    EXPECT_TRUE(cancelled);
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(5));
    const auto status = response.get();
    // Forced server shutdown may reset the HTTP/2 stream instead of sending trailers.
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED) << status.error_message();
    EXPECT_TRUE(PyGILState_Check());
    EXPECT_TRUE(op.is_server_shutdown_);
}

}  // namespace
}  // namespace rtp_llm
