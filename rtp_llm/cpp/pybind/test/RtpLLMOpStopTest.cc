#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"

#include <atomic>
#include <cstdlib>
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRouter.h"
#include <chrono>
#include <thread>
#include <pybind11/embed.h>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

struct RequestState {
    std::atomic<bool> entered{false};
    std::atomic<bool> draining{false};
    std::atomic<bool> forward_done{false};
    std::atomic<bool> completed{false};
    std::atomic<bool> engine_stopped{false};
    std::atomic<bool> stopped_before_completion{false};
};

void runCudaForward() {
    auto input  = torch::ones({64, 64}, torch::TensorOptions().device(torch::kCUDA));
    auto output = torch::mm(input, input).cpu();
    EXPECT_EQ(output.sum().item<float>(), 262144.0f);
}

class DrainingService: public LocalRpcServiceImpl {
public:
    explicit DrainingService(std::shared_ptr<RequestState> state): state_(std::move(state)) {}
    void beginShutdown() override {
        state_->draining = true;
    }
    size_t onflightRequestNum() override {
        return state_->entered && !state_->completed ? 1 : 0;
    }
    void stop() override {
        state_->stopped_before_completion = !state_->completed;
        state_->engine_stopped            = true;
    }
    grpc::Status RemoteGenerate(grpc::ServerContext* context, ServerStream* stream) override {
        state_->entered = true;
        while (!state_->forward_done && !context->IsCancelled()) {
            std::this_thread::sleep_for(1ms);
        }
        bool done = state_->forward_done;
        if (done) {
            GenerateOutputsPB response;
            stream->Write(response);
        }
        state_->completed = true;
        return done ? grpc::Status::OK : grpc::Status(grpc::StatusCode::CANCELLED, "shutdown deadline");
    }

private:
    std::shared_ptr<RequestState> state_;
};

class RtpLLMOpStopTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
        auto module  = py::module_::import("types").attr("ModuleType")("stop_test").cast<py::module_>();
        registerRtpLLMOp(module);
        py::module_::import("sys").attr("modules")["stop_test"] = module;
    }
    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    void checkStop(bool complete_forward) {
        setenv("RTP_LLM_STOP_TIMEOUT_MS", complete_forward ? "2000" : "200", 1);
        if (complete_forward) {
            runCudaForward();  // Initialize CUDA before measuring the drain budget.
        }
        auto  state           = std::make_shared<RequestState>();
        auto  object          = py::module_::import("stop_test").attr("RtpLLMOp")();
        auto& op              = object.cast<RtpLLMOp&>();
        op.model_rpc_service_ = std::make_unique<DrainingService>(state);
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(op.model_rpc_service_.get());
        op.grpc_server_ = builder.BuildAndStart();
        ASSERT_NE(op.grpc_server_, nullptr);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 6s);
        auto stub = RpcService::NewStub(
            grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
        grpc::Status status;
        bool         received = false;
        std::thread  client([&] {
            auto rpc = stub->RemoteGenerate(&context);
            rpc->WritesDone();
            GenerateOutputsPB response;
            while (rpc->Read(&response)) {
                received = true;
            }
            status = rpc->Finish();
        });
        auto         entered_deadline = std::chrono::steady_clock::now() + 2s;
        while (!state->entered && std::chrono::steady_clock::now() < entered_deadline) {
            std::this_thread::sleep_for(1ms);
        }
        EXPECT_TRUE(state->entered);

        // This is a real Python thread, resumed by beginShutdown. It must be
        // able to run Python while the production pybind stop() drains the RPC.
        py::dict globals;
        globals["wait_for_shutdown"] = py::cpp_function([state] {
            py::gil_scoped_release release;
            while (!state->draining) {
                std::this_thread::sleep_for(1ms);
            }
        });
        globals["finish_forward"]    = py::cpp_function([state] {
            runCudaForward();
            state->forward_done = true;
        });
        if (complete_forward) {
            py::exec(R"(
import threading
def forward():
    wait_for_shutdown()
    result = sum(range(1000))
    finish_forward()
worker = threading.Thread(target=forward)
worker.start()
)",
                     globals);
        }

        std::atomic<bool> stop_returned{false};
        std::atomic<bool> watchdog_fired{false};
        std::thread       watchdog([&] {
            auto deadline = std::chrono::steady_clock::now() + 4s;
            while (!stop_returned && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(5ms);
            }
            if (!stop_returned) {
                watchdog_fired = true;
                context.TryCancel();
            }
        });
        auto              start = std::chrono::steady_clock::now();
        object.attr("stop")();
        auto duration = std::chrono::steady_clock::now() - start;
        stop_returned = true;
        client.join();
        watchdog.join();
        EXPECT_FALSE(watchdog_fired);
        EXPECT_TRUE(state->completed);
        EXPECT_TRUE(state->engine_stopped);
        EXPECT_FALSE(state->stopped_before_completion);
        EXPECT_EQ(status.ok(), complete_forward);
        EXPECT_EQ(received, complete_forward);
        if (complete_forward) {
            EXPECT_TRUE(state->forward_done);
            globals["worker"].attr("join")();
            EXPECT_LT(duration, 1500ms);
        } else {
            EXPECT_GE(duration, 150ms);
            EXPECT_LT(duration, 1500ms);
        }
        object.attr("stop")();  // Repeated shutdown remains safe.
        unsetenv("RTP_LLM_STOP_TIMEOUT_MS");
    }
    inline static std::unique_ptr<py::scoped_interpreter> interpreter_;
};

TEST_F(RtpLLMOpStopTest, HttpGenerationRoutesRejectNewWorkButKeepExistingCount) {
    EngineInitParams params;
    HttpApiServer    server(nullptr, nullptr, "tcp:127.0.0.1:0", params, py::none());
    server.http_server_ = std::make_unique<http_server::HttpServer>();
    ASSERT_TRUE(server.registerInferenceService());
    ASSERT_TRUE(server.registerChatService());
    {
        CounterGuard existing(server.active_request_count_, server.admission_);
        ASSERT_TRUE(existing.admitted());
        server.beginShutdown();
        for (const auto& path : {"/", "/inference_internal", "/v1/chat/completions", "/v1/chat/render"}) {
            auto callback = server.http_server_->_router->FindRoute("POST", path);
            ASSERT_TRUE(callback.has_value());
            struct Writer: http_server::HttpResponseWriter {
                explicit Writer(bool& rejected): HttpResponseWriter(nullptr), rejected_(rejected) {}
                bool Write(const std::string&) override {
                    rejected_ = _statusCode == 503;
                    return true;
                }
                bool WriteDone() override {
                    return true;
                }
                bool& rejected_;
            };
            bool rejected = false;
            (*callback)(std::make_unique<Writer>(rejected), http_server::HttpRequest());
            EXPECT_TRUE(rejected) << path;
            EXPECT_EQ(server.onflightRequestNum(), 1);
        }
    }
    EXPECT_EQ(server.onflightRequestNum(), 0);
}

TEST_F(RtpLLMOpStopTest, PythonForwardCompletesDuringDrain) {
    checkStop(true);
}
TEST_F(RtpLLMOpStopTest, DeadlineCancelsAnUnfinishedStream) {
    checkStop(false);
}
}  // namespace
}  // namespace rtp_llm
