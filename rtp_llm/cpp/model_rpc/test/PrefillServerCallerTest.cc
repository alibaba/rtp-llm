#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"

namespace rtp_llm::test {

class FakePrefillRpcService final: public RpcService::Service {
public:
    enum class Mode {
        kFirstChunkSnapshot,
        kWaitForCancel,
        kReturnErrorWithoutChunk,
    };

public:
    explicit FakePrefillRpcService(Mode mode): mode_(mode) {}

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        started_.store(true);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            started_cv_.notify_all();
        }

        if (mode_ == Mode::kFirstChunkSnapshot) {
            GenerateOutputsPB first_response;
            first_response.set_request_id(request->request_id());
            first_response.mutable_error_info()->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
            first_response.mutable_error_info()->set_error_message("first chunk error");
            first_response.mutable_flatten_output()->add_finished(false);
            first_response.mutable_flatten_output()->add_aux_info()->set_step_output_len(1);
            if (!writer->Write(first_response)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "failed to write first response");
            }

            GenerateOutputsPB second_response;
            second_response.set_request_id(request->request_id());
            second_response.mutable_flatten_output()->add_finished(true);
            second_response.mutable_flatten_output()->add_aux_info()->set_step_output_len(2);
            if (!writer->Write(second_response)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "failed to write second response");
            }
            return grpc::Status::OK;
        }

        if (mode_ == Mode::kWaitForCancel) {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                if (context->IsCancelled()) {
                    cancelled_.store(true);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        cancel_cv_.notify_all();
                    }
                    return grpc::Status(grpc::StatusCode::CANCELLED, "cancelled by client");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "wait for cancel timed out");
        }

        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill failed before any response");
    }

    bool waitStarted(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return started_cv_.wait_for(lock, timeout, [this]() { return started_.load(); });
    }

    bool waitCancelled(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cancel_cv_.wait_for(lock, timeout, [this]() { return cancelled_.load(); });
    }

private:
    Mode                    mode_;
    std::atomic<bool>       started_{false};
    std::atomic<bool>       cancelled_{false};
    std::mutex              mutex_;
    std::condition_variable started_cv_;
    std::condition_variable cancel_cv_;
};

class FakePrefillRpcServer {
public:
    explicit FakePrefillRpcServer(std::unique_ptr<FakePrefillRpcService> service): service_(std::move(service)) {}

    ~FakePrefillRpcServer() {
        stop();
    }

    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("0.0.0.0:0", grpc::InsecureServerCredentials(), &listen_port_);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        return server_ != nullptr && listen_port_ != 0;
    }

    void stop() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
            server_.reset();
        }
    }

    uint32_t port() const {
        return listen_port_;
    }

    FakePrefillRpcService* service() const {
        return service_.get();
    }

private:
    std::unique_ptr<FakePrefillRpcService> service_;
    std::unique_ptr<grpc::Server>          server_;
    int                                    listen_port_{0};
};

class PrefillServerCallerTest: public ::testing::Test {
protected:
    std::shared_ptr<PrefillServerCallerContext> callPrefill(FakePrefillRpcServer& server,
                                                            int64_t               request_id,
                                                            const std::string&    unique_key,
                                                            int64_t               deadline_us = 5 * 1000 * 1000) {
        GenerateInputPB request;
        request.set_request_id(request_id);
        request.add_token_ids(1);
        request.mutable_generate_config()->set_timeout_ms(5000);
        return caller_.callPrefill(&request, "127.0.0.1", server.port(), unique_key, deadline_us);
    }

    bool waitDone(const std::shared_ptr<PrefillServerCallerContext>& context,
                  std::chrono::milliseconds                          timeout = std::chrono::seconds(3)) {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (context->done()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return context->done();
    }

protected:
    PrefillServerCaller caller_{"prefill-server-caller-test"};
};

TEST_F(PrefillServerCallerTest, PreserveFirstReadResponseSnapshotThroughPublicApi) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kFirstChunkSnapshot));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1001, "snapshot");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    EXPECT_TRUE(context->success());

    const auto& response = context->response();
    ASSERT_TRUE(response.has_error_info());
    EXPECT_EQ(response.error_info().error_code(), ErrorCodePB::UNKNOWN_ERROR);
    EXPECT_EQ(response.error_info().error_message(), "first chunk error");
    ASSERT_TRUE(response.has_flatten_output());
    ASSERT_EQ(response.flatten_output().aux_info_size(), 1);
    EXPECT_EQ(response.flatten_output().aux_info(0).step_output_len(), 1);
}

TEST_F(PrefillServerCallerTest, CancelMarksContextDoneAndUnsuccessful) {
    FakePrefillRpcServer server(std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kWaitForCancel));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1002, "cancel");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(server.service()->waitStarted(std::chrono::seconds(1)));

    context->cancel();

    ASSERT_TRUE(waitDone(context));
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(server.service()->waitCancelled(std::chrono::seconds(1)));
}

TEST_F(PrefillServerCallerTest, RpcFailureWithoutAnyChunkStaysUnsuccessful) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kReturnErrorWithoutChunk));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1003, "rpc-failure");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->response().has_error_info());
}

}  // namespace rtp_llm::test
