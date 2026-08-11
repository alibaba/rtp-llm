#include "gtest/gtest.h"

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

std::shared_ptr<GenerateInput> makeGenerateInput() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    return input;
}

ModelConfig makeModelConfig() {
    ModelConfig config;
    config.max_seq_len = 2048;
    return config;
}

class ObservedNormalGenerateStream final: public NormalGenerateStream {
public:
    ObservedNormalGenerateStream():
        NormalGenerateStream(
            makeGenerateInput(), makeModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr),
        first_check_future_(first_check_promise_.get_future().share()) {}

    ErrorResult<GenerateOutputs> nextOutput(const OutputCancellationCheck& is_cancelled = {}) override {
        return NormalGenerateStream::nextOutput([this, &is_cancelled]() {
            const bool observed_cancel = is_cancelled && is_cancelled();
            std::call_once(first_check_once_, [this]() { first_check_promise_.set_value(); });
            return observed_cancel;
        });
    }

    bool waitForFirstCheck(std::chrono::milliseconds timeout) const {
        return first_check_future_.wait_for(timeout) == std::future_status::ready;
    }

    void stopForCleanup() {
        reportError(ErrorCode::CANCELLED, "test cleanup");
    }

private:
    std::promise<void>        first_check_promise_;
    std::shared_future<void>  first_check_future_;
    std::once_flag            first_check_once_;
};

class LocalRpcServerHarness final: public LocalRpcServer {
public:
    grpc::Status poll(grpc::ServerContext*             context,
                      WriterInterface*                 writer,
                      std::shared_ptr<GenerateStream>& stream) {
        return pollStreamOutput(context, "stream-cancel-test", writer, stream);
    }

    ErrorInfo collect(grpc::ServerContext*             context,
                      std::shared_ptr<GenerateStream>& stream,
                      GenerateOutputs&                 outputs) {
        auto input = makeGenerateInput();
        return collectStreamOutput(context, stream, input, outputs);
    }
};

class CancellationService final: public RpcService::Service {
public:
    CancellationService():
        stream_(std::make_shared<ObservedNormalGenerateStream>()),
        stream_status_future_(stream_status_promise_.get_future()),
        batch_status_future_(batch_status_promise_.get_future()) {}

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        std::shared_ptr<GenerateStream> stream = stream_;
        const auto status = harness_.poll(context, writer, stream);
        stream_status_promise_.set_value(status);
        return status;
    }

    grpc::Status BatchGenerateCall(grpc::ServerContext*        context,
                                   const BatchGenerateInputPB*,
                                   BatchGenerateOutputsPB*) override {
        std::shared_ptr<GenerateStream> stream = stream_;
        GenerateOutputs                outputs;
        const auto                     error = harness_.collect(context, stream, outputs);
        const auto status = error.ok() ? grpc::Status::OK :
                                         grpc::Status(transErrorCodeToGrpc(error.code()), error.ToString());
        batch_status_promise_.set_value(status);
        return status;
    }

    std::shared_ptr<ObservedNormalGenerateStream> stream() const {
        return stream_;
    }

    std::future<grpc::Status>& streamStatusFuture() {
        return stream_status_future_;
    }

    std::future<grpc::Status>& batchStatusFuture() {
        return batch_status_future_;
    }

private:
    LocalRpcServerHarness                         harness_;
    std::shared_ptr<ObservedNormalGenerateStream> stream_;
    std::promise<grpc::Status>                    stream_status_promise_;
    std::future<grpc::Status>                     stream_status_future_;
    std::promise<grpc::Status>                    batch_status_promise_;
    std::future<grpc::Status>                     batch_status_future_;
};

class LoopbackServer final {
public:
    explicit LoopbackServer(grpc::Service* service) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service);
        server_ = builder.BuildAndStart();
    }

    ~LoopbackServer() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    bool started() const {
        return server_ != nullptr && port_ > 0;
    }

    std::shared_ptr<grpc::Channel> channel() const {
        return grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
    }

private:
    int                           port_{0};
    std::unique_ptr<grpc::Server> server_;
};

TEST(LocalRpcOutputCancellationTest, StreamingCancelWhileWaitingReturnsFromServerHandler) {
    CancellationService service;
    LoopbackServer      server(&service);
    ASSERT_TRUE(server.started());
    auto stub = RpcService::NewStub(server.channel());

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 5s);
    GenerateInputPB request;
    auto            reader = stub->GenerateStreamCall(&context, request);

    const bool entered_wait = service.stream()->waitForFirstCheck(2500ms);
    context.TryCancel();
    if (!entered_wait) {
        service.stream()->stopForCleanup();
    }
    const auto client_status = reader->Finish();

    auto&      handler_result = service.streamStatusFuture();
    const auto handler_ready  = handler_result.wait_for(2500ms);
    if (handler_ready != std::future_status::ready) {
        service.stream()->stopForCleanup();
        EXPECT_EQ(handler_result.wait_for(2500ms), std::future_status::ready);
    }

    EXPECT_TRUE(entered_wait);
    EXPECT_EQ(client_status.error_code(), grpc::StatusCode::CANCELLED);
    ASSERT_EQ(handler_ready, std::future_status::ready);
    EXPECT_EQ(handler_result.get().error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(service.stream()->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(LocalRpcOutputCancellationTest, BatchGenerateCancelWhileWaitingReturnsFromServerHandler) {
    CancellationService service;
    LoopbackServer      server(&service);
    ASSERT_TRUE(server.started());
    auto stub = RpcService::NewStub(server.channel());

    grpc::ClientContext    context;
    BatchGenerateInputPB   request;
    BatchGenerateOutputsPB response;
    context.set_deadline(std::chrono::system_clock::now() + 5s);
    auto client_result = std::async(
        std::launch::async, [&]() { return stub->BatchGenerateCall(&context, request, &response); });

    const bool entered_wait = service.stream()->waitForFirstCheck(2500ms);
    context.TryCancel();
    if (!entered_wait) {
        service.stream()->stopForCleanup();
    }

    auto client_ready = client_result.wait_for(2500ms);
    if (client_ready != std::future_status::ready) {
        service.stream()->stopForCleanup();
        EXPECT_EQ(client_result.wait_for(2500ms), std::future_status::ready);
    }
    auto&      handler_result = service.batchStatusFuture();
    const auto handler_ready  = handler_result.wait_for(2500ms);
    if (handler_ready != std::future_status::ready) {
        service.stream()->stopForCleanup();
        EXPECT_EQ(handler_result.wait_for(2500ms), std::future_status::ready);
    }

    EXPECT_TRUE(entered_wait);
    ASSERT_EQ(client_ready, std::future_status::ready);
    EXPECT_EQ(client_result.get().error_code(), grpc::StatusCode::CANCELLED);
    ASSERT_EQ(handler_ready, std::future_status::ready);
    EXPECT_EQ(handler_result.get().error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(service.stream()->statusInfo().code(), ErrorCode::CANCELLED);
}

}  // namespace
}  // namespace rtp_llm
