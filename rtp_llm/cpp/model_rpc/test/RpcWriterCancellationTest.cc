#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {

class RejectingWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    bool Write(const GenerateOutputsPB&, grpc::WriteOptions) override {
        ++write_calls;
        return false;
    }

    int write_calls = 0;
};

class SingleOutputStream: public GenerateStream {
public:
    SingleOutputStream(): GenerateStream(makeInput(), makeModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput() override {
        GenerateOutputs outputs;
        GenerateOutput  output;
        output.output_ids = torch::ones({1, 1}, torch::kInt32);
        output.finished   = false;
        outputs.generate_outputs.push_back(std::move(output));
        return ErrorResult<GenerateOutputs>(std::move(outputs));
    }

    void updateOutput(const StreamUpdateInfo&) override {}

private:
    static std::shared_ptr<GenerateInput> makeInput() {
        auto input             = std::make_shared<GenerateInput>();
        input->request_id      = 41;
        input->generate_config = std::make_shared<GenerateConfig>();
        input->input_ids       = torch::tensor({1}, torch::kInt32);
        return input;
    }

    static ModelConfig makeModelConfig() {
        ModelConfig config;
        config.max_seq_len = 8;
        return config;
    }
};

class SingleResponseClientStream: public grpc::ClientReaderWriterInterface<GenerateRequestPB, GenerateOutputsPB> {
public:
    bool Read(GenerateOutputsPB* response) override {
        ++read_calls;
        if (read_calls != 1) {
            return false;
        }
        response->mutable_flatten_output()->add_aux_info();
        return true;
    }

    bool NextMessageSize(uint32_t*) override {
        return false;
    }

    bool Write(const GenerateRequestPB&, grpc::WriteOptions) override {
        return true;
    }

    void WaitForInitialMetadata() override {}

    bool WritesDone() override {
        ++writes_done_calls;
        return true;
    }

    grpc::Status Finish() override {
        ++finish_calls;
        return grpc::Status(grpc::StatusCode::CANCELLED, "cancelled by client");
    }

    int read_calls        = 0;
    int writes_done_calls = 0;
    int finish_calls      = 0;
};

TEST(RpcWriterCancellationTest, LocalWriteFailureCancelsStreamAndReturnsCancelled) {
    LocalRpcServer                  server;
    RejectingWriter                 writer;
    std::shared_ptr<GenerateStream> stream = std::make_shared<SingleOutputStream>();

    const auto status = server.pollStreamOutput(nullptr, "41", &writer, stream);

    EXPECT_EQ(writer.write_calls, 1);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(RpcWriterCancellationTest, RemoteWriteFailureCancelsGrpcStreamClosure) {
    PrefillRpcServer server;
    RejectingWriter  writer;
    GenerateInputPB  request;
    request.set_request_id(42);
    RPCContext                   rpc_context{&request, &writer};
    RemoteServerResource         resource;
    kmonitor::MetricsReporterPtr metrics_reporter;
    auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
    PrefillGenerateContext       context(&resource, rpc_context, 0, nullptr, metrics_reporter, meta);
    context.stream_        = std::make_shared<SingleOutputStream>();
    context.client_context = std::make_shared<grpc::ClientContext>();
    auto client_stream     = std::make_shared<SingleResponseClientStream>();
    context.client_stream  = client_stream;

    server.pollRemoteOutput(context);

    EXPECT_EQ(client_stream->read_calls, 1);
    EXPECT_EQ(writer.write_calls, 1);
    EXPECT_TRUE(context.cancelled());
    EXPECT_EQ(context.error_status.error_code(), grpc::StatusCode::CANCELLED);
    const auto close_status = context.closeGrpcStream();
    EXPECT_EQ(close_status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(client_stream->writes_done_calls, 1);
    EXPECT_EQ(client_stream->finish_calls, 1);

    context.stream_.reset();
}

// These tests use real local gRPC calls: a closed client must interrupt the
// server even when neither the model nor Decode publishes another packet.
class ScopedCancellationRpcServer {
public:
    explicit ScopedCancellationRpcServer(grpc::Service* service) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service);
        server_ = builder.BuildAndStart();
        if (!server_ || port_ == 0) {
            throw std::runtime_error("failed to start cancellation test RPC server");
        }
    }

    ~ScopedCancellationRpcServer() {
        server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(1));
        server_->Wait();
    }

    std::shared_ptr<grpc::Channel> channel() const {
        return grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
    }

private:
    int                           port_ = 0;
    std::unique_ptr<grpc::Server> server_;
};

bool waitForCancellationFlag(const std::atomic<bool>& flag) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!flag.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return flag.load();
}

class WaitingOutputRpcService: public RpcService::Service {
public:
    WaitingOutputRpcService() {
        auto input                         = std::make_shared<GenerateInput>();
        input->request_id                  = 43;
        input->begin_time_us               = currentTimeUs();
        input->input_ids                   = torch::tensor({1}, torch::kInt32);
        input->generate_config             = std::make_shared<GenerateConfig>();
        input->generate_config->timeout_ms = 10000;
        ModelConfig config;
        config.max_seq_len = 128;
        stream_ = std::make_shared<NormalGenerateStream>(input, config, RuntimeConfig{}, ResourceContext{}, nullptr);
    }

    grpc::Status GenerateStreamCall(grpc::ServerContext* context,
                                    const GenerateInputPB*,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        entered.store(true);
        const auto status = server_.pollStreamOutput(context, "43", writer, stream_);
        returned.store(true);
        return status;
    }

    void cleanup() {
        stream_->reportError(ErrorCode::CANCELLED, "test cleanup");
    }

    bool cancelled() const {
        return stream_->hasError() && stream_->statusInfo().code() == ErrorCode::CANCELLED;
    }

    std::atomic<bool> entered{false};
    std::atomic<bool> returned{false};

private:
    LocalRpcServer                  server_;
    std::shared_ptr<GenerateStream> stream_;
};

class WaitingDecodeRpcService: public RpcService::Service {
public:
    grpc::Status RemoteGenerate(grpc::ServerContext*                                            context,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        GenerateRequestPB request;
        if (!stream->Read(&request) || !stream->Write(GenerateOutputsPB())) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "allocation closed");
        }
        allocated.store(true);
        while (!context->IsCancelled() && !stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cancelled.store(context->IsCancelled());
        return grpc::Status(grpc::StatusCode::CANCELLED, "decode cancelled");
    }

    std::atomic<bool> allocated{false};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> stop{false};
};

class RejectingDecodeRpcService: public RpcService::Service {
public:
    grpc::Status RemoteGenerate(grpc::ServerContext*,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>*) override {
        called.store(true);
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "synthetic allocation rejected");
    }
    std::atomic<bool> called{false};
};

class CompletingDecodeRpcService: public RpcService::Service {
public:
    grpc::Status RemoteGenerate(grpc::ServerContext*,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        GenerateRequestPB request;
        if (!stream->Read(&request) || !stream->Write(GenerateOutputsPB())) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "allocation closed");
        }
        GenerateOutputsPB response;
        response.set_request_id(request.request_id());
        response.mutable_flatten_output()->add_finished(true);
        response.mutable_flatten_output()->add_aux_info()->set_output_len(1);
        return stream->Write(response) ? grpc::Status::OK : grpc::Status(grpc::StatusCode::CANCELLED, "output closed");
    }
};

class WaitingPrefillRpcService: public RpcService::Service {
public:
    explicit WaitingPrefillRpcService(std::shared_ptr<grpc::Channel> channel, bool late_parent = false):
        channel_(std::move(channel)), late_parent_(late_parent) {}

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   server_context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        RPCContext             rpc_context{request, writer};
        auto                   meta = std::make_shared<RpcServerRuntimeMeta>();
        PrefillGenerateContext context(
            &server_.resource(), rpc_context, 10000, late_parent_ ? nullptr : server_context, reporter_, meta);
        context.grpc_connection.channel = channel_;
        context.grpc_connection.stub    = RpcService::NewStub(channel_);
        server_.remoteAllocateResource(context);
        if (late_parent_) {
            // EnqueueGroup creates the child before FetchResponse supplies its parent.
            context.server_context = server_context;
        }
        std::atomic_store(&child_context_, context.client_context);
        child_registered.store(true);
        if (context.hasError()) {
            returned.store(true);
            return context.error_status;
        }
        // Exercise the production forwarding loop, including blocked Read,
        // normal EOF, and cancellation after Fetch supplies its context.
        context.stream_ = std::make_shared<SingleOutputStream>();
        server_.pollRemoteOutput(context);
        const auto status = context.error_status;
        context.stream_.reset();
        returned.store(true);
        return status;
    }

    void cleanup() {
        auto child = std::atomic_load(&child_context_);
        if (child) {
            child->TryCancel();
        }
    }

    std::atomic<bool> returned{false};
    std::atomic<bool> child_registered{false};

private:
    PrefillRpcServer                     server_;
    kmonitor::MetricsReporterPtr         reporter_;
    std::shared_ptr<grpc::Channel>       channel_;
    std::shared_ptr<grpc::ClientContext> child_context_;
    bool                                 late_parent_ = false;
};

TEST(RpcWriterCancellationTest, ClientCancellationInterruptsNonStreamingOutputWait) {
    WaitingOutputRpcService     service;
    ScopedCancellationRpcServer server(&service);
    auto                        stub = RpcService::NewStub(server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    auto       reader  = stub->GenerateStreamCall(&context, GenerateInputPB());
    const bool entered = waitForCancellationFlag(service.entered);
    EXPECT_TRUE(entered);
    if (entered) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    context.TryCancel();
    const bool returned = waitForCancellationFlag(service.returned);
    EXPECT_TRUE(returned) << "cancelled model wait still blocks without output";
    if (returned) {
        EXPECT_TRUE(service.cancelled());
    }
    // Always release the baseline implementation after recording the failure.
    service.cleanup();
    EXPECT_TRUE(waitForCancellationFlag(service.returned));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_TRUE(service.cancelled());
}

TEST(RpcWriterCancellationTest, ClientCancellationInterruptsBlockedDecodeRead) {
    WaitingDecodeRpcService     decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel());
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    GenerateInputPB request;
    request.set_request_id(44);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto reader = stub->GenerateStreamCall(&context, request);
    EXPECT_TRUE(waitForCancellationFlag(decode.allocated));
    EXPECT_TRUE(waitForCancellationFlag(prefill.child_registered));
    context.TryCancel();
    EXPECT_TRUE(waitForCancellationFlag(decode.cancelled)) << "upstream cancellation never reaches Decode";
    // Cleanup is separate from the assertion so the negative control cannot hang.
    prefill.cleanup();
    decode.stop.store(true);
    EXPECT_TRUE(waitForCancellationFlag(prefill.returned));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::CANCELLED);
}
TEST(RpcWriterCancellationTest, FetchCancellationInterruptsPreviouslyAllocatedDecodeRead) {
    WaitingDecodeRpcService     decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel(), true);
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    GenerateInputPB request;
    request.set_request_id(44);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto reader = stub->GenerateStreamCall(&context, request);
    EXPECT_TRUE(waitForCancellationFlag(decode.allocated));
    EXPECT_TRUE(waitForCancellationFlag(prefill.child_registered));
    context.TryCancel();
    EXPECT_TRUE(waitForCancellationFlag(decode.cancelled)) << "upstream cancellation never reaches Decode";
    // Cleanup is separate from the assertion so the negative control cannot hang.
    prefill.cleanup();
    decode.stop.store(true);
    EXPECT_TRUE(waitForCancellationFlag(prefill.returned));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::CANCELLED);
}

TEST(RpcWriterCancellationTest, AllocatedDecodeStreamCompletesNormallyAfterFetchBindsParent) {
    CompletingDecodeRpcService  decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel(), true);
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    GenerateInputPB request;
    request.set_request_id(45);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto              reader = stub->GenerateStreamCall(&context, request);
    GenerateOutputsPB response;
    int               frames = 0;
    while (reader->Read(&response)) {
        ++frames;
        EXPECT_EQ(response.request_id(), 45);
        ASSERT_EQ(response.flatten_output().finished_size(), 1);
        EXPECT_TRUE(response.flatten_output().finished(0));
        ASSERT_EQ(response.flatten_output().aux_info_size(), 1);
        EXPECT_TRUE(response.flatten_output().aux_info(0).pd_sep());
    }
    EXPECT_TRUE(reader->Finish().ok());
    EXPECT_EQ(frames, 1);
    EXPECT_TRUE(prefill.returned.load());
}

TEST(RpcWriterCancellationTest, FetchDeadlineInterruptsPreviouslyAllocatedDecodeRead) {
    WaitingDecodeRpcService     decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel(), true);
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));
    GenerateInputPB request;
    request.set_request_id(46);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto reader = stub->GenerateStreamCall(&context, request);
    EXPECT_TRUE(waitForCancellationFlag(decode.allocated));
    EXPECT_TRUE(waitForCancellationFlag(prefill.child_registered));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(waitForCancellationFlag(decode.cancelled));
    prefill.cleanup();
    decode.stop.store(true);
    EXPECT_TRUE(waitForCancellationFlag(prefill.returned));
}

TEST(RpcWriterCancellationTest, DecodeAllocationFailurePreservesGrpcErrorAndClosesStream) {
    RejectingDecodeRpcService   decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel());
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    GenerateInputPB request;
    request.set_request_id(47);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto       reader = stub->GenerateStreamCall(&context, request);
    const auto status = reader->Finish();
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("synthetic allocation rejected"), std::string::npos);
    EXPECT_TRUE(decode.called.load());
    EXPECT_TRUE(prefill.returned.load());
}

}  // namespace
}  // namespace rtp_llm
