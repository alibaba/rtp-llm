#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"

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

TEST(RpcWriterCancellationTest, InvalidInputsReturnStructuredErrorsWithoutAnEngine) {
    LocalRpcServer   local;
    PrefillRpcServer prefill;
    for (const bool use_prefill : {false, true}) {
        for (const int tokens : {-1, 0, 1}) {
            SCOPED_TRACE(use_prefill);
            SCOPED_TRACE(tokens);
            GenerateInputPB request;
            request.set_request_id(40);
            request.add_token_ids(1);
            auto* config = request.mutable_generate_config();
            config->set_max_new_tokens(tokens);
            config->set_prefill_only(tokens == 1);
            config->set_can_use_pd_separation(true);
            const auto status = use_prefill ? prefill.GenerateStreamCall(nullptr, &request, nullptr) :
                                              local.GenerateStreamCall(nullptr, &request, nullptr);
            EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
            ErrorDetailsPB details;
            ASSERT_TRUE(details.ParseFromString(status.error_details()));
            EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
            EXPECT_NE(details.error_message().find(tokens == 0 ? "writer" : "max_new_tokens"), std::string::npos);
        }
        const auto status = use_prefill ? prefill.GenerateStreamCall(nullptr, nullptr, nullptr) :
                                          local.GenerateStreamCall(nullptr, nullptr, nullptr);
        EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }
    EXPECT_EQ(local.onflightRequestNum(), 0u);
    EXPECT_EQ(prefill.onflightRequestNum(), 0u);
}

TEST(RpcWriterCancellationTest, DecodeRejectsPrefillBeforeAllocatingResources) {
    DecodeRpcServer              server;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr metrics_reporter;
    auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
    for (const bool explicit_prefill : {false, true}) {
        DecodeGenerateContext context(rpc_context, 0, nullptr, metrics_reporter, meta);
        auto*                 request = context.allocate_request.mutable_input();
        request->add_token_ids(1);
        request->mutable_generate_config()->set_prefill_only(explicit_prefill);
        EXPECT_THROW(server.allocateResource(context), std::invalid_argument);
        EXPECT_EQ(context.getStream(), nullptr);
    }
}

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

}  // namespace
}  // namespace rtp_llm
