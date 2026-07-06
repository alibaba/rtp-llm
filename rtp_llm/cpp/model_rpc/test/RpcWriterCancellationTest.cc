#include <gtest/gtest.h>

#include <memory>

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
