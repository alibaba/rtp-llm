#include <gtest/gtest.h>
#include <memory>

#include <grpcpp/grpcpp.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

#include <torch/torch.h>

using namespace rtp_llm;

namespace {

// WriterInterface implementation that always rejects writes (returns false),
// simulating a closed or cancelled downstream consumer.
class RejectingWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    bool Write(const GenerateOutputsPB& /*outputs*/, grpc::WriteOptions /*options*/) override {
        return false;
    }
};

// Minimal GenerateStream subclass for testing pollStreamOutput.
// Provides one output so the loop body executes; the RejectingWriter then
// triggers the Write-failure code path.
class FakeStream: public GenerateStream {
public:
    explicit FakeStream(const std::shared_ptr<GenerateInput>& input):
        GenerateStream(input, createMockModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput() override {
        GenerateOutputs outputs;
        outputs.request_id = 0;
        return ErrorResult<GenerateOutputs>(std::move(outputs));
    }

    bool hasOutput() override {
        return true;
    }

    void updateOutput(const StreamUpdateInfo&) override {}

private:
    static ModelConfig createMockModelConfig() {
        ModelConfig config;
        config.max_seq_len = 4096;
        return config;
    }
};

// Exposes the protected pollStreamOutput method for testing.
class TestableLocalRpcServer: public LocalRpcServer {
public:
    using LocalRpcServer::pollStreamOutput;
};

std::shared_ptr<GenerateInput> createTestInput() {
    auto config            = std::make_shared<GenerateConfig>();
    auto input             = std::make_shared<GenerateInput>();
    input->request_id      = 1;
    input->generate_config = config;
    input->input_ids       = torch::zeros({1}, torch::kInt32);
    input->begin_time_us   = 0;
    return input;
}

}  // namespace

TEST(RpcWriterCancellationTest, LocalWriteFailureReturnsCancelled) {
    TestableLocalRpcServer server;
    auto                   input  = createTestInput();
    auto                   stream = std::make_shared<FakeStream>(input);
    RejectingWriter        writer;

    // Pass nullptr for context so the IsCancelled check is skipped.
    auto status = server.pollStreamOutput(nullptr, "test_request_key", &writer, stream);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_NE(status.error_message().find("consumer closed"), std::string::npos);
}
