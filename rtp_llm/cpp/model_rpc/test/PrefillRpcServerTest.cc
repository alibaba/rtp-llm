#include <functional>
#include <memory>

#include "gtest/gtest.h"
#define private public
#define protected public
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#undef protected
#undef private
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

class TestMultimodalProcessor: public MultimodalProcessor {
public:
    explicit TestMultimodalProcessor(ErrorCode result_code):
        MultimodalProcessor(py::none(), MMModelConfig{true, {{1}}, false}, 100), result_code_(result_code) {}

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<MultimodalInput> mm_inputs,
                                                      std::string                        ip_port = "") override {
        if (result_code_ != ErrorCode::NONE_ERROR) {
            return ErrorInfo(result_code_, "multimodal test error");
        }
        MultimodalOutput output;
        for (size_t i = 0; i < mm_inputs.size(); ++i) {
            output.mm_features.push_back(torch::zeros({2, 1}));
        }
        return output;
    }

private:
    ErrorCode result_code_;
};

grpc::Status runWithRetry(PrefillGenerateContext&                             context,
                          const std::function<void(PrefillGenerateContext&)>& operation,
                          int                                                 max_retries = 3) {
    constexpr int64_t retry_timeout_ms  = 1000;
    constexpr int64_t retry_interval_ms = 0;
    EXECUTE_WITH_RETRY(operation, context, max_retries, retry_timeout_ms, retry_interval_ms);
    return context.error_status;
}

class PrefillRpcServerTest: public DeviceTestBase {
protected:
    std::shared_ptr<GenerateInput> makeMultimodalInput() {
        auto input               = std::make_shared<GenerateInput>();
        input->input_ids         = torch::tensor({0, 1, 2}, torch::kInt32);
        input->multimodal_inputs = std::vector<MultimodalInput>{MultimodalInput("image")};
        return input;
    }

    std::unique_ptr<PrefillGenerateContext> makeContext(GenerateInputPB* request) {
        rpc_context_ = RPCContext{request, nullptr};
        return std::make_unique<PrefillGenerateContext>(
            &resource_, rpc_context_, 0, &server_context_, metrics_reporter_, nullptr);
    }

protected:
    RemoteServerResource         resource_;
    RPCContext                   rpc_context_;
    grpc::ServerContext          server_context_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
};

TEST_F(PrefillRpcServerTest, deterministicMultimodalErrorsAreNotRetryable) {
    EXPECT_FALSE(PrefillRpcServer::isRetryableMultimodalError(ErrorCode::MM_LONG_PROMPT_ERROR));
    EXPECT_FALSE(PrefillRpcServer::isRetryableMultimodalError(ErrorCode::MM_WRONG_FORMAT_ERROR));
    EXPECT_FALSE(PrefillRpcServer::isRetryableMultimodalError(ErrorCode::MM_NOT_SUPPORTED_ERROR));
}

TEST_F(PrefillRpcServerTest, transientMultimodalErrorsRemainRetryable) {
    EXPECT_TRUE(PrefillRpcServer::isRetryableMultimodalError(ErrorCode::MM_PROCESS_ERROR));
    EXPECT_TRUE(PrefillRpcServer::isRetryableMultimodalError(ErrorCode::MM_EMPTY_ENGINE_ERROR));
    EXPECT_TRUE(PrefillRpcServer::isRetryableMultimodalError(ErrorCode::MM_DOWNLOAD_FAILED));
}

TEST_F(PrefillRpcServerTest, multimodalProcessMarksDeterministicErrorNonRetryable) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();

    PrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(ErrorCode::MM_WRONG_FORMAT_ERROR);
    int  call_count      = 0;
    auto operation       = [&](PrefillGenerateContext& retry_context) {
        ++call_count;
        server.multimodalProcess(retry_context);
    };
    auto status = runWithRetry(*context, operation);

    EXPECT_EQ(call_count, 1);
    EXPECT_EQ(context->retry_times, 1);
    EXPECT_TRUE(context->hasError());
    EXPECT_FALSE(context->shouldRetry());
    EXPECT_EQ(context->error_info.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_NE(status.error_message().find("multimodal test error"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, multimodalProcessKeepsTransientErrorRetryable) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();

    PrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(ErrorCode::MM_PROCESS_ERROR);
    server.multimodalProcess(*context);

    EXPECT_TRUE(context->hasError());
    EXPECT_TRUE(context->shouldRetry());
    EXPECT_EQ(context->error_info.code(), ErrorCode::MM_PROCESS_ERROR);
}

TEST_F(PrefillRpcServerTest, multimodalProcessDoesNotMutateOriginalRequest) {
    GenerateInputPB request;
    request.set_request_id(1);
    request.add_token_ids(0);
    request.add_token_ids(1);
    request.add_token_ids(2);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();

    PrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(ErrorCode::NONE_ERROR);
    server.multimodalProcess(*context);

    ASSERT_FALSE(context->hasError());
    ASSERT_EQ(request.token_ids_size(), 3);
    EXPECT_EQ(request.token_ids(0), 0);
    EXPECT_EQ(request.token_ids(1), 1);
    EXPECT_EQ(request.token_ids(2), 2);
    EXPECT_GT(context->generate_input->input_ids.numel(), request.token_ids_size());
}

}  // namespace rtp_llm
