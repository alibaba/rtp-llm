#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

class TestDecodeRpcService final: public RpcService::Service {
public:
    explicit TestDecodeRpcService(bool fail_first_allocate):
        first_allocate_failure_(fail_first_allocate ? std::optional<grpc::Status>(grpc::Status(
                                                          grpc::StatusCode::INTERNAL, "allocate failed once")) :
                                                      std::nullopt) {}

    explicit TestDecodeRpcService(grpc::Status first_allocate_failure):
        first_allocate_failure_(std::move(first_allocate_failure)) {}

    grpc::Status RemoteGenerate(grpc::ServerContext*,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        GenerateRequestPB request;
        if (!stream->Read(&request)) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "missing allocate request");
        }
        ++allocate_count_;
        if (first_allocate_failure_.has_value() && allocate_count_ == 1) {
            return *first_allocate_failure_;
        }

        GenerateOutputsPB response;
        if (!stream->Write(response)) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "write allocate response failed");
        }
        while (stream->Read(&request)) {}
        return grpc::Status::OK;
    }

    int allocateCount() const {
        return allocate_count_.load();
    }

private:
    std::optional<grpc::Status> first_allocate_failure_;
    std::atomic<int>            allocate_count_{0};
};

class TestDecodeRpcServer {
public:
    explicit TestDecodeRpcServer(bool fail_first_allocate): service_(fail_first_allocate) {}
    explicit TestDecodeRpcServer(grpc::Status first_allocate_failure): service_(std::move(first_allocate_failure)) {}
    ~TestDecodeRpcServer() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("0.0.0.0:0", grpc::InsecureServerCredentials(), &listen_port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        return server_ != nullptr && listen_port_ != 0;
    }

    int listenPort() const {
        return listen_port_;
    }

    int allocateCount() const {
        return service_.allocateCount();
    }

private:
    TestDecodeRpcService          service_;
    std::unique_ptr<grpc::Server> server_;
    int                           listen_port_{0};
};

class TestMultimodalProcessor: public MultimodalProcessor {
public:
    explicit TestMultimodalProcessor(ErrorCode result_code):
        TestMultimodalProcessor(std::vector<ErrorCode>{result_code}) {}

    explicit TestMultimodalProcessor(std::vector<ErrorCode> result_codes):
        MultimodalProcessor(py::none(), MMModelConfig{true, {{1}}, false}, 100),
        result_codes_(std::move(result_codes)) {}

    int callCount() const {
        return call_count_;
    }

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<MultimodalInput> mm_inputs,
                                                      std::string                        ip_port = "") override {
        const auto result_code = result_codes_[std::min<size_t>(call_count_, result_codes_.size() - 1)];
        ++call_count_;
        if (result_code != ErrorCode::NONE_ERROR) {
            return ErrorInfo(result_code, "multimodal test error");
        }
        MultimodalOutput output;
        for (size_t i = 0; i < mm_inputs.size(); ++i) {
            output.mm_features.push_back(torch::zeros({2, 1}));
        }
        return output;
    }

private:
    std::vector<ErrorCode> result_codes_;
    int                    call_count_ = 0;
};

class TestEngineBase final: public EngineBase {
public:
    explicit TestEngineBase(bool is_mtp_eagle): EngineBase(EngineInitParams()), is_mtp_eagle_(is_mtp_eagle) {}

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(std::shared_ptr<GenerateStream>&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("not used by PrefillRpcServerTest");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
    bool isMTPEagle() override {
        return is_mtp_eagle_;
    }

private:
    bool is_mtp_eagle_;
};

class TestPrefillRpcServer: public PrefillRpcServer {
public:
    grpc::Status runWithRetry(PrefillGenerateContext&                             context,
                              const std::function<void(PrefillGenerateContext&)>& operation,
                              int                                                 max_retries       = 3,
                              int64_t                                             retry_timeout_ms  = 0,
                              int64_t                                             retry_interval_ms = 0) {
        EXECUTE_WITH_RETRY(operation, context, max_retries, retry_timeout_ms, retry_interval_ms);
        return context.error_status;
    }

    void setProcessIdForTest(std::string process_id) {
        process_id_ = process_id;
    }

    void setEngineForTest(bool is_mtp_eagle) {
        engine_ = std::make_shared<TestEngineBase>(is_mtp_eagle);
    }

    void prepareGenerateInputForTest(PrefillGenerateContext& context) {
        prepareGenerateInput(context);
    }

    void remoteAllocateResourceForTest(PrefillGenerateContext& context) {
        remoteAllocateResource(context);
    }

    void setContextErrorForTest(PrefillGenerateContext& context, const ErrorInfo& error_info) {
        setContextError(context, error_info);
    }

    std::chrono::system_clock::time_point decodeChannelReadyDeadlineForTest(const PrefillGenerateContext& context,
                                                                            int64_t max_rpc_timeout_ms = 0) const {
        return decodeChannelReadyDeadline(context, max_rpc_timeout_ms);
    }

    std::optional<ErrorInfo> parseDownstreamErrorForTest(const grpc::Status& status) const {
        return parseDownstreamError(status);
    }

    ErrorInfo waitStreamBeforeRunForTest(const std::shared_ptr<GenerateStream>& stream) {
        return waitStreamBeforeRun(stream);
    }

    void setMaxRpcTimeoutForTest(int64_t timeout_ms) {
        maga_init_params_.pd_sep_config.max_rpc_timeout_ms = timeout_ms;
    }

    void setPrefillMaxWaitTimeoutForTest(int64_t timeout_ms) {
        maga_init_params_.pd_sep_config.prefill_max_wait_timeout_ms = timeout_ms;
    }
};

class PrefillRpcServerTest: public DeviceTestBase {
protected:
    std::shared_ptr<GenerateInput> makeMultimodalInput() {
        auto input               = std::make_shared<GenerateInput>();
        input->generate_config   = std::make_shared<GenerateConfig>();
        input->input_ids         = torch::tensor({0, 1, 2}, torch::kInt32);
        input->multimodal_inputs = std::vector<MultimodalInput>{MultimodalInput("image")};
        return input;
    }

    std::shared_ptr<GenerateStream> makeWaitingStream() {
        auto input             = std::make_shared<GenerateInput>();
        input->generate_config = std::make_shared<GenerateConfig>();
        input->begin_time_us   = currentTimeUs();
        input->input_ids       = torch::tensor({0, 1, 2}, torch::kInt32);

        ModelConfig model_config;
        model_config.max_seq_len = 2048;
        model_config.vocab_size  = 1024;
        return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
    }

    std::unique_ptr<PrefillGenerateContext> makeContext(GenerateInputPB* request, int64_t timeout_ms = 0) {
        rpc_context_ = RPCContext{request, nullptr};
        return std::make_unique<PrefillGenerateContext>(
            &resource_, rpc_context_, timeout_ms, &server_context_, metrics_reporter_, nullptr);
    }

protected:
    RemoteServerResource         resource_;
    RPCContext                   rpc_context_;
    grpc::ServerContext          server_context_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
};

TEST_F(PrefillRpcServerTest, waitStreamBeforeRunUsesEachServerTimeout) {
    TestPrefillRpcServer first_server;
    first_server.setPrefillMaxWaitTimeoutForTest(1);
    auto first_error = first_server.waitStreamBeforeRunForTest(makeWaitingStream());
    EXPECT_EQ(first_error.code(), ErrorCode::WAIT_TO_RUN_TIMEOUT);
    EXPECT_NE(first_error.ToString().find("1000 us"), std::string::npos);

    TestPrefillRpcServer second_server;
    second_server.setPrefillMaxWaitTimeoutForTest(7);
    auto second_error = second_server.waitStreamBeforeRunForTest(makeWaitingStream());
    EXPECT_EQ(second_error.code(), ErrorCode::WAIT_TO_RUN_TIMEOUT);
    EXPECT_NE(second_error.ToString().find("7000 us"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, GenerateStreamCallRejectsNullRequestBeforeDereference) {
    TestPrefillRpcServer server;
    grpc::ServerContext  context;

    const auto status = server.GenerateStreamCall(&context, nullptr, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_NE(error_details.error_message().find("prefill generate request must not be null"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, explicitPrefillOnlyRejectsPositiveMaxNewTokensBeforeRouting) {
    GenerateInputPB request;
    request.set_request_id(1);
    request.add_token_ids(0);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(2);
    config->set_prefill_only(true);
    config->set_can_use_pd_separation(true);

    TestPrefillRpcServer server;
    server.setEngineForTest(/*is_mtp_eagle=*/false);
    grpc::ServerContext context;
    const auto          status = server.GenerateStreamCall(&context, &request, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_NE(error_details.error_message().find("prefill_only"), std::string::npos);
    EXPECT_NE(error_details.error_message().find("max_new_tokens"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, explicitPrefillOnlyRejectsReturnPromptLogitsBeforeRouting) {
    GenerateInputPB request;
    request.set_request_id(2);
    request.add_token_ids(0);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(0);
    config->set_prefill_only(true);
    config->set_return_prompt_logits(true);
    config->set_can_use_pd_separation(true);

    TestPrefillRpcServer server;
    grpc::ServerContext  context;
    const auto           status = server.GenerateStreamCall(&context, &request, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_NE(error_details.error_message().find("prefill_only"), std::string::npos);
    EXPECT_NE(error_details.error_message().find("return_prompt_logits"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, rawGrpcPromptScoringWithPositiveMaxNewTokensRoutesLocally) {
    GenerateInputPB request;
    request.set_request_id(3);
    request.add_token_ids(0);
    request.add_multimodal_inputs()->set_multimodal_url("image");
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_return_prompt_logits(true);
    config->set_is_streaming(true);
    config->set_reuse_cache(true);
    config->set_can_use_pd_separation(true);
    ASSERT_FALSE(config->prefill_only());
    ASSERT_EQ(QueryConverter::resolveMaxNewTokens(*config), 1);

    TestPrefillRpcServer server;
    server.mm_processor_          = std::make_shared<TestMultimodalProcessor>(ErrorCode::MM_WRONG_FORMAT_ERROR);
    auto                processor = std::static_pointer_cast<TestMultimodalProcessor>(server.mm_processor_);
    grpc::ServerContext context;
    const auto          status = server.GenerateStreamCall(&context, &request, nullptr);

    EXPECT_EQ(processor->callCount(), 1);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::MM_WRONG_FORMAT_ERROR));
    EXPECT_NE(error_details.error_message().find("multimodal test error"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, bareWireZeroRoutesToLocalLegacyPrefill) {
    GenerateInputPB request;
    request.set_request_id(101);
    request.add_token_ids(0);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(0);
    config->set_prefill_only(false);
    config->set_can_use_pd_separation(true);
    ASSERT_EQ(QueryConverter::resolveMaxNewTokens(*config), 0);

    std::string serialized_request;
    ASSERT_TRUE(request.SerializeToString(&serialized_request));
    GenerateInputPB wire_request;
    ASSERT_TRUE(wire_request.ParseFromString(serialized_request));
    ASSERT_EQ(wire_request.generate_config().max_new_tokens(), 0);
    ASSERT_FALSE(wire_request.generate_config().prefill_only());
    ASSERT_TRUE(wire_request.generate_config().can_use_pd_separation());

    wire_request.add_multimodal_inputs()->set_multimodal_url("image");
    TestPrefillRpcServer server;
    server.mm_processor_          = std::make_shared<TestMultimodalProcessor>(ErrorCode::MM_WRONG_FORMAT_ERROR);
    auto                processor = std::static_pointer_cast<TestMultimodalProcessor>(server.mm_processor_);
    grpc::ServerContext context;
    const auto          status = server.GenerateStreamCall(&context, &wire_request, nullptr);

    EXPECT_EQ(processor->callCount(), 1);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::MM_WRONG_FORMAT_ERROR));
    EXPECT_NE(error_details.error_message().find("multimodal test error"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, flaggedWireZeroRoutesToLocalPrefillOnly) {
    GenerateInputPB request;
    request.set_request_id(102);
    request.add_token_ids(0);
    request.add_token_ids(1);
    request.add_token_ids(2);
    request.add_multimodal_inputs()->set_multimodal_url("image");
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(0);
    config->set_prefill_only(true);
    config->set_can_use_pd_separation(true);

    std::string serialized_request;
    ASSERT_TRUE(request.SerializeToString(&serialized_request));
    GenerateInputPB wire_request;
    ASSERT_TRUE(wire_request.ParseFromString(serialized_request));
    ASSERT_EQ(wire_request.generate_config().max_new_tokens(), 0);
    ASSERT_TRUE(wire_request.generate_config().prefill_only());
    ASSERT_TRUE(wire_request.generate_config().can_use_pd_separation());

    TestPrefillRpcServer server;
    server.mm_processor_          = std::make_shared<TestMultimodalProcessor>(ErrorCode::MM_WRONG_FORMAT_ERROR);
    auto                processor = std::static_pointer_cast<TestMultimodalProcessor>(server.mm_processor_);
    grpc::ServerContext context;
    const auto          status = server.GenerateStreamCall(&context, &wire_request, nullptr);

    EXPECT_EQ(processor->callCount(), 1);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::MM_WRONG_FORMAT_ERROR));
    EXPECT_NE(error_details.error_message().find("multimodal test error"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, prepareAllocateResourceRetriesDecodeWithoutRepeatingMultimodalProcessing) {
    TestDecodeRpcServer decode_server(/*fail_first_allocate=*/true);
    ASSERT_TRUE(decode_server.start());

    GenerateInputPB request;
    request.set_request_id(1);
    request.add_token_ids(0);
    request.add_token_ids(1);
    request.add_token_ids(2);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();
    context->generate_input->generate_config->role_addrs.emplace_back(
        RoleType::DECODE, "127.0.0.1", 0, decode_server.listenPort());

    TestPrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(ErrorCode::NONE_ERROR);
    auto processor       = std::static_pointer_cast<TestMultimodalProcessor>(server.mm_processor_);
    auto operation = [&](PrefillGenerateContext& retry_context) { server.prepareAllocateResource(retry_context); };

    auto status = server.runWithRetry(*context, operation, 1);

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(context->retry_times, 2);
    EXPECT_EQ(decode_server.allocateCount(), 2);
    EXPECT_EQ(processor->callCount(), 1);
    EXPECT_TRUE(context->multimodalProcessed());
    EXPECT_TRUE(context->tokenIdsExpanded());
    EXPECT_TRUE(context->closeGrpcStream().ok());
}

TEST_F(PrefillRpcServerTest, decodeReadinessUsesSubHundredMillisecondRemainingBudget) {
    GenerateInputPB request;
    request.set_request_id(2);
    auto context = makeContext(&request, /*timeout_ms=*/40);

    TestPrefillRpcServer server;
    const auto           before    = std::chrono::system_clock::now();
    const auto           deadline  = server.decodeChannelReadyDeadlineForTest(*context);
    const auto           remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - before).count();

    EXPECT_GT(remaining, 0);
    EXPECT_LE(remaining, 40);
    EXPECT_LT(remaining, 100);
    ASSERT_TRUE(context->request_deadline.has_value());
    EXPECT_EQ(deadline, *context->request_deadline);
}

TEST_F(PrefillRpcServerTest, decodeReadinessWithoutConfiguredBudgetsUsesSafetyCap) {
    GenerateInputPB request;
    request.set_request_id(7);
    auto context = makeContext(&request);

    TestPrefillRpcServer server;
    const auto           before    = std::chrono::system_clock::now();
    const auto           deadline  = server.decodeChannelReadyDeadlineForTest(*context);
    const auto           remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - before).count();

    EXPECT_GE(remaining, 14900);
    EXPECT_LE(remaining, 15000);
}

TEST_F(PrefillRpcServerTest, decodeReadinessUsesTightestRetryAndRpcBudgets) {
    GenerateInputPB request;
    request.set_request_id(7);
    auto context = makeContext(&request, /*timeout_ms=*/500);
    context->setRetryTimeoutMs(80);

    TestPrefillRpcServer server;
    const auto           before    = std::chrono::system_clock::now();
    const auto           deadline  = server.decodeChannelReadyDeadlineForTest(*context, /*max_rpc_timeout_ms=*/200);
    const auto           remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - before).count();

    EXPECT_GT(remaining, 0);
    EXPECT_LE(remaining, 80);
    ASSERT_TRUE(context->retry_deadline.has_value());
    EXPECT_EQ(deadline, *context->retry_deadline);

    context->setRetryTimeoutMs(400);
    const auto rpc_before    = std::chrono::system_clock::now();
    const auto rpc_deadline  = server.decodeChannelReadyDeadlineForTest(*context, /*max_rpc_timeout_ms=*/30);
    const auto rpc_remaining = std::chrono::duration_cast<std::chrono::milliseconds>(rpc_deadline - rpc_before).count();
    EXPECT_GT(rpc_remaining, 0);
    EXPECT_LE(rpc_remaining, 30);
}

TEST_F(PrefillRpcServerTest, retriesReuseOneAbsoluteReadinessDeadline) {
    GenerateInputPB request;
    request.set_request_id(8);
    auto context = makeContext(&request);

    TestPrefillRpcServer                               server;
    std::vector<std::chrono::system_clock::time_point> observed_deadlines;
    auto                                               operation = [&](PrefillGenerateContext& retry_context) {
        observed_deadlines.push_back(server.decodeChannelReadyDeadlineForTest(retry_context));
        if (observed_deadlines.size() == 1) {
            server.setContextErrorForTest(retry_context,
                                          ErrorInfo(ErrorCode::GET_CONNECTION_FAILED, "retryable failure"));
        }
    };

    EXPECT_TRUE(server
                    .runWithRetry(*context,
                                  operation,
                                  /*max_retries=*/1,
                                  /*retry_timeout_ms=*/100,
                                  /*retry_interval_ms=*/0)
                    .ok());
    ASSERT_TRUE(context->retry_deadline.has_value());
    ASSERT_EQ(observed_deadlines.size(), 2);
    EXPECT_EQ(observed_deadlines[0], *context->retry_deadline);
    EXPECT_EQ(observed_deadlines[1], *context->retry_deadline);
}

TEST_F(PrefillRpcServerTest, downstreamRetriesReuseOriginalAbsoluteDeadline) {
    TestDecodeRpcServer decode_server(/*fail_first_allocate=*/true);
    ASSERT_TRUE(decode_server.start());

    GenerateInputPB request;
    request.set_request_id(3);
    auto context    = makeContext(&request, /*timeout_ms=*/1000);
    auto connection = resource_.rpc_pool.getReadyConnection("127.0.0.1:" + std::to_string(decode_server.listenPort()),
                                                            std::chrono::seconds(1));
    ASSERT_TRUE(connection.ok()) << connection.status();
    context->grpc_connection = *connection;

    TestPrefillRpcServer server;
    server.setProcessIdForTest("prefill-client");
    server.remoteAllocateResourceForTest(*context);
    ASSERT_TRUE(context->hasError());
    ASSERT_NE(context->client_context, nullptr);
    const auto first_deadline = context->client_context->deadline();

    context->reset();
    server.remoteAllocateResourceForTest(*context);
    ASSERT_FALSE(context->hasError());
    ASSERT_NE(context->client_context, nullptr);
    const auto second_deadline = context->client_context->deadline();

    ASSERT_TRUE(context->request_deadline.has_value());
    EXPECT_EQ(first_deadline, *context->request_deadline);
    EXPECT_EQ(second_deadline, *context->request_deadline);
    EXPECT_EQ(decode_server.allocateCount(), 2);
    EXPECT_TRUE(context->closeGrpcStream().ok());
}

TEST_F(PrefillRpcServerTest, downstreamWithoutRequestDeadlineKeepsMaxRpcTimeout) {
    TestDecodeRpcServer decode_server(/*fail_first_allocate=*/false);
    ASSERT_TRUE(decode_server.start());

    GenerateInputPB request;
    request.set_request_id(4);
    auto context    = makeContext(&request);
    auto connection = resource_.rpc_pool.getReadyConnection("127.0.0.1:" + std::to_string(decode_server.listenPort()),
                                                            std::chrono::seconds(1));
    ASSERT_TRUE(connection.ok()) << connection.status();
    context->grpc_connection = *connection;

    TestPrefillRpcServer server;
    server.setProcessIdForTest("prefill-client");
    server.setMaxRpcTimeoutForTest(250);
    const auto before = std::chrono::system_clock::now();
    server.remoteAllocateResourceForTest(*context);
    ASSERT_FALSE(context->hasError());
    ASSERT_NE(context->client_context, nullptr);
    const auto timeout_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(context->client_context->deadline() - before).count();

    EXPECT_GE(timeout_ms, 150);
    EXPECT_LE(timeout_ms, 300);
    EXPECT_TRUE(context->closeGrpcStream().ok());
}

TEST_F(PrefillRpcServerTest, downstreamClientContextIgnoresAllocationRetryDeadline) {
    TestDecodeRpcServer decode_server(/*fail_first_allocate=*/false);
    ASSERT_TRUE(decode_server.start());

    GenerateInputPB request;
    request.set_request_id(10);
    auto context    = makeContext(&request, /*timeout_ms=*/20000);
    auto connection = resource_.rpc_pool.getReadyConnection("127.0.0.1:" + std::to_string(decode_server.listenPort()),
                                                            std::chrono::seconds(1));
    ASSERT_TRUE(connection.ok()) << connection.status();
    context->grpc_connection = *connection;
    context->setRetryTimeoutMs(5000);

    TestPrefillRpcServer server;
    server.setProcessIdForTest("prefill-client");
    server.setMaxRpcTimeoutForTest(0);
    server.remoteAllocateResourceForTest(*context);

    ASSERT_FALSE(context->hasError());
    ASSERT_TRUE(context->request_deadline.has_value());
    ASSERT_TRUE(context->retry_deadline.has_value());
    ASSERT_NE(context->client_context, nullptr);
    EXPECT_EQ(context->client_context->deadline(), *context->request_deadline);
    EXPECT_GT(context->client_context->deadline(), *context->retry_deadline);
    EXPECT_TRUE(context->closeGrpcStream().ok());
}

TEST_F(PrefillRpcServerTest, downstreamDomainErrorOverridesTransportFallback) {
    ErrorDetailsPB details;
    details.set_error_code(static_cast<int64_t>(ErrorCode::GRAMMAR_COMPILE_OVERLOADED));
    details.set_error_message("grammar compilation capacity exhausted");
    std::string serialized_details;
    ASSERT_TRUE(details.SerializeToString(&serialized_details));
    TestDecodeRpcServer decode_server(
        grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "generic resource exhausted", serialized_details));
    ASSERT_TRUE(decode_server.start());

    GenerateInputPB request;
    request.set_request_id(9);
    auto context    = makeContext(&request);
    auto connection = resource_.rpc_pool.getReadyConnection("127.0.0.1:" + std::to_string(decode_server.listenPort()),
                                                            std::chrono::seconds(1));
    ASSERT_TRUE(connection.ok()) << connection.status();
    context->grpc_connection = *connection;

    TestPrefillRpcServer server;
    server.setProcessIdForTest("prefill-client");
    server.remoteAllocateResourceForTest(*context);

    ASSERT_TRUE(context->hasError());
    EXPECT_EQ(context->error_info.code(), ErrorCode::GRAMMAR_COMPILE_OVERLOADED);
    EXPECT_EQ(context->error_status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_NE(context->error_info.ToString().find("grammar compilation capacity exhausted"), std::string::npos);
    EXPECT_EQ(context->error_info.ToString().find("decode addr"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, malformedDownstreamDetailsUseTransportFallback) {
    TestPrefillRpcServer server;
    const grpc::Status   status(grpc::StatusCode::RESOURCE_EXHAUSTED, "generic resource exhausted", "not-a-proto");

    EXPECT_FALSE(server.parseDownstreamErrorForTest(status).has_value());
}

TEST_F(PrefillRpcServerTest, exhaustedOrCancelledRequestStartsNoDownstreamRpc) {
    GenerateInputPB request;
    request.set_request_id(5);

    TestPrefillRpcServer server;
    auto                 expired = makeContext(&request, /*timeout_ms=*/1000);
    expired->request_deadline    = std::chrono::system_clock::now() - std::chrono::milliseconds(1);
    server.remoteAllocateResourceForTest(*expired);
    EXPECT_EQ(expired->error_info.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(expired->client_context, nullptr);

    auto cancelled = makeContext(&request, /*timeout_ms=*/1000);
    cancelled->cancel_state->store(true);
    server.remoteAllocateResourceForTest(*cancelled);
    EXPECT_EQ(cancelled->error_info.code(), ErrorCode::CANCELLED);
    EXPECT_EQ(cancelled->client_context, nullptr);
}

TEST_F(PrefillRpcServerTest, retryStopsAtExpiredRequestDeadlineWithoutAnotherAttempt) {
    GenerateInputPB request;
    request.set_request_id(6);
    auto context = makeContext(&request);

    TestPrefillRpcServer server;
    int                  attempts  = 0;
    auto                 operation = [&attempts, &server](PrefillGenerateContext& retry_context) {
        ++attempts;
        retry_context.request_deadline = std::chrono::system_clock::now() - std::chrono::milliseconds(1);
        server.setContextErrorForTest(retry_context, ErrorInfo(ErrorCode::GET_CONNECTION_FAILED, "retryable failure"));
    };

    auto status =
        server.runWithRetry(*context, operation, /*max_retries=*/10, /*retry_timeout_ms=*/0, /*retry_interval_ms=*/200);

    EXPECT_EQ(attempts, 1);
    EXPECT_EQ(context->error_info.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_FALSE(status.ok());
}

TEST_F(PrefillRpcServerTest, retryStopsAtExpiredRetryDeadlineWithoutAnotherAttempt) {
    GenerateInputPB request;
    request.set_request_id(11);
    auto context = makeContext(&request);

    TestPrefillRpcServer server;
    int                  attempts  = 0;
    auto                 operation = [&attempts, &server](PrefillGenerateContext& retry_context) {
        ++attempts;
        retry_context.retry_deadline = std::chrono::system_clock::now() - std::chrono::milliseconds(1);
        server.setContextErrorForTest(retry_context, ErrorInfo(ErrorCode::GET_CONNECTION_FAILED, "retryable failure"));
    };

    auto status = server.runWithRetry(
        *context, operation, /*max_retries=*/10, /*retry_timeout_ms=*/30, /*retry_interval_ms=*/200);

    EXPECT_EQ(attempts, 1);
    EXPECT_EQ(context->error_info.code(), ErrorCode::GET_CONNECTION_FAILED);
    EXPECT_FALSE(status.ok());
}

TEST_F(PrefillRpcServerTest, retrySleepSaturatesOverflowingInterval) {
    GenerateInputPB request;
    request.set_request_id(12);
    auto context = makeContext(&request);

    EXPECT_EQ(context->cappedRetrySleepUs(std::numeric_limits<int64_t>::max()), std::numeric_limits<int64_t>::max());
}

TEST_F(PrefillRpcServerTest, mergeMultimodalLengthsUsesPrefillMetadata) {
    GenerateOutputsPB response;
    auto*             first_aux_info                   = response.mutable_flatten_output()->add_aux_info();
    auto*             second_aux_info                  = response.mutable_flatten_output()->add_aux_info();
    (*first_aux_info->mutable_multimodal_lengths())[9] = 1;

    PrefillRpcServer::mergeMultimodalLengths(response, {{0, 2752}, {1, 64}});

    ASSERT_EQ(first_aux_info->multimodal_lengths_size(), 2);
    EXPECT_EQ(first_aux_info->multimodal_lengths().at(0), 2752);
    EXPECT_EQ(first_aux_info->multimodal_lengths().at(1), 64);
    ASSERT_EQ(second_aux_info->multimodal_lengths_size(), 2);
    EXPECT_EQ(second_aux_info->multimodal_lengths().at(0), 2752);
    EXPECT_EQ(second_aux_info->multimodal_lengths().at(1), 64);
}

TEST_F(PrefillRpcServerTest, mergeCacheReuseInfoReportsCompletedDecodeHandoffForColdPrefill) {
    AuxInfoPB aux_info;
    aux_info.set_total_reuse_len(1280);
    aux_info.set_local_reuse_len(1280);

    PrefillRpcServer::mergeCacheReuseInfo(aux_info, 0, 0, 0, 0, /*use_independent_block_pools=*/true);

    EXPECT_EQ(aux_info.total_reuse_len(), 1280);
    EXPECT_EQ(aux_info.prefill_total_reuse_len(), 0);
    EXPECT_EQ(aux_info.decode_total_reuse_len(), 1280);
    EXPECT_EQ(aux_info.decode_local_reuse_len(), 1280);
}

TEST_F(PrefillRpcServerTest, mergeCacheReuseInfoKeepsLongerPrefillPrefixWithoutAddingPhases) {
    AuxInfoPB aux_info;
    aux_info.set_total_reuse_len(1280);
    aux_info.set_local_reuse_len(1280);

    PrefillRpcServer::mergeCacheReuseInfo(aux_info, 1536, 1536, 0, 1536, /*use_independent_block_pools=*/true);

    EXPECT_EQ(aux_info.total_reuse_len(), 1536);
    EXPECT_EQ(aux_info.memory_reuse_len(), 1536);
    EXPECT_EQ(aux_info.prefill_total_reuse_len(), 1536);
    EXPECT_EQ(aux_info.decode_total_reuse_len(), 1280);
}

TEST_F(PrefillRpcServerTest, mergeCacheReuseInfoPrefersPrefillAttributionOnEqualPrefix) {
    AuxInfoPB aux_info;
    aux_info.set_total_reuse_len(512);
    aux_info.set_local_reuse_len(512);

    PrefillRpcServer::mergeCacheReuseInfo(aux_info, 512, 512, 0, 512, /*use_independent_block_pools=*/true);

    EXPECT_EQ(aux_info.total_reuse_len(), 512);
    EXPECT_EQ(aux_info.memory_reuse_len(), 512);
    EXPECT_EQ(aux_info.decode_memory_reuse_len(), 0);
}

TEST_F(PrefillRpcServerTest, mergeCacheReuseInfoKeepsSharedPoolTopLevelPrefillOnly) {
    AuxInfoPB aux_info;
    aux_info.set_total_reuse_len(1280);
    aux_info.set_local_reuse_len(1280);

    PrefillRpcServer::mergeCacheReuseInfo(aux_info, 0, 0, 0, 0, /*use_independent_block_pools=*/false);

    EXPECT_EQ(aux_info.total_reuse_len(), 0);
    EXPECT_EQ(aux_info.prefill_total_reuse_len(), 0);
    EXPECT_EQ(aux_info.decode_total_reuse_len(), 1280);
}

TEST_F(PrefillRpcServerTest, multimodalProcessMarksDeterministicErrorNonRetryable) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();

    TestPrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(ErrorCode::MM_WRONG_FORMAT_ERROR);
    int  call_count      = 0;
    auto operation       = [&](PrefillGenerateContext& retry_context) {
        ++call_count;
        server.multimodalProcess(retry_context);
    };
    auto status = server.runWithRetry(*context, operation);

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

    TestPrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(ErrorCode::MM_REMOTE_RPC_FAILED);
    auto processor       = std::static_pointer_cast<TestMultimodalProcessor>(server.mm_processor_);
    auto operation       = [&](PrefillGenerateContext& retry_context) {
        if (!retry_context.generate_input) {
            retry_context.generate_input = makeMultimodalInput();
        }
        server.multimodalProcess(retry_context);
    };
    auto status = server.runWithRetry(*context, operation, 2);

    EXPECT_EQ(processor->callCount(), 3);
    EXPECT_EQ(context->retry_times, 3);
    EXPECT_TRUE(context->hasError());
    EXPECT_TRUE(context->shouldRetry());
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(context->error_info.code(), ErrorCode::MM_REMOTE_RPC_FAILED);
}

TEST_F(PrefillRpcServerTest, multimodalProcessRejectsMissingPreparedInput) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context = makeContext(&request);

    PrefillRpcServer server;
    EXPECT_THROW(server.multimodalProcess(*context), std::runtime_error);
}

TEST_F(PrefillRpcServerTest, multimodalProcessRejectsMissingProcessor) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();

    PrefillRpcServer server;
    server.multimodalProcess(*context);

    EXPECT_TRUE(context->hasError());
    EXPECT_FALSE(context->shouldRetry());
    EXPECT_EQ(context->error_info.code(), ErrorCode::MM_NOT_SUPPORTED_ERROR);
    EXPECT_NE(context->error_status.error_message().find("multimodal inputs require a configured multimodal processor"),
              std::string::npos);
    EXPECT_FALSE(context->multimodalProcessed());
    EXPECT_FALSE(context->tokenIdsExpanded());
}

TEST_F(PrefillRpcServerTest, textOnlyProcessIsRetainedAcrossRetryReset) {
    GenerateInputPB request;
    request.set_request_id(1);
    request.add_token_ids(10);
    request.add_token_ids(20);
    auto context                       = makeContext(&request);
    context->generate_input            = std::make_shared<GenerateInput>();
    context->generate_input->input_ids = torch::tensor({10, 20}, torch::kInt32);
    auto original_input                = context->generate_input;

    PrefillRpcServer server;
    server.multimodalProcess(*context);

    EXPECT_TRUE(context->multimodalProcessed());
    EXPECT_FALSE(context->tokenIdsExpanded());
    EXPECT_FALSE(context->hasError());

    context->error_status = grpc::Status(grpc::StatusCode::INTERNAL, "transient downstream failure");
    context->reset();
    EXPECT_EQ(context->generate_input, original_input);

    auto alloc_request = server.buildAllocateRequest(*context);
    ASSERT_EQ(alloc_request.input().token_ids_size(), 2);
    EXPECT_EQ(alloc_request.input().token_ids(0), 10);
    EXPECT_EQ(alloc_request.input().token_ids(1), 20);
}

TEST_F(PrefillRpcServerTest, retryRebuildsPbInputBeforeMultimodalProcessing) {
    GenerateInputPB request;
    request.set_request_id(1);
    request.add_token_ids(0);
    request.add_token_ids(1);
    request.add_token_ids(2);
    request.mutable_generate_config();
    request.add_multimodal_inputs()->set_multimodal_url("image");
    auto context = makeContext(&request);

    TestPrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(
        std::vector<ErrorCode>{ErrorCode::MM_REMOTE_RPC_FAILED, ErrorCode::NONE_ERROR});
    server.setEngineForTest(/*is_mtp_eagle=*/false);
    auto processor = std::static_pointer_cast<TestMultimodalProcessor>(server.mm_processor_);
    auto operation = [&](PrefillGenerateContext& retry_context) {
        server.prepareGenerateInputForTest(retry_context);
        server.multimodalProcess(retry_context);
    };

    auto status = server.runWithRetry(*context, operation, 1);

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(processor->callCount(), 2);
    ASSERT_NE(context->generate_input, nullptr);
    EXPECT_EQ(context->generate_input->input_ids.numel(), 4);
    EXPECT_EQ(request.token_ids_size(), 3);
    EXPECT_TRUE(context->multimodalProcessed());
    EXPECT_TRUE(context->tokenIdsExpanded());
    EXPECT_TRUE(context->generate_input->generate_config->pd_separation);
    EXPECT_TRUE(context->generate_input->generate_config->force_disable_sp_run);
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
    EXPECT_TRUE(context->multimodalProcessed());
    EXPECT_TRUE(context->tokenIdsExpanded());
    const auto& expanded_ids = context->generate_input->input_ids;
    ASSERT_EQ(expanded_ids.numel(), 4);
    EXPECT_EQ(expanded_ids.data_ptr<int32_t>()[0], 0);
    EXPECT_EQ(expanded_ids.data_ptr<int32_t>()[3], 2);
    EXPECT_EQ(context->generate_input->mm_locs.value().item<int32_t>(), 1);
    EXPECT_TRUE(
        torch::equal(context->generate_input->text_tokens_mask.value(), torch::tensor({1, 0, 0, 1}, torch::kInt32)));
}

TEST_F(PrefillRpcServerTest, successfulMultimodalProcessIsReusedAcrossRetries) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();

    PrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(ErrorCode::NONE_ERROR);
    auto processor       = std::static_pointer_cast<TestMultimodalProcessor>(server.mm_processor_);

    server.multimodalProcess(*context);
    server.multimodalProcess(*context);

    EXPECT_EQ(processor->callCount(), 1);
    EXPECT_TRUE(context->tokenIdsExpanded());
}

TEST_F(PrefillRpcServerTest, retryResetKeepsSuccessfulMultimodalResult) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();

    PrefillRpcServer server;
    server.mm_processor_ = std::make_shared<TestMultimodalProcessor>(ErrorCode::NONE_ERROR);
    auto processor       = std::static_pointer_cast<TestMultimodalProcessor>(server.mm_processor_);

    server.multimodalProcess(*context);
    auto processed_input  = context->generate_input;
    context->error_status = grpc::Status(grpc::StatusCode::INTERNAL, "transient downstream failure");
    context->reset();
    server.multimodalProcess(*context);

    EXPECT_EQ(context->generate_input, processed_input);
    EXPECT_EQ(processor->callCount(), 1);
    EXPECT_TRUE(context->multimodalProcessed());
    EXPECT_TRUE(context->tokenIdsExpanded());
}

TEST_F(PrefillRpcServerTest, retryResetDiscardsIncompleteMultimodalState) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    context->generate_input = makeMultimodalInput();
    context->error_status   = grpc::Status(grpc::StatusCode::INTERNAL, "transient multimodal failure");
    context->markMultimodalAttemptStarted();

    context->reset();

    EXPECT_EQ(context->generate_input, nullptr);
    EXPECT_FALSE(context->tokenIdsExpanded());
}

TEST_F(PrefillRpcServerTest, retryResetKeepsPreparedInputBeforeMultimodalAttempt) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    auto prepared_input     = makeMultimodalInput();
    context->generate_input = prepared_input;
    context->error_status   = grpc::Status(grpc::StatusCode::INTERNAL, "connection failed before multimodal");

    context->reset();

    EXPECT_EQ(context->generate_input, prepared_input);
}

TEST_F(PrefillRpcServerTest, retryResetKeepsInputWithoutAnError) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context            = makeContext(&request);
    auto original_input     = makeMultimodalInput();
    context->generate_input = original_input;

    context->reset();

    EXPECT_EQ(context->generate_input, original_input);
    EXPECT_FALSE(context->multimodalProcessed());
    EXPECT_FALSE(context->tokenIdsExpanded());
}

TEST_F(PrefillRpcServerTest, retryResetAllowsASecondAttemptToSucceed) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context = makeContext(&request);

    TestPrefillRpcServer server;
    int                  call_count = 0;
    auto                 operation  = [&](PrefillGenerateContext& retry_context) {
        ++call_count;
        if (call_count == 1) {
            retry_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, "transient");
        }
    };

    auto status = server.runWithRetry(*context, operation, 2);

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(call_count, 2);
    EXPECT_EQ(context->retry_times, 2);
    EXPECT_TRUE(context->shouldRetry());
}

TEST_F(PrefillRpcServerTest, zeroRetryBudgetRunsExactlyOnce) {
    GenerateInputPB request;
    request.set_request_id(1);
    auto context = makeContext(&request);

    TestPrefillRpcServer server;
    int                  call_count = 0;
    auto                 operation  = [&](PrefillGenerateContext& retry_context) {
        ++call_count;
        retry_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, "transient");
    };

    server.runWithRetry(*context, operation, 0);

    EXPECT_EQ(call_count, 1);
    EXPECT_EQ(context->retry_times, 1);
}

TEST_F(PrefillRpcServerTest, allocateRequestUsesExpandedTokenIds) {
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

    auto alloc_request = server.buildAllocateRequest(*context);

    auto expanded_ids = context->generate_input->input_ids.cpu().contiguous();
    ASSERT_EQ(expanded_ids.numel(), 4);
    ASSERT_EQ(alloc_request.input().token_ids_size(), expanded_ids.numel());
    const auto* expanded_ids_ptr = expanded_ids.data_ptr<int32_t>();
    for (int i = 0; i < alloc_request.input().token_ids_size(); ++i) {
        EXPECT_EQ(alloc_request.input().token_ids(i), expanded_ids_ptr[i]);
    }
}

TEST_F(PrefillRpcServerTest, allocateRequestKeepsOriginalIdsWithoutExpansion) {
    GenerateInputPB request;
    request.set_request_id(1);
    request.add_token_ids(10);
    request.add_token_ids(20);
    request.mutable_generate_config()->set_max_new_tokens(7);
    auto context                              = makeContext(&request);
    context->generate_input                   = std::make_shared<GenerateInput>();
    context->prefill_worker_cache_store_addrs = {"a:1", "b:2"};

    TestPrefillRpcServer server;
    server.setProcessIdForTest("prefill-client");
    auto alloc_request = server.buildAllocateRequest(*context);

    EXPECT_EQ(alloc_request.stage(), RemoteStage::ALLOCATE);
    EXPECT_EQ(alloc_request.request_id(), 1);
    EXPECT_EQ(alloc_request.client_id(), "prefill-client");
    ASSERT_EQ(alloc_request.input().token_ids_size(), 2);
    EXPECT_EQ(alloc_request.input().token_ids(0), 10);
    EXPECT_EQ(alloc_request.input().token_ids(1), 20);
    EXPECT_EQ(alloc_request.input().generate_config().max_new_tokens(), 7);
    ASSERT_EQ(alloc_request.peer_addrs_size(), 2);
    EXPECT_EQ(alloc_request.peer_addrs(0), "a:1");
    EXPECT_EQ(alloc_request.peer_addrs(1), "b:2");
}

}  // namespace rtp_llm
