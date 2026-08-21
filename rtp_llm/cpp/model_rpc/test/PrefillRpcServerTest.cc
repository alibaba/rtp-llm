#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

class TestDecodeRpcService final: public RpcService::Service {
public:
    explicit TestDecodeRpcService(bool fail_first_allocate): fail_first_allocate_(fail_first_allocate) {}

    grpc::Status RemoteGenerate(grpc::ServerContext*,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        GenerateRequestPB request;
        if (!stream->Read(&request)) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "missing allocate request");
        }
        ++allocate_count_;
        if (fail_first_allocate_ && allocate_count_ == 1) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "allocate failed once");
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
    bool             fail_first_allocate_;
    std::atomic<int> allocate_count_{0};
};

class TestDecodeRpcServer {
public:
    explicit TestDecodeRpcServer(bool fail_first_allocate): service_(fail_first_allocate) {}
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
                              int                                                 max_retries      = 3,
                              int64_t                                             retry_timeout_ms = 0) {
        constexpr int64_t retry_interval_ms = 0;
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
