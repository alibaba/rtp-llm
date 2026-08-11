#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadFence.h"
#define private public
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#undef private
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

#include <functional>
#include <stdexcept>
#include <thread>

namespace rtp_llm {

class LocalRpcServerOutcomeTest: public DeviceTestBase {
protected:
    class TestServer: public LocalRpcServer {
    public:
        grpc::Status poll(grpc::ServerContext*             context,
                          WriterInterface*                 writer,
                          std::shared_ptr<GenerateStream>& stream,
                          RemoteGenerateWaitResult&        result) {
            return pollStreamOutput(context, "test-request", writer, stream, &result);
        }
    };

    class CountingWriter: public LocalRpcServer::WriterInterface {
    public:
        bool Write(const GenerateOutputsPB& output, grpc::WriteOptions) override {
            ++write_count;
            last_output = output;
            return true;
        }

        int               write_count = 0;
        GenerateOutputsPB last_output;
    };

    class FakeClientStream: public PrefillGenerateContext::ClientStream {
    public:
        bool Write(const GenerateRequestPB&, grpc::WriteOptions) override {
            ++write_count;
            if (throw_on_write) {
                throw std::runtime_error("injected write failure");
            }
            if (on_write) {
                on_write();
            }
            return write_ok;
        }
        bool Read(GenerateOutputsPB* output) override {
            ++read_count;
            if (throw_on_read) {
                throw std::runtime_error("injected read failure");
            }
            if (read_ok) {
                output->mutable_error_info()->set_error_code(response_error);
                output->set_remote_load_quiesced(remote_load_quiesced);
            }
            return read_ok;
        }
        bool NextMessageSize(uint32_t*) override {
            return true;
        }
        void WaitForInitialMetadata() override {}
        bool WritesDone() override {
            return true;
        }
        grpc::Status Finish() override {
            ++finish_count;
            if (on_finish) {
                on_finish();
            }
            return grpc::Status::OK;
        }

        bool                  write_ok      = true;
        bool                  read_ok       = true;
        bool                  throw_on_write = false;
        bool                  throw_on_read = false;
        bool                  remote_load_quiesced = true;
        ErrorCodePB           response_error = ErrorCodePB::NONE_ERROR;
        int                   write_count   = 0;
        int                   read_count    = 0;
        int                   finish_count  = 0;
        std::function<void()> on_write;
        std::function<void()> on_finish;
    };

    class TestPrefillServer: public PrefillRpcServer {
    public:
        void loadStart(PrefillGenerateContext& context) {
            remoteLoadCacheStart(context);
        }
        void pollLocal(PrefillGenerateContext& context) {
            pollLocalOutput(context);
        }
        void loadEnd(PrefillGenerateContext& context) {
            remoteLoadCacheEnd(context);
        }
        size_t activeLeaseCount() const {
            return remote_load_leases_.activeJobsForTest();
        }
        bool waitForNoActiveLeases() const {
            for (int attempt = 0; attempt < 2000; ++attempt) {
                if (activeLeaseCount() == 0) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return false;
        }
        bool waitForLeaseRelease(const std::weak_ptr<KVCacheResource>& lease) const {
            for (int attempt = 0; attempt < 2000; ++attempt) {
                if (lease.expired()) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return false;
        }
    };

    std::shared_ptr<GenerateStream> createStream() {
        auto input             = std::make_shared<GenerateInput>();
        input->input_ids       = torch::tensor({1}, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        input->generate_config->pd_separation = true;

        ModelConfig model_config;
        model_config.max_seq_len                 = 16;
        model_config.vocab_size                  = 16;
        model_config.special_tokens.eos_token_id = 15;

        ResourceContext resource_context;
        cache_manager_ = std::make_shared<KVCacheManager>(
            test::makeSimpleMhaCacheConfig(1, 8, 2, rtp_llm::DataType::TYPE_INT8), false, nullptr);
        EXPECT_TRUE(cache_manager_->init());
        resource_context.cache_manager = cache_manager_;
        resource_context.role_type     = RoleType::PREFILL;

        auto stream = std::make_shared<NormalGenerateStream>(
            input, model_config, RuntimeConfig(), resource_context, nullptr);
        stream->setNeedReleaseResource(false);
        stream->generate_status_->status = StreamState::RUNNING;
        EXPECT_TRUE(stream->streamCacheResource().initKVBlock().ok());
        return stream;
    }

    void update(const std::shared_ptr<GenerateStream>& stream, int token, bool remote) {
        stream->update({torch::tensor({{token}}, torch::kInt32),
                        1,
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        remote,
                        false});
    }

    const KVCacheResource* pdLeaseIdentity(const std::shared_ptr<GenerateStream>& stream) const {
        return stream->streamCacheResource().pd_kvcache_ref_.get();
    }

    std::weak_ptr<KVCacheResource> observeLeaseBeforeTransfer(const std::shared_ptr<GenerateStream>& stream) {
        EXPECT_EQ(stream->holdKVCacheForPDSep(), PdSepCacheHoldResult::Held);
        auto lease = stream->streamCacheResource().pd_kvcache_ref_;
        EXPECT_NE(lease, nullptr);
        return lease;
    }

    PrefillGenerateContext& createPrefillContext(const std::shared_ptr<GenerateStream>& stream,
                                                 CountingWriter&                        writer,
                                                 const std::shared_ptr<FakeClientStream>& client_stream) {
        request_.set_request_id(1);
        request_.mutable_generate_config()->set_timeout_ms(1000);
        RPCContext rpc_context{&request_, &writer};
        prefill_context_ = std::make_unique<PrefillGenerateContext>(
            &remote_resource_, rpc_context, 1000, &server_context_, metrics_reporter_, meta_);
        prefill_context_->generate_input = stream->generateInput();
        prefill_context_->setStream(stream);
        prefill_context_->client_context = std::make_shared<grpc::ClientContext>();
        prefill_context_->client_stream  = client_stream;
        prefill_context_->load_deadline_unix_ms = currentTimeUs() / 1000 + 30'000;
        auto allocation_token = makeRemoteLoadAllocationToken(
            "test-owner", "test-allocation", prefill_context_->load_deadline_unix_ms);
        EXPECT_TRUE(allocation_token.ok());
        prefill_context_->allocation_token = allocation_token.ok() ? *allocation_token : std::string();
        prefill_context_->remote_load_quiesce = []() { return true; };
        return *prefill_context_;
    }

    void TearDown() override {
        if (prefill_context_) {
            prefill_context_->cleanupRemoteLoadCache();
            prefill_context_->getStream().reset();
            prefill_context_.reset();
        }
        EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
        DeviceTestBase::TearDown();
    }

    TestServer server_;
    TestPrefillServer prefill_server_;
    std::shared_ptr<KVCacheManager> cache_manager_;
    GenerateInputPB request_;
    grpc::ServerContext server_context_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::shared_ptr<RpcServerRuntimeMeta> meta_ = std::make_shared<RpcServerRuntimeMeta>();
    RemoteServerResource remote_resource_;
    std::unique_ptr<PrefillGenerateContext> prefill_context_;
};

TEST_F(LocalRpcServerOutcomeTest, PollsHandoffWhenRemoteGenerationIsRequired) {
    auto stream = createStream();
    update(stream, 2, true);
    grpc::ServerContext context;
    CountingWriter      writer;

    RemoteGenerateWaitResult result = RemoteGenerateWaitResult::Error;
    EXPECT_TRUE(server_.poll(&context, &writer, stream, result).ok());
    EXPECT_EQ(result, RemoteGenerateWaitResult::Handoff);
    EXPECT_EQ(writer.write_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, PollsLocalDoneWhenPrefillTerminatesLocally) {
    auto stream = createStream();
    update(stream, 15, true);
    grpc::ServerContext context;
    CountingWriter      writer;

    RemoteGenerateWaitResult result = RemoteGenerateWaitResult::Error;
    EXPECT_TRUE(server_.poll(&context, &writer, stream, result).ok());
    EXPECT_EQ(result, RemoteGenerateWaitResult::LocalDone);
    EXPECT_EQ(writer.write_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, PollsErrorWhenPrefillFails) {
    auto stream = createStream();
    stream->reportError(ErrorCode::MALLOC_FAILED, "injected failure");
    grpc::ServerContext context;
    CountingWriter      writer;

    RemoteGenerateWaitResult result = RemoteGenerateWaitResult::Handoff;
    EXPECT_FALSE(server_.poll(&context, &writer, stream, result).ok());
    EXPECT_EQ(result, RemoteGenerateWaitResult::Error);
    EXPECT_EQ(writer.write_count, 0);
}

TEST_F(LocalRpcServerOutcomeTest, LocalDoneWaitsForRemoteLoadEndBeforeFinishing) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);

    prefill_server_.loadStart(context);
    ASSERT_TRUE(context.remote_load_cache_started);
    ASSERT_NE(context.cache_lease_ticket, nullptr);
    ASSERT_EQ(prefill_server_.activeLeaseCount(), 1);
    ASSERT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());

    update(stream, 15, true);
    prefill_server_.pollLocal(context);
    EXPECT_TRUE(context.local_generate_done);
    EXPECT_FALSE(context.finished);
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);

    prefill_server_.loadEnd(context);
    EXPECT_TRUE(context.finished);
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 0);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->read_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, ErrorWaitsForRemoteLoadEndAndPreservesError) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);

    prefill_server_.loadStart(context);
    stream->reportError(ErrorCode::MALLOC_FAILED, "stage failure");
    prefill_server_.pollLocal(context);
    EXPECT_TRUE(context.error_status.ok());
    EXPECT_FALSE(context.deferred_local_status.ok());
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);

    prefill_server_.loadEnd(context);
    EXPECT_FALSE(context.error_status.ok());
    EXPECT_NE(context.error_status.error_message().find("stage failure"), std::string::npos);
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 0);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
}

TEST_F(LocalRpcServerOutcomeTest, HandoffBeforeLoadStartKeepsOneIdempotentLease) {
    auto stream = createStream();
    stream->setNeedReleaseResource(true);
    update(stream, 2, true);
    ASSERT_TRUE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    const auto* lease = pdLeaseIdentity(stream);
    ASSERT_NE(lease, nullptr);
    stream->moveToNext();
    ASSERT_EQ(stream->getStatus(), StreamState::FINISHED);
    ASSERT_TRUE(stream->streamCacheResource().isResourceReleased());

    auto client_stream = std::make_shared<FakeClientStream>();
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);

    EXPECT_TRUE(context.remote_load_cache_started);
    EXPECT_NE(context.cache_lease_ticket, nullptr);
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(pdLeaseIdentity(stream), nullptr);
    prefill_server_.loadEnd(context);
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 0);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(pdLeaseIdentity(stream), nullptr);
}

TEST_F(LocalRpcServerOutcomeTest, LoadStartBeforeHandoffUsesTheSameLease) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);

    prefill_server_.loadStart(context);
    ASSERT_EQ(prefill_server_.activeLeaseCount(), 1);
    ASSERT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    update(stream, 2, true);
    EXPECT_TRUE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_NE(pdLeaseIdentity(stream), nullptr);

    prefill_server_.pollLocal(context);
    EXPECT_FALSE(context.finished);
    prefill_server_.loadEnd(context);
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 0);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(pdLeaseIdentity(stream), nullptr);
}

TEST_F(LocalRpcServerOutcomeTest, ContextCleanupCancelsLoadAndReleasesLease) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);
    ASSERT_EQ(prefill_server_.activeLeaseCount(), 1);
    client_stream->on_finish = [this, stream]() {
        EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
        EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    };

    context.cleanupRemoteLoadCache();
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(context.remote_load_cache_started);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, WriteFailureReportsLoadErrorAndReleasesAfterQuiescing) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->write_ok = false;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    client_stream->on_finish = [this, stream]() {
        EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
        EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    };

    prefill_server_.loadStart(context);

    EXPECT_EQ(context.error_info.code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_FALSE(context.error_status.ok());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_FALSE(context.remote_load_cache_started);
    EXPECT_EQ(client_stream->write_count, 1);
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, WriteExceptionUsesTheSameFailureAndCleanupPath) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->throw_on_write = true;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);

    EXPECT_NO_THROW(prefill_server_.loadStart(context));
    EXPECT_EQ(context.error_info.code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_NE(context.error_info.ToString().find("injected write failure"), std::string::npos);
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, LocalDoneRacingWriteFailureWins) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->write_ok = false;
    client_stream->on_write = [this, stream]() {
        update(stream, 15, true);
    };
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);

    prefill_server_.loadStart(context);

    EXPECT_TRUE(context.local_generate_done);
    EXPECT_FALSE(context.finished);
    EXPECT_TRUE(context.error_status.ok());
    EXPECT_FALSE(stream->hasError());
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(writer.write_count, 0);

    prefill_server_.pollLocal(context);
    EXPECT_TRUE(context.finished);
    EXPECT_EQ(writer.write_count, 1);
    EXPECT_FALSE(writer.last_output.flatten_output().output_ids().int32_data().empty());
}

TEST_F(LocalRpcServerOutcomeTest, LocalErrorRacingWriteFailureWins) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->write_ok = false;
    client_stream->on_write = [stream]() {
        stream->reportError(ErrorCode::MALLOC_FAILED, "stage failure");
    };
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);

    prefill_server_.loadStart(context);

    EXPECT_EQ(context.error_info.code(), ErrorCode::MALLOC_FAILED);
    EXPECT_NE(context.error_status.error_message().find("stage failure"), std::string::npos);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
}

TEST_F(LocalRpcServerOutcomeTest, PublishedHandoffRacingWriteFailureBecomesLoadError) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->write_ok = false;
    client_stream->on_write = [this, stream]() {
        update(stream, 2, true);
    };
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);

    prefill_server_.loadStart(context);

    EXPECT_EQ(context.error_info.code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_FALSE(context.error_status.ok());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, HoldFailureDoesNotWriteRemoteLoadRequest) {
    auto stream = createStream();
    stream->setNeedReleaseResource(true);
    stream->releaseResource();
    ASSERT_TRUE(stream->streamCacheResource().isResourceReleased());

    auto client_stream = std::make_shared<FakeClientStream>();
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);

    EXPECT_EQ(context.error_info.code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_EQ(client_stream->write_count, 0);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
}

TEST_F(LocalRpcServerOutcomeTest, CleanupReleasesFastPublishedLeaseWithoutContextFlag) {
    auto stream = createStream();
    update(stream, 2, true);
    ASSERT_TRUE(stream->streamCacheResource().hasKVCacheHoldForPDSep());

    auto client_stream = std::make_shared<FakeClientStream>();
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    ASSERT_EQ(prefill_server_.activeLeaseCount(), 0);
    client_stream->on_finish = [stream]() {
        EXPECT_TRUE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    };

    context.cleanupRemoteLoadCache();
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, ReadFailureReportsLoadErrorAndReleasesAfterQuiescing) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->read_ok = false;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);
    update(stream, 2, true);
    prefill_server_.pollLocal(context);
    client_stream->on_finish = [this, stream]() {
        EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
        EXPECT_TRUE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    };

    prefill_server_.loadEnd(context);

    EXPECT_EQ(context.error_info.code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, ReadExceptionUsesTheSameFailureAndCleanupPath) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->throw_on_read = true;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);
    update(stream, 2, true);
    prefill_server_.pollLocal(context);

    EXPECT_NO_THROW(prefill_server_.loadEnd(context));
    EXPECT_EQ(context.error_info.code(), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    EXPECT_NE(context.error_info.ToString().find("injected read failure"), std::string::npos);
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, RemoteLoadResponseErrorReportsFailureAndReleasesLease) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->response_error = ErrorCodePB::UNKNOWN_ERROR;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);
    update(stream, 2, true);
    prefill_server_.pollLocal(context);
    client_stream->on_finish = [this, stream]() {
        EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
        EXPECT_TRUE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    };

    prefill_server_.loadEnd(context);

    EXPECT_EQ(context.error_info.code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_FALSE(context.error_status.ok());
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, LocalDoneWinsOverRemoteLoadReadFailure) {
    auto stream        = createStream();
    auto source_lease  = observeLeaseBeforeTransfer(stream);
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->read_ok = false;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_FALSE(source_lease.expired());
    update(stream, 15, true);
    prefill_server_.pollLocal(context);
    client_stream->on_finish = [this, stream, source_lease]() {
        EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
        EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
        EXPECT_FALSE(source_lease.expired());
    };

    prefill_server_.loadEnd(context);

    EXPECT_TRUE(context.finished);
    EXPECT_TRUE(context.error_status.ok());
    EXPECT_FALSE(stream->hasError());
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_TRUE(prefill_server_.waitForLeaseRelease(source_lease));
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, LocalErrorWinsOverRemoteLoadResponseError) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->response_error = ErrorCodePB::UNKNOWN_ERROR;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);
    stream->reportError(ErrorCode::MALLOC_FAILED, "stage failure");
    prefill_server_.pollLocal(context);
    client_stream->on_finish = [this, stream]() {
        EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
        EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    };

    prefill_server_.loadEnd(context);

    EXPECT_FALSE(context.error_status.ok());
    EXPECT_NE(context.error_status.error_message().find("stage failure"), std::string::npos);
    EXPECT_EQ(context.error_status.error_message().find("remote load response failed"), std::string::npos);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, LocalDoneWinsOverRemoteLoadResponseError) {
    auto stream        = createStream();
    auto source_lease  = observeLeaseBeforeTransfer(stream);
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->response_error = ErrorCodePB::UNKNOWN_ERROR;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);
    EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_FALSE(source_lease.expired());
    update(stream, 15, true);
    prefill_server_.pollLocal(context);
    client_stream->on_finish = [this, stream, source_lease]() {
        EXPECT_EQ(prefill_server_.activeLeaseCount(), 1);
        EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
        EXPECT_FALSE(source_lease.expired());
    };

    prefill_server_.loadEnd(context);

    EXPECT_TRUE(context.finished);
    EXPECT_TRUE(context.error_status.ok());
    EXPECT_FALSE(stream->hasError());
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_TRUE(prefill_server_.waitForLeaseRelease(source_lease));
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
    EXPECT_EQ(client_stream->finish_count, 1);
}

TEST_F(LocalRpcServerOutcomeTest, LocalErrorWinsOverRemoteLoadReadFailure) {
    auto stream        = createStream();
    auto client_stream = std::make_shared<FakeClientStream>();
    client_stream->read_ok = false;
    CountingWriter writer;
    auto& context = createPrefillContext(stream, writer, client_stream);
    prefill_server_.loadStart(context);
    stream->reportError(ErrorCode::MALLOC_FAILED, "stage failure");
    prefill_server_.pollLocal(context);

    prefill_server_.loadEnd(context);

    EXPECT_FALSE(context.error_status.ok());
    EXPECT_NE(context.error_status.error_message().find("stage failure"), std::string::npos);
    EXPECT_EQ(context.error_status.error_message().find("remote load response failed"), std::string::npos);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    EXPECT_TRUE(prefill_server_.waitForNoActiveLeases());
    EXPECT_FALSE(stream->streamCacheResource().hasKVCacheHoldForPDSep());
}

}  // namespace rtp_llm
