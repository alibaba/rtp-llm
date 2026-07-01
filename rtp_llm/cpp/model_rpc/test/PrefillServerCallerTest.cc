#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"

namespace rtp_llm::test {

// Pure function test: prefill produces a small set of grpc::StatusCode values
// (see transErrorCodeToGrpc in RpcErrorCode.h). Decode side must map each one
// back to a representative ErrorCode so downstream consumers see something
// other than the generic UNKNOWN_ERROR.
TEST(RpcErrorCodeReverseMapping, KnownStatusesMapToConcreteErrorCodes) {
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::CANCELLED), ErrorCode::CANCELLED);
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::RESOURCE_EXHAUSTED), ErrorCode::MALLOC_FAILED);
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::DEADLINE_EXCEEDED), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::OUT_OF_RANGE), ErrorCode::LONG_PROMPT_ERROR);
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::INVALID_ARGUMENT), ErrorCode::INVALID_PARAMS);
}

TEST(RpcErrorCodeReverseMapping, UnmappedStatusesFallToUnknownError) {
    // INTERNAL is the catch-all on the prefill side too — the original
    // ErrorCode is genuinely lost, so UNKNOWN_ERROR is the right answer.
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::INTERNAL), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::UNAVAILABLE), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::UNAUTHENTICATED), ErrorCode::UNKNOWN_ERROR);
}

TEST(RpcErrorCodeReverseMapping, RoundTripPrefillProducedStatuses) {
    // For every ErrorCode that prefill can produce a non-INTERNAL grpc status
    // for, the round-trip via reverse mapping should preserve the family
    // (e.g. MALLOC_FAILED → RESOURCE_EXHAUSTED → MALLOC_FAILED).
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::CANCELLED)), ErrorCode::CANCELLED);
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::MALLOC_FAILED)), ErrorCode::MALLOC_FAILED);
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::GENERATE_TIMEOUT)), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::INVALID_PARAMS)), ErrorCode::INVALID_PARAMS);
    // DECODE_MALLOC_FAILED collapses to MALLOC_FAILED (both share
    // RESOURCE_EXHAUSTED). This is documented loss; consumers must rely on
    // the error message string to distinguish if needed.
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::DECODE_MALLOC_FAILED)), ErrorCode::MALLOC_FAILED);
    // Likewise OUT_OF_VOCAB_RANGE collapses to LONG_PROMPT_ERROR (both share
    // OUT_OF_RANGE).
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::OUT_OF_VOCAB_RANGE)), ErrorCode::LONG_PROMPT_ERROR);
}

TEST(GrpcAddressUtilTest, FormatsRawIpv6AndRejectsInvalidPorts) {
    EXPECT_EQ(formatGrpcHostPort("127.0.0.1", 9001), "127.0.0.1:9001");
    EXPECT_EQ(formatGrpcHostPort("prefill-0.service", 9002), "prefill-0.service:9002");
    EXPECT_EQ(formatGrpcHostPort("fe80::1", 9003), "[fe80::1]:9003");
    EXPECT_EQ(formatGrpcHostPort("[::1]", 9004), "[::1]:9004");
    EXPECT_EQ(formatGrpcHostPort("127.0.0.1", 0), "");
    EXPECT_EQ(formatGrpcHostPort("127.0.0.1", 65536), "");
    EXPECT_EQ(formatGrpcHostPort("[::1", 9005), "");
}

class FakePrefillRpcService final: public RpcService::Service {
public:
    enum class Mode {
        kFirstChunkSnapshot,
        kLaterChunkSnapshot,
        kWaitForCancel,
        kErrorChunkThenWaitForCancel,
        kReturnErrorWithoutChunk,
        kCaptureForwardedRequest,
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
            auto* first_aux_info = first_response.mutable_flatten_output()->add_aux_info();
            first_aux_info->set_step_output_len(1);
            first_aux_info->set_prefill_total_reuse_len(88);
            first_aux_info->set_prefill_local_reuse_len(16);
            first_aux_info->set_prefill_remote_reuse_len(72);
            first_aux_info->set_prefill_memory_reuse_len(4);
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

        if (mode_ == Mode::kErrorChunkThenWaitForCancel) {
            GenerateOutputsPB first_response;
            first_response.set_request_id(request->request_id());
            first_response.mutable_error_info()->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
            first_response.mutable_error_info()->set_error_message("first chunk error");
            first_response.mutable_flatten_output()->add_finished(false);
            first_response.mutable_flatten_output()->add_aux_info()->set_step_output_len(1);
            if (!writer->Write(first_response)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "failed to write first response");
            }

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

        if (mode_ == Mode::kLaterChunkSnapshot) {
            GenerateOutputsPB first_response;
            first_response.set_request_id(request->request_id());
            first_response.mutable_flatten_output()->add_finished(false);
            first_response.mutable_flatten_output()->add_aux_info()->set_step_output_len(1);
            if (!writer->Write(first_response)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "failed to write first response");
            }

            GenerateOutputsPB second_response;
            second_response.set_request_id(request->request_id());
            second_response.mutable_flatten_output()->add_finished(true);
            auto* second_aux_info = second_response.mutable_flatten_output()->add_aux_info();
            second_aux_info->set_step_output_len(2);
            second_aux_info->set_prefill_total_reuse_len(96);
            second_aux_info->set_prefill_local_reuse_len(24);
            second_aux_info->set_prefill_remote_reuse_len(72);
            second_aux_info->set_prefill_memory_reuse_len(8);
            if (!writer->Write(second_response)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "failed to write second response");
            }
            return grpc::Status::OK;
        }

        if (mode_ == Mode::kCaptureForwardedRequest) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                captured_request_.CopyFrom(*request);
            }
            GenerateOutputsPB response;
            response.set_request_id(request->request_id());
            response.mutable_flatten_output()->add_finished(true);
            response.mutable_flatten_output()->add_aux_info()->set_step_output_len(1);
            if (!writer->Write(response)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "failed to write capture response");
            }
            return grpc::Status::OK;
        }

        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill failed before any response");
    }

    grpc::Status
    GetPeerInfo(grpc::ServerContext*, const GetPeerInfoRequestPB*, GetPeerInfoResponsePB* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++peer_info_call_count_;
        response->set_tp_size(peer_info_tp_size_);
        if (!peer_info_dp_addr_responses_.empty()) {
            const size_t response_index =
                std::min<size_t>(peer_info_call_count_ - 1, peer_info_dp_addr_responses_.size() - 1);
            for (const auto& addr : peer_info_dp_addr_responses_[response_index]) {
                response->add_dp_grpc_addrs(addr);
            }
        }
        return grpc::Status::OK;
    }

    void setPeerInfoResponses(int tp_size, std::vector<std::vector<std::string>> dp_addr_responses) {
        std::lock_guard<std::mutex> lock(mutex_);
        peer_info_tp_size_           = tp_size;
        peer_info_dp_addr_responses_ = std::move(dp_addr_responses);
        peer_info_call_count_        = 0;
    }

    int peerInfoCallCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return peer_info_call_count_;
    }

    bool waitStarted(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return started_cv_.wait_for(lock, timeout, [this]() { return started_.load(); });
    }

    bool waitCancelled(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cancel_cv_.wait_for(lock, timeout, [this]() { return cancelled_.load(); });
    }

    GenerateInputPB capturedRequest() {
        std::lock_guard<std::mutex> lock(mutex_);
        return captured_request_;
    }

private:
    Mode                    mode_;
    std::atomic<bool>       started_{false};
    std::atomic<bool>       cancelled_{false};
    std::mutex              mutex_;
    std::condition_variable started_cv_;
    std::condition_variable cancel_cv_;
    GenerateInputPB         captured_request_;
    int                     peer_info_tp_size_{1};
    int                     peer_info_call_count_{0};
    std::vector<std::vector<std::string>> peer_info_dp_addr_responses_;
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
    GenerateInputPB makeSyncRequest(FakePrefillRpcServer& server, int64_t request_id, int32_t timeout_ms) {
        GenerateInputPB request;
        request.set_request_id(request_id);
        request.add_token_ids(1);
        request.mutable_generate_config()->set_timeout_ms(timeout_ms);
        auto* role_addr = request.mutable_generate_config()->add_role_addrs();
        role_addr->set_role(RoleAddrPB::PREFILL);
        role_addr->set_ip("127.0.0.1");
        role_addr->set_grpc_port(server.port());
        return request;
    }

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

TEST_F(PrefillServerCallerTest, ErrorChunkMarksContextFailedAndPreservesFirstSnapshot) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kFirstChunkSnapshot));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1001, "snapshot");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->failed());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(context->errorInfo().ToString(), "first chunk error");

    const auto& response = context->response();
    ASSERT_TRUE(response.has_error_info());
    EXPECT_EQ(response.error_info().error_code(), ErrorCodePB::UNKNOWN_ERROR);
    EXPECT_EQ(response.error_info().error_message(), "first chunk error");
    ASSERT_TRUE(response.has_flatten_output());
    ASSERT_EQ(response.flatten_output().aux_info_size(), 1);
    EXPECT_EQ(response.flatten_output().aux_info(0).step_output_len(), 1);

    PrefillServerCallerContext::ReuseLensSnapshot reuse_lens;
    ASSERT_TRUE(context->getPrefillReuseLensSnapshot(reuse_lens));
    EXPECT_EQ(reuse_lens.total, 88);
    EXPECT_EQ(reuse_lens.local, 16);
    EXPECT_EQ(reuse_lens.remote, 72);
    EXPECT_EQ(reuse_lens.memory, 4);
}

TEST_F(PrefillServerCallerTest, ErrorChunkCancelsOutstandingRpcAndWaitsForFinish) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kErrorChunkThenWaitForCancel));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1011, "error-cancel");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->failed());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(context->errorInfo().ToString(), "first chunk error");
    EXPECT_TRUE(server.service()->waitCancelled(std::chrono::seconds(1)));
}

TEST_F(PrefillServerCallerTest, LaterChunkReuseLensRefreshSnapshot) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kLaterChunkSnapshot));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1007, "later-snapshot");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(context->failed());

    GenerateOutputsPB first_response;
    ASSERT_TRUE(context->takeFirstResponse(first_response));
    ASSERT_TRUE(first_response.has_flatten_output());
    ASSERT_EQ(first_response.flatten_output().aux_info_size(), 1);
    EXPECT_EQ(first_response.flatten_output().aux_info(0).step_output_len(), 1);

    const auto& response = context->response();
    ASSERT_TRUE(response.has_flatten_output());
    ASSERT_EQ(response.flatten_output().aux_info_size(), 1);
    EXPECT_EQ(response.flatten_output().aux_info(0).step_output_len(), 2);

    PrefillServerCallerContext::ReuseLensSnapshot reuse_lens;
    ASSERT_TRUE(context->getPrefillReuseLensSnapshot(reuse_lens));
    EXPECT_EQ(reuse_lens.total, 96);
    EXPECT_EQ(reuse_lens.local, 24);
    EXPECT_EQ(reuse_lens.remote, 72);
    EXPECT_EQ(reuse_lens.memory, 8);
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

TEST_F(PrefillServerCallerTest, AsyncDecodeEntrancePrefillPreservesPdSeparationRequestShape) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kCaptureForwardedRequest));
    ASSERT_TRUE(server.start());

    GenerateInputPB request;
    request.set_request_id(1010);
    request.add_token_ids(1);
    request.mutable_generate_config()->set_timeout_ms(5000);
    request.mutable_generate_config()->set_max_new_tokens(64);
    request.mutable_generate_config()->set_num_beams(1);
    request.mutable_generate_config()->set_num_return_sequences(1);
    request.mutable_generate_config()->set_can_use_pd_separation(true);

    auto context = caller_.callPrefill(&request, "127.0.0.1", server.port(), "decode-entrance-key", 5 * 1000 * 1000);
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    ASSERT_TRUE(context->success());

    const auto captured_request = server.service()->capturedRequest();
    EXPECT_EQ(captured_request.request_id(), 1010);
    EXPECT_EQ(captured_request.generate_config().max_new_tokens(), 64);
    EXPECT_TRUE(captured_request.generate_config().can_use_pd_separation());
    EXPECT_EQ(captured_request.generate_config().unique_key(), "decode-entrance-key");
}

TEST_F(PrefillServerCallerTest, DestroyContextCancelsOutstandingPrefillRpc) {
    FakePrefillRpcServer server(std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kWaitForCancel));
    ASSERT_TRUE(server.start());

    {
        auto context = callPrefill(server, 1004, "destroy-cancel");
        ASSERT_NE(context, nullptr);
        ASSERT_TRUE(server.service()->waitStarted(std::chrono::seconds(1)));
    }

    EXPECT_TRUE(server.service()->waitCancelled(std::chrono::seconds(1)));
}

TEST_F(PrefillServerCallerTest, UnstartedContextDoneReturnsImmediately) {
    auto context = std::make_shared<PrefillServerCallerContext>("127.0.0.1:1", "never-started");

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->failed());
}

TEST_F(PrefillServerCallerTest, AsyncReaderCreationFailureReturnsNullContext) {
    caller_.async_reader_factory_ = [](const std::shared_ptr<RpcService::Stub>&,
                                       grpc::ClientContext*,
                                       const GenerateInputPB&,
                                       grpc::CompletionQueue*) {
        return std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>>();
    };

    GenerateInputPB request;
    request.set_request_id(1012);
    request.add_token_ids(1);
    request.mutable_generate_config()->set_timeout_ms(5000);

    auto context = caller_.callPrefill(&request, "127.0.0.1", 1, "null-reader", 5 * 1000 * 1000);
    EXPECT_EQ(context, nullptr);
}

TEST_F(PrefillServerCallerTest, AsyncPrefillNormalizesRawIpv6TargetAddress) {
    GenerateInputPB request;
    request.set_request_id(1013);
    request.add_token_ids(1);
    request.mutable_generate_config()->set_timeout_ms(1);

    auto context = caller_.callPrefill(&request, "::1", 65535, "ipv6-target", 1000);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->prefill_addr_, "[::1]:65535");
    context->cancel();
}

TEST_F(PrefillServerCallerTest, AsyncPrefillRejectsInvalidTargetPort) {
    GenerateInputPB request;
    request.set_request_id(1014);
    request.add_token_ids(1);
    request.mutable_generate_config()->set_timeout_ms(1);

    EXPECT_EQ(caller_.callPrefill(&request, "::1", 0, "bad-port", 1000), nullptr);
}

TEST_F(PrefillServerCallerTest, InvalidatePeerInfoDropsCacheAndReprobes) {
    auto service = std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kCaptureForwardedRequest);
    service->setPeerInfoResponses(2, {{"127.0.0.1:1111"}, {"127.0.0.1:2222"}});
    FakePrefillRpcServer server(std::move(service));
    ASSERT_TRUE(server.start());

    auto first = caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0);
    ASSERT_EQ(first.tp_size, 2);
    ASSERT_EQ(first.dp_addrs.size(), 1);
    EXPECT_EQ(first.dp_addrs[0], "127.0.0.1:1111");
    EXPECT_EQ(server.service()->peerInfoCallCount(), 1);

    auto cached = caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0);
    ASSERT_EQ(cached.dp_addrs.size(), 1);
    EXPECT_EQ(cached.dp_addrs[0], "127.0.0.1:1111");
    EXPECT_EQ(server.service()->peerInfoCallCount(), 1);

    caller_.invalidatePrefillPeerInfo("127.0.0.1", server.port());

    auto refreshed = caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0);
    ASSERT_EQ(refreshed.tp_size, 2);
    ASSERT_EQ(refreshed.dp_addrs.size(), 1);
    EXPECT_EQ(refreshed.dp_addrs[0], "127.0.0.1:2222");
    EXPECT_EQ(server.service()->peerInfoCallCount(), 2);
}

TEST_F(PrefillServerCallerTest, RpcFailureWithoutAnyChunkStaysUnsuccessful) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kReturnErrorWithoutChunk));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1003, "rpc-failure");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->failed());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(context->errorInfo().ToString(), "prefill failed before any response");
    EXPECT_FALSE(context->response().has_error_info());
}

TEST_F(PrefillServerCallerTest, SyncFallbackTimeoutCancelsPrefillRpc) {
    FakePrefillRpcServer server(std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kWaitForCancel));
    ASSERT_TRUE(server.start());

    grpc::ServerContext server_context;
    auto                request = makeSyncRequest(server, 1005, 50);

    auto status = caller_.callPrefill(&server_context, &request, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(server.service()->waitCancelled(std::chrono::seconds(1)));
}

TEST_F(PrefillServerCallerTest, SyncFallbackClientCancelCancelsPrefillRpc) {
    FakePrefillRpcServer server(std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kWaitForCancel));
    ASSERT_TRUE(server.start());

    auto              request = makeSyncRequest(server, 1006, 5000);
    std::atomic<bool> started{false};
    std::atomic<bool> cancelled{false};

    std::thread cancel_thread([&]() {
        started.store(server.service()->waitStarted(std::chrono::seconds(1)));
        cancelled.store(true);
    });

    auto status = caller_.callPrefill(nullptr, &request, nullptr, [&cancelled]() { return cancelled.load(); });
    cancel_thread.join();

    EXPECT_TRUE(started.load());
    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_TRUE(server.service()->waitCancelled(std::chrono::seconds(1)));
}

}  // namespace rtp_llm::test
