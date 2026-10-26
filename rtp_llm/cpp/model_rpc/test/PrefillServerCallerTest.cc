#include <future>
#include "rtp_llm/cpp/utils/TimeUtil.h"
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
#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

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
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::UNAVAILABLE), ErrorCode::CONNECT_FAILED);
    EXPECT_EQ(transGrpcStatusToErrorCode(grpc::StatusCode::UNAUTHENTICATED), ErrorCode::UNKNOWN_ERROR);
}

TEST(RpcErrorCodeReverseMapping, RoundTripPrefillProducedStatuses) {
    // For every ErrorCode that prefill can produce a non-INTERNAL grpc status
    // for, the round-trip via reverse mapping should preserve the family
    // (e.g. MALLOC_FAILED → RESOURCE_EXHAUSTED → MALLOC_FAILED).
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::CANCELLED)), ErrorCode::CANCELLED);
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::MALLOC_FAILED)), ErrorCode::MALLOC_FAILED);
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::GENERATE_TIMEOUT)),
              ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::INVALID_PARAMS)), ErrorCode::INVALID_PARAMS);
    // DECODE_MALLOC_FAILED collapses to MALLOC_FAILED (both share
    // RESOURCE_EXHAUSTED). This is documented loss; consumers must rely on
    // the error message string to distinguish if needed.
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::DECODE_MALLOC_FAILED)),
              ErrorCode::MALLOC_FAILED);
    // Likewise OUT_OF_VOCAB_RANGE collapses to LONG_PROMPT_ERROR (both share
    // OUT_OF_RANGE).
    EXPECT_EQ(transGrpcStatusToErrorCode(transErrorCodeToGrpc(ErrorCode::OUT_OF_VOCAB_RANGE)),
              ErrorCode::LONG_PROMPT_ERROR);
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
            const auto                  trace = context->client_metadata().find("traceparent");
            if (trace != context->client_metadata().end()) {
                captured_traceparent_.assign(trace->second.data(), trace->second.size());
            }
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
            first_aux_info->set_prefill_disk_reuse_len(3);
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
            second_aux_info->set_prefill_disk_reuse_len(6);
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
        if (peer_info_fail_) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "peer info unavailable");
        }
        const auto info =
            peer_info_responses_.empty() ?
                PrefillPeerInfo{1, 1} :
                peer_info_responses_[std::min<size_t>(peer_info_call_count_ - 1, peer_info_responses_.size() - 1)];
        response->set_tp_size(info.tp_size);
        response->set_cp_size(info.cp_size);
        return grpc::Status::OK;
    }

    void setPeerInfoResponses(std::vector<PrefillPeerInfo> responses) {
        std::lock_guard<std::mutex> lock(mutex_);
        peer_info_responses_  = std::move(responses);
        peer_info_call_count_ = 0;
    }

    int peerInfoCallCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return peer_info_call_count_;
    }

    void setPeerInfoFailure(bool fail) {
        std::lock_guard<std::mutex> lock(mutex_);
        peer_info_fail_ = fail;
    }

    bool waitStarted(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return started_cv_.wait_for(lock, timeout, [this]() { return started_.load(); });
    }

    bool waitCancelled(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cancel_cv_.wait_for(lock, timeout, [this]() { return cancelled_.load(); });
    }

    std::string capturedTraceparent() {
        std::lock_guard<std::mutex> lock(mutex_);
        return captured_traceparent_;
    }

    GenerateInputPB capturedRequest() {
        std::lock_guard<std::mutex> lock(mutex_);
        return captured_request_;
    }

private:
    Mode                         mode_;
    std::atomic<bool>            started_{false};
    std::atomic<bool>            cancelled_{false};
    std::mutex                   mutex_;
    std::condition_variable      started_cv_;
    std::condition_variable      cancel_cv_;
    GenerateInputPB              captured_request_;
    std::string                  captured_traceparent_;
    int                          peer_info_call_count_{0};
    bool                         peer_info_fail_{false};
    std::vector<PrefillPeerInfo> peer_info_responses_;
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
        const auto request_deadline_ms = currentTimeMs() + deadline_us / 1000;
        return caller_.callPrefill(&request, "127.0.0.1", server.port(), unique_key, request_deadline_ms, {}).value();
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
    PrefillServerCaller caller_;
};

TEST_F(PrefillServerCallerTest, AsyncClientTracePropagatesAndWaitsForRpcTerminalState) {
    namespace memory = opentelemetry::exporter::memory;
    telemetry::TelemetryRuntime::shutdown(5000);
    std::shared_ptr<memory::InMemorySpanData> span_data;
    auto                                      exporter = memory::InMemorySpanExporterFactory::Create(span_data);
    telemetry::TelemetryConfig                config;
    config.enabled = true;
    config.role    = "test";
    ASSERT_TRUE(telemetry::TelemetryRuntime::initWithExporter(std::move(exporter), config));
    std::vector<std::string> received_contexts;
    for (auto mode : {FakePrefillRpcService::Mode::kCaptureForwardedRequest,
                      FakePrefillRpcService::Mode::kWaitForCancel,
                      FakePrefillRpcService::Mode::kErrorChunkThenWaitForCancel}) {
        FakePrefillRpcServer server(std::make_unique<FakePrefillRpcService>(mode));
        ASSERT_TRUE(server.start());
        auto request      = makeSyncRequest(server, 1020, 5000);
        auto request_span = telemetry::TelemetryRuntime::tracer()->StartSpan("decode-client-trace-test");
        auto result       = caller_.callPrefill(
            &request, "127.0.0.1", server.port(), "trace-test", currentTimeMs() + 5000, request_span);
        ASSERT_TRUE(result.ok());
        auto context = result.value();
        ASSERT_TRUE(server.service()->waitStarted(std::chrono::seconds(3)));
        if (mode == FakePrefillRpcService::Mode::kWaitForCancel) {
            // Receipt alone does not end the streaming CLIENT span.
            EXPECT_FALSE(context->done());
            ASSERT_NE(context->async_state_->trace_guard, nullptr);
            EXPECT_TRUE(context->async_state_->trace_guard->sharedSpan()->IsRecording());
            context->cancel();
        }
        ASSERT_TRUE(waitDone(context));
        EXPECT_EQ(context->success(), mode == FakePrefillRpcService::Mode::kCaptureForwardedRequest);
        received_contexts.push_back(server.service()->capturedTraceparent());
        request_span->End();
    }
    EXPECT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    const auto spans  = span_data->GetSpans();
    size_t     calls  = 0;
    size_t     errors = 0;
    for (const auto& span : spans) {
        if (span->GetName() != "rtp_llm.call_prefill")
            continue;
        ++calls;
        errors += span->GetStatus() == opentelemetry::trace::StatusCode::kError;
        EXPECT_EQ(span->GetSpanKind(), opentelemetry::trace::SpanKind::kClient);
        EXPECT_TRUE(span->GetParentSpanId().IsValid());
        char trace_id[32];
        char span_id[16];
        span->GetTraceId().ToLowerBase16(trace_id);
        span->GetSpanId().ToLowerBase16(span_id);
        const auto propagated_prefix = "00-" + std::string(trace_id, 32) + "-" + std::string(span_id, 16) + "-";
        EXPECT_EQ(std::count_if(received_contexts.begin(),
                                received_contexts.end(),
                                [&](const std::string& value) { return value.find(propagated_prefix) == 0; }),
                  1);
        bool source_found = false;
        for (const auto& source : spans) {
            if (source->GetName() != "decode-client-trace-test")
                continue;
            char source_id[16];
            char actual_source_id[16];
            source->GetSpanId().ToLowerBase16(source_id);
            span->GetParentSpanId().ToLowerBase16(actual_source_id);
            source_found |= std::string(source_id, 16) == std::string(actual_source_id, 16);
        }
        EXPECT_TRUE(source_found);
    }
    EXPECT_EQ(calls, 3u);
    EXPECT_EQ(errors, 2u);
}

TEST_F(PrefillServerCallerTest, ErrorChunkMarksContextFailed) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kFirstChunkSnapshot));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1001, "snapshot");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->failed());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(context->errorInfo().ToString(),
              "first chunk error [received_at location=PrefillServerCallerContext observer_role=DECODE "
              "operation=GenerateStreamCall peer=127.0.0.1:"
                  + std::to_string(server.port()) + " request_id=1001 key=snapshot wire_code=1]");
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
    EXPECT_EQ(context->errorInfo().ToString(),
              "first chunk error [received_at location=PrefillServerCallerContext observer_role=DECODE "
              "operation=GenerateStreamCall peer=127.0.0.1:"
                  + std::to_string(server.port()) + " request_id=1011 key=error-cancel wire_code=1]");
    EXPECT_TRUE(server.service()->waitCancelled(std::chrono::seconds(1)));
}

TEST_F(PrefillServerCallerTest, MultipleChunksCompleteSuccessfully) {
    FakePrefillRpcServer server(
        std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kLaterChunkSnapshot));
    ASSERT_TRUE(server.start());

    auto context = callPrefill(server, 1007, "later-snapshot");
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(context->failed());
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

    const auto request_deadline_ms = currentTimeMs() + 1000;
    auto       context =
        caller_.callPrefill(&request, "127.0.0.1", server.port(), "decode-entrance-key", request_deadline_ms, {})
            .value();
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitDone(context));
    ASSERT_TRUE(context->success());

    const auto captured_request = server.service()->capturedRequest();
    EXPECT_EQ(captured_request.request_id(), 1010);
    EXPECT_EQ(captured_request.generate_config().timeout_ms(), request.generate_config().timeout_ms());
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

TEST_F(PrefillServerCallerTest, AsyncReaderCreationFailurePreservesCause) {
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

    const auto request_deadline_ms = currentTimeMs() + 5000;
    auto       result = caller_.callPrefill(&request, "127.0.0.1", 1, "null-reader", request_deadline_ms, {});
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::RPC_FINISH_FAILED);
    EXPECT_NE(result.status().ToString().find("null-reader"), std::string::npos);
    EXPECT_NE(result.status().ToString().find("127.0.0.1:1"), std::string::npos);
}

TEST_F(PrefillServerCallerTest, AsyncPrefillNormalizesRawIpv6TargetAddress) {
    GenerateInputPB request;
    request.set_request_id(1013);
    request.add_token_ids(1);
    request.mutable_generate_config()->set_timeout_ms(1);

    const auto request_deadline_ms = currentTimeMs() + 5000;
    auto       context = caller_.callPrefill(&request, "::1", 65535, "ipv6-target", request_deadline_ms, {}).value();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->prefill_addr_, "[::1]:65535");
    context->cancel();
}

TEST_F(PrefillServerCallerTest, AsyncPrefillRejectsInvalidTargetPort) {
    GenerateInputPB request;
    request.set_request_id(1014);
    request.add_token_ids(1);
    request.mutable_generate_config()->set_timeout_ms(1);

    const auto request_deadline_ms = currentTimeMs() + 5000;
    auto       result              = caller_.callPrefill(&request, "::1", 0, "bad-port", request_deadline_ms, {});
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(result.status().ToString().find("bad-port"), std::string::npos);
}

TEST_F(PrefillServerCallerTest, PeerInfoCallsRpcEveryTimeAndUsesLatestResponse) {
    auto service = std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kCaptureForwardedRequest);
    service->setPeerInfoResponses({{2, 1}, {4, 2}, {8, 4}});
    FakePrefillRpcServer server(std::move(service));
    ASSERT_TRUE(server.start());

    auto first =
        caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; }).value();
    ASSERT_EQ(first.tp_size, 2);
    ASSERT_EQ(first.cp_size, 1);
    EXPECT_EQ(server.service()->peerInfoCallCount(), 1);

    auto second =
        caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; }).value();
    EXPECT_EQ(second.tp_size, 4);
    EXPECT_EQ(second.cp_size, 2);
    EXPECT_EQ(server.service()->peerInfoCallCount(), 2);

    auto third =
        caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; }).value();
    EXPECT_EQ(third.tp_size, 8);
    EXPECT_EQ(third.cp_size, 4);
    EXPECT_EQ(server.service()->peerInfoCallCount(), 3);
}

TEST_F(PrefillServerCallerTest, PeerInfoFailureDoesNotReusePreviousSuccessAndNextCallReprobes) {
    auto service = std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kCaptureForwardedRequest);
    service->setPeerInfoResponses({{2, 1}, {4, 2}});
    FakePrefillRpcServer server(std::move(service));
    ASSERT_TRUE(server.start());

    ASSERT_EQ(caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; })
                  .value()
                  .tp_size,
              2);
    EXPECT_EQ(server.service()->peerInfoCallCount(), 1);

    server.service()->setPeerInfoFailure(true);
    auto failed = caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; });
    ASSERT_FALSE(failed.ok());
    EXPECT_EQ(failed.status().code(), ErrorCode::CONNECT_FAILED);
    EXPECT_NE(failed.status().ToString().find("GetPeerInfo"), std::string::npos);
    EXPECT_NE(failed.status().ToString().find("peer info unavailable"), std::string::npos);
    EXPECT_EQ(server.service()->peerInfoCallCount(), 2);

    server.service()->setPeerInfoFailure(false);
    auto recovered =
        caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; }).value();
    EXPECT_EQ(recovered.tp_size, 4);
    EXPECT_EQ(recovered.cp_size, 2);
    EXPECT_EQ(server.service()->peerInfoCallCount(), 3);
}

TEST_F(PrefillServerCallerTest, PeerInfoCarriesCpSizeIndependentlyFromTpSize) {
    auto service = std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kCaptureForwardedRequest);
    service->setPeerInfoResponses({{4, 2}});
    FakePrefillRpcServer server(std::move(service));
    ASSERT_TRUE(server.start());

    auto info =
        caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; }).value();
    EXPECT_EQ(info.tp_size, 4);
    EXPECT_EQ(info.cp_size, 2);
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
    EXPECT_NE(context->errorInfo().ToString().find("prefill failed before any response"), std::string::npos);
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

TEST_F(PrefillServerCallerTest, ConcurrentSyncCancellationKeepsOtherRequestAndRetiresCallback) {
    FakePrefillRpcServer first(std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kWaitForCancel));
    FakePrefillRpcServer second(std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kWaitForCancel));
    ASSERT_TRUE(first.start());
    ASSERT_TRUE(second.start());
    auto              first_request  = makeSyncRequest(first, 1030, 5000);
    auto              second_request = makeSyncRequest(second, 1031, 5000);
    std::atomic<bool> cancel_first{false};
    std::atomic<bool> cancel_second{false};
    std::atomic<int>  first_callback_count{0};
    auto              first_call  = std::async(std::launch::async, [&] {
        return caller_.callPrefill(nullptr, &first_request, nullptr, [&] {
            first_callback_count.fetch_add(1);
            return cancel_first.load();
        });
    });
    auto              second_call = std::async(std::launch::async, [&] {
        return caller_.callPrefill(nullptr, &second_request, nullptr, [&] { return cancel_second.load(); });
    });
    EXPECT_TRUE(first.service()->waitStarted(std::chrono::seconds(1)));
    EXPECT_TRUE(second.service()->waitStarted(std::chrono::seconds(1)));
    cancel_first.store(true);
    EXPECT_EQ(first_call.get().error_code(), grpc::StatusCode::CANCELLED);
    const auto retired_count = first_callback_count.load();
    EXPECT_EQ(second_call.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    EXPECT_FALSE(second.service()->waitCancelled(std::chrono::milliseconds(0)));
    cancel_second.store(true);
    EXPECT_EQ(second_call.get().error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(first_callback_count.load(), retired_count);
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

namespace rtp_llm::test {
TEST_F(PrefillServerCallerTest, PeerInfoRejectsMissingCpSizeWithoutFallback) {
    auto service = std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kCaptureForwardedRequest);
    service->setPeerInfoResponses({{4, 0}});
    FakePrefillRpcServer server(std::move(service));
    ASSERT_TRUE(server.start());
    auto result = caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; });
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(result.status().ToString().find("cp_size=0"), std::string::npos);
    EXPECT_NE(result.status().ToString().find(std::to_string(server.port())), std::string::npos);
}

TEST_F(PrefillServerCallerTest, PeerInfoAcceptsTpCpWithoutDpAddresses) {
    auto service = std::make_unique<FakePrefillRpcService>(FakePrefillRpcService::Mode::kCaptureForwardedRequest);
    service->setPeerInfoResponses({{4, 1}});
    FakePrefillRpcServer server(std::move(service));
    ASSERT_TRUE(server.start());
    auto result = caller_.getPrefillPeerInfo("127.0.0.1", server.port(), 0, GenerateInputPB{}, [] { return false; });
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().tp_size, 4);
    EXPECT_EQ(result.value().cp_size, 1);
}
}  // namespace rtp_llm::test
