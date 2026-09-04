#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "opentelemetry/sdk/trace/span_data.h"
#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

namespace rtp_llm {
namespace {

class PartialEnqueueEngine: public EngineBase {
public:
    // EngineBase requires a scheduler; this inert implementation keeps the
    // batch-enqueue tests focused on their per-context cancellation path.
    class NoopScheduler: public SchedulerBase {
    public:
        absl::Status enqueue(const GenerateStreamPtr&) override {
            return absl::OkStatus();
        }
        std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
        enqueueGroup(const std::vector<GenerateStreamPtr>&) override {
            return {{}, {}};
        }
        absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
            return std::list<GenerateStreamPtr>();
        }
        absl::Status stop() override {
            return absl::OkStatus();
        }
        bool empty() override {
            return true;
        }
        int64_t lastScheduleTime() override {
            return 0;
        }
        int64_t onflightStreams() override {
            return 0;
        }
    };

    PartialEnqueueEngine(): EngineBase(EngineInitParams()) {
        scheduler_ = std::make_unique<NoopScheduler>();
    }

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void enqueue(std::shared_ptr<GenerateStream>&) override {}
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>&) override {
        ++enqueue_multiple_calls;
        return {enqueue_successes, streams};
    }
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return KVCacheInfo();
    }

    std::vector<bool>              enqueue_successes;
    std::vector<GenerateStreamPtr> streams;
    int                            enqueue_multiple_calls{0};
};

class TestPrefillBatchRpcServer: public PrefillBatchRpcServer {
public:
    grpc::Status EnqueueGroup(grpc::ServerContext*,
                              const EnqueueGroupRequestPB* request,
                              EnqueueBatchResponsePB*      response) override {
        ++enqueue_group_calls;
        captured_group_request = *request;
        response->set_batch_id(request->batch_id());
        const int result_count = request->requests_size() - (omit_last_result ? 1 : 0);
        for (int i = 0; i < result_count; ++i) {
            const auto& group_input = request->requests(i);
            if (group_input.has_input()) {
                response->add_successes()->set_request_id(group_input.input().request_id());
            } else {
                auto* error = response->add_errors();
                error->set_request_id(0);
                error->mutable_error_info()->set_error_code(grpc::StatusCode::INVALID_ARGUMENT);
                error->mutable_error_info()->set_error_message("missing input");
            }
        }
        return grpc::Status::OK;
    }

    void setParallelism(int64_t dp_size, int64_t dp_rank) {
        maga_init_params_.parallelism_config.dp_size = dp_size;
        maga_init_params_.parallelism_config.dp_rank = dp_rank;
    }

    grpc::Status outwardStatus(PrefillGenerateContext& context, const grpc::Status& fallback) {
        return preferPriorityPreemption(context, fallback);
    }

    int                   enqueue_group_calls = 0;
    bool                  omit_last_result    = false;
    EnqueueGroupRequestPB captured_group_request;
};

class TracingDecodeRpcService final: public RpcService::Service {
public:
    grpc::Status RemoteGenerate(grpc::ServerContext*                                            server_context,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        grpc::Status status = grpc::Status::OK;
        auto         span   = telemetry::startRpcServerSpan(
            "rtp_llm.decode_remote_generate", server_context, true, "RpcService/RemoteGenerate");
        telemetry::GrpcStatusSpanGuard span_guard(span, &status);

        GenerateRequestPB request;
        if (!stream->Read(&request)) {
            status = grpc::Status(grpc::StatusCode::INTERNAL, "missing allocate request");
            return status;
        }
        span_guard.setAttribute(telemetry::kAttrRequestId, std::to_string(request.request_id()));
        span_guard.setAttribute(telemetry::kAttrRtpLlmRequestId, request.request_id());
        GenerateOutputsPB response;
        if (!stream->Write(response)) {
            status = grpc::Status(grpc::StatusCode::INTERNAL, "write allocate response failed");
            return status;
        }
        while (stream->Read(&request)) {}
        return status;
    }
};

class TracingDecodeRpcServer {
public:
    ~TracingDecodeRpcServer() {
        if (server_) {
            server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
            server_->Wait();
        }
    }

    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        return server_ != nullptr && port_ > 0;
    }

    int port() const {
        return port_;
    }

private:
    TracingDecodeRpcService       service_;
    std::unique_ptr<grpc::Server> server_;
    int                           port_{0};
};

EnqueueBatchExternalInputPB* addInput(EnqueueBatchDpSlotPB* slot, int64_t request_id) {
    auto* external_input = slot->add_requests();
    external_input->mutable_input()->set_request_id(request_id);
    return external_input;
}

std::set<int64_t> successIds(const EnqueueBatchResponsePB& response) {
    std::set<int64_t> ids;
    for (const auto& success : response.successes()) {
        ids.insert(success.request_id());
    }
    return ids;
}

std::shared_ptr<DeferredPrefillContext> makeDeferred(PrefillBatchRpcServer& server, int64_t request_id) {
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    auto input   = std::make_shared<GenerateInputPB>();
    input->set_request_id(request_id);
    input->mutable_group_id()->set_value(99);
    RPCContext rpc_context{input.get(), nullptr};
    auto       context  = std::make_unique<PrefillGenerateContext>(&server.resource(),
                                                            rpc_context,
                                                            /*timeout_ms=*/0,
                                                            /*server_context=*/nullptr,
                                                            server.metrics_reporter_,
                                                            server.meta_);
    auto       deferred = std::make_shared<DeferredPrefillContext>();
    deferred->context   = std::move(context);
    deferred->input     = std::move(input);
    return deferred;
}

std::shared_ptr<GenerateInput> makeGenerateInput(int64_t request_id) {
    auto input             = std::make_shared<GenerateInput>();
    input->request_id      = request_id;
    input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    return input;
}

GenerateStreamPtr makeGenerateStream(const std::shared_ptr<GenerateInput>& input) {
    ModelConfig model_config;
    model_config.max_seq_len = 128;
    RuntimeConfig runtime_config;
    return std::make_shared<NormalGenerateStream>(
        input, model_config, runtime_config, ResourceContext{}, /*metrics_reporter=*/nullptr);
}

namespace trace_api       = opentelemetry::trace;
namespace trace_sdk       = opentelemetry::sdk::trace;
namespace memory_exporter = opentelemetry::exporter::memory;
namespace nostd           = opentelemetry::nostd;

std::string toHex(const trace_api::TraceId& trace_id) {
    char value[32];
    trace_id.ToLowerBase16(value);
    return std::string(value, 32);
}

std::string toHex(const trace_api::SpanId& span_id) {
    char value[16];
    span_id.ToLowerBase16(value);
    return std::string(value, 16);
}

std::vector<const trace_sdk::SpanData*> findSpans(const std::vector<std::unique_ptr<trace_sdk::SpanData>>& spans,
                                                  const std::string&                                       name) {
    std::vector<const trace_sdk::SpanData*> matches;
    for (const auto& span : spans) {
        if (span->GetName() == name) {
            matches.push_back(span.get());
        }
    }
    return matches;
}

void setTraceContext(GenerateInputPB& input, const std::string& trace_id, const std::string& parent_span_id) {
    auto* trace_context = input.mutable_request_info()->mutable_trace_context();
    trace_context->set_traceparent("00-" + trace_id + "-" + parent_span_id + "-01");
}

class PrefillBatchTraceTest: public ::testing::Test {
protected:
    void SetUp() override {
        telemetry::TelemetryRuntime::shutdown(5000);
        auto                       exporter = memory_exporter::InMemorySpanExporterFactory::Create(span_data_);
        telemetry::TelemetryConfig config;
        config.enabled = true;
        config.role    = "test";
        config.tp_rank = 0;
        ASSERT_TRUE(telemetry::TelemetryRuntime::initWithExporter(std::move(exporter), config));
    }

    void TearDown() override {
        telemetry::TelemetryRuntime::shutdown(5000);
    }

    std::vector<std::unique_ptr<trace_sdk::SpanData>> finishTelemetry() {
        EXPECT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
        return span_data_->GetSpans();
    }

    std::shared_ptr<memory_exporter::InMemorySpanData> span_data_;
};

void buildReadySlots(PrefillBatchRpcServer&                         server,
                     const std::vector<int64_t>&                    request_ids,
                     std::vector<PrefillBatchRpcServer::BatchSlot>& slots,
                     std::vector<PrefillBatchRpcServer::ReadySlot>& ready_slots) {
    slots.resize(request_ids.size());
    ready_slots.reserve(request_ids.size());
    for (size_t i = 0; i < request_ids.size(); ++i) {
        const auto request_id = request_ids[i];
        auto&      slot       = slots[i];
        slot.input            = std::make_shared<GenerateInputPB>();
        slot.input->set_request_id(request_id);
        RPCContext rpc_context{slot.input.get(), nullptr};
        auto       context      = std::make_unique<PrefillGenerateContext>(&server.resource(),
                                                                rpc_context,
                                                                /*timeout_ms=*/0,
                                                                /*server_context=*/nullptr,
                                                                server.metrics_reporter_,
                                                                server.meta_);
        context->generate_input = makeGenerateInput(request_id);
        slot.deferred           = std::make_shared<DeferredPrefillContext>();
        slot.deferred->context  = std::move(context);
        slot.deferred->input    = slot.input;
        ready_slots.push_back(PrefillBatchRpcServer::ReadySlot{&slot, slot.deferred});
    }
}

TEST(PrefillBatchRpcServerTest, FlattensLocalSlotsAndPropagatesFetchLease) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueBatchRequestPB request;
    request.set_batch_id(101);
    request.set_fetch_attach_timeout_ms(4321);
    auto* first_slot = request.add_dp_slots();
    first_slot->set_dp_rank(0);
    addInput(first_slot, 11);
    first_slot->add_requests();
    auto* second_slot = request.add_dp_slots();
    second_slot->set_dp_rank(0);
    addInput(second_slot, 12);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());

    EXPECT_EQ(server.enqueue_group_calls, 1);
    EXPECT_EQ(server.captured_group_request.batch_id(), 101);
    EXPECT_EQ(server.captured_group_request.dp_rank(), 0);
    EXPECT_EQ(server.captured_group_request.fetch_attach_timeout_ms(), 4321);
    ASSERT_EQ(server.captured_group_request.requests_size(), 3);
    EXPECT_EQ(server.captured_group_request.requests(0).input().request_id(), 11);
    EXPECT_FALSE(server.captured_group_request.requests(1).has_input());
    EXPECT_EQ(server.captured_group_request.requests(2).input().request_id(), 12);
    EXPECT_EQ(response.batch_id(), 101);
    EXPECT_EQ(successIds(response), (std::set<int64_t>{11, 12}));
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 0);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 3);
}

TEST(PrefillBatchRpcServerTest, RejectsInvalidRankWithoutBlockingLocalRequests) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    EnqueueBatchRequestPB request;
    auto*                 local_slot = request.add_dp_slots();
    local_slot->set_dp_rank(0);
    addInput(local_slot, 21);
    auto* invalid_slot = request.add_dp_slots();
    invalid_slot->set_dp_rank(1);
    addInput(invalid_slot, 22);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());
    EXPECT_EQ(server.enqueue_group_calls, 1);
    ASSERT_EQ(server.captured_group_request.requests_size(), 1);
    EXPECT_EQ(server.captured_group_request.requests(0).input().request_id(), 21);
    EXPECT_EQ(successIds(response), (std::set<int64_t>{21}));
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 22);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 2);
}

TEST(PrefillBatchRpcServerTest, RejectsWholeBatchWhenRequestIdIsDuplicatedAcrossSlots) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    EnqueueBatchRequestPB request;
    auto*                 first = request.add_dp_slots();
    first->set_dp_rank(0);
    addInput(first, 31);
    auto* second = request.add_dp_slots();
    second->set_dp_rank(0);
    addInput(second, 31);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());
    EXPECT_EQ(server.enqueue_group_calls, 0);
    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(response.errors_size(), 2);
    EXPECT_EQ(response.errors(0).request_id(), 31);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(response.errors(1).request_id(), 31);
    EXPECT_EQ(response.errors(1).error_info().error_code(), grpc::StatusCode::ALREADY_EXISTS);
}

TEST(PrefillBatchRpcServerTest, FailsFastWhenMultiDpIsConfigured) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/2, /*dp_rank=*/0);
    EnqueueBatchRequestPB  request;
    EnqueueBatchResponsePB response;
    EXPECT_ANY_THROW(server.EnqueueBatch(nullptr, &request, &response));
    EXPECT_EQ(server.enqueue_group_calls, 0);
}

TEST(PrefillBatchRpcServerTest, FailsFastWhenEnqueueGroupOmitsAResult) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    server.omit_last_result = true;
    EnqueueBatchRequestPB request;
    auto*                 slot = request.add_dp_slots();
    slot->set_dp_rank(0);
    addInput(slot, 51);
    addInput(slot, 52);
    EnqueueBatchResponsePB response;
    EXPECT_ANY_THROW(server.EnqueueBatch(nullptr, &request, &response));
    EXPECT_EQ(server.enqueue_group_calls, 1);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 1);
}

TEST(PrefillBatchRpcServerTest, AdmitGroupCopiesBatchMetadataAndLease) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    EnqueueGroupRequestPB request;
    request.set_batch_id(7);
    request.set_dp_rank(0);
    request.set_fetch_attach_timeout_ms(999);
    request.add_requests()->mutable_input()->set_request_id(61);
    request.add_requests()->mutable_input()->set_request_id(62);
    EnqueueBatchResponsePB                        response;
    std::vector<PrefillBatchRpcServer::BatchSlot> slots;

    ASSERT_TRUE(server.admitGroup(&request, &response, slots).ok());
    ASSERT_EQ(slots.size(), 2);
    EXPECT_EQ(slots[0].input->group_size(), 2);
    ASSERT_TRUE(slots[0].input->has_group_id());
    EXPECT_EQ(slots[0].input->group_id().value(), 7);
    EXPECT_EQ(slots[0].fetch_attach_timeout_ms, 999);
    EXPECT_EQ(slots[1].input->group_size(), 2);
    ASSERT_TRUE(slots[1].input->has_group_id());
    EXPECT_EQ(slots[1].input->group_id().value(), 7);
    EXPECT_EQ(slots[1].fetch_attach_timeout_ms, 999);
}

TEST(PrefillBatchRpcServerTest, ContextCapturesAdmittedEnvelopeBeforeQueryConversion) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    EnqueueGroupRequestPB request;
    request.set_batch_id(8);
    request.set_dp_rank(0);
    request.add_requests()->mutable_input()->set_request_id(63);
    EnqueueBatchResponsePB                        response;
    std::vector<PrefillBatchRpcServer::BatchSlot> slots;

    ASSERT_TRUE(server.admitGroup(&request, &response, slots).ok());
    ASSERT_EQ(slots.size(), 1);
    server.buildSlotContexts(slots);
    ASSERT_NE(slots[0].deferred, nullptr);
    ASSERT_EQ(slots[0].deferred->context->generate_input, nullptr);

    EXPECT_EQ(slots[0].deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    auto canceling = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_EQ(canceling.running_task_info_list[0].request_id, 63);
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, 8);
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);
}

TEST(PrefillBatchRpcServerTest, PartialSchedulerRejectionCleansRejectedPrefillResources) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<PartialEnqueueEngine>();
    server.engine_ = engine;

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {1001, 1002}, slots, ready_slots);

    engine->streams = {
        makeGenerateStream(ready_slots[0].deferred->context->generate_input),
        makeGenerateStream(ready_slots[1].deferred->context->generate_input),
    };
    engine->streams[1]->reportError(ErrorCode::MALLOC_FAILED, "scheduler rejected request");
    engine->enqueue_successes  = {true, false};
    auto accepted_deferred     = ready_slots[0].deferred;
    auto rejected_cancel_state = ready_slots[1].deferred->context->cancel_state;

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.enqueueGroupStreams(ready_slots, &response).ok());

    ASSERT_EQ(ready_slots.size(), 1);
    EXPECT_EQ(ready_slots[0].slot, &slots[0]);
    EXPECT_EQ(ready_slots[0].deferred, accepted_deferred);
    EXPECT_TRUE(rejected_cancel_state->load());
    EXPECT_EQ(engine->streams[1]->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 1002);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);

    auto schedule_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(schedule_info.running_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.running_task_info_list[0].request_id, 1001);
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.finished_task_info_list[0].request_id, 1002);
    EXPECT_EQ(schedule_info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::MALLOC_FAILED));

    accepted_deferred->context->cancel_state->store(true);
    accepted_deferred.reset();
}

TEST(PrefillBatchRpcServerTest, LatchedPriorityCancelBeforeEnqueuePreservesRaw8429) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<PartialEnqueueEngine>();
    server.engine_ = engine;

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {1013}, slots, ready_slots);
    ASSERT_EQ(ready_slots[0].deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.enqueueGroupStreams(ready_slots, &response).ok());

    EXPECT_TRUE(ready_slots.empty());
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 1013);
    EXPECT_EQ(response.errors(0).error_info().error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, RejectsWhenEnqueueMultipleReordersStreams) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<PartialEnqueueEngine>();
    server.engine_ = engine;

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {2001, 2002}, slots, ready_slots);

    engine->streams = {
        makeGenerateStream(ready_slots[1].deferred->context->generate_input),
        makeGenerateStream(ready_slots[0].deferred->context->generate_input),
    };
    engine->enqueue_successes = {true, true};

    EnqueueBatchResponsePB response;
    const auto             status = server.enqueueGroupStreams(ready_slots, &response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_NE(status.error_message().find("result order mismatch"), std::string::npos);
    for (auto& ready_slot : ready_slots) {
        server.rejectSlot(ready_slot, status, &response);
    }
    EXPECT_EQ(response.errors_size(), 2);
}

TEST(PrefillBatchRpcServerTest, DoesNotAckSuccessAfterRequestDeadlineExpires) {
    PrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {2010}, slots, ready_slots);
    auto deferred                            = ready_slots[0].deferred;
    deferred->context->request_timeout_ms    = 1;
    deferred->context->request_begin_time_us = currentTimeUs() - 10 * 1000;
    auto cancel_state                        = deferred->context->cancel_state;
    ASSERT_TRUE(server.deferred_contexts_->store(2010, deferred).ok());

    EnqueueBatchResponsePB response;
    server.publishSlot(ready_slots[0], &response);

    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(cancel_state->load());
    EXPECT_EQ(server.deferred_contexts_->size(), 0);
}

TEST(PrefillBatchRpcServerTest, DeferredContextMapTakesAndRemovesContext) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3001);

    ASSERT_TRUE(contexts->store(3001, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3001, deferred, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(deferred->finishOperation());
    EXPECT_EQ(contexts->size(), 1);
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3001, claimed).ok());
    EXPECT_EQ(claimed, deferred);
    EXPECT_EQ(contexts->size(), 0);

    std::shared_ptr<DeferredPrefillContext> duplicate_fetch;
    const auto                              duplicate_fetch_status = contexts->take(3001, duplicate_fetch);
    EXPECT_EQ(duplicate_fetch_status.error_code(), grpc::StatusCode::NOT_FOUND);
    EXPECT_EQ(duplicate_fetch, nullptr);
}

TEST(PrefillBatchRpcServerTest, ActiveOperationOwnsPriorityFinalizationOnExit) {
    DeferredPrefillContext deferred;
    EXPECT_FALSE(deferred.requestPriorityFinalization());
    EXPECT_TRUE(deferred.finishOperation());
    EXPECT_FALSE(deferred.finishOperation());
}

TEST(PrefillBatchRpcServerTest, IdleContextCanBeFinalizedWithoutOperationWaiter) {
    DeferredPrefillContext deferred;
    EXPECT_FALSE(deferred.finishOperation());
    EXPECT_TRUE(deferred.requestPriorityFinalization());
    EXPECT_FALSE(deferred.requestPriorityFinalization());
}

TEST(PrefillBatchRpcServerTest, PriorityPreemptionCancelRemainsRoutableAfterFetchTakesContext) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3009);

    ASSERT_TRUE(contexts->registerActive(3009, deferred).ok());
    ASSERT_TRUE(contexts->store(3009, deferred).ok());
    EXPECT_FALSE(deferred->finishOperation());
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3009, claimed).ok());

    ASSERT_EQ(contexts->cancelByPriorityPreemption(3009), PriorityCancelResult::ACCEPTED);
    EXPECT_TRUE(claimed->context->cancel_state->load());
    EXPECT_TRUE(claimed->context->isPriorityPreempted());

    contexts->finish(3009, claimed.get());
    // A retry joins the already-installed weak-ACK latch.
    EXPECT_EQ(contexts->cancelByPriorityPreemption(3009), PriorityCancelResult::ACCEPTED);
}

TEST(PrefillBatchRpcServerTest, LateCancelAfterNaturalFinishDoesNotInstallAbsentFence) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3025);

    ASSERT_TRUE(contexts->registerActive(3025, deferred).ok());
    ASSERT_TRUE(contexts->store(3025, deferred).ok());
    EXPECT_FALSE(deferred->finishOperation());
    std::shared_ptr<DeferredPrefillContext> fetched;
    ASSERT_TRUE(contexts->take(3025, fetched).ok());
    contexts->finish(3025, fetched.get());

    EXPECT_EQ(contexts->cancelByPriorityPreemption(3025), PriorityCancelResult::NOT_FOUND);

    // NOT_FOUND is deliberately conservative: unlike TOMBSTONED it does not
    // claim that cancel-before-enqueue was fenced, and therefore must not
    // poison a later registration with a synthetic 8429.
    auto replacement = makeDeferred(server, 3025);
    EXPECT_TRUE(contexts->registerActive(3025, replacement).ok());
}

TEST(PrefillBatchRpcServerTest, NaturalFinishAndCancelHaveOneLinearizedOutcome) {
    PrefillBatchRpcServer server;

    for (int64_t request_id = 3200; request_id < 3300; ++request_id) {
        auto contexts = std::make_shared<DeferredPrefillContextMap>();
        auto deferred = makeDeferred(server, request_id);
        ASSERT_TRUE(contexts->registerActive(request_id, deferred).ok());

        std::atomic<int>     ready{0};
        std::atomic<bool>    start{false};
        PriorityCancelResult cancel_result = PriorityCancelResult::TOMBSTONED;
        std::thread          finish_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            contexts->finish(request_id, deferred.get());
        });
        std::thread          cancel_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            cancel_result = contexts->cancelByPriorityPreemption(request_id);
        });
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        finish_thread.join();
        cancel_thread.join();

        EXPECT_NE(cancel_result, PriorityCancelResult::TOMBSTONED);
        EXPECT_TRUE(cancel_result == PriorityCancelResult::ACCEPTED
                    || cancel_result == PriorityCancelResult::NOT_FOUND);
    }
}

TEST(PrefillBatchRpcServerTest, PreparingContextIsVisibleToCancelBeforeStore) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3010);

    ASSERT_TRUE(contexts->registerActive(3010, deferred).ok());
    ASSERT_EQ(contexts->cancelByPriorityPreemption(3010), PriorityCancelResult::ACCEPTED);

    EXPECT_TRUE(deferred->context->isPriorityPreempted());
    EXPECT_TRUE(deferred->context->cancel_state->load());
    auto status_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(status_info.running_task_info_list.size(), 1);
    EXPECT_EQ(status_info.running_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELING);
    contexts->finish(3010, deferred.get());
}

TEST(PrefillBatchRpcServerTest, CancelBeforeRegisterInstallsTombstoneAndRejectsEnqueueWith8429) {
    PrefillBatchRpcServer server;
    auto                  engine = std::make_shared<PartialEnqueueEngine>();
    server.engine_               = engine;
    CancelRequestPB request;
    request.set_request_id(3011);
    CancelResponsePB response;

    ASSERT_TRUE(server.Cancel(nullptr, &request, &response).ok());
    ASSERT_EQ(response.status(), CancelStatusPB::CANCEL_STATUS_TOMBSTONED);
    CancelResponsePB retry_response;
    ASSERT_TRUE(server.Cancel(nullptr, &request, &retry_response).ok());
    EXPECT_EQ(retry_response.status(), CancelStatusPB::CANCEL_STATUS_TOMBSTONED);

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    PrefillBatchRpcServer::BatchSlot              slot;
    slot.input = std::make_shared<GenerateInputPB>();
    slot.input->set_request_id(3011);
    slots.push_back(std::move(slot));
    EnqueueBatchResponsePB enqueue_response;
    ASSERT_TRUE(server.acceptGroup(std::move(slots), &enqueue_response).ok());

    EXPECT_EQ(engine->enqueue_multiple_calls, 0);
    ASSERT_EQ(enqueue_response.errors_size(), 1);
    EXPECT_EQ(enqueue_response.errors(0).request_id(), 3011);
    EXPECT_EQ(enqueue_response.errors(0).error_info().error_code(),
              static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, CancelRejectsZeroRequestId) {
    PrefillBatchRpcServer server;
    auto                  engine = std::make_shared<PartialEnqueueEngine>();
    server.engine_               = engine;
    CancelRequestPB  request;
    CancelResponsePB response;

    auto status = server.Cancel(nullptr, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(PrefillBatchRpcServerTest, CancelActiveRequestLatches8429WithoutSchedulerIntent) {
    PrefillBatchRpcServer server;
    auto                  engine = std::make_shared<PartialEnqueueEngine>();
    server.engine_               = engine;
    auto deferred                = makeDeferred(server, 3012);
    ASSERT_TRUE(server.deferred_contexts_->registerActive(3012, deferred).ok());
    CancelRequestPB request;
    request.set_request_id(3012);
    CancelResponsePB response;

    ASSERT_TRUE(server.Cancel(nullptr, &request, &response).ok());

    EXPECT_EQ(response.status(), CancelStatusPB::CANCEL_STATUS_ACCEPTED);
    EXPECT_TRUE(deferred->context->isPriorityPreempted());

    // A tombstone retry acknowledges the already-installed weak latch but
    // must not create a scheduler intent.
    CancelResponsePB retry_response;
    ASSERT_TRUE(server.Cancel(nullptr, &request, &retry_response).ok());
    EXPECT_EQ(retry_response.status(), CancelStatusPB::CANCEL_STATUS_ACCEPTED);
    auto status_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(status_info.running_task_info_list.size(), 1);
    EXPECT_EQ(status_info.running_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELING);
    server.deferred_contexts_->finish(3012, deferred.get());
}

TEST(PrefillBatchRpcServerTest, TypedPriorityTerminalDowngradesActiveCancelAckToTombstone) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3017);
    ASSERT_TRUE(contexts->registerActive(3017, deferred).ok());

    EXPECT_EQ(contexts->cancelByPriorityPreemption(3017), PriorityCancelResult::ACCEPTED);
    contexts->publishPriorityPreemptionCanceled(3017, deferred.get());

    EXPECT_EQ(contexts->cancelByPriorityPreemption(3017), PriorityCancelResult::TOMBSTONED);
    auto replacement = makeDeferred(server, 3017);
    auto status      = contexts->registerActive(3017, replacement);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
}

TEST(PrefillBatchRpcServerTest, CancelAndRegisterHaveOneLinearizedOutcome) {
    PrefillBatchRpcServer server;

    for (int64_t request_id = 3100; request_id < 3200; ++request_id) {
        auto contexts = std::make_shared<DeferredPrefillContextMap>();
        auto deferred = makeDeferred(server, request_id);

        std::atomic<int>     ready{0};
        std::atomic<bool>    start{false};
        grpc::Status         registration_status;
        PriorityCancelResult cancel_result = PriorityCancelResult::NOT_FOUND;

        std::thread register_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            registration_status = contexts->registerActive(request_id, deferred);
        });
        std::thread cancel_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            cancel_result = contexts->cancelByPriorityPreemption(request_id);
        });
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        register_thread.join();
        cancel_thread.join();

        if (registration_status.ok()) {
            EXPECT_EQ(cancel_result, PriorityCancelResult::ACCEPTED);
            EXPECT_TRUE(deferred->context->isPriorityPreempted());
            continue;
        }

        EXPECT_EQ(cancel_result, PriorityCancelResult::TOMBSTONED);
        EXPECT_EQ(registration_status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
        ErrorDetailsPB details;
        ASSERT_TRUE(details.ParseFromString(registration_status.error_details()));
        EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
        EXPECT_FALSE(deferred->context->isPriorityPreempted());
    }
}

TEST(PrefillBatchRpcServerTest, CancelStatusWireNumbersRemainBackwardCompatible) {
    EXPECT_EQ(CancelStatusPB::CANCEL_STATUS_UNSPECIFIED, 0);
    EXPECT_EQ(CancelStatusPB::CANCEL_STATUS_ACCEPTED, 1);
    EXPECT_EQ(CancelStatusPB::CANCEL_STATUS_NOT_FOUND, 2);
    EXPECT_EQ(CancelStatusPB::CANCEL_STATUS_TOMBSTONED, 3);
}

TEST(PrefillBatchRpcServerTest, AcceptedPriorityPreemptionOverridesPrepareFailure) {
    TestPrefillBatchRpcServer server;
    auto                      contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                      deferred = makeDeferred(server, 3014);
    ASSERT_TRUE(contexts->registerActive(3014, deferred).ok());

    ASSERT_EQ(contexts->cancelByPriorityPreemption(3014), PriorityCancelResult::ACCEPTED);
    contexts->finish(3014, deferred.get());
    auto           outward = server.outwardStatus(*deferred->context,
                                        grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource failed"));
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(outward.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(contexts->cancelByPriorityPreemption(3014), PriorityCancelResult::ACCEPTED);
}

TEST(PrefillBatchRpcServerTest, OtherTerminalBeforePriorityCancelReturnsNotFound) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3018);
    ASSERT_TRUE(contexts->registerActive(3018, deferred).ok());

    ASSERT_TRUE(deferred->context->tryMarkOtherTerminal());
    bool                                    newly_installed = true;
    std::shared_ptr<DeferredPrefillContext> canceled;
    EXPECT_EQ(contexts->cancelByPriorityPreemption(3018, canceled, &newly_installed), PriorityCancelResult::NOT_FOUND);
    EXPECT_FALSE(newly_installed);
    EXPECT_EQ(canceled, nullptr);
    EXPECT_EQ(deferred->context->terminalCause(), PrefillTerminalCause::OTHER);
}

TEST(PrefillBatchRpcServerTest, PriorityCancelBeforeOtherTerminalPreserves8429) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3019);
    ASSERT_TRUE(contexts->registerActive(3019, deferred).ok());

    bool                                    newly_installed = false;
    std::shared_ptr<DeferredPrefillContext> canceled;
    ASSERT_EQ(contexts->cancelByPriorityPreemption(3019, canceled, &newly_installed), PriorityCancelResult::ACCEPTED);
    EXPECT_TRUE(newly_installed);
    EXPECT_FALSE(deferred->context->tryMarkOtherTerminal());
    EXPECT_EQ(deferred->context->terminalCause(), PrefillTerminalCause::PRIORITY_PREEMPTION);
}

TEST(PrefillBatchRpcServerTest, PriorityAndOtherTerminalBarrierHasExactlyOneWinner) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3020);
    ASSERT_TRUE(contexts->registerActive(3020, deferred).ok());

    std::atomic<int>  ready{0};
    std::atomic<bool> start{false};
    bool              other_won       = false;
    bool              priority_won    = false;
    bool              newly_installed = false;
    std::thread       other_thread([&] {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        other_won = deferred->context->tryMarkOtherTerminal();
    });
    std::thread       priority_thread([&] {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        std::shared_ptr<DeferredPrefillContext> canceled;
        priority_won =
            contexts->cancelByPriorityPreemption(3020, canceled, &newly_installed) == PriorityCancelResult::ACCEPTED;
    });
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    other_thread.join();
    priority_thread.join();

    EXPECT_NE(other_won, priority_won);
    EXPECT_EQ(priority_won, newly_installed);
    EXPECT_EQ(deferred->context->terminalCause(),
              priority_won ? PrefillTerminalCause::PRIORITY_PREEMPTION : PrefillTerminalCause::OTHER);
}

TEST(PrefillBatchRpcServerTest, CanceledPrepareSlotCanFinalizeBeforeSiblingLeavesPrepare) {
    PrefillBatchRpcServer server;
    auto                  canceled = makeDeferred(server, 3021);
    auto                  sibling  = makeDeferred(server, 3022);

    EXPECT_EQ(canceled->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    EXPECT_FALSE(canceled->requestPriorityFinalization());
    EXPECT_TRUE(canceled->finishOperation());

    // The sibling still owns PREPARE, proving finalizer ownership is per slot.
    EXPECT_FALSE(sibling->tryStartOperation().started);
    EXPECT_FALSE(sibling->finishOperation());
    EXPECT_TRUE(sibling->tryStartOperation().started);
}

TEST(PrefillBatchRpcServerTest, TerminalCauseAloneCannotStartFinalizerBeforeCancelRegistersIt) {
    PrefillBatchRpcServer server;
    auto                  deferred = makeDeferred(server, 3023);

    EXPECT_EQ(deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    // The operation may exit after the terminal CAS but before Cancel has
    // registered finalization. It must not claim the finalizer prematurely.
    EXPECT_FALSE(deferred->finishOperation());
    EXPECT_TRUE(deferred->requestPriorityFinalization());
}

TEST(PrefillBatchRpcServerTest, PriorityTerminalCannotClaimLogicalFinalizer) {
    PrefillBatchRpcServer server;
    auto                  deferred = makeDeferred(server, 3026);

    ASSERT_EQ(deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    EXPECT_FALSE(deferred->finishOperation());
    // The priority finalizer has not been registered yet, but the terminal
    // cause is already priority-owned. Logical finalization must not race it.
    EXPECT_FALSE(deferred->requestLogicalFinalization());
    EXPECT_TRUE(deferred->requestPriorityFinalization());
}

TEST(PrefillBatchRpcServerTest, FetchAfterAcceptedPriorityCancelReturns8429Tombstone) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3015);
    ASSERT_TRUE(contexts->registerActive(3015, deferred).ok());
    ASSERT_TRUE(contexts->store(3015, deferred).ok());
    EXPECT_FALSE(deferred->finishOperation());

    ASSERT_EQ(contexts->cancelByPriorityPreemption(3015), PriorityCancelResult::ACCEPTED);

    std::shared_ptr<DeferredPrefillContext> fetched;
    auto                                    status = contexts->take(3015, fetched);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(fetched, nullptr);
}

TEST(PrefillBatchRpcServerTest, PriorityPreemptionReturnsRaw8429InErrorDetails) {
    TestPrefillBatchRpcServer server;
    auto                      deferred = makeDeferred(server, 3013);
    deferred->context->requestPriorityPreempt();

    auto status =
        server.outwardStatus(*deferred->context, grpc::Status(grpc::StatusCode::CANCELLED, "downstream cancelled"));
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, PrefillFinalizerPublishesCanceled8429ExactlyOnce) {
    PrefillBatchRpcServer server;
    auto                  deferred = makeDeferred(server, 3016);
    deferred->context->requestPriorityPreempt();

    // Cancel wins before QueryConverter creates generate_input or a local
    // stream reaches RuntimeMeta. The deferred batch envelope is the only
    // available source for batch identity in this window.
    auto canceling = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, 99);
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);

    EXPECT_TRUE(deferred->context->finalizePriorityPreemption());
    EXPECT_TRUE(deferred->context->finalizePriorityPreemption());

    auto status_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(status_info.running_task_info_list.empty());
    ASSERT_EQ(status_info.finished_task_info_list.size(), 1);
    const auto& task = status_info.finished_task_info_list[0];
    EXPECT_EQ(task.request_id, 3016);
    EXPECT_EQ(task.batch_id, 99);
    EXPECT_EQ(task.priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(task.error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, PriorityFirstCauseSuppressesOrdinaryDequeueTerminal) {
    PrefillBatchRpcServer server;
    auto                  deferred    = makeDeferred(server, 3024);
    auto                  input       = makeGenerateInput(3024);
    input->group_id                   = 99;
    auto stream                       = makeGenerateStream(input);
    deferred->context->generate_input = input;
    deferred->context->setStream(stream);
    deferred->context->setLocalStreamSchedulerOwned(false);

    ASSERT_EQ(deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    deferred->context->dequeueStreamFromRuntimeMeta();

    auto canceling = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_TRUE(canceling.finished_task_info_list.empty());
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, 99);
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);

    ASSERT_TRUE(deferred->context->finalizePriorityPreemption());
    auto canceled = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(canceled.running_task_info_list.empty());
    ASSERT_EQ(canceled.finished_task_info_list.size(), 1);
    EXPECT_EQ(canceled.finished_task_info_list[0].batch_id, 99);
    EXPECT_EQ(canceled.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
}

TEST(PrefillBatchRpcServerTest, PriorityFinalizerDoesNotWaitForSchedulerRejectedStream) {
    PrefillBatchRpcServer server;
    auto                  deferred    = makeDeferred(server, 3017);
    auto                  input       = makeGenerateInput(3017);
    auto                  stream      = makeGenerateStream(input);
    deferred->context->generate_input = input;
    deferred->context->setStream(stream);
    deferred->context->setLocalStreamSchedulerOwned(false);
    deferred->context->requestPriorityPreempt();

    EXPECT_TRUE(deferred->context->finalizePriorityPreemption());

    auto status_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(status_info.running_task_info_list.empty());
    ASSERT_EQ(status_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(status_info.finished_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(status_info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, DeferredContextMapRejectsDuplicateRequestId) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  first    = makeDeferred(server, 3002);
    auto                  second   = makeDeferred(server, 3002);

    ASSERT_TRUE(contexts->store(3002, first).ok());
    const auto duplicate_status = contexts->store(3002, second);
    EXPECT_EQ(duplicate_status.error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(duplicate_status.error_message(), "request already exists in deferred context map");
    ASSERT_TRUE(contexts->armTtl(3002, first, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(first->finishOperation());
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3002, claimed).ok());
    EXPECT_EQ(claimed, first);
}

TEST(PrefillBatchRpcServerTest, ConcurrentStoreAllowsExactlyOneContextPerRequestId) {
    constexpr int                                        kThreadCount = 16;
    PrefillBatchRpcServer                                server;
    auto                                                 contexts = std::make_shared<DeferredPrefillContextMap>();
    std::vector<std::shared_ptr<DeferredPrefillContext>> candidates;
    candidates.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i) {
        candidates.push_back(makeDeferred(server, 3010));
    }

    std::atomic<int>              ready{0};
    std::atomic<bool>             start{false};
    std::vector<grpc::StatusCode> codes(kThreadCount);
    std::vector<std::thread>      threads;
    threads.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&, i] {
            ready.fetch_add(1);
            while (!start.load()) {
                std::this_thread::yield();
            }
            codes[i] = contexts->store(3010, candidates[i]).error_code();
        });
    }
    while (ready.load() != kThreadCount) {
        std::this_thread::yield();
    }
    start.store(true);
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(std::count(codes.begin(), codes.end(), grpc::StatusCode::OK), 1);
    EXPECT_EQ(std::count(codes.begin(), codes.end(), grpc::StatusCode::ALREADY_EXISTS), kThreadCount - 1);
    EXPECT_EQ(contexts->size(), 1);
}

TEST(PrefillBatchRpcServerTest, StaleRollbackCannotRemoveAReplacementContext) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  first    = makeDeferred(server, 3011);
    auto                  second   = makeDeferred(server, 3011);

    ASSERT_TRUE(contexts->store(3011, first).ok());
    EXPECT_EQ(contexts->remove(3011, first.get()), first);
    ASSERT_TRUE(contexts->store(3011, second).ok());
    EXPECT_EQ(contexts->remove(3011, first.get()), nullptr);

    ASSERT_TRUE(contexts->armTtl(3011, second, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(second->finishOperation());
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3011, claimed).ok());
    EXPECT_EQ(claimed, second);
}

TEST(PrefillBatchRpcServerTest, ConcurrentFetchAndRollbackHaveSingleOwner) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3012);
    ASSERT_TRUE(contexts->store(3012, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3012, deferred, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(deferred->finishOperation());

    std::atomic<bool>                       start{false};
    grpc::Status                            take_status;
    std::shared_ptr<DeferredPrefillContext> fetched;
    std::shared_ptr<DeferredPrefillContext> rolled_back;
    std::thread                             fetch_thread([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        take_status = contexts->take(3012, fetched);
    });
    std::thread                             rollback_thread([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        rolled_back = contexts->remove(3012, deferred.get());
    });
    start.store(true);
    fetch_thread.join();
    rollback_thread.join();

    EXPECT_EQ(static_cast<int>(take_status.ok()) + static_cast<int>(rolled_back != nullptr), 1);
    EXPECT_EQ(fetched ? fetched : rolled_back, deferred);
    EXPECT_EQ(contexts->size(), 0);
}

TEST(PrefillBatchRpcServerTest, StopAcceptingPreservesPublishedContextsForFetch) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3014);
    ASSERT_TRUE(contexts->store(3014, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3014, deferred, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(deferred->finishOperation());

    contexts->stopAccepting();
    auto rejected = makeDeferred(server, 3015);
    EXPECT_EQ(contexts->store(3015, rejected).error_code(), grpc::StatusCode::UNAVAILABLE);

    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3014, claimed).ok());
    EXPECT_EQ(claimed, deferred);
}

TEST(PrefillBatchRpcServerTest, DeferredContextMapExpiresAndCancelsUnfetchedContext) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3003);

    ASSERT_TRUE(contexts->store(3003, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3003, deferred, std::chrono::milliseconds(10)).ok());
    for (int i = 0; i < 100 && contexts->size() != 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(contexts->size(), 0);
    EXPECT_TRUE(deferred->context->cancel_state->load());
    EXPECT_EQ(deferred->context->error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
}

TEST(PrefillBatchRpcServerTest, TakingContextCancelsItsTtlWithoutCancellingRequest) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3004);

    ASSERT_TRUE(contexts->store(3004, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3004, deferred, std::chrono::milliseconds(20)).ok());
    EXPECT_FALSE(deferred->finishOperation());
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3004, claimed).ok());
    ASSERT_EQ(claimed, deferred);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    EXPECT_FALSE(deferred->context->cancel_state->load());
}

TEST(PrefillBatchRpcServerTest, TakingContextDoesNotLeakItsAlarm) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3005);

    ASSERT_TRUE(contexts->store(3005, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3005, deferred, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(deferred->finishOperation());
    std::weak_ptr<grpc::Alarm>              alarm = deferred->ttl_alarm;
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3005, claimed).ok());
    claimed.reset();
    deferred.reset();
    EXPECT_TRUE(alarm.expired());
}

TEST(PrefillBatchRpcServerTest, CancelAllClearsAndCancelsDeferredContexts) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  first    = makeDeferred(server, 3006);
    auto                  second   = makeDeferred(server, 3007);

    ASSERT_TRUE(contexts->store(3006, first).ok());
    ASSERT_TRUE(contexts->store(3007, second).ok());
    ASSERT_TRUE(contexts->armTtl(3006, first, std::chrono::seconds(1)).ok());
    ASSERT_TRUE(contexts->armTtl(3007, second, std::chrono::seconds(1)).ok());
    contexts->cancelAll(grpc::Status(grpc::StatusCode::UNAVAILABLE, "shutdown"));

    EXPECT_EQ(contexts->size(), 0);
    EXPECT_TRUE(first->context->cancel_state->load());
    EXPECT_TRUE(second->context->cancel_state->load());
    EXPECT_EQ(first->context->error_status.error_code(), grpc::StatusCode::UNAVAILABLE);

    auto       after_shutdown  = makeDeferred(server, 3008);
    const auto shutdown_status = contexts->store(3008, after_shutdown);
    EXPECT_EQ(shutdown_status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(shutdown_status.error_message(), "Prefill batch server is shutting down");
}

TEST_F(PrefillBatchTraceTest, PerRequestCarriersCreateIsolatedLogicalParentsAndP2dChild) {
    const std::string first_trace   = "11111111111111111111111111111111";
    const std::string second_trace  = "22222222222222222222222222222222";
    const std::string first_parent  = "1111111111111111";
    const std::string second_parent = "2222222222222222";

    TracingDecodeRpcServer decode_server;
    ASSERT_TRUE(decode_server.start());

    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(2);
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i].input = std::make_shared<GenerateInputPB>();
        slots[i].input->set_request_id(4001 + i);
    }
    setTraceContext(*slots[0].input, first_trace, first_parent);
    setTraceContext(*slots[1].input, second_trace, second_parent);

    server.buildSlotContexts(slots);
    ASSERT_TRUE(slots[0].deferred->context->trace_span_guard);
    ASSERT_TRUE(slots[1].deferred->context->trace_span_guard);

    auto& first_context          = *slots[0].deferred->context;
    first_context.generate_input = makeGenerateInput(4001);
    first_context.generate_input->generate_config->role_addrs.emplace_back(
        RoleType::DECODE, "127.0.0.1", /*http_port=*/0, decode_server.port());
    server.prepareAllocateResource(first_context);
    ASSERT_TRUE(first_context.error_status.ok()) << first_context.error_status.error_message();
    ASSERT_TRUE(first_context.closeGrpcStream().ok());
    const auto p2d_parent_span_id = first_context.trace_span_guard->sharedSpan()->GetContext().span_id();

    for (auto& slot : slots) {
        ASSERT_TRUE(slot.deferred->context->tryMarkOtherTerminal());
        server.finishSlotOperation(slot.input->request_id(), slot.deferred);
        slot.deferred->finishLogicalTrace();
    }

    auto spans    = finishTelemetry();
    auto logicals = findSpans(spans, "rtp_llm.prefill_batch_request");
    ASSERT_EQ(logicals.size(), 2u);
    const trace_sdk::SpanData* first_logical  = nullptr;
    const trace_sdk::SpanData* second_logical = nullptr;
    for (const auto* logical : logicals) {
        if (toHex(logical->GetTraceId()) == first_trace) {
            first_logical = logical;
        } else if (toHex(logical->GetTraceId()) == second_trace) {
            second_logical = logical;
        }
        EXPECT_EQ(logical->GetSpanKind(), trace_api::SpanKind::kInternal);
        EXPECT_EQ(logical->GetAttributes().find("rpc.response.status_code"), logical->GetAttributes().end());
        EXPECT_EQ(logical->GetAttributes().find("rpc.system"), logical->GetAttributes().end());
    }
    ASSERT_NE(first_logical, nullptr);
    ASSERT_NE(second_logical, nullptr);
    EXPECT_EQ(toHex(first_logical->GetParentSpanId()), first_parent);
    EXPECT_EQ(toHex(second_logical->GetParentSpanId()), second_parent);

    auto p2d_spans = findSpans(spans, "rtp_llm.remote_generate");
    ASSERT_EQ(p2d_spans.size(), 1u);
    EXPECT_EQ(p2d_spans[0]->GetSpanKind(), trace_api::SpanKind::kClient);
    EXPECT_EQ(p2d_spans[0]->GetParentSpanId(), p2d_parent_span_id);
    EXPECT_EQ(toHex(p2d_spans[0]->GetTraceId()), first_trace);
    auto decode_spans = findSpans(spans, "rtp_llm.decode_remote_generate");
    ASSERT_EQ(decode_spans.size(), 1u);
    EXPECT_EQ(decode_spans[0]->GetSpanKind(), trace_api::SpanKind::kServer);
    EXPECT_EQ(toHex(decode_spans[0]->GetTraceId()), first_trace);
    EXPECT_EQ(decode_spans[0]->GetParentSpanId(), p2d_spans[0]->GetSpanId());
}

TEST_F(PrefillBatchTraceTest, ActiveOnlyShutdownWaitsForOperationOwnerAndEndsOnce) {
    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(1);
    slots[0].input = std::make_shared<GenerateInputPB>();
    slots[0].input->set_request_id(4051);
    setTraceContext(*slots[0].input, "33333333333333333333333333333333", "3333333333333333");
    server.buildSlotContexts(slots);

    auto contexts = std::make_shared<DeferredPrefillContextMap>();
    auto deferred = slots[0].deferred;
    ASSERT_TRUE(contexts->registerActive(4051, deferred).ok());
    contexts->cancelAll(grpc::Status(grpc::StatusCode::UNAVAILABLE, "shutdown"));

    EXPECT_TRUE(deferred->context->error_status.ok());
    EXPECT_FALSE(deferred->context->cancel_state->load());
    EXPECT_TRUE(span_data_->GetSpans().empty());
    EXPECT_FALSE(deferred->finishOperation());
    server.finishSlotOperation(4051, deferred);
    server.finishSlotOperation(4051, deferred);

    auto spans    = finishTelemetry();
    auto logicals = findSpans(spans, "rtp_llm.prefill_batch_request");
    ASSERT_EQ(logicals.size(), 1u);
    EXPECT_EQ(logicals[0]->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(nostd::get<std::string>(logicals[0]->GetAttributes().at("error.type")), "Unavailable");
}

TEST_F(PrefillBatchTraceTest, MissingMalformedAndDisabledCarriersFailOpenWithoutLogicalRoot) {
    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(3);
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i].input = std::make_shared<GenerateInputPB>();
        slots[i].input->set_request_id(4101 + i);
    }
    setTraceContext(*slots[0].input, "33333333333333333333333333333333", "3333333333333333");
    slots[2].input->mutable_request_info()->mutable_trace_context()->set_traceparent("malformed");

    server.buildSlotContexts(slots);
    ASSERT_TRUE(slots[0].deferred->context->trace_span_guard);
    EXPECT_FALSE(slots[1].deferred->context->trace_span_guard);
    EXPECT_FALSE(slots[2].deferred->context->trace_span_guard);
    for (auto& slot : slots) {
        ASSERT_TRUE(slot.deferred->context->tryMarkOtherTerminal());
        server.finishSlotOperation(slot.input->request_id(), slot.deferred);
    }

    auto spans = finishTelemetry();
    EXPECT_EQ(findSpans(spans, "rtp_llm.prefill_batch_request").size(), 1u);

    std::vector<PrefillBatchRpcServer::BatchSlot> disabled_slots(1);
    disabled_slots[0].input = std::make_shared<GenerateInputPB>();
    disabled_slots[0].input->set_request_id(4104);
    setTraceContext(*disabled_slots[0].input, "44444444444444444444444444444444", "4444444444444444");
    server.buildSlotContexts(disabled_slots);
    EXPECT_FALSE(disabled_slots[0].deferred->context->trace_span_guard);
}

TEST_F(PrefillBatchTraceTest, TraceContextLengthBoundaryPreservesValidParent) {
    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(2);
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i].input = std::make_shared<GenerateInputPB>();
        slots[i].input->set_request_id(4110 + i);
    }

    const std::string first_trace    = "11111111111111111111111111111111";
    const std::string second_trace   = "22222222222222222222222222222222";
    const std::string first_parent   = "1111111111111111";
    const std::string second_parent  = "2222222222222222";
    const std::string tracestate_512 = "a=" + std::string(253, 'x') + ",b=" + std::string(254, 'y');
    const std::string tracestate_513 = "a=" + std::string(253, 'x') + ",b=" + std::string(255, 'y');
    ASSERT_EQ(tracestate_512.size(), 512u);
    ASSERT_EQ(tracestate_513.size(), 513u);

    setTraceContext(*slots[0].input, first_trace, first_parent);
    slots[0].input->mutable_request_info()->mutable_trace_context()->set_tracestate(tracestate_512);
    setTraceContext(*slots[1].input, second_trace, second_parent);
    slots[1].input->mutable_request_info()->mutable_trace_context()->set_tracestate(tracestate_513);

    server.buildSlotContexts(slots);
    ASSERT_TRUE(slots[0].deferred->context->trace_span_guard);
    ASSERT_TRUE(slots[1].deferred->context->trace_span_guard);
    for (auto& slot : slots) {
        ASSERT_TRUE(slot.deferred->context->tryMarkOtherTerminal());
        server.finishSlotOperation(slot.input->request_id(), slot.deferred);
    }

    auto spans    = finishTelemetry();
    auto logicals = findSpans(spans, "rtp_llm.prefill_batch_request");
    ASSERT_EQ(logicals.size(), 2u);
    for (const auto* logical : logicals) {
        if (toHex(logical->GetTraceId()) == first_trace) {
            EXPECT_EQ(toHex(logical->GetParentSpanId()), first_parent);
            EXPECT_EQ(logical->GetSpanContext().trace_state()->ToHeader(), tracestate_512);
        } else if (toHex(logical->GetTraceId()) == second_trace) {
            EXPECT_EQ(toHex(logical->GetParentSpanId()), second_parent);
            EXPECT_TRUE(logical->GetSpanContext().trace_state()->ToHeader().empty());
        } else {
            FAIL() << "unexpected trace id " << toHex(logical->GetTraceId());
        }
    }
}

TEST_F(PrefillBatchTraceTest, LogicalFailureUsesDomainStatusWithoutRpcAttributesAndEndsOnce) {
    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(1);
    slots[0].input = std::make_shared<GenerateInputPB>();
    slots[0].input->set_request_id(4201);
    setTraceContext(*slots[0].input, "55555555555555555555555555555555", "5555555555555555");
    server.buildSlotContexts(slots);

    auto& deferred = slots[0].deferred;
    ASSERT_TRUE(deferred->context->tryMarkOtherTerminal());
    deferred->commitTerminalStatus(grpc::Status(grpc::StatusCode::INTERNAL, "prepare failed"));
    EXPECT_TRUE(deferred->context->error_status.ok());
    server.finishSlotOperation(4201, deferred);
    deferred->finishLogicalTrace();
    deferred->finishLogicalTrace();

    auto spans    = finishTelemetry();
    auto logicals = findSpans(spans, "rtp_llm.prefill_batch_request");
    ASSERT_EQ(logicals.size(), 1u);
    EXPECT_EQ(logicals[0]->GetStatus(), trace_api::StatusCode::kError);
    const auto& attributes = logicals[0]->GetAttributes();
    ASSERT_NE(attributes.find("error.type"), attributes.end());
    EXPECT_EQ(nostd::get<std::string>(attributes.at("error.type")), "Internal");
    EXPECT_EQ(attributes.find("rpc.response.status_code"), attributes.end());
    EXPECT_EQ(attributes.find("rtp_llm.grpc_status_code"), attributes.end());
}

// Publishing the OTHER terminal cause is what unblocks requestLogicalFinalization()
// on every other thread -- tryMarkOtherTerminal() also returns true when the cause
// is already OTHER, so it fences nothing. finishSlotOperation() is a close-only
// path: it claims the logical finalizer and ends the span while contributing no
// status of its own. A failure status must therefore be committed before the cause
// becomes visible, or the failed request is reported as a success.
//
// Entry A: cancel() itself publishes the cause (store/outward error paths).
TEST_F(PrefillBatchTraceTest, CancelPublishingCauseItselfNeverReportsFailureAsSuccess) {
    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(1);
    slots[0].input = std::make_shared<GenerateInputPB>();
    slots[0].input->set_request_id(4601);
    setTraceContext(*slots[0].input, "77777777777777777777777777777777", "7777777777777777");
    server.buildSlotContexts(slots);
    auto deferred = slots[0].deferred;
    ASSERT_TRUE(deferred->context->trace_span_guard);

    // The cause is still ACTIVE, so no other thread can claim the finalizer yet.
    ASSERT_EQ(deferred->context->terminalCause(), PrefillTerminalCause::ACTIVE);
    EXPECT_FALSE(deferred->requestLogicalFinalization());

    deferred->cancel(grpc::Status(grpc::StatusCode::INTERNAL, "store failed"));
    // A slot operation completing right after the cause became visible.
    server.finishSlotOperation(4601, deferred);

    EXPECT_FALSE(deferred->logical_status.ok()) << "cancel status dropped";
    auto spans    = finishTelemetry();
    auto logicals = findSpans(spans, "rtp_llm.prefill_batch_request");
    ASSERT_EQ(logicals.size(), 1u);
    EXPECT_EQ(logicals[0]->GetStatus(), trace_api::StatusCode::kError) << "a cancelled request was closed as a success";
}

// Entry B: callers that publish the terminal cause themselves and only reach
// cancel() later -- expire() (marks under mu_, cancels after releasing it) and
// cancelAll() (marks every context under mu_, then loops calling cancel()).
// Real work sits in that gap, so cancel() cannot repair the ordering for them:
// they must commit the status before they publish the cause. This pins that
// call-site contract; reordering commit after the mark makes it fail.
TEST_F(PrefillBatchTraceTest, CausePublishedByCallerRequiresStatusCommittedFirst) {
    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(1);
    slots[0].input = std::make_shared<GenerateInputPB>();
    slots[0].input->set_request_id(5601);
    setTraceContext(*slots[0].input, "88888888888888888888888888888888", "8888888888888888");
    server.buildSlotContexts(slots);
    auto deferred = slots[0].deferred;
    ASSERT_TRUE(deferred->context->trace_span_guard);

    const grpc::Status shutdown_status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down");
    // The ordering expire()/cancelAll() must use while still holding mu_.
    deferred->commitTerminalStatus(shutdown_status);
    deferred->context->tryMarkOtherTerminal();
    // An in-flight slot operation completing in the gap before cancel() is reached:
    // it claims the logical finalizer and ends the span, contributing no status.
    server.finishSlotOperation(5601, deferred);
    // The caller finally reaches cancel(); the span is already closed by now.
    deferred->cancel(shutdown_status);

    auto spans    = finishTelemetry();
    auto logicals = findSpans(spans, "rtp_llm.prefill_batch_request");
    ASSERT_EQ(logicals.size(), 1u);
    EXPECT_EQ(logicals[0]->GetStatus(), trace_api::StatusCode::kError)
        << "a shutdown-cancelled request was closed as a success";
}

// The reachable production interleaving: cancelAll() marks every stored context
// under mu_, releases mu_, and only then loops calling cancel(). A slot whose
// prepare task completes in that gap runs finishSlotOperation() on a worker
// thread -- a close-only path -- sees the cause already OTHER, claims the
// logical finalizer and ends the span. If cancelAll() had not committed the
// shutdown status before marking, that span is closed as a success.
TEST_F(PrefillBatchTraceTest, ConcurrentShutdownAndSlotFinalizationNeverReportsFailureAsSuccess) {
    constexpr int kIterations = 60;
    for (int i = 0; i < kIterations; ++i) {
        TestPrefillBatchRpcServer server;
        server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
        std::vector<PrefillBatchRpcServer::BatchSlot> slots(1);
        const int64_t                                 request_id = 6600 + i;
        slots[0].input                                           = std::make_shared<GenerateInputPB>();
        slots[0].input->set_request_id(request_id);
        setTraceContext(*slots[0].input, "99999999999999999999999999999999", "9999999999999999");
        server.buildSlotContexts(slots);
        auto deferred = slots[0].deferred;
        ASSERT_TRUE(deferred->context->trace_span_guard);

        auto contexts = std::make_shared<DeferredPrefillContextMap>();
        ASSERT_TRUE(contexts->store(request_id, deferred).ok());

        std::atomic<bool> go{false};
        std::thread       shutdowner([&] {
            while (!go.load(std::memory_order_acquire)) {}
            contexts->cancelAll(grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down"));
        });
        // The prepare-pool thread finishing its slot in cancelAll()'s gap.
        std::thread finalizer([&] {
            while (!go.load(std::memory_order_acquire)) {}
            server.finishSlotOperation(request_id, deferred);
        });
        go.store(true, std::memory_order_release);
        shutdowner.join();
        finalizer.join();
        deferred->finishLogicalTrace();

        EXPECT_FALSE(deferred->logical_status.ok()) << "shutdown status dropped at iteration " << i;
    }

    auto spans    = finishTelemetry();
    auto logicals = findSpans(spans, "rtp_llm.prefill_batch_request");
    ASSERT_FALSE(logicals.empty());
    for (const auto* logical : logicals) {
        EXPECT_EQ(logical->GetStatus(), trace_api::StatusCode::kError)
            << "a request that failed during shutdown was closed as a success";
    }
}

TEST_F(PrefillBatchTraceTest, TtlStyleCancellationSynthesizesTruncatedWaitExactlyOnce) {
    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(1);
    slots[0].input = std::make_shared<GenerateInputPB>();
    slots[0].input->set_request_id(4301);
    setTraceContext(*slots[0].input, "66666666666666666666666666666666", "6666666666666666");
    server.buildSlotContexts(slots);

    auto generate_input = makeGenerateInput(4301);
    auto stream         = makeGenerateStream(generate_input);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    slots[0].deferred->context->generate_input = generate_input;
    slots[0].deferred->context->setStream(stream);
    EXPECT_FALSE(slots[0].deferred->finishOperation());

    slots[0].deferred->cancel(grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "FetchResponse context TTL expired"));
    slots[0].deferred->finishLogicalTrace();
    slots[0].deferred->finishLogicalTrace();
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);

    auto spans = finishTelemetry();
    ASSERT_EQ(findSpans(spans, "rtp_llm.prefill_batch_request").size(), 1u);
    auto wait_spans = findSpans(spans, "wait");
    ASSERT_EQ(wait_spans.size(), 1u);
    EXPECT_EQ(wait_spans[0]->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_TRUE(findSpans(spans, "prefill").empty());
    const auto& attributes = wait_spans[0]->GetAttributes();
    ASSERT_NE(attributes.find("rtp_llm.phase.truncated"), attributes.end());
    EXPECT_TRUE(nostd::get<bool>(attributes.at("rtp_llm.phase.truncated")));
}

TEST_F(PrefillBatchTraceTest, PriorityFinalizationUsesSnapshotBeforeReleasingStream) {
    TestPrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(1);
    slots[0].input = std::make_shared<GenerateInputPB>();
    slots[0].input->set_request_id(4351);
    setTraceContext(*slots[0].input, "77777777777777777777777777777777", "7777777777777777");
    server.buildSlotContexts(slots);

    auto& deferred       = slots[0].deferred;
    auto  generate_input = makeGenerateInput(4351);
    auto  stream         = makeGenerateStream(generate_input);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    deferred->context->generate_input = generate_input;
    deferred->context->setStream(stream);
    const auto time_info = stream->getTimeInfo();

    ASSERT_EQ(deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    ASSERT_TRUE(deferred->context->finalizePriorityPreemption());
    EXPECT_FALSE(deferred->context->getStream());
    deferred->commitTerminalStatus(deferred->context->error_status);
    deferred->finishLogicalTrace(&time_info);
    deferred->finishLogicalTrace(&time_info);

    auto spans    = finishTelemetry();
    auto logicals = findSpans(spans, "rtp_llm.prefill_batch_request");
    ASSERT_EQ(logicals.size(), 1u);
    EXPECT_EQ(logicals[0]->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(logicals[0]->GetDescription(), "PRIORITY_PREEMPTED");
    const auto& logical_attributes = logicals[0]->GetAttributes();
    EXPECT_EQ(nostd::get<std::string>(logical_attributes.at("error.type")), "PRIORITY_PREEMPTED");
    EXPECT_EQ(nostd::get<int64_t>(logical_attributes.at("rtp_llm.error.code")),
              static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(nostd::get<std::string>(logical_attributes.at("rtp_llm.error.reason")), "PRIORITY_PREEMPTED");
    auto wait_spans = findSpans(spans, "wait");
    ASSERT_EQ(wait_spans.size(), 1u);
    EXPECT_EQ(wait_spans[0]->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(nostd::get<std::string>(wait_spans[0]->GetAttributes().at("error.type")), "PRIORITY_PREEMPTED");
    EXPECT_TRUE(findSpans(spans, "prefill").empty());
}

TEST_F(PrefillBatchTraceTest, FetchNotFoundCreatesRealServerSpanWithTransportStatus) {
    PrefillBatchRpcServer server;
    grpc::ServerContext   server_context;
    FetchRequestPB        request;
    request.set_request_id(4401);

    auto status = server.FetchResponse(&server_context, &request, nullptr);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);

    auto spans       = finishTelemetry();
    auto fetch_spans = findSpans(spans, "rtp_llm.fetch_response");
    ASSERT_EQ(fetch_spans.size(), 1u);
    EXPECT_EQ(fetch_spans[0]->GetSpanKind(), trace_api::SpanKind::kServer);
    EXPECT_FALSE(fetch_spans[0]->GetParentSpanId().IsValid());
    EXPECT_EQ(fetch_spans[0]->GetStatus(), trace_api::StatusCode::kError);
    const auto& attributes = fetch_spans[0]->GetAttributes();
    EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.response.status_code")), "NOT_FOUND");
    EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.method")), "RpcService/FetchResponse");
}

}  // namespace
}  // namespace rtp_llm
