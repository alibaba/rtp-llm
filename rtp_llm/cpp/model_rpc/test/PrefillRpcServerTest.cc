#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "gtest/gtest.h"
#include <mutex>
#include <string>
#include <thread>
#include <atomic>

#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

namespace rtp_llm {

TEST(PrefillRpcServerTest, NullGenerateRequestReturnsInvalidParamsBeforeEngineAccess) {
    PrefillRpcServer server;
    const auto status = server.GenerateStreamCall(nullptr, nullptr, nullptr);
    EXPECT_FALSE(status.ok());
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
}

TEST(PrefillRpcServerTest, GetPeerInfoReturnsOnlySelectedPeerLayoutWithoutWorkerAddresses) {
    PrefillRpcServer server;
    auto&            pc = server.maga_init_params_.parallelism_config;
    pc.tp_size          = 4;
    pc.dp_size          = 3;
    pc.dp_rank          = 2;
    // No worker address list is needed to report the selected endpoint's layout.
    for (bool sharded : {false, true}) {
        pc.prefill_cp_config.kv_cache_sharded = sharded;
        grpc::ServerContext   context;
        GetPeerInfoRequestPB  request;
        GetPeerInfoResponsePB response;
        ASSERT_TRUE(server.GetPeerInfo(&context, &request, &response).ok());
        EXPECT_EQ(response.tp_size(), 4);
        EXPECT_EQ(response.cp_size(), sharded ? 4 : 1);
    }
}

TEST(PrefillRpcServerTest, GetPeerInfoRejectsInvalidTpSize) {
    PrefillRpcServer server;
    server.maga_init_params_.parallelism_config.tp_size = 0;
    grpc::ServerContext   context;
    GetPeerInfoRequestPB  request;
    GetPeerInfoResponsePB response;
    const auto            status = server.GetPeerInfo(&context, &request, &response);
    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("invalid tp_size=0"), std::string::npos);
}

TEST(PrefillRpcServerTest, OnflightScopeTracksStepAndCleansOnReturn) {
    PrefillRpcServer server;

    {
        PrefillRpcServer::OnflightScope scope(&server, 9001);
        {
            std::lock_guard<std::mutex> lock(server.onflight_trackers_mutex_);
            ASSERT_EQ(server.onflight_trackers_.size(), 1);
            ASSERT_NE(server.onflight_trackers_.find(9001), server.onflight_trackers_.end());
            EXPECT_EQ(server.onflight_trackers_.at(9001)->step.load(),
                      static_cast<int>(PrefillRpcServer::GenerateStreamStep::kEntry));
        }

        scope.markStep(PrefillRpcServer::GenerateStreamStep::kAfterEngineEnqueue);
        {
            std::lock_guard<std::mutex> lock(server.onflight_trackers_mutex_);
            EXPECT_EQ(server.onflight_trackers_.at(9001)->step.load(),
                      static_cast<int>(PrefillRpcServer::GenerateStreamStep::kAfterEngineEnqueue));
        }
    }

    std::lock_guard<std::mutex> lock(server.onflight_trackers_mutex_);
    EXPECT_TRUE(server.onflight_trackers_.empty());
}

TEST(PrefillRpcServerTest, StartLoadRejectsMissingEngine) {
    PrefillRpcServer                server;
    grpc::ServerContext             context;
    P2PConnectorStartLoadRequestPB  request;
    P2PConnectorStartLoadResponsePB response;

    auto status = server.StartLoad(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    const auto error = errorInfoFromGrpcStatus(status);
    EXPECT_EQ(error.code(), ErrorCode::P2P_CONNECTOR_TRANSFER_NOT_INITIALIZED);
    EXPECT_NE(error.ToString().find("stage=StartLoad"), std::string::npos);
    EXPECT_NE(error.ToString().find("engine is null"), std::string::npos);
}

TEST(PrefillRpcServerTest, BatchDuplicateIdsRejectEveryItemWithoutChangingExistingAdmission) {
    PrefillRpcServer server;
    const auto active_key = masterEnqueuedHandoffUniqueKey(987);
    server.batch_entries_[active_key].reserved = true;
    EnqueueBatchRequestPB request;
    request.set_batch_id(123);
    auto* slot = request.add_dp_slots();
    slot->set_dp_rank(0);
    for (const int64_t id : {987, 988, 987}) {
        slot->add_requests()->mutable_input()->set_request_id(id);
    }
    grpc::ServerContext context;
    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(&context, &request, &response).ok());
    EXPECT_EQ(response.batch_id(), 123);
    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(response.errors_size(), 3);
    for (int i = 0; i < response.errors_size(); ++i) {
        EXPECT_EQ(response.errors(i).request_id(), slot->requests(i).input().request_id());
        EXPECT_EQ(response.errors(i).error_info().error_code(), static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
    }
    ASSERT_EQ(server.batch_entries_.size(), 1u);
    EXPECT_TRUE(server.batch_entries_.at(active_key).reserved);
    EXPECT_EQ(server.onflight_requests_.load(), 0u);
}

TEST(PrefillRpcServerTest, BatchMissingInputAndWrongDpRankReturnPerItemErrorsBeforeEngineAccess) {
    PrefillRpcServer server;
    EnqueueBatchRequestPB request;
    request.set_batch_id(124);
    auto* slot = request.add_dp_slots();
    slot->set_dp_rank(1);
    slot->add_requests();
    slot->add_requests()->mutable_input()->set_request_id(989);
    grpc::ServerContext context;
    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(&context, &request, &response).ok());
    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(response.errors_size(), 2);
    EXPECT_EQ(response.errors(0).request_id(), 0);
    EXPECT_EQ(response.errors(1).request_id(), 989);
    for (const auto& error : response.errors()) {
        EXPECT_EQ(error.error_info().error_code(), static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
    }
    EXPECT_TRUE(server.batch_entries_.empty());
    EXPECT_EQ(server.onflight_requests_.load(), 0u);
}

TEST(PrefillRpcServerTest, BatchActiveIdRejectionPreservesTheOriginalReservation) {
    PrefillRpcServer server;
    const auto key = masterEnqueuedHandoffUniqueKey(990);
    server.batch_entries_[key].reserved = true;
    EnqueueBatchRequestPB request;
    auto* slot = request.add_dp_slots();
    slot->set_dp_rank(0);
    slot->add_requests()->mutable_input()->set_request_id(990);
    grpc::ServerContext context;
    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(&context, &request, &response).ok());
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 990);
    EXPECT_EQ(response.errors(0).error_info().error_code(), static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(server.batch_entries_.size(), 1u);
    EXPECT_TRUE(server.batch_entries_.at(key).reserved);
}

TEST(PrefillRpcServerTest, BatchRejectsUnsupportedMultipleDpBeforeEngineAccess) {
    PrefillRpcServer server;
    server.maga_init_params_.parallelism_config.dp_size = 2;
    grpc::ServerContext context;
    EnqueueBatchRequestPB request;
    EnqueueBatchResponsePB response;
    EXPECT_ANY_THROW(server.EnqueueBatch(&context, &request, &response));
    EXPECT_TRUE(server.batch_entries_.empty());
    EXPECT_EQ(server.onflightRequestNum(), 0u);
}

TEST(PrefillRpcServerTest, BatchConversionExceptionReleasesReservationForRetry) {
    PrefillRpcServer server;
    EnqueueBatchRequestPB request;
    request.set_batch_id(123);
    auto* slot = request.add_dp_slots();
    slot->set_dp_rank(0);
    auto* input = slot->add_requests()->mutable_input();
    input->set_request_id(987);
    input->add_token_ids(1);
    input->mutable_generate_config()->set_max_new_tokens(-1);
    for (int attempt = 0; attempt < 2; ++attempt) {
        grpc::ServerContext context;
        EnqueueBatchResponsePB response;
        ASSERT_TRUE(server.EnqueueBatch(&context, &request, &response).ok());
        EXPECT_TRUE(server.batch_entries_.empty());
        ASSERT_EQ(response.errors_size(), 1);
        EXPECT_EQ(response.errors(0).request_id(), 987);
        EXPECT_EQ(response.errors(0).error_info().error_code(), static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
    }
}

TEST(PrefillRpcServerTest, GenerateStreamCallRejectsMissingRequestTimeout) {
    PrefillRpcServer    server;
    grpc::ServerContext context;
    GenerateInputPB     request;
    request.set_request_id(43);
    request.add_token_ids(1);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);
    config->set_unique_key("missing_request_deadline");
    auto status = server.GenerateStreamCall(&context, &request, nullptr);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
}

TEST(PrefillRpcServerTest, GenerateStreamCallRejectsPdRequestWithoutUniqueKey) {
    PrefillRpcServer    server;
    grpc::ServerContext context;
    GenerateInputPB     request;
    request.set_request_id(42);
    request.add_token_ids(1);
    auto* config = request.mutable_generate_config();
    config->set_timeout_ms(5000);
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    auto status = server.GenerateStreamCall(&context, &request, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.error_message(), "PD handoff requires non-empty unique_key");
}


}  // namespace rtp_llm

namespace rtp_llm {

TEST(PDCancelRegistryTest, CancelBeforeAdmissionFencesRepeatedAndLateEnqueue) {
    auto             meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    EXPECT_EQ(registry.cancel(101, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}), CANCEL_STATUS_TOMBSTONED);
    EXPECT_EQ(registry.cancel(101, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}), CANCEL_STATUS_TOMBSTONED);
    GenerateInputPB request;
    request.set_request_id(101);
    PDCancelRegistry::Handle handle;
    auto status = registry.admit(request, "master_enqueued_101", "prefill:123", currentTimeMs() + 1000, handle);
    EXPECT_EQ(status.code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_FALSE(handle);
    EXPECT_TRUE(meta->getEngineScheduleInfo(0).running_task_info_list.empty());
}

TEST(PDCancelRegistryTest, AcceptedCancelRequiresLocalTransferAndDownstreamCleanup) {
    auto             meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB  request;
    request.set_request_id(102);
    request.mutable_group_id()->set_value(77);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(registry.admit(request, "handoff102", "prefill:123", currentTimeMs() + 1000, handle).ok());
    EXPECT_EQ(registry.cancel(102, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}), CANCEL_STATUS_ACCEPTED);
    EXPECT_EQ(registry.cancel(102, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}), CANCEL_STATUS_ACCEPTED);
    EXPECT_TRUE(registry.isCanceled("handoff102"));
    ASSERT_EQ(meta->getEngineScheduleInfo(0).running_task_info_list.size(), 1u);
    EXPECT_FALSE(registry.complete(handle));
    registry.beginRead("handoff102");
    registry.finishLocal(handle);
    EXPECT_FALSE(registry.complete(handle));
    registry.finishDownstream(handle);
    EXPECT_FALSE(registry.complete(handle));
    registry.endRead("handoff102");
    ASSERT_TRUE(registry.complete(handle));
    EXPECT_FALSE(registry.complete(handle));
    auto info = meta->getEngineScheduleInfo(0);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1u);
    const auto& terminal = info.finished_task_info_list.front();
    EXPECT_EQ(terminal.request_id, 102);
    EXPECT_EQ(terminal.batch_id, 77);
    EXPECT_EQ(terminal.priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(terminal.error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    PDCancelRegistry::Handle late;
    EXPECT_EQ(registry.admit(request, "retry102", "prefill:123", currentTimeMs() + 1000, late).code(),
              ErrorCode::PRIORITY_PREEMPTED);
}

TEST(PDCancelRegistryTest, FinishedDecodeReturnsNotFoundWithoutAbsentFence) {
    auto             meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB  request;
    request.set_request_id(103);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(registry.admit(request, "handoff103", "", currentTimeMs() + 1000, handle).ok());
    registry.finishLocal(handle);
    EXPECT_EQ(registry.cancel(103, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}), CANCEL_STATUS_NOT_FOUND);
    EXPECT_TRUE(registry.pending().empty());
    EXPECT_TRUE(meta->getEngineScheduleInfo(0).finished_task_info_list.empty());
}

TEST(PDCancelRegistryTest, DecodeControlSurvivesLocalCompletionUntilPrefillFinishes) {
    auto             meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB  request;
    request.set_request_id(104);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(registry.admit(request, "handoff104", "prefill:123", currentTimeMs() + 1000, handle).ok());
    registry.finishLocal(handle);
    EXPECT_EQ(registry.cancel(104, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}), CANCEL_STATUS_ACCEPTED);
    EXPECT_FALSE(registry.complete(handle));
    registry.finishDownstream(handle);
    EXPECT_TRUE(registry.complete(handle));
}

TEST(PrefillRpcServerTest, CancelValidatesRequestAndUsesRegistry) {
    PrefillRpcServer server;
    server.meta_            = std::make_shared<RpcServerRuntimeMeta>();
    server.cancel_registry_ = std::make_unique<PDCancelRegistry>(server.meta_);
    grpc::ServerContext context;
    CancelRequestPB     request;
    CancelResponsePB    response;
    EXPECT_EQ(server.Cancel(&context, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    request.set_request_id(105);
    ASSERT_TRUE(server.Cancel(&context, &request, &response).ok());
    EXPECT_EQ(response.status(), CANCEL_STATUS_TOMBSTONED);
}

}  // namespace rtp_llm

namespace rtp_llm {

TEST(PDCancelRegistryTest, RacingCancelAndAdmissionCannotLoseTheCancelIntent) {
    for (int i = 0; i < 32; ++i) {
        auto             meta = std::make_shared<RpcServerRuntimeMeta>();
        PDCancelRegistry registry(meta);
        GenerateInputPB  request;
        request.set_request_id(106);
        PDCancelRegistry::Handle handle;
        ErrorInfo                admission;
        CancelStatusPB           ack = CANCEL_STATUS_UNSPECIFIED;
        std::atomic<bool>        start{false};
        std::thread              enqueue([&]() {
            while (!start.load())
                std::this_thread::yield();
            admission = registry.admit(request, "handoff106", "prefill:123", currentTimeMs() + 1000, handle);
        });
        std::thread              cancel([&]() {
            while (!start.load())
                std::this_thread::yield();
            ack = registry.cancel(106, {ErrorCode::PRIORITY_PREEMPTED, "preempted"});
        });
        start.store(true);
        enqueue.join();
        cancel.join();
        if (admission.ok()) {
            ASSERT_TRUE(handle);
            EXPECT_EQ(ack, CANCEL_STATUS_ACCEPTED);
            EXPECT_TRUE(handle->canceled.load());
        } else {
            EXPECT_EQ(admission.code(), ErrorCode::PRIORITY_PREEMPTED);
            EXPECT_EQ(ack, CANCEL_STATUS_TOMBSTONED);
        }
    }
}

}  // namespace rtp_llm

namespace rtp_llm {
TEST(PrefillRpcServerTest, ShutdownRejectsNewBatchBeforeEngineAccess) {
    PrefillRpcServer server;
    server.beginShutdown();
    grpc::ServerContext context;
    EnqueueBatchRequestPB request;
    EnqueueBatchResponsePB response;
    EXPECT_EQ(server.EnqueueBatch(&context, &request, &response).error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(server.onflightRequestNum(), 0u);
    EXPECT_EQ(response.successes_size(), 0);
}

TEST(PrefillRpcServerTest, RemoteServiceForwardsShutdownAndUsesBatchDrainCount) {
    RemoteRpcServiceImpl service;
    auto server = std::make_shared<PrefillRpcServer>();
    service.prefill_server_ = server;
    service.local_server_ = server;
    server->batch_entries_["acknowledged"].reserved = true;
    service.beginShutdown();
    EXPECT_EQ(service.onflightRequestNum(), 1u);
    grpc::ServerContext context;
    EnqueueBatchRequestPB request;
    EnqueueBatchResponsePB response;
    EXPECT_EQ(service.EnqueueBatch(&context, &request, &response).error_code(), grpc::StatusCode::UNAVAILABLE);
    server->batch_entries_.clear();
    EXPECT_EQ(service.onflightRequestNum(), 0u);
}

TEST(PrefillRpcServerTest, ShutdownWaitCountIncludesAcknowledgedBatchLifetime) {
    PrefillRpcServer server;
    server.batch_entries_["admitting"].reserved = true;
    GenerateInputPB input;
    input.set_request_id(1001);
    server.batch_entries_["acknowledged"].context = server.makeBatchRequestContext(input, nullptr);
    server.batch_entries_["acknowledged"].context->markRpcHandlingCompleted();
    server.batch_entries_["terminal-fence"].attach_pending = true;
    server.onflight_requests_.store(1);
    server.beginShutdown();
    EXPECT_EQ(server.onflightRequestNum(), 3u);
    server.batch_entries_["admitting"].reserved = false;
    server.batch_entries_["acknowledged"].context.reset();
    EXPECT_EQ(server.onflightRequestNum(), 1u);
    server.onflight_requests_.store(0);
    EXPECT_EQ(server.onflightRequestNum(), 0u);
}

TEST(PrefillRpcServerTest, BatchAttachDeadlineUsesDefaultAndOverallCap) {
    PrefillRpcServer server;
    server.registerBatchAttach("default", nullptr, 0, 1000000, 100);
    server.registerBatchAttach("negative", nullptr, -1, 1000000, 100);
    server.registerBatchAttach("short", nullptr, 50, 1000000, 100);
    server.registerBatchAttach("capped", nullptr, 500, 120, 100);
    server.registerBatchAttach("huge", nullptr, INT64_MAX, 120, 100);
    EXPECT_EQ(server.batch_entries_.at("default").deadline_ms, 600100);
    EXPECT_EQ(server.batch_entries_.at("negative").deadline_ms, 600100);
    EXPECT_EQ(server.batch_entries_.at("short").deadline_ms, 150);
    EXPECT_EQ(server.batch_entries_.at("capped").deadline_ms, 120);
    EXPECT_EQ(server.batch_entries_.at("huge").deadline_ms, 120);
}

TEST(PrefillRpcServerTest, BatchAttachBeforeDeadlineDisarmsExpiry) {
    PrefillRpcServer server;
    server.registerBatchAttach("attached", nullptr, 50, 1000, 100);
    EXPECT_TRUE(server.attachBatch("attached", 149));
    server.expireBatchAttachments(200);
    EXPECT_EQ(server.batch_entries_.count("attached"), 0);
    EXPECT_TRUE(server.attachBatch("ordinary", 200));
}

TEST(PrefillRpcServerTest, BatchAttachDoesNotReleaseActiveAdmission) {
    PrefillRpcServer server;
    server.batch_entries_["active"].reserved = true;
    server.registerBatchAttach("active", nullptr, 50, 1000, 100);
    ASSERT_TRUE(server.attachBatch("active", 149));
    server.expireBatchAttachments(1000);
    ASSERT_EQ(server.batch_entries_.count("active"), 1);
    EXPECT_TRUE(server.batch_entries_.at("active").reserved);
    EXPECT_FALSE(server.batch_entries_.at("active").attach_pending);
}

TEST(PrefillRpcServerTest, ExpiredBatchAdmissionRetainsFenceUntilLocalCleanup) {
    PrefillRpcServer server;
    server.batch_entries_["active"].reserved = true;
    server.registerBatchAttach("active", nullptr, 50, 1000, 100);
    server.expireBatchAttachments(1000);
    ASSERT_EQ(server.batch_entries_.count("active"), 1);
    EXPECT_FALSE(server.attachBatch("active", 1000));
    server.batch_entries_.at("active").reserved = false;
    server.expireBatchAttachments(1000);
    EXPECT_EQ(server.batch_entries_.count("active"), 0);
}

TEST(PrefillRpcServerTest, LateBatchAttachCannotBeatCleanupOrReviveExpiredEntry) {
    PrefillRpcServer server;
    server.registerBatchAttach("late", nullptr, 50, 1000, 100);
    EXPECT_FALSE(server.attachBatch("late", 150));
    server.expireBatchAttachments(150);
    EXPECT_TRUE(server.batch_entries_.at("late").expired);
    EXPECT_FALSE(server.attachBatch("late", 151));
    server.expireBatchAttachments(200);
    EXPECT_FALSE(server.attachBatch("late", 200));
    server.expireBatchAttachments(1000);
    EXPECT_EQ(server.batch_entries_.count("late"), 0);
}
}  // namespace rtp_llm

namespace rtp_llm {
TEST(PDCancelRegistryTest, LocalCancelWaitsForActivePrefillReadEvenAfterLocalCompletion) {
    auto             meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB  request;
    request.set_request_id(107);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(registry.admit(request, "handoff107", "", currentTimeMs() + 1000, handle).ok());
    registry.beginRead("handoff107");
    registry.finishLocal(handle);
    EXPECT_EQ(registry.cancel(107, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}), CANCEL_STATUS_ACCEPTED);
    EXPECT_FALSE(registry.complete(handle));
    registry.endRead("handoff107");
    EXPECT_TRUE(registry.complete(handle));
}
}  // namespace rtp_llm

namespace rtp_llm {
TEST(PDCancelRegistryTest, OrdinaryCleanupRetainsCauseAndDoesNotPublishPriorityPreemption) {
    for (auto code : {ErrorCode::CANCELLED, ErrorCode::GENERATE_TIMEOUT, ErrorCode::INVALID_PARAMS}) {
        auto             meta = std::make_shared<RpcServerRuntimeMeta>();
        PDCancelRegistry registry(meta);
        GenerateInputPB  request;
        request.set_request_id(108);
        PDCancelRegistry::Handle handle;
        ASSERT_TRUE(registry.admit(request, "handoff108", "prefill:123", currentTimeMs() + 1000, handle).ok());
        EXPECT_EQ(registry.cancel(108, {code, "original failure"}), CANCEL_STATUS_ACCEPTED);
        ASSERT_TRUE(handle->canceled.load());
        const auto rpc_error = errorInfoFromGrpcStatus(PrefillRpcServer{}.serializeErrorMsg("108", handle->cancel_reason));
        EXPECT_EQ(rpc_error.code(), code);
        EXPECT_NE(rpc_error.ToString().find("original failure"), std::string::npos);
        registry.finishLocal(handle);
        EXPECT_FALSE(registry.complete(handle));
        EXPECT_TRUE(meta->getEngineScheduleInfo(0).finished_task_info_list.empty());
        registry.finishDownstream(handle);
        ASSERT_TRUE(registry.complete(handle));
        const auto info = meta->getEngineScheduleInfo(0);
        ASSERT_EQ(info.finished_task_info_list.size(), 1);
        EXPECT_EQ(info.finished_task_info_list.front().error_code, static_cast<int64_t>(code));
        EXPECT_EQ(info.finished_task_info_list.front().priority_preemption_progress, PriorityPreemptionProgress::NONE);
    }
}
TEST(PDCancelRegistryTest, CompletedRequestsExpireFromCompletionRatherThanBusinessDeadline) {
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB request;
    request.set_request_id(110);
    const auto deadline = currentTimeMs() + 2 * 60 * 60 * 1000;
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(registry.admit(request, "done110", "prefill:123", deadline, handle).ok());
    registry.finishDownstream(handle);
    registry.finishLocal(handle);
    EXPECT_LT(handle->retain_until_ms, deadline);
    {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        registry.trimLocked(handle->retain_until_ms - 1);
        EXPECT_EQ(registry.entries_.count(110), 1u);
        registry.trimLocked(handle->retain_until_ms);
        EXPECT_EQ(registry.entries_.count(110), 0u);
        EXPECT_TRUE(registry.entry_expirations_.empty());
    }
}

TEST(PDCancelRegistryTest, DownstreamCompletionReplacesEarlierExpirationWithoutStaleIndex) {
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB request;
    request.set_request_id(115);
    PDCancelRegistry::Handle handle;
    const auto deadline = currentTimeMs() + 2 * 60 * 60 * 1000;
    ASSERT_TRUE(registry.admit(request, "done115", "prefill:123", deadline, handle).ok());
    registry.finishLocal(handle);
    EXPECT_GT(handle->retain_until_ms, deadline);
    registry.finishDownstream(handle);
    EXPECT_LT(handle->retain_until_ms, deadline);
    {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        ASSERT_EQ(registry.entry_expirations_.size(), 1u);
        registry.trimLocked(handle->retain_until_ms);
        EXPECT_TRUE(registry.entries_.empty());
        EXPECT_TRUE(registry.entry_expirations_.empty());
    }
}

TEST(PDCancelRegistryTest, UnconfirmedCanceledRequestSurvivesExpirationAndRemainsPending) {
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB request;
    request.set_request_id(111);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(registry.admit(request, "pending111", "prefill:123", currentTimeMs() + 1000, handle).ok());
    ASSERT_EQ(registry.cancel(111, {ErrorCode::GENERATE_TIMEOUT, "timeout"}), CANCEL_STATUS_ACCEPTED);
    registry.finishLocal(handle);
    {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        registry.trimLocked(handle->retain_until_ms);
        EXPECT_EQ(registry.entries_.count(111), 1u);
    }
    EXPECT_TRUE(registry.isCanceled("pending111"));
    EXPECT_EQ(registry.pending().size(), 1u);
    EXPECT_FALSE(registry.complete(handle));
    registry.finishDownstream(handle);
    ASSERT_TRUE(registry.complete(handle));
    EXPECT_TRUE(registry.pending().empty());
    {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        registry.trimLocked(handle->retain_until_ms);
    }
    EXPECT_FALSE(registry.find(111));
    EXPECT_FALSE(registry.isCanceled("pending111"));
}

TEST(PDCancelRegistryTest, CancelKeyIndexKeepsOtherCanceledRequestWithSameKey) {
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB first_request;
    first_request.set_request_id(112);
    GenerateInputPB second_request;
    second_request.set_request_id(113);
    PDCancelRegistry::Handle first;
    PDCancelRegistry::Handle second;
    ASSERT_TRUE(registry.admit(first_request, "shared-key", "", currentTimeMs() + 1000, first).ok());
    ASSERT_TRUE(registry.admit(second_request, "shared-key", "", currentTimeMs() + 1000, second).ok());
    ASSERT_EQ(registry.cancel(112, {ErrorCode::CANCELLED, "cancel"}), CANCEL_STATUS_ACCEPTED);
    ASSERT_EQ(registry.cancel(113, {ErrorCode::CANCELLED, "cancel"}), CANCEL_STATUS_ACCEPTED);
    registry.finishLocal(first);
    ASSERT_TRUE(registry.complete(first));
    {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        registry.trimLocked(first->retain_until_ms);
    }
    EXPECT_TRUE(registry.isCanceled("shared-key"));
    EXPECT_EQ(registry.pending().size(), 1u);
    registry.finishLocal(second);
    ASSERT_TRUE(registry.complete(second));
    {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        registry.trimLocked(second->retain_until_ms);
    }
    EXPECT_FALSE(registry.isCanceled("shared-key"));
}

TEST(PDCancelRegistryTest, RepeatedAbsentCancelHasSingleExpiryAndPreservesFirstReason) {
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    EXPECT_EQ(registry.cancel(114, {ErrorCode::GENERATE_TIMEOUT, "first"}), CANCEL_STATUS_TOMBSTONED);
    EXPECT_EQ(registry.cancel(114, {ErrorCode::CANCELLED, "second"}), CANCEL_STATUS_TOMBSTONED);
    {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        ASSERT_EQ(registry.fence_expirations_.size(), 1u);
        const auto expiry = registry.absent_fences_.at(114).expires_at_ms;
        EXPECT_EQ(registry.absent_fences_.at(114).reason.code(), ErrorCode::GENERATE_TIMEOUT);
        registry.trimLocked(expiry);
        EXPECT_TRUE(registry.absent_fences_.empty());
        EXPECT_TRUE(registry.fence_expirations_.empty());
    }
}

TEST(PDCancelRegistryTest, LateAdmissionRetainsOrdinaryCancellationCause) {
    auto             meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    EXPECT_EQ(registry.cancel(109, {ErrorCode::GENERATE_TIMEOUT, "deadline"}), CANCEL_STATUS_TOMBSTONED);
    GenerateInputPB request;
    request.set_request_id(109);
    PDCancelRegistry::Handle handle;
    EXPECT_EQ(registry.admit(request, "late109", "", currentTimeMs() + 1000, handle).code(),
              ErrorCode::GENERATE_TIMEOUT);
}
}  // namespace rtp_llm
