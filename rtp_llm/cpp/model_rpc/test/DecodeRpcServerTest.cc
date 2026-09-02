#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {

namespace {

DecodeRpcServer::LoadKVCacheContext makeLoadContext(const std::string&               request_key,
                                                    const std::vector<std::string>&  peer_addrs,
                                                    const std::vector<CacheKeyType>& cache_keys,
                                                    const GroupBlockIds&             block_ids_by_group,
                                                    int32_t                          prefill_cp_size,
                                                    int64_t                          reuse_block_size = 0) {
    return {/*request_id=*/42,
            request_key,
            peer_addrs,
            cache_keys,
            block_ids_by_group,
            reuse_block_size,
            /*timeout_ms=*/1000,
            /*partition_count=*/1,
            /*partition_id=*/0,
            /*server_context=*/nullptr,
            prefill_cp_size};
}

GroupBase makeRpcGroup(std::string tag, std::vector<int> layer_ids) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
    spec->seq_size_per_block = 8;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 8;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = 8;
    return group;
}

// Tokens per logical (cache-key sized) block, and the CP-scaled tokens per block
// of a compacted fixed/state group under prefill CP=2 (cp.scale_seq_size).
constexpr size_t kBaseSeqSizePerBlock    = 64;
constexpr size_t kCompactSeqSizePerBlock = kBaseSeqSizePerBlock * 2;

CacheGroupPolicy makeCompactStatePolicy(uint32_t active_tail_blocks) {
    auto policy               = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.active_tail_blocks = active_tail_blocks;
    policy.cp_slice           = CpBlockSliceMode::PAYLOAD_BYTES;
    EXPECT_EQ(policy.cp_mapping, CpBlockMappingMode::COMPACT_LAST_RANK);
    return policy;
}

using KeyOffsetPairs = std::vector<std::pair<int, int>>;

KeyOffsetPairs keyOffsetPairs(const std::vector<CacheStoreBlockPair>& plan) {
    KeyOffsetPairs pairs;
    pairs.reserve(plan.size());
    for (const auto& pair : plan) {
        pairs.emplace_back(pair.key_index, pair.offset_index);
    }
    return pairs;
}

}  // namespace

TEST(ModelRpcProtoTest, GroupedCacheFieldsPreserveLegacyNumbers) {
    const auto* broadcast = BroadcastLoadRequestPB::descriptor();
    ASSERT_NE(broadcast, nullptr);
    EXPECT_TRUE(broadcast->IsReservedNumber(5));
    EXPECT_TRUE(broadcast->IsReservedNumber(12));
    EXPECT_EQ(broadcast->FindFieldByName("block_num")->number(), 6);
    EXPECT_EQ(broadcast->FindFieldByName("reuse_block_size")->number(), 7);
    EXPECT_EQ(broadcast->FindFieldByName("timeout_ms")->number(), 8);
    EXPECT_EQ(broadcast->FindFieldByName("dp_rank")->number(), 9);
    EXPECT_EQ(broadcast->FindFieldByName("partition_count")->number(), 10);
    EXPECT_EQ(broadcast->FindFieldByName("partition_id")->number(), 11);
    EXPECT_EQ(broadcast->FindFieldByName("prefill_cp_size")->number(), 13);
    EXPECT_EQ(broadcast->FindFieldByName("tagged_group_block_ids")->number(), 14);

    const auto* remote = RemoteOperationRequestPB::descriptor();
    ASSERT_NE(remote, nullptr);
    EXPECT_TRUE(remote->IsReservedNumber(3));
    EXPECT_EQ(remote->FindFieldByName("group_ids"), nullptr);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
}

TEST(DecodeRpcServerTest, CPShardedLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101, 102};
    const GroupBlockIds             block_ids_by_group;
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/2, /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
    ASSERT_EQ(request.cache_keys_size(), 2);
    EXPECT_EQ(request.cache_keys(0), 101);
    EXPECT_EQ(request.cache_keys(1), 102);
}

TEST(DecodeRpcServerTest, CPShardedMlaLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const GroupBlockIds             block_ids_by_group;
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/2, /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/1, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
}

TEST(DecodeRpcServerTest, TaggedBlockRowsResolveByLocalTagOrder) {
    auto                   topology = CacheTopology::create({makeRpcGroup("linear", {0}), makeRpcGroup("full", {1})},
                                                            {{0, {"linear"}}, {1, {"full"}}});
    BroadcastLoadRequestPB request;
    auto*                  full = request.add_tagged_group_block_ids();
    full->set_tag("full");
    full->add_block_ids(10);
    auto* linear = request.add_tagged_group_block_ids();
    linear->set_tag("linear");
    linear->add_block_ids(20);

    const auto blocks = DecodeRpcServer::decodeGroupBlockIds(request, *topology);
    EXPECT_EQ(blocks[topology->groupIdForTag("full")]->blocks(), (BlockIndicesType{10}));
    EXPECT_EQ(blocks[topology->groupIdForTag("linear")]->blocks(), (BlockIndicesType{20}));

    auto reordered = CacheTopology::create({makeRpcGroup("full", {1}), makeRpcGroup("linear", {0})},
                                           {{0, {"linear"}}, {1, {"full"}}});
    EXPECT_NE(topology->groupIdForTag("full"), reordered->groupIdForTag("full"));
    EXPECT_EQ(DecodeRpcServer::makeTaggedRequestKey(42, 1, topology->group("full").tag),
              DecodeRpcServer::makeTaggedRequestKey(42, 1, reordered->group("full").tag));
}

TEST(DecodeRpcServerTest, EmptyTaggedBlockRowsAreRejected) {
    auto                   topology = CacheTopology::create({makeRpcGroup("full", {0})}, {{0, {"full"}}});
    BroadcastLoadRequestPB request;
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(request, *topology));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsRejectTopologyMismatch) {
    auto topology =
        CacheTopology::create({makeRpcGroup("full", {0}), makeRpcGroup("linear", {0})}, {{0, {"full", "linear"}}});
    BroadcastLoadRequestPB missing_tag;
    auto*                  row = missing_tag.add_tagged_group_block_ids();
    row->set_tag("full");
    row->add_block_ids(1);

    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(missing_tag, *topology));
}

TEST(DecodeRpcServerTest, MtpCacheKeyUsesSharedBaseModelIdForEverySlot) {
    constexpr size_t mtp_base_model_id = 17;

    for (size_t mtp_model_id = 0; mtp_model_id < 2; ++mtp_model_id) {
        EXPECT_EQ(DecodeRpcServer::makeMTPModuleCacheKey(mtp_base_model_id, "101", /*layer_id=*/0),
                  "model_id_17_token_id_str_101_layer_id_0")
            << "mtp_model_id=" << mtp_model_id;
    }
}

TEST(DecodeRpcServerTest, MtpLoadPlanContainsOnlyModule0) {
    auto module0          = std::make_unique<EngineInitParams>();
    module0->model_id     = 17;
    auto module1          = std::make_unique<EngineInitParams>();
    module1->model_id     = 23;
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::move(module0));
    mtp_model_params->push_back(std::move(module1));
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].module_index, 0);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
    EXPECT_EQ(plan[0].cache_model_id, 17);
}

TEST(DecodeRpcServerTest, MtpLoadPlanRejectsMissingModule0) {
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(nullptr).empty());

    ProposeModelEngineInitParams missing_params;
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&missing_params).empty());

    auto                         empty_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    ProposeModelEngineInitParams no_modules(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(empty_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&no_modules).empty());

    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(nullptr);
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    ProposeModelEngineInitParams null_module0(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&null_module0).empty());
}

TEST(DecodeRpcServerTest, MtpLoadPlanIgnoresInactiveModules) {
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    mtp_model_params->push_back(nullptr);
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
}

TEST(DecodeRpcServerTest, ReadFailureLogContainsPeerErrorAndEveryBlockKey) {
    test::TestLogCapture log_capture("read_cache_failure");
    DecodeRpcServer::logReadFailures(/*request_id=*/42,
                                     "127.0.0.1:1:2",
                                     ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED,
                                     "connect failed",
                                     {"blocks={kv_key_0,kv_key_1}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("PD_CACHE_KEY_READ_FAILED"), std::string::npos);
    EXPECT_NE(log_content.find("127.0.0.1:1:2"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_0"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_1"), std::string::npos);
}

TEST(DecodeRpcServerTest, ReadTimeoutLogsKeysAndCancellationIsSilent) {
    test::TestLogCapture log_capture("read_cache_timeout_cancel");
    DecodeRpcServer::logReadFailures(
        /*request_id=*/43, "peer", ErrorCode::LOAD_CACHE_TIMEOUT, "timeout", {"blocks={timeout_key}"});
    DecodeRpcServer::logReadFailures(
        /*request_id=*/44, "peer", ErrorCode::CANCELLED, "cancelled", {"blocks={cancelled_key}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("timeout_key"), std::string::npos);
    EXPECT_EQ(log_content.find("cancelled_key"), std::string::npos);
}

TEST(DecodeRpcServerTest, CancelledGenerateRequestReadUsesCancelledStatus) {
    const auto status = DecodeRpcServer::generateRequestReadFailureStatus(/*cancelled=*/true);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(status.error_message(), "request is cancelled");
}

TEST(DecodeRpcServerTest, NonCancelledGenerateRequestReadPreservesFailure) {
    const auto status = DecodeRpcServer::generateRequestReadFailureStatus(/*cancelled=*/false);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "poll generate request failed");
}

TEST(DecodeRpcServerTest, CacheLoadTimeoutClassifiedAsDependencyFailure) {
    // The cache timeouts map onto DEADLINE_EXCEEDED, so a predicate keyed on the
    // gRPC status would mislabel this upstream KV-transfer failure as a local
    // deadline of the decode node.
    const ErrorInfo    error_info(ErrorCode::LOAD_CACHE_TIMEOUT, "load cache timeout");
    const grpc::Status error_status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
    ASSERT_EQ(error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::loadCacheFromPrefill, error_info, error_status),
                 "DependencyFailure");
}

TEST(DecodeRpcServerTest, CacheStoreLoadBufferTimeoutClassifiedAsDependencyFailure) {
    const ErrorInfo    error_info(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, "load buffer timeout");
    const grpc::Status error_status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
    ASSERT_EQ(error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::loadCacheFromPrefill, error_info, error_status),
                 "DependencyFailure");
}

TEST(DecodeRpcServerTest, CancelledDuringCacheLoadKeepsCancelledClassification) {
    // A client going away while KV cache is still arriving is not a failure of the
    // Prefill dependency, so the stage alone must not decide the classification.
    const ErrorInfo    error_info(ErrorCode::CANCELLED, "request is cancelled");
    const grpc::Status error_status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
    ASSERT_EQ(error_status.error_code(), grpc::StatusCode::CANCELLED);

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::loadCacheFromPrefill, error_info, error_status),
                 "Cancelled");
}

TEST(DecodeRpcServerTest, CancelledCacheLoadClassificationDoesNotRequireTransportStatus) {
    const ErrorInfo error_info(ErrorCode::CANCELLED, "request is cancelled");

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::loadCacheFromPrefill, error_info, grpc::Status::OK),
                 "Cancelled");
}

TEST(DecodeRpcServerTest, DeadlineOutsideCacheLoadKeepsStatusClassification) {
    const ErrorInfo    error_info(ErrorCode::GENERATE_TIMEOUT, "generate timeout");
    const grpc::Status error_status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
    ASSERT_EQ(error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::localGenerate, error_info, error_status),
                 "DeadlineExceeded");
}

TEST(DecodeRpcServerTest, ExceptionUnwindingWithoutStatusIsClassifiedAsException) {
    // PhaseSpanSynthesisScope reports unwinding while error_status is still OK:
    // nothing set a gRPC status on the way out.
    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::localGenerate, ErrorInfo::OkStatus(), grpc::Status::OK),
                 "Exception");
}

TEST(DecodeRpcServerTest, SuccessfulRequestHasNoPhaseErrorType) {
    EXPECT_EQ(DecodeRpcServer::phaseErrorType(/*request_ok=*/true,
                                              DecodeStatInfo::loadCacheFromPrefill,
                                              ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, "ignored"),
                                              grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "ignored")),
              nullptr);
}

TEST(DecodeRpcServerTest, CompactStateGroupLoadsGlobalTailKeysIntoCanonicalSlots) {
    // 11 logical blocks under prefill CP=2 compact into a 6-slot state table:
    // slot j covers logical blocks [2j, 2j+1], so the two active tail slots 4 and
    // 5 must be filled from the *global* cache keys 9 and 10 - not from keys 4
    // and 5, which is what indexing the compacted table with logical positions
    // (or the table length with the logical key count) would produce.
    const auto plan = DecodeRpcServer::buildGroupLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                                          /*local_block_num=*/6,
                                                          /*cache_key_count=*/11,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kCompactSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{9, 4}, {10, 5}}));
}

TEST(DecodeRpcServerTest, UnshardedFullGroupUsesTerminalKeyOfEachTagBlock) {
    auto       policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/3,
                                                          /*cache_key_count=*/6,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          /*group_seq_size_per_block=*/2,
                                                          /*base_seq_size_per_block=*/1,
                                                          /*physical_cp_size=*/1);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{1, 0}, {3, 1}, {5, 2}}));
}

TEST(DecodeRpcServerTest, UnshardedSameGeometryKeepsCompleteKeyNamespace) {
    auto       policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/44,
                                                          /*cache_key_count=*/44,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/false,
                                                          /*group_seq_size_per_block=*/8,
                                                          /*base_seq_size_per_block=*/8,
                                                          /*physical_cp_size=*/1);

    ASSERT_EQ(plan.size(), 44u);
    for (size_t i = 0; i < plan.size(); ++i) {
        EXPECT_EQ(plan[i].key_index, static_cast<int>(i));
        EXPECT_EQ(plan[i].offset_index, static_cast<int>(i));
    }
}

TEST(DecodeRpcServerTest, PhysicallyShardedFullGroupKeepsLogicalPlanForPeerSplit) {
    auto       policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/11,
                                                          /*cache_key_count=*/11,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/false,
                                                          /*group_seq_size_per_block=*/8,
                                                          /*base_seq_size_per_block=*/4,
                                                          /*physical_cp_size=*/2);

    ASSERT_EQ(plan.size(), 11u);
    for (size_t i = 0; i < plan.size(); ++i) {
        EXPECT_EQ(plan[i].key_index, static_cast<int>(i));
        EXPECT_EQ(plan[i].offset_index, static_cast<int>(i));
    }
}

TEST(DecodeRpcServerTest, CompactStateGroupLoadPlanMatchesProducerStorePlan) {
    // The consumer must project exactly like the producer: same (key, offset)
    // pairs, or the decode reads a key the prefill never registered.
    const auto policy        = makeCompactStatePolicy(/*active_tail_blocks=*/2);
    const auto decode_plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                                 /*local_block_num=*/6,
                                                                 /*cache_key_count=*/11,
                                                                 /*reuse_block_size=*/0,
                                                                 /*use_hybrid=*/true,
                                                                 kCompactSeqSizePerBlock,
                                                                 kBaseSeqSizePerBlock);
    const auto producer_plan = buildCacheStorePlan(policy,
                                                   /*total_logical_blocks=*/11,
                                                   /*reuse_block_size=*/0,
                                                   /*use_hybrid=*/true,
                                                   /*cp_rank=*/1,
                                                   /*cp_size=*/2);

    EXPECT_EQ(keyOffsetPairs(decode_plan), keyOffsetPairs(producer_plan));
}

TEST(DecodeRpcServerTest, CompactStateGroupIgnoresSpeculativeReserveTailSlots) {
    // MTP reserve slots make the positional table longer than the canonical slots
    // the sequence actually backs. The destinations must stay 4 and 5; picking the
    // tail of the table would target the reserve slot 6 and duplicate key 10.
    const auto plan = DecodeRpcServer::buildGroupLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                                          /*local_block_num=*/7,
                                                          /*cache_key_count=*/11,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kCompactSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{9, 4}, {10, 5}}));
}

TEST(DecodeRpcServerTest, CompactOneTailGroupLoadsOnlyTheLastCanonicalSlot) {
    // hca_state declares active_tail_blocks=1; explicit_block_num only sizes its
    // pool and must not widen the per-request projection.
    auto policy               = makeCompactStatePolicy(/*active_tail_blocks=*/1);
    policy.explicit_block_num = 256;

    const auto plan = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/6,
                                                          /*cache_key_count=*/11,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kCompactSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{10, 5}}));
}

TEST(DecodeRpcServerTest, CompactStateGroupWithSingleLogicalBlockLoadsKeyZero) {
    const auto plan = DecodeRpcServer::buildGroupLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                                          /*local_block_num=*/1,
                                                          /*cache_key_count=*/1,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kCompactSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{0, 0}}));
}

TEST(DecodeRpcServerTest, UnscaledSwaGroupKeepsLogicalTailPositions) {
    // A COMPACT_LAST_RANK policy on a group whose block still covers one logical
    // block has a flat table: the tail slots are logical positions 3 and 4.
    const auto plan = DecodeRpcServer::buildGroupLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                                          /*local_block_num=*/5,
                                                          /*cache_key_count=*/5,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kBaseSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{3, 3}, {4, 4}}));
}

TEST(DecodeRpcServerTest, TailGroupReserveSlotsDoNotStarveTheLoad) {
    // The last sequence-backed slot is 2; a reserve slot 3 must not shift the
    // one-block tail window past the stored cache keys and drop the load.
    auto policy               = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    policy.active_tail_blocks = 1;
    ASSERT_EQ(policy.cp_mapping, CpBlockMappingMode::NONE);

    const auto plan = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/4,
                                                          /*cache_key_count=*/3,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kBaseSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{2, 2}}));
}

TEST(DecodeRpcServerTest, FullGroupKeepsWholeLogicalBlocksAfterReuse) {
    // Decode owns whole logical blocks of a BLOCK_ROUND_ROBIN group: the plan must
    // not shard it (the per-peer split happens later, per block), and reused
    // blocks are skipped.
    const auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    ASSERT_EQ(policy.cp_mapping, CpBlockMappingMode::BLOCK_ROUND_ROBIN);

    const auto plan = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/5,
                                                          /*cache_key_count=*/4,
                                                          /*reuse_block_size=*/2,
                                                          /*use_hybrid=*/false,
                                                          kBaseSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{2, 2}, {3, 3}}));
}

TEST(DecodeRpcServerTest, EmptyTableOrMissingCacheKeysYieldNoLoad) {
    const auto policy = makeCompactStatePolicy(/*active_tail_blocks=*/2);

    EXPECT_TRUE(DecodeRpcServer::buildGroupLoadPlan(policy,
                                                    /*local_block_num=*/0,
                                                    /*cache_key_count=*/11,
                                                    /*reuse_block_size=*/0,
                                                    /*use_hybrid=*/true,
                                                    kCompactSeqSizePerBlock,
                                                    kBaseSeqSizePerBlock)
                    .empty());
    EXPECT_TRUE(DecodeRpcServer::buildGroupLoadPlan(policy,
                                                    /*local_block_num=*/6,
                                                    /*cache_key_count=*/0,
                                                    /*reuse_block_size=*/0,
                                                    /*use_hybrid=*/true,
                                                    kCompactSeqSizePerBlock,
                                                    kBaseSeqSizePerBlock)
                    .empty());
}

}  // namespace rtp_llm
