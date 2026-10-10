#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <unordered_map>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/distribute/context_parallel/CpTransferProjection.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

namespace {

DecodeRpcServer::LoadKVCacheContext makeLoadContext(std::string              request_key,
                                                    std::vector<std::string> peer_addrs,
                                                    CacheKeysType            cache_keys,
                                                    GroupBlockIds            group_block_ids,
                                                    int32_t                  prefill_cp_size,
                                                    int64_t                  reuse_block_size = 0) {
    return {/*request_id=*/42,
            request_key,
            peer_addrs,
            cache_keys,
            std::move(group_block_ids),
            reuse_block_size,
            /*timeout_ms=*/1000,
            /*partition_count=*/1,
            /*partition_id=*/0,
            /*server_context=*/nullptr,
            prefill_cp_size};
}

class TestDecodeRpcService final: public RpcService::Service {
public:
    TestDecodeRpcService() {
        server_.meta_ = meta_;
    }

    grpc::Status RemoteGenerate(grpc::ServerContext*                                            context,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        return server_.RemoteGenerate(context, stream);
    }

    EngineScheduleInfo scheduleInfo() const {
        return meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    }

private:
    std::shared_ptr<RpcServerRuntimeMeta> meta_ = std::make_shared<RpcServerRuntimeMeta>();
    DecodeRpcServer                       server_;
};

GroupBase makeRpcGroup(std::string tag) {
    auto spec = std::make_shared<MHAKVCacheSpec>(tag, 8, 8, 1);

    GroupBase group;
    group.tag       = std::move(tag);
    group.spec      = std::move(spec);
    group.policy    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.block_num = 8;
    return group;
}

// Tokens per logical (cache-key sized) block, and the CP-scaled tokens per block
// of a compacted fixed/state group under prefill CP=2 (cp.scale_seq_size).
constexpr size_t kBaseSeqSizePerBlock    = 64;
constexpr size_t kCompactSeqSizePerBlock = kBaseSeqSizePerBlock * 2;

CacheGroupPolicy makeCompactStatePolicy(uint32_t active_tail_blocks) {
    auto policy               = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.cp_mapping         = CpBlockMappingMode::COMPACT_LAST_RANK;
    policy.active_tail_blocks = active_tail_blocks;
    policy.cp_slice           = CpBlockSliceMode::PAYLOAD_BYTES;
    EXPECT_EQ(policy.cp_mapping, CpBlockMappingMode::COMPACT_LAST_RANK);
    return policy;
}

// Test the selected logical pages. Physical key/slot/byte mapping is covered
// by cache-region tests and the handoff load fixture below.
std::vector<size_t> testLoadPlan(const CacheGroupPolicy& policy,
                                 size_t                  local_block_num,
                                 size_t                  cache_key_count,
                                 size_t                  reuse_block_size,
                                 bool                    use_hybrid,
                                 size_t                  group_seq_size_per_block,
                                 size_t                  base_seq_size_per_block) {
    const size_t key_scale = group_seq_size_per_block / base_seq_size_per_block;
    const bool   compact_tail     = policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK && key_scale > 1;
    const int    producer_cp_size = compact_tail ? static_cast<int>(key_scale) : 1;
    return selectLoadTransferPages(policy,
                                   cache_key_count / key_scale + (cache_key_count % key_scale != 0),
                                   local_block_num,
                                   compact_tail ? reuse_block_size : reuse_block_size / key_scale,
                                   use_hybrid,
                                   compact_tail,
                                   producer_cp_size,
                                   producer_cp_size,
                                   producer_cp_size - 1)
        .logical_pages;
}

std::shared_ptr<NormalGenerateStream> makeGenerateStream(int seq_length) {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::zeros({seq_length}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = seq_length + 16;
    return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
}

class HandoffTestEngine final: public EngineBase {
public:
    HandoffTestEngine(const EngineInitParams& params,
                      std::shared_ptr<KVCacheManager> manager,
                      bool                            has_mtp = false):
        EngineBase(params), has_mtp_(has_mtp) {
        resource_context_.cache_manager = std::move(manager);
    }
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void enqueue(std::shared_ptr<GenerateStream>&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in cache handoff test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
    bool isMTPEagle() override {
        return has_mtp_;
    }

private:
    bool has_mtp_;
};

class CapturingHandoffLoadStore final: public NormalCacheStore {
public:
    CapturingHandoffLoadStore() {
        // NormalCacheStore's destructor stops this holder. No transport or
        // worker threads are started by this synchronous capture fixture.
        request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(nullptr);
    }
    struct Block {
        void*    addr = nullptr;
        uint32_t bytes = 0;
    };
    std::unordered_map<std::string, std::vector<std::string>> visited_groups;
    std::unordered_map<std::string, std::unordered_map<std::string, Block>> loaded_blocks;

    void load(const std::shared_ptr<RequestBlockBuffer>& buffer,
              CacheStoreLoadDoneCallback callback,
              const std::string& ip,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        for (const auto& [key, block] : buffer->getBlocks()) {
            loaded_blocks[ip].emplace(key, Block{block->addr.get(), block->len});
        }
        callback(true, CacheStoreErrorCode::None);
    }
    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& buffers,
                                             const std::string& ip,
                                             uint32_t port,
                                             uint32_t rdma_port,
                                             int64_t timeout_ms,
                                             LoadContext::CheckCancelFunc cancel,
                                             int partition_count,
                                             int partition_id) override {
        for (const auto& buffer : buffers) {
            visited_groups[ip].push_back(buffer->getRequestKey());
        }
        auto context = std::make_shared<LoadContext>(shared_from_this(), /*combine_load=*/false);
        context->load(buffers, ip, port, rdma_port, timeout_ms, cancel, partition_count, partition_id);
        return context;
    }
};

}  // namespace

TEST(DecodeRpcServerTest, TimeoutLearnedAfterConstructionKeepsAbsoluteDeadline) {
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr metrics_reporter;
    DecodeGenerateContext        context(rpc_context, /*timeout_ms=*/0, nullptr, metrics_reporter, nullptr);

    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(1000);
    ASSERT_TRUE(context.request_deadline.has_value());
    const auto absolute_deadline = *context.request_deadline;
    EXPECT_GT(absolute_deadline, std::chrono::system_clock::now());
    EXPECT_EQ(context.request_timeout_ms, 1000);

    context.setRequestTimeoutMs(1000);
    EXPECT_EQ(context.request_deadline, absolute_deadline) << "reapplying a timeout must not extend its budget";

    context.setRequestTimeoutMs(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    EXPECT_TRUE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(0);
    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(-1);
    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());
}

TEST(DecodeRpcServerTest, invalidPrefillOnlyUsesInvalidArgumentWithDetails) {
    TestDecodeRpcService service;
    int                  listen_port = 0;
    grpc::ServerBuilder  builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &listen_port);
    builder.RegisterService(&service);
    auto grpc_server = builder.BuildAndStart();
    ASSERT_NE(grpc_server, nullptr);
    ASSERT_NE(listen_port, 0);

    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(listen_port), grpc::InsecureChannelCredentials());
    auto stub    = RpcService::NewStub(channel);
    grpc::ClientContext context;
    auto                stream = stub->RemoteGenerate(&context);
    GenerateRequestPB   request;
    request.set_stage(RemoteStage::ALLOCATE);
    request.set_client_id("test");
    request.set_request_id(1);
    request.mutable_input()->add_token_ids(0);
    request.mutable_input()->mutable_generate_config()->set_max_new_tokens(1);
    request.mutable_input()->mutable_generate_config()->set_prefill_only(true);

    EXPECT_TRUE(stream->Write(request));
    stream->WritesDone();
    GenerateOutputsPB response;
    EXPECT_FALSE(stream->Read(&response));
    const auto status = stream->Finish();
    grpc_server->Shutdown();
    grpc_server->Wait();

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_NE(error_details.error_message().find("max_new_tokens"), std::string::npos);

    const auto schedule_info = service.scheduleInfo();
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.finished_task_info_list.front().request_id, 1);
    EXPECT_EQ(schedule_info.finished_task_info_list.front().error_code,
              static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
}

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

    const auto* response = BroadcastLoadResponsePB::descriptor();
    ASSERT_NE(response, nullptr);
    EXPECT_EQ(response->FindFieldByName("loaded_cache_block_count")->number(), 3);

    const auto* remote = RemoteOperationRequestPB::descriptor();
    ASSERT_NE(remote, nullptr);
    EXPECT_TRUE(remote->IsReservedNumber(3));
    EXPECT_EQ(remote->FindFieldByName("group_ids"), nullptr);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
}

TEST(DecodeRpcServerTest, HeterogeneousPhysicalBlockMarksExactCoveredBaseKeys) {
    auto       policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    CacheConfig config;
    config.seq_size_per_block = 4;
    config.setTopology({makeRpcGroup("full")}, {{0, {"full"}}});
    const size_t key_scale = config.groupKeyScale("full");
    const auto plan   = testLoadPlan(policy,
                                   /*local_block_num=*/3,
                                   /*cache_key_count=*/5,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   8,
                                   4);
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan, (std::vector<size_t>{0, 1, 2}));
    std::vector<size_t> transferred(5, 0);
    for (size_t page : plan) {
        const size_t begin = page * key_scale;
        const size_t end   = std::min(begin + key_scale, transferred.size());
        for (size_t key = begin; key < end; ++key) {
            ++transferred[key];
        }
    }

    EXPECT_EQ(transferred, (std::vector<size_t>{1, 1, 1, 1, 1}));
}

TEST(DecodeRpcServerTest, CompactPhysicalBlockCoversEveryBaseKey) {
    auto       policy = makeCompactStatePolicy(/*active_tail_blocks=*/2);
    CacheConfig config;
    config.seq_size_per_block = kBaseSeqSizePerBlock;
    auto group                = makeRpcGroup("state");
    group.spec   = std::make_shared<MHAKVCacheSpec>("state", kCompactSeqSizePerBlock, kCompactSeqSizePerBlock, 1);
    group.policy = policy;
    config.setTopology({std::move(group)}, {{0, {"state"}}});
    const size_t key_scale = config.groupKeyScale("state");
    const auto plan   = testLoadPlan(policy,
                                   /*local_block_num=*/6,
                                   /*cache_key_count=*/11,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   kCompactSeqSizePerBlock,
                                   kBaseSeqSizePerBlock);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan, (std::vector<size_t>{4, 5}));
    std::vector<size_t> transferred(11, 0);
    for (size_t page : plan) {
        const size_t begin = page * key_scale;
        const size_t end   = std::min(begin + key_scale, transferred.size());
        for (size_t key = begin; key < end; ++key) {
            ++transferred[key];
        }
    }

    EXPECT_EQ(transferred, (std::vector<size_t>{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}));
}

TEST(DecodeRpcServerTest, CompletedHandoffPrefixRequiresEveryTransferObligation) {
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(
                  /*already_reused_blocks=*/2,
                  /*required=*/{0, 0, 2, 2, 2, 2},
                  /*transferred=*/{0, 0, 2, 2, 1, 2}),
              4u);
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(
                  /*already_reused_blocks=*/0, /*required=*/{2, 2}, /*transferred=*/{1, 2}),
              0u);
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(
                  /*already_reused_blocks=*/2, /*required=*/{0, 0, 0, 0}, /*transferred=*/{0, 0, 0, 0}),
              0u);
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(
                  /*already_reused_blocks=*/0, /*required=*/{1}, /*transferred=*/{}),
              0u);

    std::vector<size_t> arrivals{0, 0};
    const std::vector<size_t> required{2, 2};  // Two peer slices per key.
    ++arrivals[1];  // Later key arrives first.
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(0, required, arrivals), 0u);
    ++arrivals[0];
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(0, required, arrivals), 0u);
    ++arrivals[1];
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(0, required, arrivals), 0u);
    ++arrivals[0];
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(0, required, arrivals), 2u);
}

TEST(DecodeRpcServerTest, MultiRankHandoffUsesMinimumPrefix) {
    EXPECT_EQ(DecodeRpcServer::minLoadedCacheBlockCount({6, 4, 5}), 4u);
    EXPECT_EQ(DecodeRpcServer::minLoadedCacheBlockCount({}), 0u);
}

TEST(DecodeRpcServerTest, EmptyRemoteLoadTopologyFailsBeforePartitionArithmetic) {
    EXPECT_EQ(DecodeRpcServer::validateRemoteLoadTopology(/*worker_size=*/0, /*peer_size=*/1).code(),
              ErrorCode::LOAD_KV_CACHE_FAILED);
    EXPECT_EQ(DecodeRpcServer::validateRemoteLoadTopology(/*worker_size=*/1, /*peer_size=*/0).code(),
              ErrorCode::LOAD_KV_CACHE_FAILED);
    EXPECT_TRUE(DecodeRpcServer::validateRemoteLoadTopology(/*worker_size=*/4, /*peer_size=*/2).ok());
    EXPECT_EQ(DecodeRpcServer::validateRemoteLoadTopology(/*worker_size=*/3, /*peer_size=*/2).code(),
              ErrorCode::LOAD_KV_CACHE_FAILED);
}

TEST(DecodeRpcServerTest, OddTpWorkersWaitForEveryCompletionQueueResponse) {
    EXPECT_EQ(DecodeRpcServer::completionQueueExpectedResponseCounts(3), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(DecodeRpcServer::completionQueueExpectedResponseCounts(5), (std::vector<size_t>{2, 2, 1}));
    EXPECT_EQ(DecodeRpcServer::completionQueueExpectedResponseCounts(4), (std::vector<size_t>{2, 2}));
    EXPECT_TRUE(DecodeRpcServer::completionQueueExpectedResponseCounts(0).empty());
}

TEST(DecodeRpcServerTest, CPShardedLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101, 102};
    GroupBlockIds                   group_block_ids;
    const auto                      load_context = makeLoadContext(request_key,
                                              peer_addrs,
                                              cache_keys,
                                              std::move(group_block_ids),
                                              /*cp_size=*/2,
                                              /*reuse=*/3);

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
    GroupBlockIds                   group_block_ids;
    const auto                      load_context = makeLoadContext(request_key,
                                              peer_addrs,
                                              cache_keys,
                                              std::move(group_block_ids),
                                              /*cp_size=*/2,
                                              /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/1, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
}

TEST(DecodeRpcServerTest, NonCpLoadRequestKeepsOrdinaryAndMlaPeerRoutes) {
    DecodeRpcServer server;
    server.resource_.workers               = {"decode-0", "decode-1"};
    const std::vector<std::string> peers   = {"prefill-0", "prefill-1", "prefill-2", "prefill-3"};
    const auto                     context = makeLoadContext("request", peers, {101}, GroupBlockIds{}, /*cp_size=*/1);

    const auto ordinary = server.constructRemoteLoadRequest(context, /*index=*/1, peers);
    EXPECT_EQ(ordinary.partition_count(), 1);
    EXPECT_EQ(ordinary.partition_id(), 0);
    ASSERT_EQ(ordinary.peer_addrs_size(), 2);
    EXPECT_EQ(ordinary.peer_addrs(0), "prefill-2");
    EXPECT_EQ(ordinary.peer_addrs(1), "prefill-3");

    const auto mla = server.constructRemoteLoadRequestForMla(context, /*index=*/1, peers);
    EXPECT_EQ(mla.partition_count(), 1);
    EXPECT_EQ(mla.partition_id(), 0);
    ASSERT_EQ(mla.peer_addrs_size(), 1);
    EXPECT_EQ(mla.peer_addrs(0), "prefill-2");

    server.resource_.workers                   = {"decode-0", "decode-1", "decode-2", "decode-3"};
    const std::vector<std::string> fewer_peers = {"prefill-0", "prefill-1"};
    const auto fewer_context = makeLoadContext("request", fewer_peers, {101}, GroupBlockIds{}, /*cp_size=*/1);
    const auto split         = server.constructRemoteLoadRequest(fewer_context, /*index=*/3, fewer_peers);
    EXPECT_EQ(split.partition_count(), 2);
    EXPECT_EQ(split.partition_id(), 1);
    ASSERT_EQ(split.peer_addrs_size(), 1);
    EXPECT_EQ(split.peer_addrs(0), "prefill-1");

    const auto mla_split = server.constructRemoteLoadRequestForMla(fewer_context, /*index=*/3, fewer_peers);
    EXPECT_EQ(mla_split.partition_count(), 1);
    EXPECT_EQ(mla_split.partition_id(), 0);
    ASSERT_EQ(mla_split.peer_addrs_size(), 1);
    EXPECT_EQ(mla_split.peer_addrs(0), "prefill-1");

    server.maga_init_params_.parallelism_config.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
    const auto unsharded_cp = server.constructRemoteLoadRequest(fewer_context, /*index=*/3, fewer_peers);
    EXPECT_EQ(unsharded_cp.partition_count(), 4);
    EXPECT_EQ(unsharded_cp.partition_id(), 3);
    ASSERT_EQ(unsharded_cp.peer_addrs_size(), 1);
    EXPECT_EQ(unsharded_cp.peer_addrs(0), "prefill-1");
}

TEST(DecodeRpcServerTest, TaggedBlockRowsPreserveReceivedOrderAcrossTopologies) {
    auto topology =
        CacheTopology::create({makeRpcGroup("linear"), makeRpcGroup("full")}, {{0, {"linear"}}, {1, {"full"}}});
    BroadcastLoadRequestPB request;
    auto*                  full = request.add_tagged_group_block_ids();
    full->set_tag("full");
    full->add_block_ids(10);
    auto* linear = request.add_tagged_group_block_ids();
    linear->set_tag("linear");
    linear->add_block_ids(20);

    const auto blocks = DecodeRpcServer::decodeGroupBlockIds(request, *topology);
    EXPECT_EQ(blocks.orderedTags(), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(blocks.blocks("full"), (BlockIndicesType{10}));
    EXPECT_EQ(blocks.blocks("linear"), (BlockIndicesType{20}));

    auto reordered =
        CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"linear"}}, {1, {"full"}}});
    EXPECT_NE(topology->groupTags(), reordered->groupTags());
    const auto reordered_blocks = DecodeRpcServer::decodeGroupBlockIds(request, *reordered);
    EXPECT_EQ(reordered_blocks.orderedTags(), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(reordered_blocks.blocks("full"), blocks.blocks("full"));
    EXPECT_EQ(DecodeRpcServer::makeTaggedRequestKey(42, 1, topology->group("full").tag),
              DecodeRpcServer::makeTaggedRequestKey(42, 1, reordered->group("full").tag));
}

TEST(DecodeRpcServerTest, LoadContextOwnsGroupIdentityAndRequestData) {
    auto full_blocks = std::make_shared<BlockIds>();
    full_blocks->assign({10});
    auto linear_blocks = std::make_shared<BlockIds>();
    linear_blocks->assign({20, NULL_BLOCK_IDX});
    GroupBlockIds blocks{{{"linear", 0}, {"full", 1}}, {linear_blocks, full_blocks}};
    auto          context = makeLoadContext("request", {"peer"}, {101}, blocks, /*prefill_cp_size=*/1);
    EXPECT_EQ(context.groupBlockIds().rows_[1], full_blocks);
    // The context owns its index, while the mutable BlockIds holders remain shared.
    blocks.tag_to_index_.clear();
    blocks.rows_.clear();
    full_blocks->assign({11});
    full_blocks.reset();
    linear_blocks.reset();

    EXPECT_EQ(context.request_key, "request");
    EXPECT_EQ(context.peer_addrs, (std::vector<std::string>{"peer"}));
    EXPECT_EQ(context.cache_keys, (CacheKeysType{101}));
    EXPECT_EQ(context.groupBlockIds().orderedTags(), (std::vector<std::string>{"linear", "full"}));
    EXPECT_EQ(context.blockIdsForGroup("full").blocks(), (BlockIndicesType{11}));
    EXPECT_EQ(context.blockIdsForGroup("linear").blocks(), (BlockIndicesType{20, NULL_BLOCK_IDX}));
    EXPECT_ANY_THROW(context.blockIdsForGroup("missing"));
}

TEST(DecodeRpcServerTest, ResourceRowsKeepTheirOwnOrderAndSharedLifetime) {
    auto resource = std::make_unique<KVCacheResource>();
    resource->initGroups(
        CacheTopology::create({makeRpcGroup("linear"), makeRpcGroup("full")}, {{0, {"full", "linear"}}}));
    resource->mutableBlockIds("linear").assign({20});
    resource->mutableBlockIds("full").assign({10});
    auto context = makeLoadContext("request", {"peer"}, {}, resource->groupBlockIds(), 1);
    EXPECT_NE(&context.groupBlockIds().tag_to_index_, &resource->groupBlockIds().tag_to_index_);
    EXPECT_EQ(context.groupBlockIds().rows_[1], resource->groupBlockIds().rows_[1]);
    resource->mutableBlockIds("full").assign({11, NULL_BLOCK_IDX});
    resource.reset();
    auto sender_topology =
        CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"full", "linear"}}});
    BroadcastLoadRequestPB request;
    DecodeRpcServer::appendGroupBlockIds(context, *sender_topology, request);
    EXPECT_EQ(context.groupBlockIds().orderedTags(), (std::vector<std::string>{"linear", "full"}));
    EXPECT_EQ(request.tagged_group_block_ids(0).tag(), "linear");
    EXPECT_EQ(context.blockIdsForGroup("full").blocks(), (BlockIndicesType{11, NULL_BLOCK_IDX}));
}

TEST(DecodeRpcServerTest, EmptyTaggedBlockRowsAreRejected) {
    auto                   topology = CacheTopology::create({makeRpcGroup("full")}, {{0, {"full"}}});
    BroadcastLoadRequestPB request;
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(request, *topology));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsRejectTopologyMismatchWithoutReplacingResult) {
    auto topology = CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"full", "linear"}}});
    auto holder   = std::make_shared<BlockIds>();
    holder->assign({42});
    GroupBlockIds          previous{{{"unchanged", 0}}, {holder}};
    BroadcastLoadRequestPB missing_tag;
    auto*                  row = missing_tag.add_tagged_group_block_ids();
    row->set_tag("full");
    row->add_block_ids(1);

    EXPECT_ANY_THROW(previous = DecodeRpcServer::decodeGroupBlockIds(missing_tag, *topology));
    EXPECT_EQ(previous.blocks("unchanged"), (BlockIndicesType{42}));

    BroadcastLoadRequestPB duplicate;
    for (int i = 0; i < 2; ++i) {
        auto* duplicate_row = duplicate.add_tagged_group_block_ids();
        duplicate_row->set_tag("full");
        duplicate_row->add_block_ids(i + 1);
    }
    EXPECT_ANY_THROW(previous = DecodeRpcServer::decodeGroupBlockIds(duplicate, *topology));
    EXPECT_EQ(previous.blocks("unchanged"), (BlockIndicesType{42}));

    BroadcastLoadRequestPB unknown;
    auto*                  unknown_row = unknown.add_tagged_group_block_ids();
    unknown_row->set_tag("foreign");
    unknown_row->add_block_ids(1);
    EXPECT_ANY_THROW(previous = DecodeRpcServer::decodeGroupBlockIds(unknown, *topology));
    EXPECT_EQ(previous.blocks("unchanged"), (BlockIndicesType{42}));
}

TEST(DecodeRpcServerTest, DenseContextRejectsInvalidIdentityBeforeUse) {
    auto                             holder = std::make_shared<BlockIds>();
    const std::vector<GroupBlockIds> malformed{
        {{}, {holder}},                                    // Missing identity.
        {{{"", 0}}, {holder}},                             // Empty tag.
        {{{"full", 0}, {"linear", 0}}, {holder, holder}},  // Duplicate index and uncovered row.
        {{{"full", 1}}, {holder}},                         // Out-of-range index.
        {{{"full", 0}, {"linear", 2}}, {holder, holder}},  // Hole in the dense index.
        {{{"full", 0}}, {nullptr}},                        // Null holder.
        {{{"full", 0}}, {}}};                              // Index without a row.
    for (const auto& blocks : malformed) {
        EXPECT_ANY_THROW(makeLoadContext("request", {"peer"}, {}, blocks, 1));
    }
}

TEST(DecodeRpcServerTest, DenseContextSerializesOwnOrderAndPreservesEmptyRows) {
    auto topology = CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"full", "linear"}}});
    auto full     = std::make_shared<BlockIds>();
    full->assign({10, NULL_BLOCK_IDX});
    auto linear  = std::make_shared<BlockIds>();
    auto context = makeLoadContext("request", {"peer"}, {}, {{{"linear", 0}, {"full", 1}}, {linear, full}}, 1);
    BroadcastLoadRequestPB request;
    DecodeRpcServer::appendGroupBlockIds(context, *topology, request);
    ASSERT_EQ(request.tagged_group_block_ids_size(), 2);
    EXPECT_EQ(request.tagged_group_block_ids(0).tag(), "linear");
    EXPECT_EQ(request.tagged_group_block_ids(0).block_ids_size(), 0);
    EXPECT_EQ(request.tagged_group_block_ids(1).tag(), "full");
    auto rows     = DecodeRpcServer::decodeGroupBlockIds(request, *topology);
    auto received = makeLoadContext("received", {"peer"}, {}, std::move(rows), 1);
    EXPECT_EQ(received.groupBlockIds().orderedTags(), context.groupBlockIds().orderedTags());
    EXPECT_TRUE(received.blockIdsForGroup("linear").blocks().empty());
    EXPECT_EQ(received.blockIdsForGroup("full").blocks(), (BlockIndicesType{10, NULL_BLOCK_IDX}));
    EXPECT_NE(received.groupBlockIds().rows_[1], full);
}

TEST(DecodeRpcServerTest, DenseContextTopologyFailureDoesNotPublishWireRows) {
    auto topology = CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"full", "linear"}}});
    auto holder   = std::make_shared<BlockIds>();
    const std::vector<GroupBlockIds> mismatched{{{{"full", 0}}, {holder}},
                                                {{{"full", 0}, {"foreign", 1}}, {holder, holder}}};
    for (const auto& blocks : mismatched) {
        auto                   context = makeLoadContext("request", {"peer"}, {}, blocks, 1);
        BroadcastLoadRequestPB request;
        EXPECT_ANY_THROW(DecodeRpcServer::appendGroupBlockIds(context, *topology, request));
        EXPECT_EQ(request.tagged_group_block_ids_size(), 0);
    }
    BroadcastLoadRequestPB request;
    request.add_tagged_group_block_ids()->set_tag("full");
    request.add_tagged_group_block_ids()->set_tag("");
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(request, *topology));
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

class PrefillCompletionRpcTest: public DeviceTestBase {};

TEST_F(PrefillCompletionRpcTest, SettlesWithoutDecodeAndPreservesProtocolFailures) {
    class CompletionService final: public RpcService::Service {
    public:
        grpc::Status RemoteGenerate(grpc::ServerContext* server_context, ServerStream* rpc_stream) override {
            DecodeRpcContext             rpc_context{rpc_stream};
            kmonitor::MetricsReporterPtr reporter;
            auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
            DecodeGenerateContext        context(rpc_context, 0, server_context, reporter, meta);
            context.request_id = 42;
            context.allocate_request.set_client_id("prefill");
            auto cache = std::make_shared<KVCacheManager>(
                test::makeSimpleMhaCacheConfig(1, 8, 2, DataType::TYPE_FP16), false, nullptr);
            EXPECT_TRUE(cache->init());
            ResourceContext resources;
            resources.cache_manager = cache;
            resources.role_type     = RoleType::DECODE;
            auto input              = std::make_shared<GenerateInput>();
            input->request_id       = 42;
            input->begin_time_us    = currentTimeUs();
            input->generate_config  = std::make_shared<GenerateConfig>();
            if (prefill_only) {
                GenerateConfigPB wire_config;
                wire_config.set_max_new_tokens(0);
                wire_config.set_prefill_only(explicit_prefill_only);
                input->generate_config = QueryConverter::transGenerateConfig(&wire_config);
                EXPECT_TRUE(input->generate_config->isPrefillOnly());
            }
            input->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
            ModelConfig config;
            config.max_seq_len = 16;
            auto stream = std::make_shared<NormalGenerateStream>(input, config, RuntimeConfig{}, resources, nullptr);
            EXPECT_TRUE(stream->initKVBlock().ok());
            EXPECT_GT(stream->stream_cache_resource_->curBlocksNum(), 0);
            if (running) {
                stream->generate_status_->status.store(StreamState::RUNNING);
            }
            context.setStream(stream);
            if (expired) {
                context.request_deadline = std::chrono::system_clock::now() - std::chrono::seconds(1);
            }
            if (stream_error) {
                stream->reportError(ErrorCode::MALLOC_FAILED, "test allocation failure");
            }
            DecodeRpcServer server;
            server.localGenerate(context);
            status                  = context.error_status;
            finished                = stream->isFinished();
            had_error               = stream->hasError();
            resource_released       = stream->stream_cache_resource_->isResourceReleased();
            blocks_after_completion = stream->stream_cache_resource_->curBlocksNum();
            EXPECT_EQ(stream->outputTokenLen(), 0);
            const auto timing       = stream->getTimeInfo();
            executed                = timing.running_started || timing.first_token_committed || timing.generation_done;
            RpcMetricsCollector metrics;
            context.collectBasicMetrics(metrics);
            error_qps = metrics.error_qps;
            if (running) {
                // This fixture has no scheduler to drain the deliberately invalid running stream.
                stream->reportError(ErrorCode::CANCELLED, "test cleanup");
                stream->moveToNext();
            }
            context.stopStream();
            error_after_cleanup = stream->hasError();
            remaining_requests  = meta->getEngineScheduleInfo(-1).running_task_info_list.size();
            return status;
        }
        bool         prefill_only          = false;
        bool         explicit_prefill_only = true;
        bool         expired               = false;
        bool         stream_error          = false;
        bool         running               = false;
        grpc::Status status;
        bool         finished = false, had_error = false, resource_released = false;
        bool         executed = false, error_qps = false, error_after_cleanup = false;
        size_t       remaining_requests      = 1;
        size_t       blocks_after_completion = 1;
    };

    for (const std::string mode : {"complete",
                                   "prefill_only",
                                   "prefill_only_generate",
                                   "bare_zero_generate",
                                   "eof",
                                   "wrong_request",
                                   "wrong_client",
                                   "wrong_stage",
                                   "expired",
                                   "stream_error",
                                   "running"}) {
        SCOPED_TRACE(mode);
        CompletionService service;
        service.prefill_only =
            mode == "prefill_only" || mode == "prefill_only_generate" || mode == "bare_zero_generate";
        service.explicit_prefill_only = mode != "bare_zero_generate";
        service.expired               = mode == "expired";
        service.stream_error          = mode == "stream_error";
        service.running               = mode == "running";
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        auto server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        auto stub = RpcService::NewStub(
            grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto rpc = stub->RemoteGenerate(&context);
        if (mode != "eof") {
            GenerateRequestPB request;
            request.set_stage(mode == "wrong_stage" ? RemoteStage::LOAD :
                              mode == "prefill_only_generate" || mode == "bare_zero_generate" ?
                                                      RemoteStage::GENERATE :
                                                      RemoteStage::PREFILL_COMPLETE);
            request.set_request_id(mode == "wrong_request" ? 43 : 42);
            request.set_client_id(mode == "wrong_client" ? "other" : "prefill");
            EXPECT_TRUE(rpc->Write(request));
        }
        rpc->WritesDone();
        const auto status = rpc->Finish();
        server->Shutdown();
        server->Wait();
        EXPECT_EQ(service.remaining_requests, 0u);
        EXPECT_FALSE(service.executed);
        if (mode == "complete" || mode == "prefill_only") {
            EXPECT_TRUE(status.ok());
            EXPECT_TRUE(service.finished);
            EXPECT_TRUE(service.resource_released);
            EXPECT_EQ(service.blocks_after_completion, 0u);
            EXPECT_FALSE(service.had_error);
            EXPECT_FALSE(service.error_qps);
            EXPECT_FALSE(service.error_after_cleanup);
        } else {
            EXPECT_FALSE(status.ok());
            EXPECT_TRUE(service.error_qps);
            if (mode == "eof" || mode == "wrong_stage") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
            } else if (mode == "prefill_only_generate" || mode == "bare_zero_generate") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
                ErrorDetailsPB details;
                ASSERT_TRUE(details.ParseFromString(status.error_details()));
                EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
                EXPECT_NE(details.error_message().find("prefill-only"), std::string::npos);
            } else if (mode == "wrong_request" || mode == "wrong_client") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
            } else if (mode == "expired") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
            } else if (mode == "running") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
            }
        }
    }
}

TEST(DecodeRpcServerTest, CacheLoadClientErrorPreservesCodeWithoutTopologyDetails) {
    const auto client_error =
        DecodeRpcServer::cacheLoadClientError(/*request_id=*/12345, ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED);

    EXPECT_EQ(client_error.code(), ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED);
    EXPECT_EQ(client_error.ToString(), "cache load failed; correlation_id=12345");
    EXPECT_EQ(client_error.ToString().find("10.0.0.8:1234"), std::string::npos);
    EXPECT_EQ(client_error.ToString().find("rdma"), std::string::npos);
}

TEST(DecodeRpcServerTest, SerializedAllocationErrorCarriesGrammarDomainCode) {
    DecodeRpcServer server;
    const auto      status = server.serializeErrorMsg(
        "request", RequestInfo{}, ErrorInfo(ErrorCode::GRAMMAR_COMPILE_OVERLOADED, "grammar queue full"));

    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::GRAMMAR_COMPILE_OVERLOADED));
    EXPECT_NE(details.error_message().find("grammar queue full"), std::string::npos);
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

TEST(DecodeRpcServerTest, HeterogeneousFullGroupUsesGlobalEndpointKeys) {
    const auto policy      = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto decode_plan = testLoadPlan(policy,
                                          /*local_block_num=*/3,
                                          /*cache_key_count=*/6,
                                          /*reuse_block_size=*/0,
                                          /*use_hybrid=*/true,
                                          /*group_seq_size_per_block=*/2,
                                          /*base_seq_size_per_block=*/1);

    EXPECT_EQ(decode_plan, (std::vector<size_t>{0, 1, 2}));
}

TEST(DecodeRpcServerTest, HeterogeneousPartialBlockUsesLastAvailableKey) {
    const auto policy      = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto decode_plan = testLoadPlan(policy,
                                          /*local_block_num=*/3,
                                          /*cache_key_count=*/5,
                                          /*reuse_block_size=*/0,
                                          /*use_hybrid=*/true,
                                          /*group_seq_size_per_block=*/2,
                                          /*base_seq_size_per_block=*/1);

    EXPECT_EQ(decode_plan, (std::vector<size_t>{0, 1, 2}));
}

TEST(DecodeRpcServerTest, CompactStateGroupLoadsGlobalTailKeysIntoCanonicalSlots) {
    // 11 logical blocks under prefill CP=2 compact into a 6-slot state table:
    // slot j covers logical blocks [2j, 2j+1], so the two active tail slots 4 and
    // 5 must be filled from the *global* cache keys 9 and 10 - not from keys 4
    // and 5, which is what indexing the compacted table with logical positions
    // (or the table length with the logical key count) would produce.
    const auto plan = testLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                   /*local_block_num=*/6,
                                   /*cache_key_count=*/11,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   kCompactSeqSizePerBlock,
                                   kBaseSeqSizePerBlock);

    EXPECT_EQ(plan, (std::vector<size_t>{4, 5}));
}

TEST(DecodeRpcServerTest, CompactStateGroupLoadPlanMatchesProducerStorePlan) {
    // Both sides must retain the old fixed global key and canonical slot pairs.
    const auto policy      = makeCompactStatePolicy(/*active_tail_blocks=*/2);
    const auto decode_plan = testLoadPlan(policy,
                                          /*local_block_num=*/6,
                                          /*cache_key_count=*/11,
                                          /*reuse_block_size=*/0,
                                          /*use_hybrid=*/true,
                                          kCompactSeqSizePerBlock,
                                          kBaseSeqSizePerBlock);
    EXPECT_EQ(decode_plan, (std::vector<size_t>{4, 5}));
}

TEST(DecodeRpcServerTest, CompactStateGroupIgnoresSpeculativeReserveTailSlots) {
    // MTP reserve slots make the positional table longer than the canonical slots
    // the sequence actually backs. The destinations must stay 4 and 5; picking the
    // tail of the table would target the reserve slot 6 and duplicate key 10.
    const auto plan = testLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                   /*local_block_num=*/7,
                                   /*cache_key_count=*/11,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   kCompactSeqSizePerBlock,
                                   kBaseSeqSizePerBlock);

    EXPECT_EQ(plan, (std::vector<size_t>{4, 5}));
}

TEST(DecodeRpcServerTest, CpFourCompactTailLoadsPartialBundleWithoutReserveSlot) {
    // Nine B=64 keys occupy three P=4 compact slots. The third slot contains
    // only key 8; slot 3 is a speculative reserve and has no producer data.
    const auto plan = testLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                   /*local_block_num=*/4,
                                   /*cache_key_count=*/9,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   /*group_seq_size_per_block=*/256,
                                   /*base_seq_size_per_block=*/64);
    EXPECT_EQ(plan, (std::vector<size_t>{1, 2}));
}

TEST(DecodeRpcServerTest, CompactOneTailGroupLoadsOnlyTheLastCanonicalSlot) {
    // hca_state declares active_tail_blocks=1; explicit_block_num only sizes its
    // pool and must not widen the per-request projection.
    auto policy               = makeCompactStatePolicy(/*active_tail_blocks=*/1);
    policy.explicit_block_num = 256;

    const auto plan = testLoadPlan(policy,
                                   /*local_block_num=*/6,
                                   /*cache_key_count=*/11,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   kCompactSeqSizePerBlock,
                                   kBaseSeqSizePerBlock);

    EXPECT_EQ(plan, (std::vector<size_t>{5}));
}

TEST(DecodeRpcServerTest, CompactHybridWithoutDeclaredTailKeepsLastCanonicalSlot) {
    const auto policy = makeCompactStatePolicy(/*active_tail_blocks=*/0);
    const auto producer =
        selectProducerTransferPages(policy, /*logical_page_count=*/6, /*use_hybrid=*/true, /*compact_tail=*/true);
    const auto decode = selectLoadTransferPages(policy,
                                                /*logical_page_count=*/6,
                                                /*max_destination_slots=*/6,
                                                /*first_page_after_reuse=*/0,
                                                /*use_hybrid=*/true,
                                                /*compact_tail=*/true,
                                                /*producer_cp_size=*/2,
                                                /*decode_peer_count=*/2,
                                                /*peer_index=*/1);
    EXPECT_EQ(producer, (std::vector<size_t>{5}));
    EXPECT_EQ(decode.logical_pages, (std::vector<size_t>{5}));
}

TEST(DecodeRpcServerTest, StateAndSwaTailSelectionsIncludeEverySlicedPeer) {
    for (int cp_size : {2, 4}) {
        for (const auto slice_mode : {CpBlockSliceMode::PAYLOAD_BYTES, CpBlockSliceMode::EQUAL_BYTES}) {
            auto policy                = makeCompactStatePolicy(/*active_tail_blocks=*/1);
            policy.enable_prefix_reuse = false;  // HCA_STATE is still a PD obligation.
            policy.cp_slice            = slice_mode;
            const size_t key_count     = 9;
            const size_t logical_page_count =
                key_count / static_cast<size_t>(cp_size) + (key_count % static_cast<size_t>(cp_size) != 0);
            for (int peer = 0; peer < cp_size; ++peer) {
                const auto plan = selectLoadTransferPages(policy,
                                                          logical_page_count,
                                                          logical_page_count + 1,
                                                          /*first_page_after_reuse=*/0,
                                                          /*use_hybrid=*/true,
                                                          /*compact_tail=*/true,
                                                          cp_size,
                                                          cp_size,
                                                          peer);
                ASSERT_TRUE(plan.peer_participates);
                ASSERT_EQ(plan.logical_pages, (std::vector<size_t>{logical_page_count - 1}));
                EXPECT_TRUE(transferPageReadFromPeer(policy, plan.logical_pages.front(), true, cp_size, cp_size, peer));
            }
        }
    }
}

TEST(DecodeRpcServerTest, SingleGroupNonFullKeepsLegacyPeerSelection) {
    const auto policy = makeCompactStatePolicy(/*active_tail_blocks=*/1);
    for (int peer = 0; peer < 2; ++peer) {
        const auto plan = selectLoadTransferPages(policy,
                                                  /*logical_page_count=*/2,
                                                  /*max_destination_slots=*/2,
                                                  /*first_page_after_reuse=*/0,
                                                  /*use_hybrid=*/false,
                                                  /*compact_tail=*/true,
                                                  /*producer_cp_size=*/2,
                                                  /*decode_peer_count=*/2,
                                                  peer);
        ASSERT_EQ(plan.logical_pages, (std::vector<size_t>{0, 1}));
        EXPECT_EQ(transferPageReadFromPeer(policy, 0, false, 2, 2, peer), peer == 0);
        EXPECT_EQ(transferPageReadFromPeer(policy, 1, false, 2, 2, peer), peer == 1);
    }
}

class HybridHandoffLoadTest: public DeviceTestBase {};

TEST_F(HybridHandoffLoadTest, LoadCacheMainAndDraftUsePhysicalRowsAndPeerObligations) {
    constexpr uint32_t base_tokens = 4;
    CacheConfig config;
    config.dtype                     = DataType::TYPE_UINT8;
    config.layer_num                 = 1;
    config.seq_size_per_block        = base_tokens;
    config.use_opaque_kv_cache_store = true;
    auto full_spec = test::makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, base_tokens, "full");
    auto state_spec = test::makeResolvedOpaqueSpec(true, "state", DataType::TYPE_UINT8, 32, base_tokens * 2);
    auto swa_spec = test::makeResolvedOpaqueSpec(true, "swa", DataType::TYPE_UINT8, 33, base_tokens * 2);
    auto plain_spec = test::makeResolvedOpaqueSpec(true, "plain", DataType::TYPE_UINT8, 16, base_tokens);
    config.fromGroupedSpecs({full_spec, state_spec, swa_spec, plain_spec},
                            {{0}, {0}, {0}, {0}},
                            {CacheGroupType::FULL, CacheGroupType::SWA, CacheGroupType::SWA, CacheGroupType::SWA},
                            {"full", "state", "swa", "plain"});
    auto groups = config.topology().groups();
    for (auto& group : groups) {
        group.block_num = 8;
        if (group.tag == "state") {
            group.policy.cp_slice           = CpBlockSliceMode::PAYLOAD_BYTES;
            group.policy.active_tail_blocks = 1;
        } else if (group.tag == "swa") {
            group.policy.cp_slice           = CpBlockSliceMode::EQUAL_BYTES;
            group.policy.active_tail_blocks = 1;
            test::setGroupLayout(group, /*kv_stride=*/40, /*scale_stride=*/0);
        } else if (group.tag == "plain") {
            group.policy.active_tail_blocks = 1;
        }
    }
    config.setTopology(std::move(groups), config.topology().layers());

    // MTP module0 shares the logical group/table contract, while its physical
    // row sizes and remote model namespace belong to the draft child.
    auto draft_config = config;
    auto draft_groups = draft_config.topology().groups();
    for (auto& group : draft_groups) {
        if (group.tag == "state") {
            group.spec = test::makeResolvedOpaqueSpec(true, "state", DataType::TYPE_UINT8, 48, base_tokens * 2);
        } else if (group.tag == "swa") {
            group.spec = test::makeResolvedOpaqueSpec(true, "swa", DataType::TYPE_UINT8, 41, base_tokens * 2);
            test::setGroupLayout(group, /*kv_stride=*/48, /*scale_stride=*/0);
        }
    }
    draft_config.setTopology(std::move(draft_groups), draft_config.topology().layers());
    config.mtp_sub_configs.push_back(config.mergeMTPModule(draft_config, /*module_index=*/0, /*main_layer_num=*/1));

    auto cache_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    ASSERT_TRUE(cache_manager->init());
    EngineInitParams params;
    params.model_config_.num_layers            = 1;
    params.model_id                            = 7;
    params.parallelism_config.local_world_size = 1;
    params.parallelism_config.world_size       = 1;
    params.parallelism_config.role_type        = RoleType::DECODE;
    auto module                                = std::make_unique<EngineInitParams>();
    module->model_id                           = 17;
    module->model_config_.num_layers           = 1;
    auto modules                               = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    modules->push_back(std::move(module));
    ProposeModelEngineInitParams proposal(SP_TYPE_MTP, /*gen_num_per_cycle=*/1, std::move(modules));
    DecodeRpcServer              server;
    server.engine_                   = std::make_shared<HandoffTestEngine>(params, cache_manager, /*has_mtp=*/true);
    server.maga_init_params_         = params;
    server.propose_maga_init_params_ = &proposal;
    auto capture                     = std::make_shared<CapturingHandoffLoadStore>();
    server.resource_.cache_store     = capture;

    std::vector<std::shared_ptr<BlockIds>>  rows;
    std::unordered_map<std::string, size_t> index;
    for (const auto& tag : config.groupTags()) {
        index.emplace(tag, rows.size());
        auto blocks = std::make_shared<BlockIds>();
        blocks->assign(tag == "full" ? BlockIndicesType{0, 1} : BlockIndicesType{0});
        rows.push_back(std::move(blocks));
    }
    grpc::ServerContext rpc_context;
    const auto          load_context = makeLoadContext("hybrid",
                                              {"peer0:1:1", "peer1:1:1"},
                                              {111, 222},
                                              GroupBlockIds{std::move(index), std::move(rows)},
                                              /*prefill_cp_size=*/2);
    // makeLoadContext deliberately leaves the server context null; loadCache's
    // cancellation callback needs a live context for the synchronous mock.
    auto live_context           = load_context;
    live_context.server_context = &rpc_context;
    const auto result           = server.loadCache(live_context);
    ASSERT_TRUE(result.ok()) << result.error_info.ToString();
    EXPECT_EQ(result.loaded_cache_block_count, 2u);

    const auto expectGroup = [&](const std::string& peer, const std::string& tag, bool present) {
        const auto  key  = DecodeRpcServer::makeTaggedRequestKey(42, 0, tag);
        const auto& seen = capture->visited_groups.at(peer);
        EXPECT_EQ(std::find(seen.begin(), seen.end(), key) != seen.end(), present) << peer << " " << tag;
    };
    for (const auto& peer : {"peer0", "peer1"}) {
        expectGroup(peer, "full", true);
        expectGroup(peer, "state", true);
        expectGroup(peer, "swa", true);
    }
    expectGroup("peer0", "plain", true);
    expectGroup("peer1", "plain", false);
    EXPECT_EQ(capture->visited_groups.at("peer0").size(), 8u);
    EXPECT_EQ(capture->visited_groups.at("peer1").size(), 6u);

    for (const auto& [global_layer, model_id] : {std::pair<int, size_t>{0, 7}, {1, 17}}) {
        for (const auto& tag : {"state", "swa"}) {
            const size_t local_bytes = global_layer == 0 ? (std::string(tag) == "state" ? 16 : 20) : 24;
            const auto   full        = cache_manager->convertIndexToBuffer(global_layer, tag, 0);
            ASSERT_EQ(full.size(), 1u);
            for (int peer = 0; peer < 2; ++peer) {
                const auto  peer_name = "peer" + std::to_string(peer);
                const auto  key       = "kv_" + makeCacheKey(model_id, "222", 0, tag);
                const auto& loaded    = capture->loaded_blocks.at(peer_name).at(key);
                EXPECT_EQ(loaded.addr, static_cast<uint8_t*>(full[0].addr) + peer * local_bytes);
                EXPECT_EQ(loaded.bytes, local_bytes);
            }
        }
    }

    // A missing destination still contributes a required key obligation. The
    // complete first B key may be reused, but key 222 cannot be published as
    // loaded merely because the other STATE/SWA groups transferred it.
    std::vector<std::shared_ptr<BlockIds>>  null_rows;
    std::unordered_map<std::string, size_t> null_index;
    for (const auto& tag : config.groupTags()) {
        null_index.emplace(tag, null_rows.size());
        auto blocks = std::make_shared<BlockIds>();
        blocks->assign(tag == "full" ? BlockIndicesType{0, NULL_BLOCK_IDX} : BlockIndicesType{0});
        null_rows.push_back(std::move(blocks));
    }
    auto null_context           = makeLoadContext("hybrid-null",
                                        {"peer0:1:1", "peer1:1:1"},
                                        {111, 222},
                                        GroupBlockIds{std::move(null_index), std::move(null_rows)},
                                        /*prefill_cp_size=*/2);
    null_context.server_context = &rpc_context;
    capture->loaded_blocks.clear();
    const auto null_result = server.loadCache(null_context);
    ASSERT_TRUE(null_result.ok()) << null_result.error_info.ToString();
    EXPECT_EQ(null_result.loaded_cache_block_count, 1u);
    const auto missing_full = "kv_" + makeCacheKey(/*model_id=*/7, "222", /*layer_id=*/0, "full");
    EXPECT_EQ(capture->loaded_blocks.at("peer1").count(missing_full), 0u);
}

TEST(DecodeRpcServerTest, CompactStateGroupWithSingleLogicalBlockLoadsKeyZero) {
    const auto plan = testLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                   /*local_block_num=*/1,
                                   /*cache_key_count=*/1,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   kCompactSeqSizePerBlock,
                                   kBaseSeqSizePerBlock);

    EXPECT_EQ(plan, (std::vector<size_t>{0}));
}

TEST(DecodeRpcServerTest, UnscaledSwaGroupKeepsLogicalTailPositions) {
    // A COMPACT_LAST_RANK policy on a group whose block still covers one logical
    // block has a flat table: the tail slots are logical positions 3 and 4.
    const auto plan = testLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                   /*local_block_num=*/5,
                                   /*cache_key_count=*/5,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   kBaseSeqSizePerBlock,
                                   kBaseSeqSizePerBlock);

    EXPECT_EQ(plan, (std::vector<size_t>{3, 4}));
}

TEST(DecodeRpcServerTest, TailGroupReserveSlotsDoNotStarveTheLoad) {
    // The last sequence-backed slot is 2; a reserve slot 3 must not shift the
    // one-block tail window past the stored cache keys and drop the load.
    auto policy               = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    policy.active_tail_blocks = 1;
    ASSERT_EQ(policy.cp_mapping, CpBlockMappingMode::NONE);

    const auto plan = testLoadPlan(policy,
                                   /*local_block_num=*/4,
                                   /*cache_key_count=*/3,
                                   /*reuse_block_size=*/0,
                                   /*use_hybrid=*/true,
                                   kBaseSeqSizePerBlock,
                                   kBaseSeqSizePerBlock);

    EXPECT_EQ(plan, (std::vector<size_t>{2}));
}

TEST(DecodeRpcServerTest, FullGroupKeepsWholeLogicalBlocksAfterReuse) {
    // Decode owns whole logical blocks of a BLOCK_ROUND_ROBIN group: the plan must
    // not shard it (the per-peer split happens later, per block), and reused
    // blocks are skipped.
    auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    ASSERT_EQ(policy.cp_mapping, CpBlockMappingMode::BLOCK_ROUND_ROBIN);

    const auto plan = testLoadPlan(policy,
                                   /*local_block_num=*/5,
                                   /*cache_key_count=*/4,
                                   /*reuse_block_size=*/2,
                                   /*use_hybrid=*/false,
                                   kBaseSeqSizePerBlock,
                                   kBaseSeqSizePerBlock);

    EXPECT_EQ(plan, (std::vector<size_t>{2, 3}));
}

TEST(DecodeRpcServerTest, FullReaderRoutingIsIndependentOfProducerOwnership) {
    auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    const auto pages =
        selectProducerTransferPages(policy, /*logical_page_count=*/5, /*use_hybrid=*/false, /*compact_tail=*/false);
    EXPECT_EQ(pages, (std::vector<size_t>{0, 1, 2, 3, 4}));
    for (int peer = 0; peer < 4; ++peer) {
        EXPECT_EQ(transferPageReadFromPeer(policy, 4, false, 4, 4, peer), peer == 0);
    }

    // Reusing two B keys with B_g=2B starts from logical page one.
    const auto load = selectLoadTransferPages(policy,
                                              /*logical_page_count=*/3,
                                              /*max_destination_slots=*/3,
                                              /*first_page_after_reuse=*/1,
                                              /*use_hybrid=*/false,
                                              /*compact_tail=*/false,
                                              /*producer_cp_size=*/2,
                                              /*decode_peer_count=*/2,
                                              /*peer_index=*/0);
    EXPECT_EQ(load.logical_pages, (std::vector<size_t>{1, 2}));

    // An unsharded producer still routes one reader per page when D has one
    // peer for each producer rank. Unequal peer counts keep all pages.
    policy.cp_mapping = CpBlockMappingMode::NONE;
    EXPECT_TRUE(transferPageReadFromPeer(policy, 1, false, 2, 2, 1));
    EXPECT_FALSE(transferPageReadFromPeer(policy, 1, false, 2, 2, 0));
    EXPECT_TRUE(transferPageReadFromPeer(policy, 1, false, 2, 1, 0));
}

TEST(DecodeRpcServerTest, EmptyTableOrMissingCacheKeysYieldNoLoad) {
    const auto policy = makeCompactStatePolicy(/*active_tail_blocks=*/2);

    EXPECT_TRUE(testLoadPlan(policy,
                             /*local_block_num=*/0,
                             /*cache_key_count=*/11,
                             /*reuse_block_size=*/0,
                             /*use_hybrid=*/true,
                             kCompactSeqSizePerBlock,
                             kBaseSeqSizePerBlock)
                    .empty());
    EXPECT_TRUE(testLoadPlan(policy,
                             /*local_block_num=*/6,
                             /*cache_key_count=*/0,
                             /*reuse_block_size=*/0,
                             /*use_hybrid=*/true,
                             kCompactSeqSizePerBlock,
                             kBaseSeqSizePerBlock)
                    .empty());
}

}  // namespace rtp_llm
