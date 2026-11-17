#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/LinearStateCacheTransfer.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
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

class AllocationErrorContext final: public LoadAsyncContext {
public:
    AllocationErrorContext(const std::shared_ptr<LoadContextCoordinator>& coordinator, ErrorInfo error):
        LoadAsyncContext({}, {}, 0, 1, coordinator), error_(std::move(error)) {}

    ErrorInfo errorInfo() const override {
        return error_;
    }

private:
    ErrorInfo error_;
};

// Allocate and free real blocks; inject only the completed backend load result.
class AllocationErrorCacheManager final: public CoordinatorCacheManager {
public:
    AllocationErrorCacheManager(const CacheConfig& config, ErrorInfo error):
        CoordinatorCacheManager(config), error_(std::move(error)) {}

    size_t allocated_blocks{0};
    size_t released_blocks{0};

    void free(const FreeInfo& info) override {
        released_blocks += info.batch_kv_cache_resource->curBlocksNum();
        CoordinatorCacheManager::free(info);
    }

protected:
    MallocResult initMallocForCommonLen(const MallocInfo& info) override {
        auto result = CoordinatorCacheManager::initMallocForCommonLen(info);
        if (result.success) {
            auto coordinator = std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
            auto context     = std::make_shared<AllocationErrorContext>(coordinator, error_);
            RTP_LLM_CHECK(coordinator->registerContext(context));
            context->setSettlementReadyCallback([](const auto& ready) { EXPECT_TRUE(ready->settle(false)); });
            result.async_context  = std::move(context);
            result.load_attempted = true;
        }
        return result;
    }

    MallocResult incrMalloc(const MallocInfo& info) override {
        auto result      = CoordinatorCacheManager::incrMalloc(info);
        allocated_blocks = info.batch_kv_cache_resource->curBlocksNum();
        return result;
    }

private:
    ErrorInfo error_;
};

class AllocationErrorEngine final: public EngineBase {
public:
    static EngineInitParams parameters() {
        EngineInitParams params;
        params.model_config_.max_seq_len = 16;
        return params;
    }

    explicit AllocationErrorEngine(ErrorInfo error): EngineBase(parameters()) {
        const auto config = test::makeSimpleMhaCacheConfig(1, 8, 2, DataType::TYPE_FP16);
        cache             = std::make_shared<KVCacheManager>(config, false, nullptr);
        if (!cache->init()) {
            throw std::runtime_error("allocation test cache initialization failed");
        }
        allocator = std::make_shared<AllocationErrorCacheManager>(config, std::move(error));
        if (!allocator->init()) {
            throw std::runtime_error("allocation test allocator initialization failed");
        }
        cache->coordinator_manager_     = allocator;
        resource_context_.cache_manager = cache;
        resource_context_.role_type     = RoleType::DECODE;
    }

    GenerateStreamPtr makeStream(const std::shared_ptr<GenerateInput>& input) override {
        const auto params = parameters();
        stream            = std::make_shared<NormalGenerateStream>(
            input, params.model_config_, params.runtime_config, resource_context_, nullptr);
        return stream;
    }
    GenerateStreamPtr enqueue(const std::shared_ptr<GenerateInput>& input) override {
        ADD_FAILURE() << "failed allocation reached engine enqueue";
        return makeStream(input);
    }
    void enqueue(GenerateStreamPtr&) override {
        ADD_FAILURE() << "failed allocation reached engine enqueue";
    }
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in allocation test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

    std::shared_ptr<KVCacheManager>              cache;
    std::shared_ptr<AllocationErrorCacheManager> allocator;
    GenerateStreamPtr                            stream;
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

CacheConfig makeSplitLoadTestConfig(bool single = false) {
    CacheConfig config;
    config.layer_num                 = 1;
    config.seq_size_per_block        = 64;
    config.dtype                     = DataType::TYPE_BF16;
    config.use_opaque_kv_cache_store = true;
    ParallelismConfig parallel;
    parallel.tp_size = 2;
    SpecBuildContext context;
    context.dtype              = DataType::TYPE_BF16;
    context.seq_size_per_block = 64;
    context.parallelism_config = &parallel;
    std::vector<GroupBase> groups;
    for (const auto& tag : {std::string("linear"), std::string("linear_ssm"), std::string("linear_replay")}) {
        const bool      conv = tag == "linear";
        const bool      ssm  = tag == "linear_ssm";
        KVCacheSpecDesc desc;
        desc.tag        = tag;
        desc.cache_type = KVCacheSpecType::OpaqueState;
        desc.dtype = desc.entry_dtype = conv ? DataType::TYPE_BF16 : DataType::TYPE_FP32;
        desc.entry_elems              = ssm ? 32 : 16;
        desc.explicit_entry_count     = 1;
        desc.is_state_cache           = true;
        desc.group_type               = conv ? CacheGroupType::LINEAR : CacheGroupType::REQUEST_STATE;
        desc.request_state_slots      = conv ? 0 : (ssm && !single ? 2 : 1);
        desc.linear_state_layout =
            conv ? LinearStateLayout::DOUBLE_BUFFER_CONV :
                   (ssm ? (single ? LinearStateLayout::SINGLE_SSM : LinearStateLayout::DOUBLE_BUFFER_SSM) :
                          LinearStateLayout::DOUBLE_BUFFER_REPLAY);
        desc.reuse = CacheReusePolicyDesc{false};
        GroupBase group;
        group.tag       = tag;
        group.spec      = SpecBuilder::build(desc, context);
        group.policy    = SpecBuilder::groupPolicy(desc);
        group.block_num = 8;
        groups.push_back(std::move(group));
    }
    config.setTopology(std::move(groups), {{0, {"linear", "linear_ssm", "linear_replay"}}});
    return config;
}

struct HostSplitLoadCache {
    CacheConfig                              config;
    std::shared_ptr<KVCacheManager>          manager = std::make_shared<KVCacheManager>(config);
    std::shared_ptr<CoordinatorCacheManager> allocator =
        std::make_shared<CoordinatorCacheManager>(config, AllocationType::HOST);
    KVCacheResource               resource;
    std::vector<BlockIndicesType> allocated_rows;
    bool                          request_owns_blocks = true;

    explicit HostSplitLoadCache(bool single = false): config(makeSplitLoadTestConfig(single)) {
        if (!allocator->init()) {
            throw std::runtime_error("host split cache initialization failed");
        }
        manager->coordinator_manager_ = allocator;
        resource.initGroups(config.topologyPtr());
        resource.setCacheKeys({9001});
        for (size_t group_id = 0; group_id < config.topology().groups().size(); ++group_id) {
            const auto& group     = config.topology().groups()[group_id];
            const auto  count     = group.policy.request_state_slots > 0 ? group.policy.request_state_slots : 1;
            const auto& pool      = allocator->groupBlockPools()[group_id];
            auto        allocated = pool->malloc(count);
            if (!allocated) {
                throw std::runtime_error("host split cache allocation failed");
            }
            allocated_rows.push_back(*allocated);
            pool->incRef(*allocated);
            resource.mutableBlockIds(group.tag).assign(*allocated);
        }
    }

    ~HostSplitLoadCache() {
        releaseRequest();
    }

    void releaseRequest() {
        if (request_owns_blocks) {
            for (size_t group_id = 0; group_id < allocated_rows.size(); ++group_id) {
                allocator->groupBlockPools()[group_id]->decRef(allocated_rows[group_id]);
            }
            request_owns_blocks = false;
        }
    }

    size_t refCount(const std::string& tag, size_t bank = 0) const {
        const auto group = config.topology().groupIdForTag(tag);
        return allocator->groupBlockPools()[group]->refCount(allocated_rows[group][bank]);
    }

    bool allBlocksReleased() const {
        // refCount() requires an allocated block; the final decRef() removes
        // that allocation. Inspect every original ID through the locked API,
        // including both SSM banks and the separately released replay group.
        for (size_t group = 0; group < allocated_rows.size(); ++group) {
            const auto& pool = allocator->groupBlockPools()[group];
            for (const auto block : allocated_rows[group]) {
                if (pool->isAllocated(block)) {
                    return false;
                }
            }
        }
        return true;
    }
};

class SplitLoadTestEngine final: public EngineBase {
public:
    explicit SplitLoadTestEngine(std::shared_ptr<KVCacheManager> cache): EngineBase(EngineInitParams{}) {
        resource_context_.cache_manager = std::move(cache);
    }
    GenerateStreamPtr enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(GenerateStreamPtr&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in load test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
};

class DelayedLoadCacheStore final: public NormalCacheStore {
public:
    DelayedLoadCacheStore() {
        // The production base destructor stops this store even when only the
        // transport is replaced by the deterministic test implementation.
        request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(nullptr);
    }

    void load(const std::shared_ptr<RequestBlockBuffer>& buffer,
              CacheStoreLoadDoneCallback                 callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.emplace_back(buffer, std::move(callback));
        condition_.notify_all();
    }

    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& buffers,
                                             const std::string&                                      ip,
                                             uint32_t                                                port,
                                             uint32_t                                                rdma_port,
                                             int64_t                                                 timeout_ms,
                                             LoadContext::CheckCancelFunc                            cancelled,
                                             int                                                     partition_count,
                                             int partition_id) override {
        auto context = std::make_shared<LoadContext>(shared_from_this(), false);
        context->load(buffers, ip, port, rdma_port, timeout_ms, std::move(cancelled), partition_count, partition_id);
        return context;
    }

    bool waitPending(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(5), [&] { return pending_.size() >= count; });
    }

    void completeAll(bool success = true) {
        std::vector<std::pair<std::shared_ptr<RequestBlockBuffer>, CacheStoreLoadDoneCallback>> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending.swap(pending_);
        }
        for (const auto& entry : pending) {
            if (success) {
                for (const auto& block : entry.first->getBlocks()) {
                    EXPECT_FALSE(block.second->gpu_mem);
                    std::memset(block.second->addr.get(), 0x5a, block.second->len);
                }
            }
            entry.second(success, success ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadBufferTimeout);
        }
    }

private:
    std::mutex                                                                              mutex_;
    std::condition_variable                                                                 condition_;
    std::vector<std::pair<std::shared_ptr<RequestBlockBuffer>, CacheStoreLoadDoneCallback>> pending_;
};

class ControlledLoadRpcService final: public RpcService::Service {
public:
    grpc::Status RemoteLoad(grpc::ServerContext*,
                            const BroadcastLoadRequestPB* request,
                            BroadcastLoadResponsePB*      response) override {
        std::unique_lock<std::mutex> lock(mutex_);
        received_ = true;
        condition_.notify_all();
        condition_.wait(lock, [&] { return released_; });
        response->set_loaded_cache_block_count(request->cache_keys_size());
        response->set_done_time_us(currentTimeUs());
        response->mutable_error_info()->set_error_code(ErrorCodePB::NONE_ERROR);
        return status_;
    }
    bool waitReceived() {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(5), [&] { return received_; });
    }
    void release(grpc::Status status = grpc::Status::OK) {
        std::lock_guard<std::mutex> lock(mutex_);
        status_   = std::move(status);
        released_ = true;
        condition_.notify_all();
    }

private:
    std::mutex              mutex_;
    std::condition_variable condition_;
    bool                    received_ = false;
    bool                    released_ = false;
    grpc::Status            status_;
};

class CallbackLoadRpcService final: public RpcService::Service {
public:
    std::function<grpc::Status(grpc::ServerContext*)> callback;
    grpc::Status
    RemoteLoad(grpc::ServerContext* context, const BroadcastLoadRequestPB*, BroadcastLoadResponsePB*) override {
        return callback(context);
    }
};

class DecodeLoadRpcService final: public RpcService::Service {
public:
    explicit DecodeLoadRpcService(DecodeRpcServer& server): server_(server) {}
    grpc::Status RemoteLoad(grpc::ServerContext*          context,
                            const BroadcastLoadRequestPB* request,
                            BroadcastLoadResponsePB*      response) override {
        return server_.RemoteLoad(context, request, response);
    }

private:
    DecodeRpcServer& server_;
};

struct SplitLoadRpcPeers {
    ControlledLoadRpcService      services[2];
    std::unique_ptr<grpc::Server> servers[2];
    std::vector<std::string>      addresses;
    SplitLoadRpcPeers() {
        for (size_t rank = 0; rank < 2; ++rank) {
            grpc::ServerBuilder builder;
            int                 port = 0;
            builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
            builder.RegisterService(&services[rank]);
            servers[rank] = builder.BuildAndStart();
            if (!servers[rank] || port == 0) {
                throw std::runtime_error("split test RPC listener failed");
            }
            addresses.push_back("127.0.0.1:" + std::to_string(port));
        }
    }
    ~SplitLoadRpcPeers() {
        for (auto& service : services) {
            service.release();
        }
        for (auto& server : servers) {
            if (server) {
                server->Shutdown();
                server->Wait();
            }
        }
    }
};

bool waitForCondition(const std::function<bool()>& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return predicate();
}

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

TEST(DecodeRpcServerTest, SplitLoadLeaseSnapshotsBothBanksAndReplayAfterRequestRowsChange) {
    HostSplitLoadCache cache;
    auto context = makeLoadContext("snapshot", {"peer"}, cache.resource.cacheKeys(), cache.resource.groupBlockIds(), 1);
    auto lease   = DecodeRpcServer::acquireCacheLoadLease(cache.manager, context);
    ASSERT_NE(lease, nullptr);
    EXPECT_EQ(lease->blocks("linear_ssm"), cache.resource.blocks("linear_ssm"));
    EXPECT_EQ(lease->blocks("linear_replay"), cache.resource.blocks("linear_replay"));
    cache.resource.mutableBlockIds("linear_ssm").assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    cache.resource.mutableBlockIds("linear_replay").assign({NULL_BLOCK_IDX});
    cache.releaseRequest();
    EXPECT_EQ(cache.refCount("linear"), 1u);
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
    EXPECT_EQ(cache.refCount("linear_ssm", 1), 1u);
    EXPECT_EQ(cache.refCount("linear_replay"), 1u);
    lease.reset();
    EXPECT_TRUE(cache.allBlocksReleased());
}

TEST(DecodeRpcServerTest, SingleStateLoadLeasePinsSoleSsmAndReplayAfterRequestRelease) {
    HostSplitLoadCache cache(true);
    auto context = makeLoadContext("snapshot", {"peer"}, cache.resource.cacheKeys(), cache.resource.groupBlockIds(), 1);
    auto lease   = DecodeRpcServer::acquireCacheLoadLease(cache.manager, context);
    ASSERT_NE(lease, nullptr);
    EXPECT_EQ(lease->blocks("linear_ssm"), cache.resource.blocks("linear_ssm"));
    EXPECT_EQ(lease->blocks("linear_replay"), cache.resource.blocks("linear_replay"));
    cache.resource.mutableBlockIds("linear_ssm").assign({NULL_BLOCK_IDX});
    cache.resource.mutableBlockIds("linear_replay").assign({NULL_BLOCK_IDX});
    cache.releaseRequest();
    EXPECT_EQ(cache.refCount("linear"), 1u);
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
    EXPECT_EQ(cache.refCount("linear_replay"), 1u);
    lease.reset();
    EXPECT_TRUE(cache.allBlocksReleased());
}

TEST(DecodeRpcServerTest, LoadContextDrainsRealCallbacksAfterTimeoutAndCancellation) {
    for (bool cancel : {false, true}) {
        SCOPED_TRACE(cancel);
        auto                                store      = std::make_shared<DelayedLoadCacheStore>();
        auto                                context    = std::make_shared<LoadContext>(store, false);
        auto                                bytes      = std::make_shared<std::vector<uint8_t>>(8, 0);
        std::weak_ptr<std::vector<uint8_t>> weak_bytes = bytes;
        auto                                buffer     = std::make_shared<RequestBlockBuffer>("drain");
        buffer->addBlock("payload", std::shared_ptr<void>(bytes, bytes->data()), bytes->size(), false, true);
        context->load({buffer}, "peer", 1, 2, cancel ? 1000 : 0, [cancel] { return cancel; }, 1, 0);
        buffer.reset();
        bytes.reset();
        context->waitDone();
        EXPECT_EQ(context->getErrorInfo().code(),
                  cancel ? ErrorCode::CANCELLED : ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
        auto drained = std::async(std::launch::async, [context] { context->waitCallbacksDrained(); });
        EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
        EXPECT_FALSE(weak_bytes.expired());
        store->completeAll();
        EXPECT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        drained.get();
        ASSERT_NE(weak_bytes.lock(), nullptr);
        EXPECT_EQ(weak_bytes.lock()->front(), 0x5a);
        context.reset();
        EXPECT_TRUE(weak_bytes.expired());
    }
}

TEST(DecodeRpcServerTest, QuarantinePinsUniqueIdsWithoutRetainingAllocatorOrManagerOwners) {
    HostSplitLoadCache cache;
    auto               context =
        makeLoadContext("quarantine-owner", {"peer"}, cache.resource.cacheKeys(), cache.resource.groupBlockIds(), 1);
    auto lease = DecodeRpcServer::acquireCacheLoadLease(cache.manager, context);
    cache.releaseRequest();
    auto quarantine = [&] { cache.manager->quarantineCacheLoad(*lease, "quarantine-owner", "unknown completion"); };
    auto first      = std::async(std::launch::async, quarantine);
    auto duplicate  = std::async(std::launch::async, quarantine);
    first.get();
    duplicate.get();
    EXPECT_EQ(cache.allocator->quarantinedCacheLoadCount(), 1u);
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 2u);  // Original lease plus one independent quarantine reference.
    EXPECT_EQ(cache.refCount("linear_ssm", 1), 2u);
    lease.reset();
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
    EXPECT_EQ(cache.refCount("linear_ssm", 1), 1u);

    const auto                             group          = cache.config.topology().groupIdForTag("linear_ssm");
    auto                                   pool           = cache.allocator->groupBlockPools()[group];
    const auto                             banks          = cache.allocated_rows[group];
    std::weak_ptr<KVCacheManager>          weak_manager   = cache.manager;
    std::weak_ptr<CoordinatorCacheManager> weak_allocator = cache.allocator;
    cache.manager.reset();
    EXPECT_TRUE(weak_manager.expired());
    EXPECT_FALSE(weak_allocator.expired()) << "an external allocator owner remains";
    EXPECT_EQ(pool->refCount(banks[0]), 1u);
    cache.allocator.reset();
    EXPECT_TRUE(weak_allocator.expired()) << "plain quarantine snapshots must not create an owner cycle";
    EXPECT_EQ(pool->refCount(banks[0]), 1u);
    EXPECT_EQ(pool->refCount(banks[1]), 1u);
    auto reusable = pool->malloc(pool->freeBlocksNum());
    ASSERT_TRUE(reusable.has_value());
    for (auto block : *reusable) {
        EXPECT_NE(block, banks[0]);
        EXPECT_NE(block, banks[1]);
    }
    pool->incRef(*reusable);
    pool->decRef(*reusable);
    std::weak_ptr<DeviceBlockPool> weak_pool = pool;
    pool.reset();
    EXPECT_TRUE(weak_pool.expired());
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
    auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    EXPECT_EQ(DecodeRpcServer::cacheKeysPerPhysicalBlock(8, 4), 2u);
    EXPECT_EQ(DecodeRpcServer::keyBlocksPerLogicalBlock(policy, 8, 4), 2u);

    std::vector<size_t> transferred(5, 0);
    DecodeRpcServer::markCacheKeyRange(transferred, /*endpoint_key_index=*/1, /*block_offset_index=*/0, /*key_span=*/2);
    DecodeRpcServer::markCacheKeyRange(transferred, /*endpoint_key_index=*/3, /*block_offset_index=*/1, /*key_span=*/2);
    DecodeRpcServer::markCacheKeyRange(transferred, /*endpoint_key_index=*/4, /*block_offset_index=*/2, /*key_span=*/2);

    EXPECT_EQ(transferred, (std::vector<size_t>{1, 1, 1, 1, 1}));
}

TEST(DecodeRpcServerTest, CompactPhysicalBlockCoversEveryBaseKey) {
    auto policy = makeCompactStatePolicy(/*active_tail_blocks=*/2);
    EXPECT_EQ(DecodeRpcServer::cacheKeysPerPhysicalBlock(kCompactSeqSizePerBlock, kBaseSeqSizePerBlock), 2u);
    EXPECT_EQ(DecodeRpcServer::keyBlocksPerLogicalBlock(policy, kCompactSeqSizePerBlock, kBaseSeqSizePerBlock), 1u);

    std::vector<size_t> transferred(11, 0);
    DecodeRpcServer::markCacheKeyRange(transferred, /*endpoint_key_index=*/9, /*block_offset_index=*/4, /*key_span=*/2);
    DecodeRpcServer::markCacheKeyRange(
        transferred, /*endpoint_key_index=*/10, /*block_offset_index=*/5, /*key_span=*/2);

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

TEST_F(PrefillCompletionRpcTest, SplitRemoteLoadReportsRealCallbackResultAfterDeadline) {
    for (bool callback_success : {false, true}) {
        SCOPED_TRACE(callback_success);
        HostSplitLoadCache cache;
        DecodeRpcServer    server;
        server.engine_                                      = std::make_shared<SplitLoadTestEngine>(cache.manager);
        server.maga_init_params_.parallelism_config.tp_size = 2;
        server.maga_init_params_.model_config_.num_layers   = 1;
        server.model_id                                     = 9;
        server.resource_.workers                            = {"d0", "d1"};
        auto store                                          = std::make_shared<DelayedLoadCacheStore>();
        server.resource_.cache_store                        = store;
        auto context                                        = makeLoadContext("late-local",
                                                                              {"127.0.0.1:1:2", "127.0.0.1:3:4"},
                                       cache.resource.cacheKeys(),
                                       cache.resource.groupBlockIds(),
                                       1);
        context.timeout_ms                                  = 0;
        auto                    request = server.constructRemoteLoadRequest(context, 0, context.peer_addrs);
        BroadcastLoadResponsePB response;
        auto rpc = std::async(std::launch::async, [&] { return server.RemoteLoad(nullptr, &request, &response); });
        EXPECT_TRUE(store->waitPending(2));
        cache.releaseRequest();
        EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
        EXPECT_EQ(cache.refCount("linear_ssm", 1), 1u);
        EXPECT_EQ(cache.refCount("linear_replay"), 1u);
        EXPECT_EQ(rpc.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
        store->completeAll(callback_success);
        EXPECT_EQ(rpc.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_TRUE(rpc.get().ok());
        EXPECT_EQ(response.error_info().error_code(),
                  callback_success ? ErrorCodePB::NONE_ERROR :
                                     transErrorCodeToRPC(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT));
        EXPECT_EQ(response.loaded_cache_block_count(), callback_success ? 1 : 0);
        EXPECT_TRUE(cache.allBlocksReleased());
    }
}

TEST_F(PrefillCompletionRpcTest, SplitFanoutTimeoutKeepsBothBanksUntilAllRanksSucceedAfterServerDestruction) {
    HostSplitLoadCache cache;
    SplitLoadRpcPeers  peers;
    auto               server = std::make_unique<DecodeRpcServer>();
    server->engine_           = std::make_shared<SplitLoadTestEngine>(cache.manager);
    server->resource_.workers = server->resource_.grpc_workers = peers.addresses;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    DecodeGenerateContext        decode(rpc_context, 0, nullptr, reporter, nullptr);
    decode.peer_addrs = {"p0:1:2", "p1:1:2"};
    auto context      = makeLoadContext(
        "late-fanout", decode.peer_addrs, cache.resource.cacheKeys(), cache.resource.groupBlockIds(), 1);
    context.timeout_ms = 1;
    const auto start   = std::chrono::steady_clock::now();
    const auto result  = server->loadCacheAsyncForTpDrained(decode, context);
    EXPECT_EQ(result.error_info.code(), ErrorCode::LOAD_CACHE_TIMEOUT);
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
    EXPECT_TRUE(peers.services[0].waitReceived());
    EXPECT_TRUE(peers.services[1].waitReceived());
    cache.releaseRequest();
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
    EXPECT_EQ(cache.refCount("linear_ssm", 1), 1u);
    EXPECT_FALSE(cache.manager->hasQuarantinedCacheLoads());
    const auto                             ssm_group      = cache.config.topology().groupIdForTag("linear_ssm");
    auto                                   pool           = cache.allocator->groupBlockPools()[ssm_group];
    const auto                             banks          = cache.allocated_rows[ssm_group];
    std::weak_ptr<CoordinatorCacheManager> weak_allocator = cache.allocator;
    server.reset();
    cache.manager.reset();
    cache.allocator.reset();
    EXPECT_FALSE(weak_allocator.expired()) << "background fanout must own the allocator, not the destroyed server";
    peers.services[0].release();
    EXPECT_EQ(pool->refCount(banks[0]), 1u);
    EXPECT_EQ(pool->refCount(banks[1]), 1u);
    peers.services[1].release();
    EXPECT_TRUE(waitForCondition([&] { return weak_allocator.expired(); }));
    EXPECT_FALSE(pool->isAllocated(banks[0]));
    EXPECT_FALSE(pool->isAllocated(banks[1]));
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
    // Reusable banks after late success also prove the allocator was not quarantined.
}

TEST_F(PrefillCompletionRpcTest, SingleStateFanoutTimeoutPinsSoleSsmUntilAllRankCallbacksDrain) {
    HostSplitLoadCache cache(true);
    SplitLoadRpcPeers  peers;
    auto               server = std::make_unique<DecodeRpcServer>();
    server->engine_           = std::make_shared<SplitLoadTestEngine>(cache.manager);
    server->resource_.workers = server->resource_.grpc_workers = peers.addresses;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    DecodeGenerateContext        decode(rpc_context, 0, nullptr, reporter, nullptr);
    decode.peer_addrs = {"p0:1:2", "p1:1:2"};
    auto context      = makeLoadContext(
        "late-fanout", decode.peer_addrs, cache.resource.cacheKeys(), cache.resource.groupBlockIds(), 1);
    context.timeout_ms = 1;
    const auto start   = std::chrono::steady_clock::now();
    const auto result  = server->loadCacheAsyncForTpDrained(decode, context);
    EXPECT_EQ(result.error_info.code(), ErrorCode::LOAD_CACHE_TIMEOUT);
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
    EXPECT_TRUE(peers.services[0].waitReceived());
    EXPECT_TRUE(peers.services[1].waitReceived());
    cache.releaseRequest();
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
    EXPECT_FALSE(cache.manager->hasQuarantinedCacheLoads());
    const auto                             ssm_group      = cache.config.topology().groupIdForTag("linear_ssm");
    auto                                   pool           = cache.allocator->groupBlockPools()[ssm_group];
    const auto                             banks          = cache.allocated_rows[ssm_group];
    std::weak_ptr<CoordinatorCacheManager> weak_allocator = cache.allocator;
    server.reset();
    cache.manager.reset();
    cache.allocator.reset();
    EXPECT_FALSE(weak_allocator.expired()) << "background fanout must own the allocator, not the destroyed server";
    peers.services[0].release();
    EXPECT_EQ(pool->refCount(banks[0]), 1u);
    peers.services[1].release();
    EXPECT_TRUE(waitForCondition([&] { return weak_allocator.expired(); }));
    EXPECT_FALSE(pool->isAllocated(banks[0]));
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
    // Reusable banks after late success also prove the allocator was not quarantined.
}

TEST_F(PrefillCompletionRpcTest, SplitFanoutAfterFrontendTimeoutUsesActualLateCallbackResult) {
    for (bool callback_success : {false, true}) {
        SCOPED_TRACE(callback_success);
        HostSplitLoadCache cache;
        SplitLoadRpcPeers  controlled_peers;
        DecodeRpcServer    server;
        server.engine_                                      = std::make_shared<SplitLoadTestEngine>(cache.manager);
        server.maga_init_params_.parallelism_config.tp_size = 2;
        server.maga_init_params_.model_config_.num_layers   = 1;
        auto store                                          = std::make_shared<DelayedLoadCacheStore>();
        server.resource_.cache_store                        = store;
        DecodeLoadRpcService local_service(server);
        grpc::ServerBuilder  builder;
        int                  port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&local_service);
        auto local_server = builder.BuildAndStart();
        ASSERT_NE(local_server, nullptr);
        server.resource_.workers =
            server.resource_.grpc_workers = {"127.0.0.1:" + std::to_string(port), controlled_peers.addresses[1]};

        DecodeRpcContext             rpc_context{nullptr};
        kmonitor::MetricsReporterPtr reporter;
        DecodeGenerateContext        decode(rpc_context, 0, nullptr, reporter, nullptr);
        decode.peer_addrs = {"127.0.0.1:1:2", "127.0.0.1:3:4"};
        auto context      = makeLoadContext(
            "real-late-fanout", decode.peer_addrs, cache.resource.cacheKeys(), cache.resource.groupBlockIds(), 1);
        context.timeout_ms = 1;
        const auto result  = server.loadCacheAsyncForTpDrained(decode, context);
        EXPECT_EQ(result.error_info.code(), ErrorCode::LOAD_CACHE_TIMEOUT);
        EXPECT_TRUE(store->waitPending(2));
        EXPECT_TRUE(controlled_peers.services[1].waitReceived());
        cache.releaseRequest();
        EXPECT_EQ(cache.refCount("linear_ssm", 0), 2u) << "local callback and fanout each retain a real lease";
        EXPECT_EQ(cache.refCount("linear_ssm", 1), 2u);
        EXPECT_FALSE(cache.manager->hasQuarantinedCacheLoads());
        controlled_peers.services[1].release();
        store->completeAll(callback_success);
        if (callback_success) {
            EXPECT_TRUE(waitForCondition([&] { return cache.allBlocksReleased(); }));
            EXPECT_FALSE(cache.manager->hasQuarantinedCacheLoads());
        } else {
            EXPECT_TRUE(waitForCondition([&] {
                return cache.manager->hasQuarantinedCacheLoads() && cache.refCount("linear", 0) == 1
                       && cache.refCount("linear_ssm", 0) == 1 && cache.refCount("linear_ssm", 1) == 1
                       && cache.refCount("linear_replay", 0) == 1;
            }));
            EXPECT_FALSE(cache.allBlocksReleased());
        }
        local_server->Shutdown();
        local_server->Wait();
    }
}

TEST_F(PrefillCompletionRpcTest, SplitFanoutTransportFailureQuarantinesAndRejectsAnotherLoad) {
    HostSplitLoadCache cache;
    SplitLoadRpcPeers  peers;
    auto               server = std::make_unique<DecodeRpcServer>();
    server->engine_           = std::make_shared<SplitLoadTestEngine>(cache.manager);
    server->resource_.workers = server->resource_.grpc_workers = peers.addresses;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    DecodeGenerateContext        decode(rpc_context, 0, nullptr, reporter, nullptr);
    decode.peer_addrs = {"p0:1:2", "p1:1:2"};
    auto context      = makeLoadContext(
        "failed-fanout", decode.peer_addrs, cache.resource.cacheKeys(), cache.resource.groupBlockIds(), 1);
    context.timeout_ms = 1;
    peers.services[0].release(grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected lost completion"));
    server->loadCacheAsyncForTpDrained(decode, context);
    cache.releaseRequest();
    EXPECT_TRUE(waitForCondition([&] { return cache.manager->hasQuarantinedCacheLoads(); }));
    const auto rejected = server->loadCacheAsyncForTpDrained(decode, context);
    EXPECT_EQ(rejected.error_info.code(), ErrorCode::LOAD_KV_CACHE_FAILED);
    EXPECT_NE(rejected.error_info.ToString().find("restart required"), std::string::npos);
    EXPECT_GE(cache.refCount("linear_ssm", 0), 1u);
    EXPECT_GE(cache.refCount("linear_ssm", 1), 1u);
    peers.services[1].release();
    // The successful rank cannot resolve ownership of the failed rank.
    EXPECT_GE(cache.refCount("linear_ssm", 0), 1u);
    server.reset();
    EXPECT_TRUE(waitForCondition([&] { return cache.refCount("linear_ssm", 0) == 1; }));
    EXPECT_EQ(cache.refCount("linear_ssm", 1), 1u);
    // Destroying the RPC server does not recover an externally owned allocator.
    DecodeRpcServer replacement;
    replacement.engine_           = std::make_shared<SplitLoadTestEngine>(cache.manager);
    replacement.resource_.workers = replacement.resource_.grpc_workers = peers.addresses;
    const auto replacement_result = replacement.loadCacheAsyncForTpDrained(decode, context);
    EXPECT_EQ(replacement_result.error_info.code(), ErrorCode::LOAD_KV_CACHE_FAILED);
    EXPECT_NE(replacement_result.error_info.ToString().find("restart required"), std::string::npos);
    auto empty_resource = std::make_shared<BatchKVCacheResource>();
    empty_resource->resetBatchSize(1);
    empty_resource->initGroups(cache.config.topologyPtr());
    empty_resource->setBatchCacheKeys(0, {1234});
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::arange(64, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    auto tokens            = std::make_shared<CompleteTokenIds>(1, 1, 128, 64);
    tokens->init(input);
    MallocInfo allocation{empty_resource, tokens};
    allocation.reuse_cache = allocation.enable_cache_lookup = false;
    const auto admission                                    = cache.allocator->malloc(allocation);
    EXPECT_FALSE(admission.success);
    EXPECT_EQ(admission.status, MallocStatus::PERMANENT_RESOURCE_EXHAUSTED);
    EXPECT_EQ(cache.allocator->quarantinedCacheLoadCount(), 1u);
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
}

TEST_F(PrefillCompletionRpcTest, SplitFanoutFrontendCancellationDrainsRpcWithoutQuarantine) {
    HostSplitLoadCache cache;
    SplitLoadRpcPeers  peers;
    DecodeRpcServer    server;
    server.engine_           = std::make_shared<SplitLoadTestEngine>(cache.manager);
    server.resource_.workers = server.resource_.grpc_workers = peers.addresses;
    std::promise<ErrorCode> frontend_result;
    auto                    frontend_done = frontend_result.get_future();
    CallbackLoadRpcService  frontend;
    frontend.callback = [&](grpc::ServerContext* server_context) {
        DecodeRpcContext             rpc_context{nullptr};
        kmonitor::MetricsReporterPtr reporter;
        DecodeGenerateContext        decode(rpc_context, 0, server_context, reporter, nullptr);
        decode.peer_addrs = {"p0:1:2", "p1:1:2"};
        auto context      = makeLoadContext(
            "cancelled-fanout", decode.peer_addrs, cache.resource.cacheKeys(), cache.resource.groupBlockIds(), 1);
        context.server_context = server_context;
        context.timeout_ms     = 5000;
        const auto result      = server.loadCacheAsyncForTpDrained(decode, context);
        frontend_result.set_value(result.error_info.code());
        return grpc::Status::OK;
    };
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&frontend);
    auto frontend_server = builder.BuildAndStart();
    ASSERT_NE(frontend_server, nullptr);
    auto stub = RpcService::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
    grpc::ClientContext     client_context;
    BroadcastLoadRequestPB  request;
    BroadcastLoadResponsePB response;
    auto client = std::async(std::launch::async, [&] { return stub->RemoteLoad(&client_context, request, &response); });
    EXPECT_TRUE(peers.services[0].waitReceived());
    EXPECT_TRUE(peers.services[1].waitReceived());
    client_context.TryCancel();
    EXPECT_EQ(client.get().error_code(), grpc::StatusCode::CANCELLED);
    const auto completed = frontend_done.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(completed, std::future_status::ready);
    if (completed == std::future_status::ready) {
        EXPECT_EQ(frontend_done.get(), ErrorCode::CANCELLED);
    }
    cache.releaseRequest();
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
    EXPECT_EQ(cache.refCount("linear_ssm", 1), 1u);
    peers.services[0].release();
    EXPECT_EQ(cache.refCount("linear_ssm", 0), 1u);
    peers.services[1].release();
    EXPECT_TRUE(waitForCondition([&] { return cache.allBlocksReleased(); }));
    EXPECT_FALSE(cache.manager->hasQuarantinedCacheLoads());
    frontend_server->Shutdown();
    frontend_server->Wait();
}

TEST_F(PrefillCompletionRpcTest, AllocationLoadTimeoutPreservesDeadlineAndReleasesResources) {
    for (bool timeout : {false, true}) {
        SCOPED_TRACE(timeout);
        const auto backend_code = timeout ? ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION;
        auto engine = std::make_shared<AllocationErrorEngine>(ErrorInfo(backend_code, "injected backend load failure"));
        const auto      initial_free_blocks = engine->cache->freeBlocksNum();
        DecodeRpcServer server;
        server.engine_ = engine;
        DecodeRpcContext             rpc_context{nullptr};
        kmonitor::MetricsReporterPtr reporter;
        auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
        DecodeGenerateContext        context(rpc_context, 0, nullptr, reporter, meta);
        context.request_id  = 45;
        context.request_key = "allocation-load-failure";
        auto* input         = context.allocate_request.mutable_input();
        input->set_request_id(context.request_id);
        input->add_token_ids(1);
        input->add_token_ids(2);
        input->add_token_ids(3);
        input->mutable_generate_config()->set_max_new_tokens(1);
        input->mutable_generate_config()->set_num_beams(1);
        input->mutable_generate_config()->set_num_return_sequences(1);

        server.allocateResource(context);

        EXPECT_EQ(context.error_status.error_code(),
                  timeout ? grpc::StatusCode::DEADLINE_EXCEEDED : grpc::StatusCode::INTERNAL);
        EXPECT_EQ(context.error_info.code(), timeout ? ErrorCode::LOAD_CACHE_TIMEOUT : ErrorCode::MALLOC_FAILED);
        EXPECT_NE(context.error_status.error_message().find("injected backend load failure"), std::string::npos);
        ASSERT_NE(engine->stream, nullptr);
        EXPECT_TRUE(engine->stream->isFinished());
        EXPECT_TRUE(engine->stream->streamCacheResource().isResourceReleased());
        EXPECT_EQ(engine->stream->streamCacheResource().curBlocksNum(), 0);
        EXPECT_GT(engine->allocator->allocated_blocks, 0u);
        EXPECT_EQ(engine->allocator->released_blocks, engine->allocator->allocated_blocks);
        EXPECT_EQ(engine->cache->freeBlocksNum(), initial_free_blocks);
        EXPECT_EQ(context.retry_times, 1);
        EXPECT_FALSE(engine->stream->hasEvent(StreamEvents::LoadInitiated));
        context.markRpcHandlingCompleted();
    }
}

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
            const auto timing = stream->getTimeInfo();
            executed          = timing.running_started || timing.first_token_committed || timing.generation_done;
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
    const auto policy        = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto producer_plan = buildCacheStorePlan(policy,
                                                   /*total_logical_blocks=*/3,
                                                   /*reuse_block_size=*/0,
                                                   /*use_hybrid=*/true,
                                                   /*cp_rank=*/0,
                                                   /*cp_size=*/1,
                                                   /*key_blocks_per_logical_block=*/2,
                                                   /*cache_key_count=*/6);
    const auto decode_plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                                 /*local_block_num=*/3,
                                                                 /*cache_key_count=*/6,
                                                                 /*reuse_block_size=*/0,
                                                                 /*use_hybrid=*/true,
                                                                 /*group_seq_size_per_block=*/2,
                                                                 /*base_seq_size_per_block=*/1);

    EXPECT_EQ(keyOffsetPairs(producer_plan), (KeyOffsetPairs{{1, 0}, {3, 1}, {5, 2}}));
    EXPECT_EQ(keyOffsetPairs(decode_plan), keyOffsetPairs(producer_plan));
}

TEST(DecodeRpcServerTest, HeterogeneousPartialBlockUsesLastAvailableKey) {
    const auto policy        = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto producer_plan = buildCacheStorePlan(policy,
                                                   /*total_logical_blocks=*/3,
                                                   /*reuse_block_size=*/0,
                                                   /*use_hybrid=*/true,
                                                   /*cp_rank=*/0,
                                                   /*cp_size=*/1,
                                                   /*key_blocks_per_logical_block=*/2,
                                                   /*cache_key_count=*/5);
    const auto decode_plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                                 /*local_block_num=*/3,
                                                                 /*cache_key_count=*/5,
                                                                 /*reuse_block_size=*/0,
                                                                 /*use_hybrid=*/true,
                                                                 /*group_seq_size_per_block=*/2,
                                                                 /*base_seq_size_per_block=*/1);

    EXPECT_EQ(keyOffsetPairs(producer_plan), (KeyOffsetPairs{{1, 0}, {3, 1}, {4, 2}}));
    EXPECT_EQ(keyOffsetPairs(decode_plan), keyOffsetPairs(producer_plan));
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

TEST(DecodeRpcServerTest, DoubleBufferedSsmLoadsLastPromptKeyIntoCommittedBank) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::REQUEST_STATE);
    policy.linear_state_layout = LinearStateLayout::DOUBLE_BUFFER_SSM;
    policy.request_state_slots = 2;
    for (const size_t prompt_pages : {1u, 127u, 128u, 129u}) {
        for (const size_t reuse_pages : {size_t{0}, prompt_pages / 2, prompt_pages}) {
            const auto plan = DecodeRpcServer::buildGroupLoadPlan(policy, 2, prompt_pages, reuse_pages, true, 64, 64);
            ASSERT_EQ(plan.size(), 1u);
            EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{static_cast<int>(prompt_pages - 1), 0}}));
            const auto producer = buildCacheStorePlan(
                defaultCacheGroupPolicy(CacheGroupType::LINEAR), prompt_pages, 0, true, 0, 1, 1, prompt_pages);
            ASSERT_EQ(producer.size(), 1u);
            EXPECT_EQ(plan[0].key_index, producer[0].key_index);
            EXPECT_EQ(plan[0].offset_index, 0);  // Never the scratch bank or prompt-page index.
        }
    }
}

TEST(DecodeRpcServerTest, SingleSsmLoadsLastPromptKeyIntoSoleAllocatorRow) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::REQUEST_STATE);
    policy.linear_state_layout = LinearStateLayout::SINGLE_SSM;
    policy.request_state_slots = 1;
    for (const size_t prompt_pages : {1u, 127u, 128u, 129u}) {
        for (const size_t reuse_pages : {size_t{0}, prompt_pages / 2, prompt_pages}) {
            const auto plan = DecodeRpcServer::buildGroupLoadPlan(policy, 1, prompt_pages, reuse_pages, true, 64, 64);
            ASSERT_EQ(plan.size(), 1u);
            EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{static_cast<int>(prompt_pages - 1), 0}}));
            const auto producer = buildCacheStorePlan(
                defaultCacheGroupPolicy(CacheGroupType::LINEAR), prompt_pages, 0, true, 0, 1, 1, prompt_pages);
            ASSERT_EQ(producer.size(), 1u);
            EXPECT_EQ(plan[0].key_index, producer[0].key_index);
            EXPECT_EQ(plan[0].offset_index, 0);  // Never the scratch bank or prompt-page index.
        }
    }
}

TEST(DecodeRpcServerTest, DoubleBufferedConvKeepsLogicalTailAndReplayHasNoTransfer) {
    auto conv                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    conv.linear_state_layout = LinearStateLayout::DOUBLE_BUFFER_CONV;
    const auto plan          = DecodeRpcServer::buildGroupLoadPlan(conv, 131, 128, 96, true, 64, 64);
    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{127, 127}}));
    EXPECT_THROW(DecodeRpcServer::buildGroupLoadPlan(conv, 2, 128, 96, true, 64, 64), std::invalid_argument);
    auto replay                = defaultCacheGroupPolicy(CacheGroupType::REQUEST_STATE);
    replay.linear_state_layout = LinearStateLayout::DOUBLE_BUFFER_REPLAY;
    replay.request_state_slots = 1;
    EXPECT_TRUE(DecodeRpcServer::buildGroupLoadPlan(replay, 1, 128, 96, true, 64, 64).empty());
    EXPECT_THROW(linearStateCacheWireTag(replay.linear_state_layout, "linear_replay"), std::invalid_argument);
}

TEST(DecodeRpcServerTest, SingleSsmRejectsExtraSlotAndInvalidTransferDestination) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::REQUEST_STATE);
    policy.linear_state_layout = LinearStateLayout::SINGLE_SSM;
    policy.request_state_slots = 1;
    EXPECT_EQ(linearStateCacheWireTag(policy.linear_state_layout, "linear_ssm"), "linear.ssm");
    for (const size_t count : {0u, 2u, 3u}) {
        EXPECT_THROW(DecodeRpcServer::buildGroupLoadPlan(policy, count, 128, 96, true, 64, 64), std::invalid_argument);
    }
    policy.request_state_slots = 2;
    EXPECT_THROW(DecodeRpcServer::buildGroupLoadPlan(policy, 1, 128, 0, true, 64, 64), std::invalid_argument);
    EXPECT_THROW(validateLinearStateDestinationBlock(LinearStateLayout::SINGLE_SSM, -1), std::invalid_argument);
    EXPECT_THROW(validateLinearStateDestinationBlock(LinearStateLayout::SINGLE_SSM, -2), std::invalid_argument);
}

TEST(DecodeRpcServerTest, DoubleBufferedSsmRejectsMissingBanksAndNullDestinations) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::REQUEST_STATE);
    policy.linear_state_layout = LinearStateLayout::DOUBLE_BUFFER_SSM;
    policy.request_state_slots = 2;
    for (const size_t bank_count : {0u, 1u, 3u}) {
        EXPECT_THROW(DecodeRpcServer::buildGroupLoadPlan(policy, bank_count, 128, 0, true, 64, 64),
                     std::invalid_argument);
    }
    policy.request_state_slots = 1;
    EXPECT_THROW(DecodeRpcServer::buildGroupLoadPlan(policy, 2, 128, 0, true, 64, 64), std::invalid_argument);
    policy.request_state_slots = 2;
    EXPECT_THROW(DecodeRpcServer::buildGroupLoadPlan(policy, 2, 128, 0, true, 128, 64), std::invalid_argument);
    EXPECT_TRUE(DecodeRpcServer::buildGroupLoadPlan(policy, 2, 0, 0, true, 64, 64).empty());
    for (const auto layout : {LinearStateLayout::DOUBLE_BUFFER_SSM, LinearStateLayout::DOUBLE_BUFFER_CONV}) {
        EXPECT_THROW(validateLinearStateDestinationBlock(layout, NULL_BLOCK_IDX), std::invalid_argument);
        EXPECT_THROW(validateLinearStateDestinationBlock(layout, -2), std::invalid_argument);
        EXPECT_NO_THROW(validateLinearStateDestinationBlock(layout, 7));
    }
    EXPECT_NO_THROW(validateLinearStateDestinationBlock(LinearStateLayout::LEGACY, NULL_BLOCK_IDX));
}

TEST(DecodeRpcServerTest, DoubleBufferedLinearTransferRejectsHeterogeneousTpAndCp) {
    EXPECT_NO_THROW(validateSplitLinearGlobalTopology(2, 2, 2, 1));
    EXPECT_NO_THROW(validateSplitLinearTransfer(2, 1, 1, 1, 0));
    EXPECT_THROW(validateSplitLinearGlobalTopology(2, 2, 4, 1), std::invalid_argument);
    EXPECT_THROW(validateSplitLinearGlobalTopology(4, 4, 2, 1), std::invalid_argument);
    EXPECT_THROW(validateSplitLinearGlobalTopology(2, 2, 2, 2), std::invalid_argument);
    EXPECT_THROW(validateSplitLinearTransfer(2, 2, 1, 1, 0), std::invalid_argument);
    EXPECT_THROW(validateSplitLinearTransfer(2, 1, 1, 2, 0), std::invalid_argument);
    EXPECT_THROW(validateSplitLinearTransfer(2, 1, 2, 1, 0), std::invalid_argument);
    EXPECT_THROW(validateSplitLinearTransfer(1, 1, 1, 1, 0), std::invalid_argument);
    EXPECT_THROW(validateSplitLinearTransfer(2, 1, 1, 1, 1), std::invalid_argument);
}

TEST(DecodeRpcServerTest, LinearStateWireSlicesPreservePayloadAndRejectOverflow) {
    const auto slices = linearStateSplitExportSlices(3 * 192, 192, 2, 128, 64);
    EXPECT_EQ(slices[0].tag, "linear.ssm");
    EXPECT_EQ(slices[0].slice.offset, 2 * 192u);
    EXPECT_EQ(slices[0].slice.bytes, 128u);
    EXPECT_EQ(slices[1].tag, "linear.conv");
    EXPECT_EQ(slices[1].slice.offset, 2 * 192u + 128u);
    EXPECT_EQ(slices[1].slice.bytes, 64u);
    EXPECT_EQ(linearStateCacheWireTag(LinearStateLayout::DOUBLE_BUFFER_SSM, "linear_ssm"), slices[0].tag);
    EXPECT_EQ(linearStateCacheWireTag(LinearStateLayout::DOUBLE_BUFFER_CONV, "linear"), slices[1].tag);
    EXPECT_EQ(linearStateCacheWireTag(LinearStateLayout::LEGACY, "linear"), "linear");
    EXPECT_THROW(linearStateCacheWireTag(LinearStateLayout::DOUBLE_BUFFER_SSM, "linear"), std::invalid_argument);
    EXPECT_THROW(linearStateSplitExportSlices(3 * 192, 192, -1, 128, 64), std::invalid_argument);
    EXPECT_THROW(linearStateSplitExportSlices(3 * 192, 192, 3, 128, 64), std::out_of_range);
    EXPECT_THROW(linearStateSplitExportSlices(3 * 192 - 1, 192, 2, 128, 64), std::out_of_range);
    EXPECT_THROW(linearStateSplitExportSlices(3 * 192, 192, 2, 193, 64), std::invalid_argument);
    EXPECT_THROW(linearStateSplitExportSlices(3 * 192, 192, 2, 128, 65), std::invalid_argument);
    EXPECT_THROW(linearStateSplitExportSlices(3 * 192, 192, 2, 0, 64), std::invalid_argument);
    EXPECT_THROW(
        checkedLinearStateCacheSlice(std::numeric_limits<size_t>::max(), 16, std::numeric_limits<int64_t>::max(), 0, 8),
        std::overflow_error);
    EXPECT_THROW(checkedLinearStateCacheSlice(std::numeric_limits<size_t>::max(),
                                              std::numeric_limits<size_t>::max(),
                                              0,
                                              0,
                                              static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1),
                 std::invalid_argument);
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
