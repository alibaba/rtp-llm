
#include <set>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockCoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <limits>
#include <numeric>
#include <thread>

using namespace std;

namespace rtp_llm {

class ImmediateAllocatorContext: public AsyncContext {
public:
    explicit ImmediateAllocatorContext(bool success, bool done = true): success_(success), done_(done) {}

    void waitDone() override {
        done_ = true;
    }

    void onDone(DoneCallback callback) override {
        if (done_ && callback) {
            callback(success_ ? ErrorInfo::OkStatus() : ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "allocator failed"));
        }
    }

    bool done() const override {
        return done_;
    }

    bool success() const override {
        return success_;
    }

    void setDone(bool done) {
        done_ = done;
    }

private:
    bool success_{false};
    bool done_{true};
};

class DestructionObserverContext: public AsyncContext {
public:
    explicit DestructionObserverContext(std::function<void()> observer): observer_(std::move(observer)) {}

    ~DestructionObserverContext() override {
        observer_();
    }

    void waitDone() override {}
    void onDone(DoneCallback) override {}
    bool done() const override {
        return false;
    }
    bool success() const override {
        return false;
    }

private:
    std::function<void()> observer_;
};

std::shared_ptr<LoadAsyncContext>
makeAllocatorLoadContext(size_t matched_blocks, const std::vector<Tier>& source_tiers, bool commit = true) {
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
    std::vector<TransferDescriptor> descriptors;
    descriptors.reserve(source_tiers.size());
    for (size_t path_index = 0; path_index < source_tiers.size(); ++path_index) {
        descriptors.emplace_back(
            nullptr, /*group_set_id=*/0, path_index, source_tiers[path_index], Tier::DEVICE, BlockIndicesType{1});
    }
    auto context =
        coordinator->create(std::move(descriptors), std::vector<bool>(source_tiers.size(), false), matched_blocks);
    EXPECT_TRUE(coordinator->registerContext(context));
    if (commit) {
        EXPECT_TRUE(context->commit());
    }
    return context;
}

class StreamReadStorageBackend: public StorageBackend {
public:
    ~StreamReadStorageBackend() override {
        shutdown();
    }

    void blockMatches() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_matches_ = false;
    }

    void releaseMatches() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_matches_ = true;
        cv_.notify_all();
    }

    void blockReads() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_reads_ = false;
    }

    void releaseReads() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_reads_ = true;
        cv_.notify_all();
    }

    void waitForMatches(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return match_calls_ >= count; });
    }

    void waitForReads(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return read_calls_ >= count; });
    }

    size_t writeCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return write_calls_;
    }
    size_t writtenKeys() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return written_keys_;
    }

    size_t matchCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return match_calls_;
    }

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        std::unique_lock<std::mutex> lock(mutex_);
        ++match_calls_;
        cv_.notify_all();
        cv_.wait(lock, [&] { return release_matches_; });
        return {request.handles.size(), nullptr};
    }

    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {
        std::unique_lock<std::mutex> lock(mutex_);
        ++read_calls_;
        cv_.notify_all();
        cv_.wait(lock, [&] { return release_reads_; });
    }

    void writeImpl(const StorageRequest& request) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++write_calls_;
        written_keys_ += request.handles.size();
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    release_matches_{true};
    bool                    release_reads_{true};
    size_t                  match_calls_{0};
    size_t                  read_calls_{0};
    size_t                  write_calls_{0};
    size_t                  written_keys_{0};
};

class StreamCacheResourceTest: public DeviceTestBase {
protected:
    StreamCacheResourceTest(): perf_scope("PERF_TEST", "1") {}

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(/*layer_num=*/3,
                                              /*block_num=*/9,
                                              /*tokens_per_block=*/2,
                                              rtp_llm::DataType::TYPE_INT8);
    }

    CacheConfig init_cp_config(int cp_size = 2, CpBlockMappingMode mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN) {
        ModelConfig model;
        model.num_layers                   = 3;
        model.attn_config.head_num         = cp_size;
        model.attn_config.kv_head_num      = cp_size;
        model.attn_config.size_per_head    = 1;
        model.attn_config.tokens_per_block = 2;
        KVCacheSpecDesc desc{"default", KVCacheSpecType::MultiHeadAttention};
        desc.dtype       = DataType::TYPE_INT8;
        desc.cp          = CacheCpPolicyDesc{};
        desc.cp->mapping = mapping;
        model.kv_cache_spec_descs.assign(3, {desc});
        ParallelismConfig parallelism;
        parallelism.tp_size                            = cp_size;
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        parallelism.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
        auto config = CacheConfigCreator::createConfig(model, parallelism, KVCacheConfig{});
        config.finalizeBlockNums(9, RuntimeConfig{});
        return config;
    }

    void prepareResource(bool reuse_cache = false, RoleType role_type = RoleType::PDFUSION) {
        prepareResourceWithInputTokens(/*input_tokens=*/{1, 2, 3, 4, 5, 6}, reuse_cache, role_type);
    }

    void prepareHybridResource(bool reuse_cache = false, RoleType role_type = RoleType::PDFUSION) {
        prepareHybridResourceWithInputTokens(/*input_tokens=*/{1, 2, 3, 4, 5, 6}, reuse_cache, role_type);
    }

    void prepareResourceWithInputTokens(const std::vector<int>& input_tokens,
                                        bool                    reuse_cache = false,
                                        RoleType                role_type   = RoleType::PDFUSION) {
        prepareResourceWithCacheConfig(init_config(), input_tokens, reuse_cache, role_type);
    }

    void prepareHybridResourceWithInputTokens(const std::vector<int>& input_tokens,
                                              bool                    reuse_cache = false,
                                              RoleType                role_type   = RoleType::PDFUSION) {
        prepareResourceWithCacheConfig(test::makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                                            /*block_num=*/9,
                                                                            /*tokens_per_block=*/2,
                                                                            rtp_llm::DataType::TYPE_FP16,
                                                                            /*group_layer_num=*/2),
                                       input_tokens,
                                       reuse_cache,
                                       role_type);
    }

    void prepareResourceWithCacheConfig(const CacheConfig&      cache_config,
                                        const std::vector<int>& input_tokens,
                                        bool                    reuse_cache,
                                        RoleType                role_type,
                                        const KVCacheConfig&    kv_cache_config = {},
                                        int                     tp_rank         = 0) {
        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role_type;
        ParallelismConfig parallelism_config;
        parallelism_config.tp_rank = tp_rank;
        cache_manager_             = std::make_shared<KVCacheManager>(cache_config,
                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          kv_cache_config,
                                                          parallelism_config,
                                                          RuntimeConfig{},
                                                          SpeculativeExecutionConfig{},
                                                          pd_sep_config);
        ASSERT_TRUE(cache_manager_->init());
        size_t expected_free_blocks = 0;
        for (const auto& group : cache_manager_->cacheConfig().topology().groups()) {
            expected_free_blocks += group.block_num - 1;
        }
        ASSERT_EQ(cache_manager_->freeBlocksNum(), expected_free_blocks);
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = reuse_cache;
        resource_context.role_type     = role_type;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_tokens.begin(), input_tokens.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig model_config;
        model_config.attn_config.tokens_per_block = 2;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        stream_                  = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
        stream_->generate_status_->status = StreamState::RUNNING;
    }

    std::shared_ptr<StreamReadStorageBackend> prepareStorageBackendResource(bool     block_matches,
                                                                            bool     seed_host = false,
                                                                            RoleType role_type = RoleType::PDFUSION,
                                                                            int      cp_size   = 0) {
        KVCacheConfig kv_cache_config;
        kv_cache_config.enable_remote_cache   = true;
        kv_cache_config.enable_memory_cache   = seed_host;
        kv_cache_config.memory_cache_size_mb  = seed_host ? 1 : 0;
        KVCacheConfig manager_kv_cache_config = kv_cache_config;
        // The manager must not construct a real KVCM client before this test
        // replaces its BlockTreeCache with the controllable fake backend.
        manager_kv_cache_config.enable_remote_cache = false;
        auto cache_config                           = cp_size > 0 ? init_cp_config(cp_size) : init_config();
        prepareResourceWithCacheConfig(cache_config,
                                       {1, 2, 3, 4, 5, 6},
                                       /*reuse_cache=*/true,
                                       role_type,
                                       manager_kv_cache_config);

        auto backend = std::make_shared<StreamReadStorageBackend>();
        if (block_matches) {
            backend->blockMatches();
        }
        cache_manager_->coordinator_manager_->block_tree_cache_.reset();
        cache_manager_->block_tree_cache_.reset();
        auto cache = createBlockTreeCache(cache_manager_->cacheConfig(),
                                          kv_cache_config,
                                          cache_manager_->coordinator_manager_,
                                          ParallelismConfig{},
                                          backend);
        EXPECT_NE(cache, nullptr);
        cache_manager_->block_tree_cache_ = cache;
        cache_manager_->coordinator_manager_->attachBlockTreeCache(cache);

        if (seed_host) {
            auto& resource = stream_->streamCacheResource();
            initCacheKeys(resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), 2);
            const GroupSetPtr& group = cache->groupSets().front();
            const BlockIdxType block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            RTP_LLM_CHECK(!isNullBlockIdx(block));
            // Seed through the production store so the record has the integrity
            // footer required by this build before it is published in the tree.
            BlockIndicesType device_blocks;
            for (const auto& pool : group->devicePools()) {
                const auto allocated = pool->malloc(1);
                RTP_LLM_CHECK(allocated.has_value());
                pool->incRef(*allocated);
                device_blocks.push_back(allocated->front());
            }
            const bool stored = cache
                                    ->executeTransfer(TransferTask(
                                        {TransferDescriptor::deviceToHost(group->groupSetId(), device_blocks, block)},
                                        std::chrono::seconds(30)))
                                    .ok();
            for (size_t member = 0; member < device_blocks.size(); ++member) {
                group->devicePools()[member]->decRef(device_blocks[member]);
            }
            RTP_LLM_CHECK(stored);
            std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
            resources[0][0].host_block  = block;
            const size_t seed_key_index = cp_size > 1 ? static_cast<size_t>(cp_size - 1) : 0;
            RTP_LLM_CHECK(cache->tree()
                              ->insertNode({resource.batch_kv_cache_resource_->cacheKeys(0).at(seed_key_index)},
                                           resources,
                                           false,
                                           /*is_resident=*/false)
                              .accepted_resource_count
                          == 1);
            group->releaseSingleBlock(Tier::HOST, block, BlockTreeRefType::CACHE);
        }

        auto& resource                                                 = stream_->streamCacheResource();
        resource.resource_context_.enable_remote_cache                 = true;
        stream_->generate_input_->generate_config->enable_remote_cache = true;
        if (cp_size == 0 || !seed_host) {
            resource.kvCacheMutable().setBatchCacheKeys(0, {100, 200, 300});
        }
        return backend;
    }

    void checkBlockFunc(BatchKVCacheResource& batch_resource, int outter_size, int inner_size) {
        ASSERT_EQ(batch_resource.batchSize(), outter_size);
        for (int i = 0; i < outter_size; ++i) {
            ASSERT_EQ(batch_resource.blocks(i, "default").size(), inner_size);
        }
    };

#define CHECK_BLOCK(block_vec, outter_size, inner_size)                                                                \
    do {                                                                                                               \
        SCOPED_TRACE("checkBlockFunc");                                                                                \
        checkBlockFunc(block_vec, outter_size, inner_size);                                                            \
    } while (0)

protected:
    autil::EnvGuard                 perf_scope;
    GenerateStreamPtr               stream_;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

TEST_F(StreamCacheResourceTest, testResourceContextReadsIgnoreRequestCacheSwitchesEnv) {
    {
        autil::EnvGuard env_guard("RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES", "0");
        ResourceContext resource_context;
        KVCacheConfig   kv_cache_config;
        resource_context.initCacheConfig(kv_cache_config);
        EXPECT_FALSE(resource_context.ignore_request_cache_switches);
        EXPECT_FALSE(resource_context.enable_prefill_partial_tail_reuse);
    }
    {
        autil::EnvGuard env_guard("RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES", "1");
        ResourceContext resource_context;
        KVCacheConfig   kv_cache_config;
        kv_cache_config.enable_prefill_partial_tail_reuse = true;
        resource_context.initCacheConfig(kv_cache_config);
        EXPECT_TRUE(resource_context.ignore_request_cache_switches);
        EXPECT_TRUE(resource_context.enable_prefill_partial_tail_reuse);
    }
}

TEST_F(StreamCacheResourceTest, testWarmUpFakeInitUsesTaggedTopology) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    stream_ =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.kvCache().groupNums(), 1);
    EXPECT_EQ(resource.kvCache().cacheResource().soleGroupTagForLayer(0), "__warmup__");
    EXPECT_EQ(resource.curBlocksNum(), 0);

    stream_->fakeInitKVBlock(2);
    EXPECT_EQ(resource.kvCache().blocks(0, "__warmup__").size(), 2);
}

TEST_F(StreamCacheResourceTest, SwapLinearBlocksUsesPolicyAndTagAfterGroupReordering) {
    for (const bool reversed : {false, true}) {
        auto            config = test::makeSimpleHybridMhaCacheConfig(4, 9, 2, DataType::TYPE_FP16, 2);
        ResourceContext context;
        context.cache_manager = std::make_shared<KVCacheManager>(config);
        StreamCacheResource resource(nullptr, context, /*need_release_resource=*/false);
        resource.init(1);
        if (reversed) {
            auto groups = config.topology().groups();
            std::reverse(groups.begin(), groups.end());
            config.setTopology(std::move(groups), config.topology().layers());
        }
        auto& batch = resource.kvCacheMutable();
        // Only the resource order changes; the manager keeps its original topology.
        batch.initGroups(config.topologyPtr());
        for (const auto& group : config.topology().groups()) {
            batch.mutableBlockIds(0, group.tag).assign({2, 5});
        }
        resource.swapLinearBlocks(0, 0, 1);
        for (const auto& group : config.topology().groups()) {
            EXPECT_EQ(batch.blocks(0, group.tag),
                      group.policy.group_type == CacheGroupType::LINEAR ? (BlockIndicesType{5, 2}) :
                                                                          (BlockIndicesType{2, 5}));
        }
    }
}

TEST_F(StreamCacheResourceTest, PrefillIncrementAfterHostLoadPreservesLinearReuseState) {
    for (const int tail_blocks : {2, 8}) {
        SCOPED_TRACE(tail_blocks);
        auto config                    = test::makeSimpleHybridMhaCacheConfig(4, 16, 2, DataType::TYPE_FP16, 2);
        config.reuse_cache_tail_blocks = tail_blocks;
        KVCacheConfig tiered_config;
        tiered_config.enable_device_cache  = false;
        tiered_config.enable_memory_cache  = true;
        tiered_config.memory_cache_size_mb = 1;
        prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5, 6, 7, 8, 9}, true, RoleType::PDFUSION, tiered_config);
        auto& resource                                         = stream_->streamCacheResource();
        resource.resource_context_.enable_device_cache         = false;
        resource.resource_context_.enable_memory_cache         = true;
        stream_->generate_input_->generate_config->reuse_cache = true;
        initCacheKeys(resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), 2);
        const auto&                                cache = cache_manager_->block_tree_cache_;
        std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(cache->groupSets().size()));
        for (const auto& group : cache->groupSets()) {
            const auto host_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_FALSE(isNullBlockIdx(host_block));
            BlockIndicesType device_blocks;
            for (const auto& pool : group->devicePools()) {
                const auto allocated = pool->malloc(1);
                ASSERT_TRUE(allocated.has_value());
                pool->incRef(*allocated);
                device_blocks.push_back(allocated->front());
            }
            ASSERT_TRUE(cache
                            ->executeTransfer(TransferTask(
                                {TransferDescriptor::deviceToHost(group->groupSetId(), device_blocks, host_block)},
                                std::chrono::seconds(30)))
                            .ok());
            for (size_t member = 0; member < device_blocks.size(); ++member) {
                group->devicePools()[member]->decRef(device_blocks[member]);
            }
            slots[0][group->groupSetId()].host_block = host_block;
        }
        ASSERT_EQ(cache->tree()
                      ->insertNode({resource.batch_kv_cache_resource_->cacheKeys(0).front()}, slots, false, false)
                      .accepted_resource_count,
                  cache->groupSets().size());
        for (const auto& group : cache->groupSets()) {
            group->releaseSingleBlock(Tier::HOST, slots[0][group->groupSetId()].host_block, BlockTreeRefType::CACHE);
        }

        ASSERT_TRUE(resource.initKVBlock().ok());
        ASSERT_NE(resource.allocator_load_context_, nullptr);
        resource.allocator_load_context_->waitDone();
        ASSERT_TRUE(resource.loadCacheDone());
        ASSERT_EQ(stream_->reuseLength(), 2);
        ASSERT_TRUE(stream_->isContextStream());
        ASSERT_FALSE(stream_->chunkedPrefillEnabled());
        const auto before = resource.kvCache().blocks(0, "linear");
        ASSERT_EQ(before.size(), 5u);
        ASSERT_GT(before[0], 0);
        ASSERT_TRUE(resource.incrKVBlock().ok());
        EXPECT_EQ(resource.kvCache().blocks(0, "linear"), before);
        EXPECT_EQ(resource.kvCache().cacheResource(0).kernelBlocks("linear"), before);

        stream_->setIsContextStream(false);
        ASSERT_TRUE(resource.incrKVBlock().ok());
        EXPECT_TRUE(isNullBlockIdx(resource.kvCache().blocks(0, "linear")[0]));
        resource.releaseResource();
    }
}

TEST_F(StreamCacheResourceTest, testAllocateResource) {
    for (const bool with_backend : {false, true}) {
        for (const bool ignore_request_switches : {false, true}) {
            for (const bool request_reuse : {false, true}) {
                for (const auto status : {StreamState::RUNNING, StreamState::FINISHED}) {
                    SCOPED_TRACE("backend=" + std::to_string(with_backend) + " ignore="
                                 + std::to_string(ignore_request_switches) + " reuse=" + std::to_string(request_reuse)
                                 + " status=" + StreamStateToString(status));
                    std::shared_ptr<StreamReadStorageBackend> backend;
                    if (with_backend) {
                        backend = prepareStorageBackendResource(/*block_matches=*/false);
                    } else {
                        prepareResource(/*reuse_cache=*/true);
                    }

                    auto& resource = stream_->streamCacheResource();

                    auto& request               = *stream_->generate_input_->generate_config;
                    request.enable_remote_cache = false;
                    // Allocate fresh blocks so the fake backend's match result does not seed the test.
                    resource.resource_context_.reuse_cache = false;
                    ASSERT_TRUE(resource.initKVBlock().ok());
                    ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
                    ASSERT_EQ(resource.curBlocksNum(), 3);
                    auto& blocks = resource.kvCacheMutable();
                    CHECK_BLOCK(blocks, 2, 3);

                    stream_->setSeqLength(7);
                    stream_->setIsContextStream(false);
                    ASSERT_TRUE(resource.incrKVBlock().ok());
                    ASSERT_EQ(cache_manager_->freeBlocksNum(), 3);

                    CHECK_BLOCK(blocks, 2, 4);

                    resource.resource_context_.reuse_cache                   = true;
                    resource.resource_context_.ignore_request_cache_switches = ignore_request_switches;
                    request.reuse_cache                                      = request_reuse;
                    stream_->generate_status_->status                        = status;
                    stream_->releaseResource();
                    if (backend) {
                        backend->shutdown();
                    }
                    const bool stored = status == StreamState::FINISHED && (request_reuse || ignore_request_switches);
                    EXPECT_EQ(cache_manager_->freeBlocksNum(), stored ? 5 : 8);
                    if (backend) {
                        EXPECT_EQ(backend->writeCalls(), stored ? 1u : 0u);
                        EXPECT_EQ(backend->writtenKeys(), stored ? 3u : 0u);
                    }

                    CHECK_BLOCK(blocks, 2, 0);
                }
            }
        }
    }
}

TEST_F(StreamCacheResourceTest, CacheOwnedLayoutDrivesStreamInitIncrementAndRelease) {
    prepareResourceWithCacheConfig(
        init_cp_config(), {1, 2, 3, 4, 5, 6, 7, 8}, /*reuse_cache=*/false, RoleType::PDFUSION);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(cache_manager_->cacheConfig().request_projection_required);
    EXPECT_EQ(cache_manager_->canonicalSizePerBlock(), 4);
    const size_t  free_before    = cache_manager_->freeBlocksNum();
    const int64_t before_init_us = currentTimeUs();
    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_GE(resource.malloc_begin_time_us_, before_init_us);
    EXPECT_LE(resource.malloc_begin_time_us_, currentTimeUs());
    EXPECT_EQ(resource.kvCache().blocksNum(0, "default"), 2);
    EXPECT_EQ(resource.kvCache().blocksNum(1, "default"), 2);
    EXPECT_EQ(resource.kvCache().cacheKeys(0).size(), 4u);
    EXPECT_EQ(resource.singleBatchNeedBlocks(/*seq_len=*/9, /*reserve_step=*/0), 1);

    const auto original_keys = resource.kvCache().cacheKeys(0);
    for (int batch = 0; batch < resource.kvCache().batchSize(); ++batch) {
        stream_->completeTokenIdsPtr()->data(batch)[8] = 9;
    }
    stream_->setSeqLength(9);
    ASSERT_TRUE(resource.incrKVBlock().ok());
    EXPECT_EQ(resource.kvCache().blocksNum(0, "default"), 3);
    EXPECT_EQ(resource.kvCache().blocksNum(1, "default"), 3);
    // Runtime/base keys include the new partial block (T02/T06 P-to-D
    // contract); CP canonical pages and complete-only tree exports differ.
    ASSERT_EQ(resource.kvCache().cacheKeys(0).size(), 5u);
    EXPECT_TRUE(std::equal(original_keys.begin(), original_keys.end(), resource.kvCache().cacheKeys(0).begin()));
    EXPECT_FALSE(resource.kvCache().lastBlockAligned());
    auto fresh_keys = std::make_shared<BatchKVCacheResource>();
    fresh_keys->resetBatchSize(resource.kvCache().batchSize());
    fresh_keys->initGroups(cache_manager_->cacheConfig().topologyPtr());
    initCacheKeys(fresh_keys, stream_->completeTokenIdsPtr(), cache_manager_->cacheConfig().seq_size_per_block);
    for (int batch = 0; batch < resource.kvCache().batchSize(); ++batch) {
        EXPECT_EQ(resource.kvCache().cacheKeys(batch), fresh_keys->cacheKeys(batch));
    }

    stream_->generate_status_->status = StreamState::FINISHED;
    stream_->releaseResource();
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

TEST_F(StreamCacheResourceTest, PartialTailRegistrationIsGatedAndIndependentOfObservation) {
    {
        prepareResourceWithInputTokens({1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
        auto& resource                                               = stream_->streamCacheResource();
        resource.resource_context_.ignore_request_cache_switches     = true;
        resource.resource_context_.enable_prefill_partial_tail_reuse = true;
        ASSERT_TRUE(resource.initKVBlock().ok());
        ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
        const auto original = *resource.prefixCacheRegistration();
        EXPECT_EQ(original.original_input_tokens, 5u);
        EXPECT_EQ(original.token_ids, (std::vector<int>{1, 2, 3, 4, 5}));
        ASSERT_EQ(original.points.size(), 3u);
        EXPECT_EQ(original.points[0].endpoint.pos, 2u);
        EXPECT_EQ(original.points[1].endpoint.pos, 4u);
        EXPECT_EQ(original.points[2].endpoint.pos, 5u);
        EXPECT_EQ(original.points[2].endpoint.key, original.base_keys.back());
        ASSERT_EQ(original.points[2].group_bindings.size(), cache_manager_->blockTreeCache()->groupSets().size());
        for (const auto& binding : original.points[2].group_bindings) {
            EXPECT_EQ(binding.state, PrefixCacheSaveState::REGISTERED);
            EXPECT_EQ(binding.ownership, PrefixCacheSaveOwnership::NONE);
            EXPECT_FALSE(binding.resource.hasTier(Tier::DEVICE));
            EXPECT_TRUE(binding.unfulfilled_reason.empty());
        }

        // The live stream and its last partial key may change after prefill.
        stream_->completeTokenIdsPtr()->data(0)[5] = 6;
        stream_->setSeqLength(6);
        stream_->setIsContextStream(false);
        ASSERT_TRUE(resource.incrKVBlock().ok());
        const auto& retained = *resource.prefixCacheRegistration();
        EXPECT_EQ(retained.original_input_tokens, original.original_input_tokens);
        EXPECT_EQ(retained.token_ids, original.token_ids);
        EXPECT_EQ(retained.base_keys, original.base_keys);
        EXPECT_EQ(retained.points.back().endpoint.pos, 5u);
        stream_->releaseResource();
    }
}

TEST_F(StreamCacheResourceTest, RegisteredShortSaveOwnsOptionalBlockUntilReleased) {
    auto config = test::makeSimpleLinearCacheConfig(3, 12, 2, DataType::TYPE_FP16);
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prefixCacheRegistration().has_value());

    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
    resource.prepareRegisteredSave();
    const auto& registration = *resource.prefixCacheRegistration();
    ASSERT_TRUE(registration.prepared);
    ASSERT_EQ(registration.points.size(), 3u);
    const auto& tail = registration.points.back().group_bindings.front();
    ASSERT_EQ(tail.ownership, PrefixCacheSaveOwnership::OPTIONAL_REQUEST);
    ASSERT_EQ(tail.state, PrefixCacheSaveState::REGISTERED);
    ASSERT_TRUE(tail.resource.hasCompleteDeviceValue());
    const auto block = tail.resource.device_blocks.front();
    const auto pool  = cache_manager_->blockTreeCache()->groupSets().front()->devicePools().front();
    EXPECT_EQ(pool->refCount(block), 1u);

    cache_manager_->releaseRegisteredSave(*resource.prefix_cache_registration_);
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_FALSE(resource.prefix_cache_registration_->prepared);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, LegacyDenseSwaContinuationPreservesLaterCachedPayload) {
    for (const bool registered_snapshot : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "registered=" << registered_snapshot);
        CacheConfig config;
        config.dtype               = DataType::TYPE_FP16;
        config.layer_num           = 1;
        config.seq_size_per_block  = 4;
        auto policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
        policy.enable_prefix_reuse = true;
        policy.sliding_window_size = 4;
        config.fromGroupedSpecs(
            {test::makeMhaSpec("swa", 16, DataType::TYPE_FP16, 1, 1)}, {{0}}, {CacheGroupType::SWA}, {"swa"}, {policy});
        config.finalizeBlockNums(24, RuntimeConfig{});
        ParallelismConfig parallelism;
        parallelism.tp_size                            = 2;
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        parallelism.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
        config.finalizeStorageLayout(parallelism);
        prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5, 6, 7, 8, 99}, true, RoleType::PREFILL);
        const auto pool        = cache_manager_->blockTreeCache()->groupSets().front()->devicePools().front();
        const auto free_before = pool->freeBlocksNum();
        auto       tokens      = std::make_shared<CompleteTokenIds>(1, 1, 128, 4);
        auto       input       = std::make_shared<GenerateInput>();
        input->input_ids       = torch::arange(1, 17, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        tokens->init(input);
        auto source = std::make_shared<BatchKVCacheResource>();
        source->resetBatchSize(1);
        source->initGroups(config.topologyPtr());
        initCacheKeys(source, tokens, 4);
        MallocInfo cold{source, tokens};
        cold.enable_cache_lookup = false;
        cold.request_layout      = test::snapshotGlobalCacheRequest(*source, *tokens);
        ASSERT_TRUE(cache_manager_->malloc(cold).success);
        const auto cached_block = source->blocks(0, "swa").front();
        auto       payload      = [&](BlockIdxType block) {
            return torch::from_blob(cache_manager_->convertIndexToBuffer(0, "swa", block).front().addr,
                                               {32},
                                    torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
        };
        payload(cached_block).fill_(0x44);
        runtimeSyncAndCheck();
        size_t                  resident = 0;
        PrefixCacheRegistration saved;
        InsertInfo              insert{source, tokens, false};
        if (registered_snapshot) {
            saved.original_input_tokens       = 16;
            saved.token_ids                   = tokens->completeTokenIdsVec(0);
            saved.base_keys                   = source->cacheKeys(0);
            saved.required_continuation_ready = true;
            for (const size_t endpoint : {8u, 16u}) {
                PrefixCacheRegistrationPoint point;
                point.endpoint.pos = endpoint;
                point.group_bindings.resize(1);
                saved.points.push_back(std::move(point));
            }
            cache_manager_->prepareRegisteredSave(saved, *source, 0);
            ASSERT_TRUE(saved.prepared);
            for (auto& point : saved.points) {
                auto& binding = point.group_bindings.front();
                ASSERT_TRUE(binding.resource.hasCompleteDeviceValue());
                ASSERT_EQ(binding.resource.device_blocks.front(), cached_block);
                binding.state = PrefixCacheSaveState::CONTENT_COMPLETE;
            }
            saved.content_completed          = true;
            insert.prefix_cache_registration = &saved;
        }
        cache_manager_->insertIntoCache(insert, resident);
        cache_manager_->releaseRegisteredSave(saved);
        cache_manager_->free(FreeInfo{source, tokens});
        auto& resource                                               = stream_->streamCacheResource();
        resource.resource_context_.ignore_request_cache_switches     = true;
        resource.resource_context_.enable_prefill_partial_tail_reuse = registered_snapshot;
        stream_->generate_status_->status                            = StreamState::WAITING;
        stream_->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream_->moveToNext(), StreamState::RUNNING);
        EXPECT_EQ(resource.prefixCacheRegistration().has_value(), registered_snapshot);
        EXPECT_EQ(stream_->initialReuseLength(), registered_snapshot ? 8 : 0);
        ModelConfig model;
        model.input_vocab_size             = 128;
        model.attn_config.tokens_per_block = 4;
        model.max_seq_len                  = 2048;
        NormalBatchStreamProcessor processor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config, false);
        TensorHolder               holder;
        auto                       gathered = processor.gatherModelInput(StreamGroups({stream_}), holder);
        ASSERT_TRUE(gathered.ok()) << gathered.status();
        if (gathered->kv_cache_update_mapping.defined() && gathered->kv_cache_update_mapping.numel() > 0) {
            cache_manager_->blockBatchCopyForForward(
                gathered->kv_cache_update_mapping, gathered->kv_cache_group_tags, true);
        }
        ASSERT_TRUE(gathered->kv_cache_block_id.defined());
        const auto writable = gathered->kv_cache_block_id.cpu().flatten()[0].item<int32_t>();
        EXPECT_TRUE(stream_->initialReuseLength() == 0 || writable != cached_block)
            << "reuse=" << stream_->initialReuseLength() << " writable=" << writable << " cached16=" << cached_block;
        if (registered_snapshot) {
            ASSERT_TRUE(gathered->kv_cache_update_mapping.defined());
            EXPECT_GT(gathered->kv_cache_update_mapping.numel(), 0);
            EXPECT_TRUE(payload(writable).slice(0, 0, 16).cpu().eq(0x44).all().item<bool>());
        }
        payload(writable).slice(0, 16, 18).fill_(0xcc);
        runtimeSyncAndCheck();
        EXPECT_TRUE(payload(cached_block).cpu().eq(0x44).all().item<bool>());
        stream_->releaseResource();
        auto final_tokens = std::make_shared<CompleteTokenIds>(1, 1, 128, 4);
        input->input_ids  = torch::arange(1, 18, torch::kInt32);
        final_tokens->init(input);
        auto final_resource = std::make_shared<BatchKVCacheResource>();
        final_resource->resetBatchSize(1);
        final_resource->initGroups(config.topologyPtr());
        initCacheKeys(final_resource, final_tokens, 4);
        MallocInfo final_lookup{final_resource, final_tokens};
        final_lookup.request_layout = test::snapshotGlobalCacheRequest(*final_resource, *final_tokens);
        const auto final_hit        = cache_manager_->malloc(final_lookup);
        ASSERT_TRUE(final_hit.success);
        EXPECT_EQ(final_hit.reuse_len, 16);
        EXPECT_TRUE(payload(final_resource->blocks(0, "swa").front()).cpu().eq(0x44).all().item<bool>());
        cache_manager_->free(FreeInfo{final_resource, final_tokens});
        cache_manager_->blockTreeCache()->evictForGroup("swa", pool->totalBlocksNum());
        EXPECT_EQ(pool->freeBlocksNum(), free_before);
    }
}

TEST_F(StreamCacheResourceTest, PartialTailRunningTransitionPreparesRequiredContinuation) {
    prepareResourceWithInputTokens({1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    stream_->generate_status_->status                            = StreamState::WAITING;
    stream_->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream_->moveToNext(), StreamState::RUNNING);
    ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
    EXPECT_TRUE(resource.prefixCacheRegistration()->required_continuation_ready);
    EXPECT_FALSE(resource.prefixCacheRegistration()->prepared);
    resource.prepareRegisteredSave();
    EXPECT_TRUE(resource.prefixCacheRegistration()->prepared);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, DenseTailSaveReusesPrivateContinuationWithoutSecondCopy) {
    auto config = test::makeSimpleMhaCacheConfig(3, 16, 4, DataType::TYPE_FP16);
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
    ASSERT_EQ(resource.prefixCacheRegistration()->points.back().endpoint.pos, 6u);

    const auto&  tag    = cache_manager_->cacheConfig().groupTags().front();
    auto         pool   = cache_manager_->blockTreeCache()->groupSets().front()->devicePools().front();
    const size_t slot   = cache_manager_->cacheConfig().endpointBlockPosition(tag, 5);
    const auto   source = resource.kvCache().blocks(0, tag).at(slot);
    ASSERT_FALSE(isNullBlockIdx(source));
    pool->incTreeRef(source, BlockTreeRefType::CACHE);
    stream_->setInitialReuseLength(5);
    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());

    const auto running = resource.kvCache().blocks(0, tag).at(slot);
    ASSERT_NE(running, source);
    const auto& required_copies = resource.registeredSaveCopyMapping();
    EXPECT_TRUE(std::any_of(required_copies.begin(), required_copies.end(), [&](const auto& copy) {
        return copy.tag == tag && copy.src == source && copy.dst == running;
    }));
    const auto&  source_pins      = resource.prefixCacheRegistration()->source_pins;
    const size_t source_pin_count = std::count_if(
        source_pins.begin(), source_pins.end(), [&](const auto& pin) { return pin.tag == tag && pin.block == source; });
    ASSERT_GE(source_pin_count, 1u);
    EXPECT_EQ(pool->refCount(source), 1u + source_pin_count);  // CACHE plus each request source pin

    resource.prepareRegisteredSave();
    const auto& registration = *resource.prefixCacheRegistration();
    ASSERT_TRUE(registration.prepared);
    const auto& tail = registration.points.back().group_bindings.front();
    ASSERT_EQ(tail.ownership, PrefixCacheSaveOwnership::REQUIRED_RUNTIME);
    ASSERT_TRUE(tail.resource.hasCompleteDeviceValue());
    const auto saved = tail.resource.device_blocks.front();
    EXPECT_NE(saved, source);
    EXPECT_EQ(saved, running);
    const auto& all_copies = resource.registeredSaveCopyMapping();
    ASSERT_EQ(all_copies.size(), resource.kvCache().batchSize());
    std::set<BlockIdxType> private_destinations;
    for (int batch = 0; batch < resource.kvCache().batchSize(); ++batch) {
        const auto destination = resource.kvCache().blocks(batch, tag).at(slot);
        EXPECT_NE(destination, source);
        EXPECT_TRUE(private_destinations.insert(destination).second);
        EXPECT_EQ(std::count_if(all_copies.begin(),
                                all_copies.end(),
                                [&](const auto& copy) {
                                    return copy.tag == tag && copy.src == source && copy.dst == destination;
                                }),
                  1);
    }
    EXPECT_FALSE(std::any_of(all_copies.begin(), all_copies.end(), [&](const auto& copy) {
        return copy.tag == tag && copy.src == running && copy.dst == saved;
    }));
    EXPECT_EQ(tail.state, PrefixCacheSaveState::REGISTERED);
    ASSERT_EQ(registration.points.front().endpoint.pos, 4u);
    EXPECT_EQ(registration.points.front().group_bindings.front().state, PrefixCacheSaveState::CONTENT_COMPLETE);
    EXPECT_TRUE(registration.content_completed);  // The recovered P4 is ready; future P6 is not.
    InsertInfo info{resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false, Tier::DEVICE};
    info.prefix_cache_registration = &registration;
    size_t resident                = 0;
    cache_manager_->insertIntoCache(info, resident);
    const auto published = cache_manager_->blockTreeCache()->tree()->findNode(registration.base_keys);
    ASSERT_EQ(published.size(), 1u);
    EXPECT_EQ(published.front()->token_end, 4u);  // Never publish the uncomputed suffix at P6.

    pool->decTreeRef(source, BlockTreeRefType::CACHE);
    EXPECT_TRUE(pool->isAllocated(source));  // A remains pinned until the required copy completes or request release.
    stream_->releaseResource();
    EXPECT_FALSE(pool->isAllocated(source));
    EXPECT_FALSE(pool->isAllocated(running));
    EXPECT_FALSE(pool->isAllocated(saved));
}

TEST_F(StreamCacheResourceTest, OptionalGroupMemberFailureRollsBackEarlierMember) {
    // Two LINEAR tags share an atomic GroupSet and need distinct exact states.
    CacheConfig config;
    config.dtype              = DataType::TYPE_FP16;
    config.layer_num          = 2;
    config.seq_size_per_block = 2;
    auto first                = test::makeLinearSpec("first", 2, DataType::TYPE_FP16, 1, 1);
    auto second               = test::makeLinearSpec("second", 2, DataType::TYPE_FP16, 1, 1);
    config.fromGroupedSpecs(
        {first, second}, {{0}, {1}}, {CacheGroupType::LINEAR, CacheGroupType::LINEAR}, {"first", "second"});
    config.finalizeBlockNums(12, RuntimeConfig{});
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());

    const auto& group_sets = cache_manager_->blockTreeCache()->groupSets();
    const auto  group_it   = std::find_if(
        group_sets.begin(), group_sets.end(), [](const auto& group) { return group->groupTags().size() >= 2; });
    ASSERT_NE(group_it, group_sets.end());
    const auto&  group             = *group_it;
    auto         first_pool        = group->devicePools()[0];
    auto         second_pool       = group->devicePools()[1];
    const size_t first_free_before = first_pool->freeBlocksNum();
    const auto   held              = second_pool->malloc(second_pool->freeBlocksNum());
    ASSERT_TRUE(held.has_value());
    second_pool->incRef(*held);

    resource.prepareRegisteredSave();
    ASSERT_TRUE(resource.prefixCacheRegistration()->prepared);
    const auto& binding = resource.prefixCacheRegistration()->points.back().group_bindings[group->groupSetId()];
    EXPECT_EQ(binding.state, PrefixCacheSaveState::UNFULFILLED);
    EXPECT_EQ(binding.unfulfilled_reason, "optional_device_capacity");
    EXPECT_TRUE(binding.resource.is_empty());
    EXPECT_EQ(first_pool->freeBlocksNum(), first_free_before);

    second_pool->decRef(*held);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, CompletedHistoryPinsDoNotBlockFallbackRuntimeGrowth) {
    auto config = test::makeSimpleLinearCacheConfig(3, 8, 4, DataType::TYPE_FP16);
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    resource.init(1);
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());

    const auto&  tag    = cache_manager_->cacheConfig().groupTags().front();
    auto         pool   = cache_manager_->blockTreeCache()->groupSets().front()->devicePools().front();
    const size_t slot   = cache_manager_->cacheConfig().endpointBlockPosition(tag, 5);
    const auto   source = resource.kvCache().blocks(0, tag).at(slot);
    pool->incTreeRef(source, BlockTreeRefType::CACHE);
    stream_->setInitialReuseLength(5);
    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
    resource.prepareRegisteredSave();
    const auto& registration = *resource.prefixCacheRegistration();
    ASSERT_TRUE(registration.prepared);
    ASSERT_FALSE(registration.source_pins.empty());
    ASSERT_TRUE(std::any_of(registration.copy_mappings.begin(),
                            registration.copy_mappings.end(),
                            [&](const auto& copy) { return copy.tag == tag && copy.src == source; }));
    ASSERT_EQ(registration.points.back().group_bindings.front().ownership, PrefixCacheSaveOwnership::OPTIONAL_REQUEST);
    const auto original_key = registration.points.back().endpoint.key;

    // Keep the cached A allocated independently of this request while the
    // fallback allocator tries to reclaim its completed optional D.
    pool->incRef(source);
    const auto   before_completion = pool->refCount(source);
    const size_t source_pin_count =
        std::count_if(registration.source_pins.begin(), registration.source_pins.end(), [&](const auto& pin) {
            return pin.tag == tag && pin.block == source;
        });
    resource.completeRegisteredSave(/*completed_endpoint=*/6);  // Inject T05's post-copy/writer completion signal.
    EXPECT_TRUE(resource.prefixCacheRegistration()->source_pins.empty());
    EXPECT_EQ(pool->refCount(source), before_completion - source_pin_count);

    const auto held = pool->malloc(pool->freeBlocksNum());
    ASSERT_TRUE(held.has_value());
    pool->incRef(*held);
    auto* ids = stream_->completeTokenIdsPtr()->data(0);
    ids[6]    = 7;
    ids[7]    = 8;
    ids[8]    = 9;
    stream_->setSeqLength(9);
    ASSERT_TRUE(resource.incrKVBlock().ok());
    const auto& retained = *resource.prefixCacheRegistration();
    EXPECT_EQ(retained.original_input_tokens, 6u);
    EXPECT_EQ(retained.points.back().endpoint.key, original_key);
    EXPECT_EQ(retained.points.back().group_bindings.front().state, PrefixCacheSaveState::UNFULFILLED);
    EXPECT_EQ(retained.points.back().group_bindings.front().unfulfilled_reason, "runtime_priority");

    pool->decRef(*held);
    pool->decRef(source);
    pool->decTreeRef(source, BlockTreeRefType::CACHE);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, CompletionOnlyMarksActuallyComputedSaveEndpoints) {
    auto config = test::makeSimpleMhaCacheConfig(3, 12, 2, DataType::TYPE_FP16);
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    resource.init(1);
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
    resource.prepareRegisteredSave();
    const auto& initial = *resource.prefixCacheRegistration();
    ASSERT_EQ(initial.points.size(), 3u);
    EXPECT_EQ(initial.points[0].endpoint.pos, 2u);
    EXPECT_EQ(initial.points[1].endpoint.pos, 4u);
    EXPECT_EQ(initial.points[2].endpoint.pos, 5u);

    resource.completeRegisteredSave(/*completed_endpoint=*/3);
    const auto& after_first_chunk = *resource.prefixCacheRegistration();
    EXPECT_EQ(after_first_chunk.points[0].group_bindings.front().state, PrefixCacheSaveState::CONTENT_COMPLETE);
    EXPECT_EQ(after_first_chunk.points[1].group_bindings.front().state, PrefixCacheSaveState::REGISTERED);
    EXPECT_EQ(after_first_chunk.points[2].group_bindings.front().state, PrefixCacheSaveState::REGISTERED);

    resource.completeRegisteredSave(/*completed_endpoint=*/5);
    const auto& after_final_chunk = *resource.prefixCacheRegistration();
    EXPECT_EQ(after_final_chunk.points[1].group_bindings.front().state, PrefixCacheSaveState::CONTENT_COMPLETE);
    EXPECT_EQ(after_final_chunk.points[2].group_bindings.front().state, PrefixCacheSaveState::CONTENT_COMPLETE);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, RegisteredHistoricalGroupsRecordMissingWithoutReplay) {
    auto config = test::makeSimpleLinearCacheConfig(3, 12, 2, DataType::TYPE_FP16);
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
    const auto   original    = *resource.prefixCacheRegistration();
    const auto   cache       = cache_manager_->blockTreeCache();
    const auto   pool        = cache->groupSets().front()->devicePools().front();
    auto&        runtime     = *resource.batch_kv_cache_resource_;
    auto         missing     = original;
    const size_t free_before = pool->freeBlocksNum();
    cache_manager_->prepareRegisteredSave(missing, runtime, 5);
    for (const auto& point : missing.points) {
        for (const auto& binding : point.group_bindings) {
            EXPECT_EQ(binding.state, PrefixCacheSaveState::UNFULFILLED);
            EXPECT_EQ(binding.unfulfilled_reason, "missing_history_before_resume");
            EXPECT_TRUE(binding.resource.is_empty());
        }
    }
    EXPECT_TRUE(missing.copy_mappings.empty());
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
    cache_manager_->releaseRegisteredSave(missing);

    // This cache ownership fixture supplies the existing completion signal;
    // model writer correctness is covered by the CUDA integration gates.
    resource.prepareRegisteredSave();
    resource.completeRegisteredSave(5);
    InsertInfo info{resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false, Tier::DEVICE};
    info.prefix_cache_registration = &*resource.prefix_cache_registration_;
    size_t resident                = 0;
    cache_manager_->insertIntoCache(info, resident);
    ASSERT_EQ(cache->tree()->size(), 3u);

    auto         cached      = original;
    const size_t cached_free = pool->freeBlocksNum();
    cache_manager_->prepareRegisteredSave(cached, runtime, 5);
    for (const auto& point : cached.points) {
        for (const auto& binding : point.group_bindings) {
            EXPECT_EQ(binding.state, PrefixCacheSaveState::CACHED_REF);
            EXPECT_TRUE(binding.unfulfilled_reason.empty());
            EXPECT_EQ(binding.ownership, PrefixCacheSaveOwnership::CACHED_REQUEST);
            EXPECT_TRUE(binding.resource.hasCompleteDeviceValue());
            EXPECT_FALSE(binding.cached_path_indices.empty());
        }
    }
    EXPECT_TRUE(cached.copy_mappings.empty());
    EXPECT_TRUE(cached.source_pins.empty());
    EXPECT_TRUE(cached.content_completed);            // Existing saved content satisfies the registered intent.
    EXPECT_EQ(cached.cached_source_pins.size(), 3u);  // Shared across the FULL prefixes 2/4/5.
    EXPECT_EQ(pool->freeBlocksNum(), cached_free);
    cache_manager_->releaseRegisteredSave(cached);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, CachedRegistrationPinsSwaHistoryAndYieldsToRuntimeWithoutReplay) {
    for (const bool yield_to_runtime : {false, true}) {
        for (const bool missing_middle : {false, true}) {
            for (const size_t resume : {6u, 7u}) {
                SCOPED_TRACE(::testing::Message()
                             << "yield=" << yield_to_runtime << "missing=" << missing_middle << "resume=" << resume);
                CacheConfig config;
                config.dtype               = DataType::TYPE_FP16;
                config.layer_num           = 2;
                config.seq_size_per_block  = 2;
                auto             full_spec = test::makeMhaSpec("full", 2, DataType::TYPE_FP16, 1, 1);
                auto             swa_spec  = test::makeMhaSpec("swa", 2, DataType::TYPE_FP16, 1, 1);
                CacheGroupPolicy full_policy;
                CacheGroupPolicy swa_policy;
                swa_policy.group_type          = CacheGroupType::SWA;
                swa_policy.sliding_window_size = 4;
                swa_policy.active_tail_blocks  = 2;
                config.fromGroupedSpecs({full_spec, swa_spec},
                                        {{0}, {1}},
                                        {CacheGroupType::FULL, CacheGroupType::SWA},
                                        {"full", "swa"},
                                        {full_policy, swa_policy});
                config.finalizeBlockNums(32, RuntimeConfig{});
                prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5, 6, 7}, true, RoleType::PREFILL);
                auto& resource                                               = stream_->streamCacheResource();
                resource.resource_context_.ignore_request_cache_switches     = true;
                resource.resource_context_.enable_prefill_partial_tail_reuse = true;
                ASSERT_TRUE(resource.initKVBlock().ok());
                ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
                auto       probe = *resource.prefixCacheRegistration();
                const auto cache = cache_manager_->blockTreeCache();
                ASSERT_EQ(cache->groupSets().size(), 2u);
                size_t full_id = 0;
                size_t swa_id  = 0;
                for (const auto& group : cache->groupSets()) {
                    if (group->groupType() == CacheGroupType::FULL)
                        full_id = group->groupSetId();
                    if (group->groupType() == CacheGroupType::SWA)
                        swa_id = group->groupSetId();
                }
                ASSERT_NE(full_id, swa_id);
                const auto full_pool   = cache->groupSets()[full_id]->devicePools().front();
                const auto swa_pool    = cache->groupSets()[swa_id]->devicePools().front();
                const auto full_blocks = full_pool->malloc(4);
                const auto swa_blocks  = swa_pool->malloc(4);
                ASSERT_TRUE(full_blocks.has_value() && swa_blocks.has_value());
                full_pool->incRef(*full_blocks);
                swa_pool->incRef(*swa_blocks);
                std::vector<std::vector<GroupSetResource>> stored(4, std::vector<GroupSetResource>(2));
                for (size_t index = 0; index < 4; ++index) {
                    stored[index][full_id].device_blocks = {(*full_blocks)[index]};
                    if (!missing_middle || index != 1) {
                        stored[index][swa_id].device_blocks = {(*swa_blocks)[index]};
                    }
                }
                const std::vector<std::vector<int>> tokens{{1, 2}, {3, 4}, {5, 6}, {7}};
                const std::vector<size_t>           endpoints{2, 4, 6, 7};
                cache->insert(probe.base_keys, stored, Tier::DEVICE, false, &tokens, &endpoints);
                full_pool->decRef(*full_blocks);
                swa_pool->decRef(*swa_blocks);
                ASSERT_EQ(cache->tree()->size(), 4u);
                if (missing_middle) {
                    // This fixture initialized a fresh runtime before supplying G.
                    // Reflect the same missing historical page there as on the tree;
                    // uncomputed fresh allocation must not stand in for restored KV.
                    auto&      runtime_ids = resource.batch_kv_cache_resource_->mutableBlockIds(0, "swa");
                    const auto unused      = runtime_ids.blocks().at(1);
                    ASSERT_FALSE(isNullBlockIdx(unused));
                    runtime_ids.setAt(1, NULL_BLOCK_IDX);
                    swa_pool->decRef(unused);
                }
                const auto baseline_free = cache_manager_->freeBlocksNum();
                cache_manager_->prepareRegisteredSave(probe, *resource.batch_kv_cache_resource_, resume);
                EXPECT_TRUE(probe.copy_mappings.empty());
                EXPECT_EQ(cache_manager_->freeBlocksNum(), baseline_free);
                // N7 is already cached even when G6<N7: it needs no producer destination.
                for (const auto& point : probe.points) {
                    for (const auto& binding : point.group_bindings) {
                        EXPECT_NE(binding.state, PrefixCacheSaveState::REGISTERED);
                        EXPECT_NE(binding.ownership, PrefixCacheSaveOwnership::OPTIONAL_REQUEST);
                    }
                }
                EXPECT_EQ(probe.cached_source_pins.size(), missing_middle ? 7u : 8u);
                for (size_t index = 0; index < 4; ++index) {
                    EXPECT_EQ(probe.points[index].group_bindings[full_id].state, PrefixCacheSaveState::CACHED_REF);
                    const auto& binding         = probe.points[index].group_bindings[swa_id];
                    const bool  missing_history = missing_middle && (index == 1 || index == 2);
                    EXPECT_EQ(binding.state,
                              missing_history ? PrefixCacheSaveState::UNFULFILLED : PrefixCacheSaveState::CACHED_REF);
                    EXPECT_EQ(binding.unfulfilled_reason, missing_history ? "missing_history_before_resume" : "");
                }
                const auto& swa_tail = probe.points.back().group_bindings[swa_id];
                ASSERT_EQ(swa_tail.cached_path_indices.size(), 2u);  // Previous and current SWA physical nodes.
                for (const auto& pin : probe.cached_source_pins) {
                    const auto pool = pin.tag == "full" ? full_pool : swa_pool;
                    EXPECT_EQ(pool->refCount(pin.block), 2u);  // One tree + one request ref, even across points.
                }
                cache->evictForGroup("full", 32);
                EXPECT_EQ(cache->tree()->size(), 0u);
                for (const auto& pin : probe.cached_source_pins) {
                    const auto pool = pin.tag == "full" ? full_pool : swa_pool;
                    ASSERT_TRUE(pool->isAllocated(pin.block));
                    EXPECT_EQ(pool->refCount(pin.block), 1u);
                }
                if (yield_to_runtime) {
                    // A mandatory continuation source remains protected while its
                    // copy descriptor is live; optional saved-state pins may yield.
                    const auto& full_runtime = resource.batch_kv_cache_resource_->blocks(0, "full");
                    ASSERT_GE(full_runtime.size(), 2u);
                    const auto required_source = full_runtime[0];
                    const auto required_before = full_pool->refCount(required_source);
                    full_pool->incRef(required_source);
                    probe.source_pins.push_back({"full", 0, required_source});
                    probe.copy_mappings.push_back({"full", required_source, full_runtime[1]});
                    EXPECT_TRUE(cache_manager_->relinquishOptionalRegisteredSave(probe));
                    EXPECT_TRUE(probe.cached_source_pins.empty());
                    EXPECT_TRUE(probe.cached_path_resources.empty());
                    ASSERT_EQ(probe.copy_mappings.size(), 1u);
                    ASSERT_EQ(probe.source_pins.size(), 1u);
                    EXPECT_EQ(full_pool->refCount(required_source), required_before + 1);
                    EXPECT_EQ(probe.points.back().group_bindings[swa_id].unfulfilled_reason, "runtime_priority");
                    for (auto block : *full_blocks)
                        EXPECT_FALSE(full_pool->isAllocated(block));
                    for (auto block : *swa_blocks)
                        EXPECT_FALSE(swa_pool->isAllocated(block));
                } else {
                    // Rebuild the evicted FULL/SWA path from pinned immutable state,
                    // including older SWA nodes; no model replay or new snapshot.
                    InsertInfo info{
                        resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false, Tier::DEVICE};
                    info.prefix_cache_registration = &probe;
                    size_t resident                = 0;
                    cache_manager_->insertIntoCache(info, resident);
                    ASSERT_EQ(cache->tree()->size(), 4u);
                    const auto path = cache->tree()->findNode(probe.base_keys);
                    ASSERT_EQ(path.size(), 4u);
                    EXPECT_EQ(path[2]->group_set_resources[swa_id].device_blocks, (BlockIndicesType{(*swa_blocks)[2]}));
                    EXPECT_EQ(path[3]->group_set_resources[swa_id].device_blocks, (BlockIndicesType{(*swa_blocks)[3]}));
                }
                cache_manager_->releaseRegisteredSave(probe);
                EXPECT_TRUE(probe.cached_source_pins.empty());
                cache_manager_->releaseRegisteredSave(probe);  // Reference cleanup is idempotent.
                stream_->releaseResource();
            }
        }
    }
}

TEST_F(StreamCacheResourceTest, RegisteredSavePublishesOnlyAfterCompletionAndReleasesConcurrentLoser) {
    auto config = test::makeSimpleLinearCacheConfig(3, 32, 2, DataType::TYPE_FP16);
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());

    auto& runtime   = *resource.batch_kv_cache_resource_;
    auto  first     = *resource.prefixCacheRegistration();
    auto  second    = first;
    auto  cancelled = first;
    cache_manager_->prepareRegisteredSave(first, runtime, 0);
    cache_manager_->prepareRegisteredSave(second, runtime, 0);
    cache_manager_->prepareRegisteredSave(cancelled, runtime, 0);
    ASSERT_TRUE(first.prepared && second.prepared && cancelled.prepared);
    const auto cache          = cache_manager_->blockTreeCache();
    const auto pool           = cache->groupSets().front()->devicePools().front();
    const auto first_tail     = first.points.back().group_bindings.front().resource.device_blocks.front();
    const auto second_tail    = second.points.back().group_bindings.front().resource.device_blocks.front();
    const auto cancelled_tail = cancelled.points.back().group_bindings.front().resource.device_blocks.front();
    ASSERT_NE(first_tail, second_tail);

    auto insert = [&](const PrefixCacheRegistration& registration) {
        InsertInfo info{resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false, Tier::DEVICE};
        info.prefix_cache_registration = &registration;
        size_t resident                = 0;
        cache_manager_->insertIntoCache(info, resident);
    };
    insert(cancelled);
    EXPECT_EQ(cache->tree()->size(), 0u);  // Allocated IDs alone cannot publish.
    cache_manager_->releaseRegisteredSave(cancelled);
    EXPECT_FALSE(pool->isAllocated(cancelled_tail));

    auto mark_complete = [](PrefixCacheRegistration& registration) {
        for (auto& point : registration.points) {
            for (auto& binding : point.group_bindings) {
                if (binding.resource.hasCompleteDeviceValue()) {
                    binding.state = PrefixCacheSaveState::CONTENT_COMPLETE;
                }
            }
        }
        registration.content_completed = true;
    };
    mark_complete(first);
    mark_complete(second);
    std::thread a([&] { insert(first); });
    std::thread b([&] { insert(second); });
    a.join();
    b.join();
    EXPECT_EQ(cache->tree()->size(), 3u);

    cache_manager_->releaseRegisteredSave(first);
    cache_manager_->releaseRegisteredSave(second);
    const bool first_retained  = pool->isAllocated(first_tail);
    const bool second_retained = pool->isAllocated(second_tail);
    EXPECT_NE(first_retained, second_retained);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, OptionalSaveYieldsToFallbackRuntimeGrowth) {
    auto config = test::makeSimpleLinearCacheConfig(3, 5, 2, DataType::TYPE_FP16);
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    resource.init(1);
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
    resource.prepareRegisteredSave();
    ASSERT_TRUE(resource.prefixCacheRegistration()->prepared);
    const auto& points = resource.prefixCacheRegistration()->points;
    ASSERT_EQ(points.size(), 3u);
    EXPECT_EQ(points[0].group_bindings.front().ownership, PrefixCacheSaveOwnership::REQUIRED_RUNTIME);
    EXPECT_EQ(points[1].group_bindings.front().ownership, PrefixCacheSaveOwnership::REQUIRED_RUNTIME);
    ASSERT_EQ(points[2].group_bindings.front().ownership, PrefixCacheSaveOwnership::OPTIONAL_REQUEST);
    const auto optional = points[2].group_bindings.front().resource.device_blocks.front();
    auto       pool     = cache_manager_->blockTreeCache()->groupSets().front()->devicePools().front();
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    resource.completeRegisteredSave(
        /*completed_endpoint=*/5);  // Inject the T05 completion signal for this ownership test.

    auto* ids = stream_->completeTokenIdsPtr()->data(0);
    ids[5]    = 6;
    ids[6]    = 7;
    stream_->setSeqLength(7);
    ASSERT_TRUE(resource.incrKVBlock().ok());
    const auto& runtime_blocks = resource.kvCache().blocks(0, cache_manager_->cacheConfig().groupTags().front());
    EXPECT_EQ(runtime_blocks.size(), 4u);
    const auto& retained = *resource.prefixCacheRegistration();
    EXPECT_EQ(retained.points[0].group_bindings.front().state, PrefixCacheSaveState::CONTENT_COMPLETE);
    EXPECT_EQ(retained.points[1].group_bindings.front().state, PrefixCacheSaveState::CONTENT_COMPLETE);
    EXPECT_EQ(retained.points[2].group_bindings.front().state, PrefixCacheSaveState::UNFULFILLED);
    EXPECT_EQ(retained.points[2].group_bindings.front().unfulfilled_reason, "runtime_priority");
    EXPECT_EQ(retained.points[2].group_bindings.front().ownership, PrefixCacheSaveOwnership::NONE);
    EXPECT_TRUE(retained.points[2].group_bindings.front().resource.is_empty());
    EXPECT_TRUE(std::none_of(retained.copy_mappings.begin(), retained.copy_mappings.end(), [&](const auto& copy) {
        return copy.dst == optional;
    }));
    if (pool->isAllocated(optional)) {
        // The freed optional numeric ID may be immediately reused by mandatory runtime growth.
        EXPECT_NE(std::find(runtime_blocks.begin(), runtime_blocks.end(), optional), runtime_blocks.end());
        EXPECT_EQ(pool->refCount(optional), 1u);
    }
    EXPECT_TRUE(retained.content_completed);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, PartialTailRegistrationDeduplicatesAlignedEndAndHonorsRole) {
    prepareResourceWithInputTokens({1, 2, 3, 4, 5}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& default_off                                           = stream_->streamCacheResource();
    default_off.resource_context_.ignore_request_cache_switches = true;
    ASSERT_TRUE(default_off.initKVBlock().ok());
    EXPECT_FALSE(default_off.prefixCacheRegistration().has_value());
    stream_->releaseResource();

    for (const auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
        prepareResourceWithInputTokens({1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true, role);
        auto& resource                                               = stream_->streamCacheResource();
        resource.resource_context_.ignore_request_cache_switches     = true;
        resource.resource_context_.enable_prefill_partial_tail_reuse = true;
        ASSERT_TRUE(resource.initKVBlock().ok());
        if (role == RoleType::PREFILL) {
            ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
            const auto& points = resource.prefixCacheRegistration()->points;
            ASSERT_EQ(points.size(), 3u);
            EXPECT_EQ(points.back().endpoint.pos, 6u);
        } else {
            EXPECT_FALSE(resource.prefixCacheRegistration().has_value());
        }
        stream_->releaseResource();
    }
}

TEST_F(StreamCacheResourceTest, PartialTailRegistrationUsesGlobalCpCheckpointUnit) {
    prepareResourceWithCacheConfig(init_cp_config(),
                                   {1, 2, 3, 4, 5, 6, 7},
                                   /*reuse_cache=*/true,
                                   RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_EQ(cache_manager_->canonicalSizePerBlock(), 4);
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
    const auto& points = resource.prefixCacheRegistration()->points;
    ASSERT_EQ(points.size(), 2u);
    EXPECT_EQ(points[0].endpoint.pos, 4u);
    EXPECT_EQ(points[0].endpoint.index, 1u);
    EXPECT_EQ(points[1].endpoint.pos, 7u);
    EXPECT_EQ(points[1].endpoint.index, 3u);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, PartialTailRegistrationCoversEveryFullUnitBoundary) {
    auto config = test::makeSimpleHybridMhaCacheConfig(4, 12, 2, DataType::TYPE_FP16, 2);
    prepareResourceWithCacheConfig(config, {1, 2, 3, 4, 5, 6, 7}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
#if USING_CUDA
    ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
    const auto& points = resource.prefixCacheRegistration()->points;
    // LINEAR materializes every slot while prefix reuse is on and sparse step
    // checkpoints no longer exist: every full-unit boundary plus the tail.
    ASSERT_EQ(points.size(), 4u);
    EXPECT_EQ(points[0].endpoint.pos, 2u);
    EXPECT_EQ(points[1].endpoint.pos, 4u);
    EXPECT_EQ(points[2].endpoint.pos, 6u);
    EXPECT_EQ(points[3].endpoint.pos, 7u);
#else
    // Precise LINEAR tail snapshots are unavailable outside NVIDIA CUDA.
    EXPECT_FALSE(resource.prefixCacheRegistration().has_value());
#endif
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, PartialTailRegistrationDoesNotAddInternalMatchPosition) {
    std::vector<int> input_tokens(900);
    std::iota(input_tokens.begin(), input_tokens.end(), 1);
    auto config = test::makeSimpleMhaCacheConfig(3, 16, 256, DataType::TYPE_INT8);
    prepareResourceWithCacheConfig(config, input_tokens, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource                                               = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches     = true;
    resource.resource_context_.enable_prefill_partial_tail_reuse = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    // Matching M=650 is an independent later phase, not a registration input.
    stream_->setInitialReuseLength(650);
    ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
    const auto& points = resource.prefixCacheRegistration()->points;
    ASSERT_EQ(points.size(), 4u);
    EXPECT_EQ(points[0].endpoint.pos, 256u);
    EXPECT_EQ(points[1].endpoint.pos, 512u);
    EXPECT_EQ(points[2].endpoint.pos, 768u);
    EXPECT_EQ(points[3].endpoint.pos, 900u);
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, ManagerMallocStampsBeginTimeForInitAndIncr) {
    prepareResourceWithInputTokens({1, 2, 3, 4, 5, 6, 7, 8}, /*reuse_cache=*/false);
    auto&      resource = stream_->streamCacheResource();
    MallocInfo info{resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr()};
    info.reuse_cache         = false;
    info.enable_cache_lookup = false;

    const int64_t legacy_before_us = currentTimeUs();
    const auto    first            = cache_manager_->malloc(info);
    const int64_t legacy_after_us  = currentTimeUs();
    ASSERT_TRUE(first.success);
    EXPECT_GE(first.malloc_begin_time_us, legacy_before_us);
    EXPECT_LE(first.malloc_begin_time_us, legacy_after_us);

    // The manager stamps the incremental attempt itself; the caller cannot
    // override the window.
    stream_->setSeqLength(9);
    const int64_t incr_before_us = currentTimeUs();
    const auto    incremental    = cache_manager_->malloc(info);
    const int64_t incr_after_us  = currentTimeUs();
    ASSERT_TRUE(incremental.success);
    EXPECT_GE(incremental.malloc_begin_time_us, incr_before_us);
    EXPECT_LE(incremental.malloc_begin_time_us, incr_after_us);

    stream_->generate_status_->status = StreamState::FINISHED;
    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, CacheOwnedLayoutPublishesAndFindsSameCanonicalPrefix) {
    auto config = init_cp_config();
    prepareResourceWithCacheConfig(config,
                                   {1, 2, 3, 4, 5, 6, 7, 8},
                                   /*reuse_cache=*/true,
                                   RoleType::PDFUSION,
                                   KVCacheConfig{});
    ASSERT_TRUE(cache_manager_->cacheConfig().request_projection_required);
    auto& first = stream_->streamCacheResource();
    ASSERT_TRUE(first.initKVBlock().ok());
    ASSERT_EQ(first.kvCache().blocksNum(0, "default"), 2);
    const auto full_keys = first.kvCache().cacheKeys(0);
    ASSERT_EQ(full_keys.size(), 4u);
    stream_->generate_status_->status = StreamState::FINISHED;
    stream_->releaseResource();
    EXPECT_EQ(cache_manager_->blockTreeCache()->getKeySnapshot().keys, (CacheKeysType{full_keys[1], full_keys[3]}));

    auto second_input             = std::make_shared<GenerateInput>();
    second_input->input_ids       = torch::tensor(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, torch::kInt32);
    second_input->generate_config = std::make_shared<GenerateConfig>();
    ModelConfig model_config;
    model_config.attn_config.tokens_per_block = 2;
    model_config.max_seq_len                  = 2048;
    auto second_stream                        = std::make_shared<NormalGenerateStream>(
        second_input, model_config, RuntimeConfig{}, stream_->resourceContext(), nullptr);
    second_stream->generate_status_->status = StreamState::RUNNING;
    auto& second                            = second_stream->streamCacheResource();
    ASSERT_TRUE(second.initKVBlock().ok());
    EXPECT_EQ(second_stream->reuseLength(), 8);
    EXPECT_EQ(second.kvCache().blocksNum(0, "default"), 3);
    second_stream->releaseResource();
}

// TEST_F(StreamCacheResourceTest, testFallbackWithFastGen) {
//     prepareResource();
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
//     auto& resource            = stream_->streamCacheResource();
//     stream_->enable_fast_gen_ = true;

//     // first chunk: 分块场景下 current_chunk_len 会被设置为 >0
//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 6);
//     ASSERT_GT(stream_->currentChunkLen(), 0);

//     int old_max_blocks = resource.maxBlockSize();
//     int released       = resource.tryReleaseKVBlock(old_max_blocks);
//     stream_->setPaused();

//     ASSERT_EQ(released, old_max_blocks);
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
//     // fast_gen 模式下，fallback 之后 chunk 长度会被重置为 0
//     ASSERT_EQ(stream_->currentChunkLen(), 0);
// }

// TEST_F(StreamCacheResourceTest, testReleaseSequenceKVCache) {
//     prepareResource();
//     auto& resource = stream_->streamCacheResource();

//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     stream_->setSeqLength(7);
//     stream_->setIsContextStream(false);
//     ASSERT_TRUE(resource.incrKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 3);
//     ASSERT_EQ(resource.maxBlockSize(), 4);

//     auto status = resource.releaseSequenceKVCache(7, 7);
//     ASSERT_TRUE(status.ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
// }

// TEST_F(StreamCacheResourceTest, testQueryLevelReuseCacheControl) {
//     // Test query-level reuse_cache control when engine-level is enabled
//     prepareResource(true);  // Enable engine-level reuse_cache
//     auto& resource = stream_->streamCacheResource();

//     // Test with query-level reuse_cache = true
//     stream_->generate_input_->generate_config->reuse_cache = true;
//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     // Test with query-level reuse_cache = false
//     stream_->releaseResource();
//     // Re-initialize batch resource after release
//     resource.init(stream_->currentBatchSize());
//     size_t baseline_free_blocks                            = cache_manager_->freeBlocksNum();
//     stream_->generate_input_->generate_config->reuse_cache = false;
//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(),
//               baseline_free_blocks >= 3 ? baseline_free_blocks - 3 : baseline_free_blocks);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     stream_->releaseResource();
// }

// TEST_F(StreamCacheResourceTest, testQueryLevelReuseCacheMasterSwitch) {
//     // Test that query-level reuse_cache is ignored when engine-level is disabled
//     prepareResource(false);  // Disable engine-level reuse_cache
//     auto& resource = stream_->streamCacheResource();

//     // Test with query-level reuse_cache = true, but should be ignored
//     stream_->generate_input_->generate_config->reuse_cache = true;
//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     // Test with query-level reuse_cache = false, should also be ignored
//     stream_->releaseResource();
//     // Re-initialize batch resource after release
//     resource.init(stream_->currentBatchSize());
//     stream_->generate_input_->generate_config->reuse_cache = false;
//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     stream_->releaseResource();
// }

TEST_F(StreamCacheResourceTest, testStreamCacheResourceReuseCacheMethod) {
    // engine=true, query=true -> true
    prepareResource(true);
    auto& resource                                         = stream_->streamCacheResource();
    stream_->generate_input_->generate_config->reuse_cache = true;
    ASSERT_TRUE(resource.reuseCache());

    // prefill-only makes reuse ineffective without mutating the requested setting
    stream_->generate_input_->generate_config->max_new_tokens = 0;
    ASSERT_TRUE(stream_->generate_input_->generate_config->reuse_cache);
    ASSERT_FALSE(resource.reuseCache());
    stream_->generate_input_->generate_config->max_new_tokens = 1;
    ASSERT_TRUE(resource.reuseCache());

    // engine=true, query=false -> false
    stream_->generate_input_->generate_config->reuse_cache = false;
    ASSERT_FALSE(resource.reuseCache());

    // engine=false, query=true -> false
    resource.resource_context_.reuse_cache                 = false;
    stream_->generate_input_->generate_config->reuse_cache = true;
    ASSERT_FALSE(resource.reuseCache());

    // engine=false, query=false -> false
    stream_->generate_input_->generate_config->reuse_cache = false;
    ASSERT_FALSE(resource.reuseCache());
}

TEST_F(StreamCacheResourceTest, testPrefillOnlyDisablesPrefixAndDeviceCacheReuseForAllocation) {
    for (const bool ignore_request_switches : {false, true}) {
        SCOPED_TRACE(ignore_request_switches);
        prepareResource(/*reuse_cache=*/true);
        auto& resource = stream_->streamCacheResource();

        stream_->generateConfig()->max_new_tokens                = 0;
        stream_->generateConfig()->num_return_sequences          = 1;
        stream_->generateConfig()->reuse_cache                   = true;
        resource.resource_context_.ignore_request_cache_switches = ignore_request_switches;
        resource.resource_context_.enable_device_cache           = true;
        resource.resource_context_.enable_memory_cache           = true;
        resource.resource_context_.enable_disk_cache             = true;
        resource.resource_context_.enable_remote_cache           = true;

        const auto real_allocator = cache_manager_->coordinator_manager_;
        auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
        cache_manager_->coordinator_manager_ = allocator;
        EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
            .WillOnce(testing::Invoke([real_allocator](const MallocInfo& info) {
                EXPECT_FALSE(info.reuse_cache);
                EXPECT_FALSE(info.enable_cache_lookup);
                return real_allocator->initMallocForCommonLen(info);
            }));
        EXPECT_CALL(*allocator, incrMalloc(testing::_))
            .Times(2)
            .WillRepeatedly(testing::Invoke([real_allocator](const MallocInfo& info) {
                EXPECT_FALSE(info.reuse_cache);
                EXPECT_FALSE(info.enable_cache_lookup);
                return real_allocator->incrMalloc(info);
            }));

        const auto init_status = resource.initKVBlock();
        EXPECT_TRUE(init_status.ok()) << init_status;
        EXPECT_GT(resource.curBlocksNum(), 0);
        EXPECT_TRUE(resource.incrKVBlock().ok());
        EXPECT_FALSE(resource.asyncLoadCache());
        EXPECT_TRUE(resource.waitForAllocatorLoad().ok());
        EXPECT_EQ(resource.allocator_load_context_, nullptr);
        EXPECT_EQ(stream_->reuseLength(), 0);
        EXPECT_EQ(resource.storeTarget(), Tier::NONE);
        EXPECT_TRUE(stream_->generateConfig()->reuse_cache);
        cache_manager_->coordinator_manager_ = real_allocator;
        stream_->releaseResource();
        EXPECT_EQ(cache_manager_->freeBlocksNum(), 8u);
    }
}

TEST_F(StreamCacheResourceTest, testReuseCacheIgnoresPerRequestSwitchWhenConfigured) {
    prepareResource(true);
    auto& resource                                           = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches = true;

    stream_->generate_input_->generate_config->reuse_cache = false;
    ASSERT_TRUE(resource.reuseCache());

    resource.resource_context_.reuse_cache = false;
    ASSERT_FALSE(resource.reuseCache());
}

TEST_F(StreamCacheResourceTest, testCacheLookupIgnoresPerRequestTierSwitches) {
    prepareResource(true);
    auto& resource   = stream_->streamCacheResource();
    auto& request    = *stream_->generate_input_->generate_config;
    auto& deployment = resource.resource_context_;

    request.reuse_cache         = true;
    request.enable_device_cache = false;
    request.enable_memory_cache = false;
    request.enable_disk_cache   = false;
    request.enable_remote_cache = false;

    for (const bool device_on : {false, true}) {
        for (const bool host_on : {false, true}) {
            for (const bool disk_on : {false, true}) {
                for (const bool remote_on : {false, true}) {
                    SCOPED_TRACE("L1=" + std::to_string(device_on) + " L2=" + std::to_string(host_on)
                                 + " L3=" + std::to_string(disk_on) + " remote=" + std::to_string(remote_on));
                    deployment.enable_device_cache = device_on;
                    deployment.enable_memory_cache = host_on;
                    deployment.enable_disk_cache   = disk_on;
                    deployment.enable_remote_cache = remote_on;
                    EXPECT_EQ(resource.enableCacheLookup(), device_on || host_on || disk_on || remote_on);
                }
            }
        }
    }

    deployment.enable_device_cache = true;
    request.reuse_cache            = false;
    EXPECT_FALSE(resource.enableCacheLookup());

    deployment.ignore_request_cache_switches = true;
    EXPECT_TRUE(resource.enableCacheLookup());
}

TEST_F(StreamCacheResourceTest, testStoreTargetUsesDeploymentLocalTiers) {
    prepareResource(true);
    auto& resource   = stream_->streamCacheResource();
    auto& request    = *stream_->generate_input_->generate_config;
    auto& deployment = resource.resource_context_;

    struct DeploymentCase {
        bool device;
        bool host;
        bool disk;
        Tier target;
    };
    const DeploymentCase cases[] = {{false, false, false, Tier::NONE},
                                    {false, false, true, Tier::DISK},
                                    {false, true, false, Tier::HOST},
                                    {false, true, true, Tier::HOST},
                                    {true, false, false, Tier::DEVICE},
                                    {true, false, true, Tier::DEVICE},
                                    {true, true, false, Tier::DEVICE},
                                    {true, true, true, Tier::DEVICE}};
    deployment.reuse_cache       = true;
    for (const bool remote_on : {false, true}) {
        deployment.enable_remote_cache = remote_on;
        SCOPED_TRACE(remote_on);
        for (const auto& test_case : cases) {
            deployment.enable_device_cache = test_case.device;
            deployment.enable_memory_cache = test_case.host;
            deployment.enable_disk_cache   = test_case.disk;
            for (unsigned request_mask = 0; request_mask < 16; ++request_mask) {
                request.enable_device_cache = (request_mask & 1) != 0;
                request.enable_memory_cache = (request_mask & 2) != 0;
                request.enable_disk_cache   = (request_mask & 4) != 0;
                request.enable_remote_cache = (request_mask & 8) != 0;
                for (const bool ignore_request_switches : {false, true}) {
                    SCOPED_TRACE("request_mask=" + std::to_string(request_mask)
                                 + " ignore=" + std::to_string(ignore_request_switches)
                                 + " target=" + std::to_string(static_cast<int>(test_case.target)));
                    deployment.ignore_request_cache_switches = ignore_request_switches;
                    request.reuse_cache                      = true;
                    EXPECT_EQ(resource.storeTarget(), test_case.target);
                    EXPECT_EQ(resource.enableDeviceCache(), test_case.device);
                    EXPECT_EQ(resource.enableMemoryCache(), test_case.host);
                    EXPECT_EQ(resource.enableDiskCache(), test_case.disk);

                    request.reuse_cache = false;
                    EXPECT_EQ(resource.storeTarget(), ignore_request_switches ? test_case.target : Tier::NONE);
                }
            }
        }
    }
}

TEST_F(StreamCacheResourceTest, testStoreTargetPreservesReuseCacheOverride) {
    prepareResource(true);
    auto& resource   = stream_->streamCacheResource();
    auto& request    = *stream_->generate_input_->generate_config;
    auto& deployment = resource.resource_context_;

    deployment.ignore_request_cache_switches = true;
    request.reuse_cache                      = false;
    request.enable_device_cache              = false;
    request.enable_memory_cache              = false;
    request.enable_disk_cache                = false;
    deployment.enable_device_cache           = true;
    deployment.enable_memory_cache           = true;
    deployment.enable_disk_cache             = true;
    EXPECT_EQ(resource.storeTarget(), Tier::DEVICE);

    deployment.enable_device_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::HOST);

    deployment.enable_memory_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::DISK);

    deployment.enable_disk_cache   = false;
    deployment.enable_remote_cache = true;
    EXPECT_EQ(resource.storeTarget(), Tier::NONE);

    deployment.enable_remote_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::NONE);
}

TEST_F(StreamCacheResourceTest, testDecodeInitKVBlock_DisablesDeviceCacheOnlyForFirstMalloc) {
    auto cache_config                                     = test::makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                             /*block_num=*/9,
                                                             /*tokens_per_block=*/2,
                                                             rtp_llm::DataType::TYPE_FP16,
                                                             /*group_layer_num=*/2);
    cache_config.disable_decode_first_malloc_device_reuse = true;
    prepareResourceWithCacheConfig(cache_config, {1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true, RoleType::DECODE);
    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(cache_manager_->cacheConfig().groupNums(), 1);

    // Enable query-level reuse/device cache, but decode initKVBlock should still force device cache off.
    stream_->generate_input_->generate_config->reuse_cache         = true;
    stream_->generate_input_->generate_config->enable_device_cache = true;
    resource.resource_context_.enable_device_cache                 = true;

    auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
    cache_manager_->coordinator_manager_ = allocator;

    testing::InSequence seq;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            EXPECT_FALSE(info.reuse_cache);
            EXPECT_FALSE(info.enable_cache_lookup);
            return {true, 0};
        }));

    EXPECT_CALL(*allocator, incrMalloc(testing::_))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            // initKVBlock should force-disable cache reuse on the first malloc for decode hybrid.
            EXPECT_FALSE(info.reuse_cache);
            EXPECT_FALSE(info.enable_cache_lookup);
            // Simulate a successful allocation so subsequent calls go through incrMalloc path.
            for (int b = 0; b < info.batch_kv_cache_resource->batchSize(); ++b) {
                auto& block_ids = info.batch_kv_cache_resource->mutableBlockIds(b, "linear");
                block_ids.assign(BlockIndicesType{/*block=*/1});
            }
            return {true, 0};
        }))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            // incrKVBlock uses the runtime lookup policy.
            EXPECT_TRUE(info.enable_cache_lookup);
            return {true, 0};
        }));

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.incrKVBlock().ok());
}

TEST_F(StreamCacheResourceTest, testPrefillOnlySkipsBlockTreeLoadsAndStores) {
    for (const bool ignore_request_switches : {false, true}) {
        for (const bool seed_host : {false, true}) {
            SCOPED_TRACE("ignore=" + std::to_string(ignore_request_switches) + " host=" + std::to_string(seed_host));
            auto  backend                = prepareStorageBackendResource(/*block_matches=*/false, seed_host);
            auto& resource               = stream_->streamCacheResource();
            auto  config                 = stream_->generateConfig();
            config->max_new_tokens       = 0;
            config->num_return_sequences = 1;
            config->reuse_cache          = true;
            resource.resource_context_.ignore_request_cache_switches = ignore_request_switches;
            resource.resource_context_.enable_memory_cache           = true;
            resource.resource_context_.enable_disk_cache             = true;

            ASSERT_TRUE(resource.initKVBlock().ok());
            ASSERT_GT(resource.curBlocksNum(), 0);
            EXPECT_FALSE(resource.asyncLoadCache());
            EXPECT_TRUE(resource.waitForAllocatorLoad().ok());
            EXPECT_EQ(resource.allocator_load_context_, nullptr);
            EXPECT_EQ(stream_->reuseLength(), 0);
            EXPECT_EQ(stream_->hostReuseLength(), 0);
            EXPECT_EQ(stream_->diskReuseLength(), 0);
            EXPECT_EQ(stream_->remoteReuseLength(), 0);
            EXPECT_EQ(backend->matchCalls(), 0u);

            const auto real_allocator = cache_manager_->coordinator_manager_;
            auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
            EXPECT_CALL(*allocator, insertIntoCache(testing::_, testing::_)).Times(0);
            EXPECT_CALL(*allocator, free(testing::_)).WillOnce(testing::Invoke([real_allocator](const FreeInfo& info) {
                real_allocator->free(info);
            }));
            cache_manager_->coordinator_manager_ = allocator;
            stream_->generate_status_->status    = StreamState::FINISHED;
            stream_->fillSubGenerateStatus(StreamState::FINISHED);
            stream_->releaseResource();
            cache_manager_->coordinator_manager_ = real_allocator;
            backend->shutdown();

            EXPECT_TRUE(resource.isResourceReleased());
            EXPECT_EQ(resource.curBlocksNum(), 0);
            EXPECT_EQ(cache_manager_->freeBlocksNum(), 8u);
            EXPECT_EQ(backend->writeCalls(), 0u);
            EXPECT_TRUE(config->reuse_cache);
        }
    }
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_WithoutAllocatorContext_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // No allocator-owned load context is in flight.
    ASSERT_FALSE(resource.asyncLoadCache());
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_NoContext_ReturnsTrue) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // No allocator load context means the load phase is immediately done.
    ASSERT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, CacheReuseMetricsSkipWarmupWithoutReporter) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    stream_->setMetricsReporter(nullptr);
    resource.resource_context_.cache_manager.reset();

    resource.reportCacheReuseMetrics();

    EXPECT_FALSE(resource.cache_reuse_metrics_.report_reuse_metrics);
}

TEST_F(StreamCacheResourceTest, CacheReuseMetricsReportWithoutCacheManager) {
    prepareResourceWithInputTokens(std::vector<int>(32, 1), /*reuse_cache=*/true);
    auto&                 resource = stream_->streamCacheResource();
    kmonitor::MetricsTags tags;
    stream_->setMetricsReporter(std::make_shared<kmonitor::MetricsReporter>("", "", tags));
    resource.resource_context_.cache_manager.reset();
    stream_->setInitialReuseLength(20);
    stream_->setLocalReuseLength(12);
    stream_->setHostReuseLength(4);
    stream_->setDiskReuseLength(6);
    stream_->setRemoteReuseLength(8);

    resource.reportCacheReuseMetrics();

    const auto& metrics = resource.cache_reuse_metrics_;
    EXPECT_TRUE(metrics.report_reuse_metrics);
    EXPECT_EQ(metrics.kv_cache_reuse_length, 20);
    EXPECT_EQ(metrics.stream_cache_device_reuse_length, 2);
    EXPECT_EQ(metrics.stream_cache_memory_reuse_length, 4);
    EXPECT_EQ(metrics.stream_cache_disk_reuse_length, 6);
    EXPECT_EQ(metrics.stream_cache_remote_reuse_length, 8);
}

TEST_F(StreamCacheResourceTest, AllocatorReuseRejectsInvalidCountsBeforePublishing) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    for (const size_t total : {size_t{0}, static_cast<size_t>(std::numeric_limits<int>::max())}) {
        auto context = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::DEVICE});
        ASSERT_TRUE(context->success());
        context->matched_blocks_         = total;
        resource.allocator_load_context_ = context;

        EXPECT_ANY_THROW((void)resource.finalizeAllocatorLoad());
        EXPECT_EQ(stream_->reuseLength(), 0);
        resource.allocator_load_context_.reset();
    }
}

TEST_F(StreamCacheResourceTest, testCacheReuseMetricsKeepBlockAlignedInputLength) {
    prepareResource(/*reuse_cache=*/true);
    StreamCacheResource& resource = stream_->streamCacheResource();

    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_EQ(resource.cache_reuse_metrics_.device_input_length, 6);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_metrics);
    EXPECT_FALSE(resource.cache_reuse_metrics_.load_attempted);
    EXPECT_TRUE(resource.cache_reuse_metrics_.report_match_to_ready_latency);
}

TEST_F(StreamCacheResourceTest, testCacheLoadLatencySegmentsCoverMatchToReady) {
    prepareResource(/*reuse_cache=*/true);
    StreamCacheResource& resource             = stream_->streamCacheResource();
    const int64_t        malloc_begin_time_us = currentTimeUs() - 400;
    MallocResult         result{true, 2};
    result.match_cost_time_us      = 100;
    result.match_end_time_us       = malloc_begin_time_us + 100;
    result.malloc_begin_time_us    = malloc_begin_time_us;
    result.load_prepare_latency_us = 300;
    result.load_attempted          = true;

    resource.cache_reuse_metrics_.load_success                  = false;
    resource.cache_reuse_metrics_.report_load_metrics           = true;
    resource.cache_reuse_metrics_.report_load_wait_latency      = true;
    resource.cache_reuse_metrics_.load_wait_latency_us          = 123;
    resource.cache_reuse_metrics_.report_match_to_ready_latency = true;
    resource.cache_reuse_metrics_.match_to_ready_latency_us     = 456;
    resource.recordCacheReuseMallocResult(result);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_metrics);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_wait_latency);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_match_to_ready_latency);
    EXPECT_EQ(resource.cache_reuse_metrics_.load_wait_latency_us, 0);
    EXPECT_EQ(resource.cache_reuse_metrics_.match_to_ready_latency_us, 0);
    resource.allocator_load_context_ = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::DEVICE});
    ASSERT_TRUE(resource.loadCacheDone());

    const RtpLLMCacheReuseMetricsCollector& metrics = resource.cache_reuse_metrics_;
    EXPECT_TRUE(metrics.report_match_cost_time);
    EXPECT_TRUE(metrics.report_load_metrics);
    EXPECT_TRUE(metrics.load_success);
    EXPECT_TRUE(metrics.load_attempted);
    EXPECT_TRUE(metrics.report_load_wait_latency);
    EXPECT_TRUE(metrics.report_match_to_ready_latency);
    EXPECT_EQ(metrics.load_prepare_latency_us, result.load_prepare_latency_us);
    EXPECT_GE(metrics.match_to_ready_latency_us,
              metrics.match_cost_time_us + metrics.load_prepare_latency_us + metrics.load_wait_latency_us);
}

TEST_F(StreamCacheResourceTest, testCacheLoadPrepareFailureHasNoWaitLatency) {
    prepareResource(/*reuse_cache=*/true);
    StreamCacheResource& resource             = stream_->streamCacheResource();
    const int64_t        malloc_begin_time_us = currentTimeUs() - 400;
    MallocResult         result{false, 0};
    result.match_cost_time_us      = 100;
    result.match_end_time_us       = malloc_begin_time_us + 100;
    result.malloc_begin_time_us    = malloc_begin_time_us;
    result.load_prepare_latency_us = 300;
    result.load_attempted          = true;

    resource.recordCacheReuseMallocResult(result);

    const RtpLLMCacheReuseMetricsCollector& metrics = resource.cache_reuse_metrics_;
    EXPECT_TRUE(metrics.report_load_metrics);
    EXPECT_FALSE(metrics.load_success);
    EXPECT_TRUE(metrics.load_attempted);
    EXPECT_FALSE(metrics.report_load_wait_latency);
    EXPECT_TRUE(metrics.report_match_to_ready_latency);
    EXPECT_EQ(metrics.load_prepare_latency_us, result.load_prepare_latency_us);
    EXPECT_GE(metrics.match_to_ready_latency_us, metrics.match_cost_time_us + metrics.load_prepare_latency_us);
}

TEST_F(StreamCacheResourceTest, testCacheLoadFailureKeepsDeviceReuseMetrics) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    StreamCacheResource&         resource = stream_->streamCacheResource();
    kmonitor::MetricsTags        kmon_tags;
    kmonitor::MetricsReporterPtr reporter = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);
    stream_->setMetricsReporter(reporter);

    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);
    stream_->setHostReuseLength(0);
    stream_->setDiskReuseLength(0);
    resource.cache_reuse_metrics_.device_input_length = 6;
    resource.load_wait_begin_time_us_                 = currentTimeUs();
    resource.malloc_begin_time_us_                    = resource.load_wait_begin_time_us_;

    auto load_context = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::HOST});
    ASSERT_TRUE(load_context->completeTransfers(1, false));
    ASSERT_EQ(load_context->mallocStatus(), MallocStatus::NONE);
    resource.allocator_load_context_ = load_context;
    ASSERT_TRUE(resource.loadCacheDone());

    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->initialReuseLength(), 2);
    EXPECT_EQ(stream_->localReuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_EQ(stream_->hostReuseLength(), 0);
    EXPECT_EQ(stream_->diskReuseLength(), 0);

    resource.reportCacheReuseMetrics();

    const RtpLLMCacheReuseMetricsCollector& metrics = resource.cache_reuse_metrics_;
    EXPECT_EQ(metrics.device_input_length, 6);
    EXPECT_EQ(metrics.kv_cache_reuse_length, 2);
    EXPECT_EQ(metrics.stream_cache_device_reuse_length, 2);
    EXPECT_EQ(metrics.stream_cache_memory_reuse_length, 0);
    EXPECT_EQ(metrics.stream_cache_disk_reuse_length, 0);
    EXPECT_TRUE(metrics.report_reuse_metrics);
    EXPECT_TRUE(metrics.report_load_metrics);
    EXPECT_FALSE(metrics.load_success);
}

TEST_F(StreamCacheResourceTest, testReleasePendingAllocatorLoadDoesNotFinalizeMetrics) {
    prepareResource(/*reuse_cache=*/true);
    StreamCacheResource& resource             = stream_->streamCacheResource();
    const int64_t        malloc_begin_time_us = currentTimeUs();
    MallocResult         result{true, 0};
    result.match_end_time_us    = malloc_begin_time_us;
    result.malloc_begin_time_us = malloc_begin_time_us;
    result.load_attempted       = true;
    resource.recordCacheReuseMallocResult(result);
    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false, false);

    resource.releaseResource();

    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_metrics);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_wait_latency);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_match_to_ready_latency);
}

// Serialization contract only: injected binding states below are not evidence
// that a producer completed, a copy succeeded, or a save entered the tree.

TEST_F(StreamCacheResourceTest, testLoadCacheDone_PendingAllocatorLoad_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    auto load_context                = std::make_shared<ImmediateAllocatorContext>(true, false);
    resource.allocator_load_context_ = load_context;

    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_FALSE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, load_context);
}

TEST_F(StreamCacheResourceTest, StorageMatchDefersReusePublicationUntilReadCompletes) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/false,
                                                 /*seed_host=*/false,
                                                 RoleType::PDFUSION,
                                                 /*cp_size=*/0);
    auto& resource = stream_->streamCacheResource();
    backend->blockReads();

    ASSERT_TRUE(resource.initKVBlock().ok());
    backend->waitForReads(1);
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(resource.allocator_load_context_);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), 0u);
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(resource.kvCache().cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).memoryReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).diskReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).storageBackendReuseBlockNum(), 0u);
    EXPECT_FALSE(resource.loadCacheDone());

    backend->releaseReads();
    context->waitDone();
    ASSERT_TRUE(context->success());
    ASSERT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(stream_->reuseLength(), 4);
    EXPECT_EQ(stream_->localReuseLength(), 0);
    EXPECT_EQ(stream_->remoteReuseLength(), 4);
    EXPECT_EQ(resource.kvCache().cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).memoryReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).diskReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).storageBackendReuseBlockNum(), 2u);
}

TEST_F(StreamCacheResourceTest, RemoteQuerySelectingLocalShortLeafKeepsExactDeviceReuse) {
    auto  backend     = prepareStorageBackendResource(/*block_matches=*/false,
                                                 /*seed_host=*/false,
                                                 RoleType::PREFILL,
                                                 /*cp_size=*/0);
    auto& resource    = stream_->streamCacheResource();
    auto  coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1234}), {{{"default", NULL_BLOCK_IDX}}}};
    auto           context = coordinator->create({}, {}, /*matched_blocks=*/0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setLocalFallback(/*token_endpoint=*/5, /*reuse_unit_tokens=*/2);
    context->setMatchCallback([&resource](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 0u);
        EXPECT_TRUE(current.localFallbackSelected());
        // Coordinator transfers the pinned short-leaf references to the
        // request before the Stream publishes the final source counters.
        resource.kvCacheMutable().cacheResource(0).setDeviceReuseBlockNum(2);
        return current.commit();
    });
    resource.allocator_load_context_ = context;
    context->startBackendMatch();
    context->waitDone();
    ASSERT_TRUE(context->success());
    ASSERT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(context->matchedTokenEndpoint(2), 5u);
    EXPECT_EQ(stream_->initialReuseLength(), 5);
    EXPECT_EQ(stream_->localReuseLength(), 5);
    EXPECT_EQ(stream_->deviceReuseLength(), 5);
    EXPECT_EQ(stream_->remoteReuseLength(), 0);
    EXPECT_EQ(resource.kvCache().cacheResource(0).deviceReuseBlockNum(), 2u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).storageBackendReuseBlockNum(), 0u);
    coordinator->shutdown();
}

TEST_F(StreamCacheResourceTest, StorageMatchWithExternalCpRequestLayoutWaitsForRemoteCompletion) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/false,
                                                 /*seed_host=*/false,
                                                 RoleType::PDFUSION,
                                                 /*cp_size=*/2);
    auto& resource = stream_->streamCacheResource();
    backend->blockReads();

    ASSERT_TRUE(resource.initKVBlock().ok());
    backend->waitForReads(1);
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(resource.allocator_load_context_);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_FALSE(resource.loadCacheDone());

    backend->releaseReads();
    context->waitDone();
    ASSERT_TRUE(context->success());
    ASSERT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(stream_->reuseLength(), 4);
    EXPECT_EQ(stream_->remoteReuseLength(), 4);
    EXPECT_EQ(resource.kvCache().cacheResource(0).storageBackendReuseBlockNum(), 1u);
}

TEST_F(StreamCacheResourceTest, ExternalCpRequestLayoutLoadsCanonicalPrefixFromHost) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/false,
                                                 /*seed_host=*/true,
                                                 RoleType::PDFUSION,
                                                 /*cp_size=*/2);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.asyncLoadCache());
    EXPECT_EQ(stream_->reuseLength(), 0);
    ASSERT_TRUE(resource.waitForAllocatorLoad().ok());
    EXPECT_EQ(stream_->reuseLength(), 4);
    EXPECT_EQ(stream_->hostReuseLength(), 4);
    EXPECT_EQ(stream_->remoteReuseLength(), 0);
    EXPECT_EQ(resource.kvCache().cacheResource(0).memoryReuseBlockNum(), 1u);
    EXPECT_EQ(backend->matchCalls(), 0u);
}

TEST_F(StreamCacheResourceTest, ExternalCpRequestLayoutReleaseDuringRemoteReadDoesNotPublishReuse) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/false,
                                                 /*seed_host=*/false,
                                                 RoleType::PDFUSION,
                                                 /*cp_size=*/2);
    auto& resource = stream_->streamCacheResource();
    backend->blockReads();
    const size_t free_before = cache_manager_->freeBlocksNum();
    ASSERT_TRUE(resource.initKVBlock().ok());
    backend->waitForReads(1);
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(resource.allocator_load_context_);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(stream_->reuseLength(), 0);

    resource.releaseResource();
    EXPECT_EQ(resource.curBlocksNum(), 0);
    backend->releaseReads();
    context->waitDone();
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

TEST_F(StreamCacheResourceTest, StorageMatchDoesNotBlockIndependentAllocation) {
    auto  backend        = prepareStorageBackendResource(/*block_matches=*/true);
    auto& first_resource = stream_->streamCacheResource();
    ASSERT_TRUE(first_resource.initKVBlock().ok());
    backend->waitForMatches(1);

    ResourceContext second_context             = stream_->resourceContext();
    second_context.enable_remote_cache         = true;
    auto second_input                          = std::make_shared<GenerateInput>();
    second_input->input_ids                    = torch::tensor(std::vector<int32_t>{7, 8, 9, 10}, torch::kInt32);
    second_input->generate_config              = std::make_shared<GenerateConfig>();
    second_input->generate_config->reuse_cache = true;
    second_input->generate_config->enable_remote_cache = false;
    ModelConfig model_config;
    model_config.attn_config.tokens_per_block = 2;
    model_config.max_seq_len                  = 2048;
    auto second_stream =
        std::make_shared<NormalGenerateStream>(second_input, model_config, RuntimeConfig{}, second_context, nullptr);
    second_stream->generate_status_->status = StreamState::RUNNING;
    second_stream->streamCacheResource().kvCacheMutable().setBatchCacheKeys(0, {400, 500});

    ASSERT_TRUE(second_stream->streamCacheResource().initKVBlock().ok());
    EXPECT_TRUE(second_stream->streamCacheResource().asyncLoadCache());

    backend->releaseMatches();
    ASSERT_TRUE(first_resource.waitForAllocatorLoad().ok());
    ASSERT_TRUE(second_stream->streamCacheResource().waitForAllocatorLoad().ok());
    // Both requests use the deployment backend; check the count after asynchronous loads finish.
    EXPECT_EQ(backend->matchCalls(), 2u);
}

TEST_F(StreamCacheResourceTest, DeferredResultPublishesOnlyDeviceReadyReuseLength) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/true, /*seed_host=*/true);
    auto& resource = stream_->streamCacheResource();

    MallocInfo info;
    info.batch_kv_cache_resource = resource.batch_kv_cache_resource_;
    info.complete_token_ids      = stream_->completeTokenIdsPtr();
    info.reuse_cache             = true;
    info.enable_cache_lookup     = true;
    MallocResult result          = cache_manager_->malloc(info);
    ASSERT_TRUE(result.success);
    backend->waitForMatches(1);
    ASSERT_NE(result.async_context, nullptr);
    EXPECT_EQ(result.reuse_len, 0);

    backend->releaseMatches();
    result.async_context->waitDone();
    ASSERT_TRUE(result.async_context->success());
    cache_manager_->free(FreeInfo{resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr()});
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_CompletedAllocatorLoad_ClearsContext) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    resource.allocator_load_context_ = makeAllocatorLoadContext(/*matched_blocks=*/0, {});
    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadContextGatesReadinessUntilDone) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    auto load_context                = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::HOST});
    resource.allocator_load_context_ = load_context;

    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_FALSE(resource.loadCacheDone());
    EXPECT_TRUE(load_context->completeTransfers(1, true));
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_FALSE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadSuccessCommitsCompleteReuse) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();

    auto load_context =
        makeAllocatorLoadContext(/*matched_blocks=*/3, {Tier::DEVICE, Tier::HOST, Tier::DISK}, /*commit=*/false);
    auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
    cache_manager_->coordinator_manager_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Return(MallocResult{true, /*reuse_len=*/2, 0, load_context}));
    EXPECT_CALL(*allocator, incrMalloc(testing::_)).WillOnce(testing::Return(MallocResult{true, 0}));

    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->hostReuseLength(), 0);
    EXPECT_EQ(stream_->diskReuseLength(), 0);

    EXPECT_TRUE(load_context->completeTransfers(1, true));
    EXPECT_TRUE(load_context->completeTransfers(1, true));
    ASSERT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(stream_->reuseLength(), 6);
    EXPECT_EQ(stream_->initialReuseLength(), 6);
    EXPECT_EQ(stream_->localReuseLength(), 6);
    EXPECT_EQ(stream_->hostReuseLength(), 2);
    EXPECT_EQ(stream_->diskReuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
}

TEST_F(StreamCacheResourceTest, testInitRejectsSuccessfulNonLoadAllocatorContext) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();

    auto context   = std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
    auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
    cache_manager_->coordinator_manager_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Return(MallocResult{true, /*reuse_len=*/0, 0, context}));
    EXPECT_CALL(*allocator, incrMalloc(testing::_)).WillOnce(testing::Return(MallocResult{true, 0}));
    EXPECT_CALL(*allocator, free(testing::_)).Times(1);

    EXPECT_FALSE(resource.initKVBlock().ok());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadSuccessUsesCanonicalReuseUnit) {
    auto cp_config = init_cp_config(/*cp_size=*/2, CpBlockMappingMode::BLOCK_ROUND_ROBIN);
    prepareResourceWithCacheConfig(cp_config,
                                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                   /*reuse_cache=*/true,
                                   RoleType::PREFILL,
                                   KVCacheConfig{});
    auto& resource = stream_->streamCacheResource();
    EXPECT_EQ(cache_manager_->canonicalSizePerBlock(), 4);

    auto load_context = makeAllocatorLoadContext(/*matched_blocks=*/2, {Tier::DEVICE, Tier::HOST}, /*commit=*/false);
    auto allocator    = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
    cache_manager_->coordinator_manager_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Return(MallocResult{true, /*reuse_len=*/4, 0, load_context}));
    EXPECT_CALL(*allocator, incrMalloc(testing::_)).WillOnce(testing::Return(MallocResult{true, 0}));

    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_EQ(stream_->reuseLength(), 4);
    EXPECT_TRUE(load_context->completeTransfers(1, true));
    ASSERT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(stream_->reuseLength(), 8);
    EXPECT_EQ(stream_->initialReuseLength(), 8);
    EXPECT_EQ(stream_->localReuseLength(), 8);
    EXPECT_EQ(stream_->hostReuseLength(), 4);
    EXPECT_EQ(stream_->deviceReuseLength(), 4);
}

TEST_F(StreamCacheResourceTest, ShardedRequestUsesCacheBoundReuseUnit) {
    prepareResourceWithCacheConfig(init_cp_config(), {1, 2, 3, 4}, /*reuse_cache=*/false, RoleType::PREFILL);
    EXPECT_EQ(stream_->streamCacheResource().reuseBlockTokens(), 4);
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadPendingPublishesZeroDeviceReadyReuse) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(6);
    stream_->setHostReuseLength(2);
    stream_->setDiskReuseLength(2);

    auto load_context = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::HOST}, /*commit=*/false);
    auto allocator    = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
    cache_manager_->coordinator_manager_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Return(MallocResult{true, /*reuse_len=*/0, 0, load_context}));
    EXPECT_CALL(*allocator, incrMalloc(testing::_)).WillOnce(testing::Return(MallocResult{true, 0}));

    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_EQ(resource.allocator_load_context_, load_context);
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(stream_->initialReuseLength(), 0);
    EXPECT_EQ(stream_->localReuseLength(), 0);
    EXPECT_EQ(stream_->deviceReuseLength(), 0);
    EXPECT_EQ(stream_->hostReuseLength(), 0);
    EXPECT_EQ(stream_->diskReuseLength(), 0);
}

TEST_F(StreamCacheResourceTest, testPrefillAllocatorLoadFailureKeepsDeviceReadyReuse) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);
    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);

    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->initialReuseLength(), 2);
    EXPECT_EQ(stream_->localReuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_EQ(stream_->hostReuseLength(), 0);
    EXPECT_EQ(stream_->diskReuseLength(), 0);
    EXPECT_FALSE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testPrefillLoadTimeoutDetachesFrozenSuffixAndRecomputes) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    const auto original = resource.kvCache().blocks(0, "default");
    ASSERT_EQ(original.size(), 3u);
    const auto pool = cache_manager_->coordinator_manager_->groupBlockPools().front();
    pool->freezeBlock(original[1]);
    stream_->setReuseLength(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);
    resource.allocator_load_context_ =
        std::make_shared<CompletedAsyncContext>(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "KVCM read timed out"));

    EXPECT_TRUE(resource.waitForAllocatorLoad().ok());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_NE(resource.kvCache().blocks(0, "default")[1], original[1]);
    EXPECT_TRUE(pool->isFrozen(original[1]));
    EXPECT_TRUE(resource.cache_reuse_metrics_.report_load_metrics);
    EXPECT_FALSE(resource.cache_reuse_metrics_.load_success);
}

TEST_F(StreamCacheResourceTest, testPrefillLoadFailureDetachesSharedSuffix) {
    for (bool cache_owned : {false, true}) {
        SCOPED_TRACE(cache_owned);
        prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
        auto& resource = stream_->streamCacheResource();
        ASSERT_TRUE(resource.initKVBlock().ok());
        auto&      batch    = resource.kvCacheMutable();
        const auto original = batch.blocks(0, "default");
        ASSERT_EQ(original.size(), 3u);
        const auto pool = cache_manager_->coordinator_manager_->groupBlockPools().front();
        // Keep the device-ready prefix shared; only the writable suffix must detach.
        pool->incTreeRef(original[0], BlockTreeRefType::CACHE);
        if (cache_owned) {
            pool->incTreeRef(original[1], BlockTreeRefType::CACHE);
        } else {
            pool->incRef(original[1]);  // Another request joined the failed load.
        }
        stream_->setReuseLength(2);
        stream_->setInitialReuseLength(2);
        stream_->setLocalReuseLength(2);
        resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);

        ASSERT_TRUE(resource.loadCacheDone());
        EXPECT_FALSE(stream_->hasError());
        EXPECT_EQ(stream_->reuseLength(), 2);
        const auto replacement = batch.blocks(0, "default")[1];
        EXPECT_NE(replacement, original[1]);
        for (int b = 0; b < batch.batchSize(); ++b) {
            EXPECT_EQ(batch.blocks(b, "default")[0], original[0]);
            EXPECT_EQ(batch.blocks(b, "default")[1], replacement);
            EXPECT_EQ(batch.blocks(b, "default")[2], original[2]);
            EXPECT_EQ(batch.mutableBlockIds(b, "default").kernelBlocks()[1], replacement);
        }
        EXPECT_EQ(pool->refCount(original[1]), 1u);
        EXPECT_EQ(pool->refCount(replacement), batch.batchSize());
        pool->decTreeRef(original[0], BlockTreeRefType::CACHE);
        if (cache_owned) {
            pool->decTreeRef(original[1], BlockTreeRefType::CACHE);
        } else {
            pool->decRef(original[1]);
        }
        stream_->releaseResource();
        EXPECT_EQ(cache_manager_->freeBlocksNum(), 8u);
    }
}

TEST_F(StreamCacheResourceTest, testPrefillLoadFailureDetachShortfallRearmsAllocation) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    const auto blocks = resource.kvCache().blocks(0, "default");
    const auto pool   = cache_manager_->coordinator_manager_->groupBlockPools().front();
    pool->incTreeRef(blocks[1], BlockTreeRefType::CACHE);
    pool->incTreeRef(blocks[2], BlockTreeRefType::CACHE);
    // Leave room for only one of the two replacements; the batch cannot fit.
    auto held_blocks = pool->malloc(pool->freeBlocksNum() - 1);
    ASSERT_TRUE(held_blocks.has_value());
    pool->incRef(*held_blocks);
    stream_->setReuseLength(2);
    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);
    stream_->reportEvent(StreamEvents::CanRun);
    stream_->reportEvent(StreamEvents::LoadInitiated);
    stream_->generate_status_->status = StreamState::LOADING_CACHE;

    EXPECT_EQ(stream_->moveToNext(), StreamState::WAITING);
    EXPECT_FALSE(stream_->hasEvent(StreamEvents::LoadInitiated));
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(resource.mallocFailedTimes(), 1);
    EXPECT_EQ(pool->refCount(blocks[1]), 1u);
    EXPECT_EQ(pool->refCount(blocks[2]), 1u);
    EXPECT_EQ(pool->freeBlocksNum(), 2u);  // Unused free block plus released prefix.
    pool->decRef(*held_blocks);
    pool->decTreeRef(blocks[1], BlockTreeRefType::CACHE);
    pool->decTreeRef(blocks[2], BlockTreeRefType::CACHE);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), 8u);
    ASSERT_TRUE(resource.initKVBlock().ok());
}

TEST_F(StreamCacheResourceTest, testPrefillMaterializationShortfallRearmsAllocatorLoad) {
    auto backend = prepareStorageBackendResource(/*block_matches=*/false, /*seed_host=*/false, RoleType::PREFILL);
    kmonitor::MetricsTags tags;
    auto                  reporter = std::make_shared<kmonitor::MetricsReporter>("", "", tags);
    stream_->setMetricsReporter(reporter);
    auto* metrics = reporter->getMetricsGroup<RtpLLMCacheOperationMetrics>();
    ASSERT_NE(metrics, nullptr);
    ASSERT_NE(metrics->malloc_retry_qps_metric, nullptr);
    auto expect_retry_qps = [&](double expected) {
        auto* series = metrics->malloc_retry_qps_metric->DeclareMetric(&tags);
        ASSERT_NE(series, nullptr);
        kmonitor::MetricsRecord record(nullptr, nullptr, 0);
        series->Snapshot(&record, 1000);
        EXPECT_TRUE(metrics->malloc_retry_qps_metric->UndeclareMetric(series));
        ASSERT_EQ(record.Values().size(), 1u);
        EXPECT_DOUBLE_EQ(std::stod(record.Values().front()->Value()), expected);
    };
    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.resourceContext().role_type, RoleType::PREFILL);

    struct LoadCounts {
        size_t commits = 0;
        size_t aborts  = 0;
    };
    auto counts      = std::make_shared<LoadCounts>();
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [counts](const std::shared_ptr<LoadAsyncContext>&) {
            ++counts->commits;
            return true;
        },
        [counts](LoadAsyncContext&) { ++counts->aborts; });
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1234}), {{{"default", NULL_BLOCK_IDX}}}};
    auto           context = coordinator->create({}, {}, /*matched_blocks=*/0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback(
        [](LoadAsyncContext&, size_t) { return LoadMatchResult{false, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED}; });

    resource.allocator_load_context_ = context;
    stream_->reportEvent(StreamEvents::CanRun);
    stream_->reportEvent(StreamEvents::LoadInitiated);
    stream_->generate_status_->status = StreamState::LOADING_CACHE;
    resource.malloc_failed_times_     = 9;
    EXPECT_FALSE(resource.loadCacheDone());
    EXPECT_FALSE(resource.loadCacheDone());
    ASSERT_NO_FATAL_FAILURE(expect_retry_qps(0));
    context->startBackendMatch();
    context->waitDone();
    ASSERT_FALSE(context->success());

    EXPECT_EQ(stream_->moveToNext(), StreamState::WAITING);
    // Admission ownership belongs to the scheduler: ordinary loading streams
    // clear CanRun when they move back to WAITING, while explicit groups keep
    // their admission and retry in the group lane.
    EXPECT_TRUE(stream_->hasEvent(StreamEvents::CanRun));
    EXPECT_FALSE(stream_->hasEvent(StreamEvents::LoadInitiated));
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(resource.mallocFailedTimes(), 10);
    ASSERT_NO_FATAL_FAILURE(expect_retry_qps(1));
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_TRUE(resource.loadCacheDone());
    ASSERT_NO_FATAL_FAILURE(expect_retry_qps(0));
    EXPECT_EQ(counts->commits, 0u);
    EXPECT_EQ(counts->aborts, 1u);

    auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
    cache_manager_->coordinator_manager_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Invoke([](const MallocInfo& info) -> MallocResult {
            EXPECT_FALSE(info.verbose);
            return {true, 0};
        }));
    EXPECT_CALL(*allocator, incrMalloc(testing::_))
        .WillOnce(testing::Invoke([](const MallocInfo& info) -> MallocResult {
            info.batch_kv_cache_resource->mutableBlockIds(0, "default").assign({1});
            return {true, 0};
        }));

    EXPECT_EQ(stream_->moveToNext(), StreamState::RUNNING);
    EXPECT_TRUE(stream_->hasEvent(StreamEvents::LoadInitiated));
    ASSERT_NO_FATAL_FAILURE(expect_retry_qps(0));
    EXPECT_EQ(resource.curBlocksNum(), 1);
    coordinator->shutdown();
}

TEST_F(StreamCacheResourceTest, testPrefillPermanentMaterializationFailureTerminates) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/false, /*seed_host=*/false, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.resourceContext().role_type, RoleType::PREFILL);

    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{5678}), {{{"default", NULL_BLOCK_IDX}}}};
    auto           context = coordinator->create({}, {}, /*matched_blocks=*/0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback(
        [](LoadAsyncContext&, size_t) { return LoadMatchResult{false, MallocStatus::PERMANENT_RESOURCE_EXHAUSTED}; });

    resource.allocator_load_context_ = context;
    stream_->reportEvent(StreamEvents::CanRun);
    stream_->reportEvent(StreamEvents::LoadInitiated);
    stream_->generate_status_->status = StreamState::LOADING_CACHE;
    context->startBackendMatch();
    context->waitDone();

    EXPECT_EQ(stream_->moveToNext(), StreamState::WAITING);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    coordinator->shutdown();
}

TEST_F(StreamCacheResourceTest, testPrefillCoordinatorCommitFailureTerminates) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/false, /*seed_host=*/false, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.resourceContext().role_type, RoleType::PREFILL);

    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return false; }, [](LoadAsyncContext&) {});
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{9012}), {{{"default", NULL_BLOCK_IDX}}}};
    auto           context = coordinator->create({}, {}, /*matched_blocks=*/0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([](LoadAsyncContext& current, size_t) { return current.commit(); });

    resource.allocator_load_context_ = context;
    stream_->reportEvent(StreamEvents::CanRun);
    stream_->reportEvent(StreamEvents::LoadInitiated);
    stream_->generate_status_->status = StreamState::LOADING_CACHE;
    context->startBackendMatch();
    context->waitDone();
    ASSERT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);

    EXPECT_EQ(stream_->moveToNext(), StreamState::WAITING);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    coordinator->shutdown();
}

TEST_F(StreamCacheResourceTest, testPrefillWaitForAllocatorLoadFailureKeepsDeviceReadyReuse) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);
    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);

    EXPECT_TRUE(resource.waitForAllocatorLoad().ok());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_FALSE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadFailureIsTerminal) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);

    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);
    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(stream_->initialReuseLength(), 0);
    EXPECT_EQ(stream_->localReuseLength(), 0);
    EXPECT_EQ(stream_->deviceReuseLength(), 0);
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadTimeoutPreservesDeadlineStatus) {
    for (const auto role : {RoleType::DECODE, RoleType::PDFUSION}) {
        SCOPED_TRACE(static_cast<int>(role));
        prepareResource(/*reuse_cache=*/true, role);
        auto& resource = stream_->streamCacheResource();
        resource.allocator_load_context_ =
            std::make_shared<CompletedAsyncContext>(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "KVCM read timed out"));

        const auto status = resource.waitForAllocatorLoad();
        EXPECT_TRUE(absl::IsDeadlineExceeded(status));
        EXPECT_NE(std::string(status.message()).find("KVCM read timed out"), std::string::npos);
        EXPECT_EQ(resource.allocator_load_context_, nullptr);
        EXPECT_TRUE(resource.cache_reuse_metrics_.report_load_metrics);
        EXPECT_FALSE(resource.cache_reuse_metrics_.load_success);
    }
}

TEST_F(StreamCacheResourceTest, testLoadCacheDoneTimeoutReportsLoadCacheTimeout) {
    prepareResource(/*reuse_cache=*/true, RoleType::DECODE);
    auto& resource = stream_->streamCacheResource();
    resource.allocator_load_context_ =
        std::make_shared<CompletedAsyncContext>(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "KVCM read timed out"));

    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_TRUE(stream_->hasError());
    EXPECT_EQ(stream_->statusInfo().code(), ErrorCode::LOAD_CACHE_TIMEOUT);
    EXPECT_NE(stream_->statusInfo().ToString().find("KVCM read timed out"), std::string::npos);
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
}

TEST_F(StreamCacheResourceTest, testReleaseResetsAllocatorContextBeforeFreeingRequestBlocks) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_GT(resource.curBlocksNum(), 0);
    ASSERT_LT(cache_manager_->freeBlocksNum(), 8u);

    // The observer may outlive this test body if releaseResource regresses.
    // Avoid references to stack state or fixture members in that failure path.
    struct ReleaseObservation {
        bool context_destroyed = false;
        bool blocks_present    = false;
    };
    auto                          observed    = std::make_shared<ReleaseObservation>();
    std::weak_ptr<GenerateStream> weak_stream = stream_;
    resource.allocator_load_context_ =
        std::make_shared<DestructionObserverContext>([observed, weak_stream, manager = cache_manager_] {
            observed->context_destroyed = true;
            const auto stream           = weak_stream.lock();
            observed->blocks_present =
                stream && stream->streamCacheResource().curBlocksNum() > 0 && manager->freeBlocksNum() < 8u;
        });
    std::weak_ptr<AsyncContext> weak_context = resource.allocator_load_context_;

    stream_->releaseResource();

    EXPECT_TRUE(observed->context_destroyed);
    EXPECT_TRUE(observed->blocks_present);
    EXPECT_TRUE(weak_context.expired());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), 8u);
}

TEST_F(StreamCacheResourceTest, PollAllocatorLoadPreservesRetryableMaterializationStatus) {
    auto  backend     = prepareStorageBackendResource(false, false, RoleType::DECODE);
    auto& resource    = stream_->streamCacheResource();
    auto  coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1234}), {{{"default", NULL_BLOCK_IDX}}}};
    auto           context = coordinator->create({}, {}, 0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback(
        [](LoadAsyncContext&, size_t) { return LoadMatchResult{false, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED}; });
    resource.allocator_load_context_ = context;
    EXPECT_FALSE(resource.pollAllocatorLoad().has_value());
    context->startBackendMatch();
    context->waitDone();
    const auto status = resource.pollAllocatorLoad();
    ASSERT_TRUE(status.has_value());
    EXPECT_TRUE(absl::IsUnavailable(*status));
    EXPECT_FALSE(stream_->hasEvent(StreamEvents::LoadInitiated));
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
}

TEST_F(StreamCacheResourceTest, PollAllocatorLoadCompletesOnlyAfterTransfersSettle) {
    prepareResource(true);
    auto& resource                   = stream_->streamCacheResource();
    auto  context                    = makeAllocatorLoadContext(1, {Tier::HOST});
    resource.allocator_load_context_ = context;
    EXPECT_FALSE(resource.pollAllocatorLoad().has_value());
    ASSERT_TRUE(context->completeTransfers(1, true));
    auto status = resource.pollAllocatorLoad();
    ASSERT_TRUE(status.has_value());
    EXPECT_TRUE(status->ok());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
}

}  // namespace rtp_llm
