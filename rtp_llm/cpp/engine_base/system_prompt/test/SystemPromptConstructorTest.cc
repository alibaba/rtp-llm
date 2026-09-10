
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

using namespace std;

namespace rtp_llm {

namespace {

class FailSecondPreRunEngine: public NormalEngine {
public:
    using NormalEngine::NormalEngine;

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& input, preRunMode mode) override {
        if (pre_run_calls_++ == 1) {
            return absl::InternalError("injected second system prompt failure");
        }
        return NormalEngine::preRun(input, mode);
    }

private:
    size_t pre_run_calls_{0};
};

class CountingReadyContext: public LoadAsyncContext {
public:
    static std::shared_ptr<CountingReadyContext> create() {
        auto coordinator = std::make_shared<LoadContextCoordinator>(
            [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
        auto context = std::shared_ptr<CountingReadyContext>(new CountingReadyContext(coordinator));
        EXPECT_TRUE(coordinator->registerContext(context));
        return context;
    }

    void waitDone() override {
        ++wait_calls_;
        LoadAsyncContext::waitDone();
    }
    size_t waitCalls() const {
        return wait_calls_;
    }

private:
    explicit CountingReadyContext(const std::shared_ptr<LoadContextCoordinator>& coordinator):
        LoadAsyncContext({}, {}, /*matched_blocks=*/0, /*context_id=*/1, coordinator) {}

    size_t wait_calls_{0};
};

template<typename EngineType>
std::shared_ptr<EngineType> createFocusedEngine(int64_t max_context_batch_size = 128, int64_t max_batch_tokens = 4096) {
    CustomConfig  config;
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    config.reuse_cache = true;
    auto params        = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.runtime_config.fifo_scheduler_config.max_context_batch_size = max_context_batch_size;
    params.runtime_config.fifo_scheduler_config.max_batch_tokens_size  = max_batch_tokens;

    NormalExecutor::test_model_factory = [vocab_size = model_config.vocab_size](const GptModelInitParams&) {
        return std::unique_ptr<ModelBase>(new MockModel(vocab_size));
    };
    auto engine                        = std::make_shared<EngineType>(params, nullptr);
    NormalExecutor::test_model_factory = nullptr;
    return engine;
}

std::shared_ptr<GenerateInput> makeSystemPromptInput() {
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    return input;
}

}  // namespace

class SystemPromptConstructorTest: public DeviceTestBase {};

TEST_F(SystemPromptConstructorTest, testMultiTaskPromptConstruct) {
    SystemPromptConstructor constructor;
    KVCacheConfig           kv_cache_config;
    vector<int>             prompt_1         = {1, 2, 3};
    vector<int>             prompt_2         = {4, 5, 6, 7};
    kv_cache_config.multi_task_prompt_tokens = {{"1", prompt_1}, {"2", prompt_2}};
    CustomConfig config;
    auto         engine = createMockEngine(config);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlocksNum(), 99);
    const_cast<ResourceContext*>(&engine->resourceContext())->reuse_cache = true;
    auto result_status =
        constructor.construct(kv_cache_config, engine.get(), engine->resourceContext().cache_manager.get(), true);
    ASSERT_EQ(result_status.ok(), true);
    auto result = result_status.value();
    ASSERT_EQ(result.size(), 2);
    // TODO(chanyin): last partial block will be wasted when need_release_resource is false
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlocksNum(), 95);  // 99 - (2 + 1) cached - 1 wasted

    const auto& item1 = result["1"];
    ASSERT_EQ(item1.prompt_tokens.size(), 3);
    ASSERT_TRUE(!item1.block_ids.empty());
    ASSERT_EQ(item1.prompt_tokens, prompt_1);

    const auto& item2 = result["2"];
    ASSERT_EQ(item2.prompt_tokens.size(), 4);
    ASSERT_TRUE(!item2.block_ids.empty());
    ASSERT_EQ(item2.prompt_tokens, prompt_2);
}

TEST_F(SystemPromptConstructorTest, testResidentInsertFailureFailsWarmupAndReleasesRequestOwnership) {
    const std::shared_ptr<NormalEngine>   engine         = createFocusedEngine<NormalEngine>();
    const std::shared_ptr<KVCacheManager> engine_manager = engine->resourceContext().cache_manager;
    const DeviceBlockPoolPtr pool = engine_manager->blockTreeCache()->groupSets().front()->devicePools().front();
    const size_t             request_blocks_before = pool->referencedBlocksNum();
    auto                     insert_manager = std::make_shared<KVCacheManager>(engine_manager->cacheConfig(), true);
    auto allocator = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(insert_manager->cacheConfig());
    insert_manager->allocator_ = allocator;
    EXPECT_CALL(*allocator, insertIntoCache(testing::_, testing::_))
        .WillOnce(testing::Invoke([](const InsertInfo& info, size_t& resident_prefix_length) {
            EXPECT_TRUE(info.is_resident);
            resident_prefix_length = 0;
        }));
    KVCacheConfig config;
    config.multi_task_prompt_tokens = {{"blocked_prompt", {1, 2, 3}}};
    SystemPromptConstructor                                                   constructor;
    const absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> result =
        constructor.construct(config, engine.get(), insert_manager.get(), true);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_NE(result.status().message().find("blocked_prompt"), std::string::npos);
    EXPECT_EQ(pool->referencedBlocksNum(), request_blocks_before);
}

TEST_F(SystemPromptConstructorTest, testSecondTaskFailureReleasesEarlierRequestOwnership) {
    auto engine  = createFocusedEngine<FailSecondPreRunEngine>();
    auto manager = engine->resourceContext().cache_manager;
    ASSERT_NE(manager->blockTreeCache(), nullptr);
    const size_t free_before      = manager->freeBlocksNum();
    const size_t available_before = manager->availableBlocksNum();

    KVCacheConfig config;
    config.multi_task_prompt_tokens = {
        {"1", {1, 2, 3}},
        {"2", {1, 2, 4}},
    };

    SystemPromptConstructor constructor;
    const auto result = constructor.construct(config, engine.get(), manager.get(), /*insert_kv_cache=*/true);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.status().message().find("injected second system prompt failure"), std::string::npos);

    // Failure releases request refs and the partial tail; the first prompt
    // remains resident with only the tree's CACHE ownership.
    EXPECT_EQ(manager->freeBlocksNum(), free_before - 1);
    ASSERT_EQ(manager->blockTreeCache()->groupSets().size(), 1u);
    EXPECT_EQ(manager->availableBlocksNum(), available_before);
    const DeviceBlockPoolPtr& pool = manager->blockTreeCache()->groupSets().front()->devicePools().front();
    EXPECT_EQ(pool->referencedBlocksNum(), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
    EXPECT_EQ(manager->blockTreeCache()->getStats().device_heap_total_size, 0u);

    EXPECT_EQ(block_tree_cache_test::BlockTreeCacheTestPeer::reclaimBlocksForTest(
                  *manager->blockTreeCache(), /*num_blocks=*/100, Tier::DEVICE),
              0);
    EXPECT_EQ(manager->freeBlocksNum(), free_before - 1);
}

TEST_F(SystemPromptConstructorTest, testNormalEnginePreservesSchedulerReserveWithoutPrefillOverride) {
    auto engine = createFocusedEngine<NormalEngine>(/*max_context_batch_size=*/3, /*max_batch_tokens=*/17);

    auto manager = engine->resourceContext().cache_manager;
    ASSERT_NE(manager, nullptr);
    ASSERT_EQ(manager->freeBlocksNum(), 99u);
    EXPECT_EQ(manager->reserveBlocksNum(), 4u);

    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(manager->cacheConfig().topologyPtr());
    auto input       = makeSystemPromptInput();
    input->input_ids = torch::arange(190, torch::kInt32);
    auto tokens      = std::make_shared<CompleteTokenIds>(1, 1, 198, manager->cacheConfig().seq_size_per_block);
    tokens->init(input);
    MallocInfo info{resource, tokens};
    info.reuse_cache         = false;
    info.enable_cache_lookup = false;

    ASSERT_TRUE(manager->malloc(info).success);
    EXPECT_EQ(resource->blocksNum(0, 0), 95);
    EXPECT_EQ(manager->freeBlocksNum(), 4u);

    tokens->setSeqLength(198);
    ASSERT_TRUE(manager->malloc(info).success);
    EXPECT_EQ(resource->blocksNum(0, 0), 99);
    EXPECT_EQ(manager->freeBlocksNum(), 0u);
    manager->free(FreeInfo{resource, tokens});
    EXPECT_EQ(manager->freeBlocksNum(), 99u);
}

TEST_F(SystemPromptConstructorTest, testNormalEngineWaitsForAllocatorObserverBeforeSystemPromptExecution) {
    auto engine         = createFocusedEngine<NormalEngine>();
    auto manager        = engine->resourceContext().cache_manager;
    auto real_allocator = manager->allocator_;
    auto context        = CountingReadyContext::create();
    auto mock_allocator = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(manager->config_);

    ON_CALL(*mock_allocator, initMallocForCommonLen(testing::_))
        .WillByDefault(testing::Return(MallocResult{true, 0, 0, context}));
    ON_CALL(*mock_allocator, incrMalloc(testing::_)).WillByDefault(testing::Invoke([&](const MallocInfo& info) {
        return real_allocator->malloc(info);
    }));
    ON_CALL(*mock_allocator, free(testing::_)).WillByDefault(testing::Invoke([&](const FreeInfo& info) {
        real_allocator->free(info);
    }));
    ON_CALL(*mock_allocator, convertIndexToAddr(testing::_, testing::_))
        .WillByDefault(testing::Invoke(
            [&](int layer_id, int block_id) { return real_allocator->convertIndexToAddr(layer_id, block_id); }));
    ON_CALL(*mock_allocator, convertIndexToBuffer(testing::_, testing::_))
        .WillByDefault(testing::Invoke(
            [&](int layer_id, int block_id) { return real_allocator->convertIndexToBuffer(layer_id, block_id); }));
    ON_CALL(*mock_allocator, convertIndexToBuffer(testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](int layer_id, int block_id, int partition_count, int partition_id) {
            return real_allocator->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
        }));
    ON_CALL(*mock_allocator, allLayerCacheBase()).WillByDefault(testing::Invoke([&] {
        return real_allocator->allLayerCacheBase();
    }));
    ON_CALL(*mock_allocator, seqSizePerBlock()).WillByDefault(testing::Invoke([&] {
        return real_allocator->seqSizePerBlock();
    }));
    ON_CALL(*mock_allocator, singleBatchNeedBlocks(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const BatchKVCacheResourcePtr& resource, int seq_len, int reserve_step) {
            return real_allocator->singleBatchNeedBlocks(resource, seq_len, reserve_step);
        }));

    manager->allocator_ = mock_allocator;
    auto stream_status  = engine->preRun(makeSystemPromptInput(), preRunMode::build_system_prompt);
    ASSERT_TRUE(stream_status.ok()) << stream_status.status();
    EXPECT_EQ(context->waitCalls(), 1u);
    EXPECT_EQ(stream_status.value()->streamCacheResource().allocator_load_context_, nullptr);

    stream_status.value().reset();
    manager->allocator_ = real_allocator;
}

}  // namespace rtp_llm
