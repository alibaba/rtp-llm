
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
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

class CountingReadyContext: public AsyncContext {
public:
    void waitDone() override {
        ++wait_calls_;
    }
    bool done() const override {
        return true;
    }
    bool success() const override {
        return true;
    }
    size_t waitCalls() const {
        return wait_calls_;
    }

private:
    size_t wait_calls_{0};
};

template<typename EngineType>
std::shared_ptr<EngineType> createFocusedEngine(int64_t device_min_free_blocks,
                                                int64_t max_context_batch_size = 128,
                                                int64_t max_batch_tokens       = 4096) {
    CustomConfig  config;
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    config.reuse_cache = true;
    auto params        = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.kv_cache_config.device_cache_min_free_blocks                = device_min_free_blocks;
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

TEST_F(SystemPromptConstructorTest, testSecondTaskFailureReleasesEarlierRequestOwnership) {
    auto engine  = createFocusedEngine<FailSecondPreRunEngine>(/*device_min_free_blocks=*/1);
    auto manager = engine->resourceContext().cache_manager;
    ASSERT_NE(manager->blockTreeCache(), nullptr);
    const size_t free_before = manager->freeBlocksNum();

    KVCacheConfig config;
    config.multi_task_prompt_tokens = {
        {"1", {1, 2, 3}},
        {"2", {1, 2, 4}},
    };

    SystemPromptConstructor constructor;
    const auto result = constructor.construct(config, engine.get(), manager.get(), /*insert_kv_cache=*/true);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.status().message().find("injected second system prompt failure"), std::string::npos);

    // The first task was already inserted, but its request ownership is not
    // committed until every task succeeds. On failure only its tree holder
    // remains; the partial tail and all request refs are released.
    EXPECT_EQ(manager->freeBlocksNum(), free_before - 1);
    ASSERT_EQ(manager->blockTreeCache()->componentGroups().size(), 1u);
    EXPECT_EQ(manager->blockTreeCache()->componentGroups().front()->devicePools().front()->activeTreeCachedBlocksNum(),
              0u);
    EXPECT_EQ(manager->blockTreeCache()->getStats().device_heap_total_size, 1u);

    EXPECT_EQ(manager->blockTreeCache()->reclaimBlocks(/*num_blocks=*/100, Tier::DEVICE), 1);
    EXPECT_EQ(manager->freeBlocksNum(), free_before);
}

TEST_F(SystemPromptConstructorTest, testNormalEngineResolvesAbsoluteDeviceReserveBeforeManagerInit) {
    auto engine = createFocusedEngine<NormalEngine>(
        /*device_min_free_blocks=*/0, /*max_context_batch_size=*/3, /*max_batch_tokens=*/17);

    // max_prefill=min(3*20,17)=17 tokens and block width is 2, so the
    // resolved absolute headroom is ceil(17/2)=9 blocks.
    EXPECT_EQ(engine->kv_cache_config.device_cache_min_free_blocks, 9);
    ASSERT_NE(engine->resourceContext().cache_manager, nullptr);
    ASSERT_NE(engine->resourceContext().cache_manager->blockTreeCache(), nullptr);
    EXPECT_EQ(engine->resourceContext().cache_manager->blockTreeCache()->config().device_min_free_blocks, 9u);
}

TEST_F(SystemPromptConstructorTest, testNormalEngineWaitsForAllocatorObserverBeforeSystemPromptExecution) {
    auto engine         = createFocusedEngine<NormalEngine>(/*device_min_free_blocks=*/1);
    auto manager        = engine->resourceContext().cache_manager;
    auto real_allocator = manager->allocator_;
    auto context        = std::make_shared<CountingReadyContext>();
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
