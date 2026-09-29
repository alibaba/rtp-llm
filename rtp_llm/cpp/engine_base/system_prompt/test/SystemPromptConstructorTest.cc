
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockCoordinatorCacheManager.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cuda_runtime.h>

#include <algorithm>
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
    static std::shared_ptr<CountingReadyContext> create(bool fail_load = false) {
        auto coordinator = std::make_shared<LoadContextCoordinator>(
            [fail_load](const std::shared_ptr<LoadAsyncContext>& context) {
                if (fail_load) {
                    context->onTaskFail();
                }
                return true;
            }, [](LoadAsyncContext&) {});
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
std::shared_ptr<EngineType>
createFocusedEngine(int64_t max_context_batch_size = 128, int64_t max_batch_tokens = 4096, bool multi_group = false) {
    CustomConfig  config;
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    config.reuse_cache = true;
    auto params        = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.runtime_config.fifo_scheduler_config.max_context_batch_size = max_context_batch_size;
    params.runtime_config.fifo_scheduler_config.max_batch_tokens_size  = max_batch_tokens;
    if (multi_group) {
        params.model_config_.kv_cache_spec_descs[0][0].tag = "first";
        auto& second                                       = params.model_config_.kv_cache_spec_descs[1][0];
        second.tag                                         = "second";
        second.group_type                                  = CacheGroupType::SWA;
        second.reuse                                       = CacheReusePolicyDesc{true};
        params.model_config_.attn_config.sliding_window    = 4;
    }

    NormalExecutor::test_model_factory = [vocab_size = model_config.vocab_size](const GptModelInitParams&) {
        return std::unique_ptr<ModelBase>(new MockModel(vocab_size));
    };
    auto engine                        = std::make_shared<EngineType>(params, nullptr);
    NormalExecutor::test_model_factory = nullptr;
    return engine;
}

class DirectBuildExecutor: public Executor {
public:
    explicit DirectBuildExecutor(int fail_on_build = 0): fail_on_build_(fail_on_build) {}

    absl::Status process(const ScheduleOutput& output, int64_t = 0) override {
        retired_ids.insert(retired_ids.end(), output.finished_request_ids.begin(), output.finished_request_ids.end());
        for (const auto& stream : output.streams) {
            ++build_calls;
            EXPECT_EQ(stream->streamCacheResource().allocator_load_context_, nullptr);
            EXPECT_TRUE(stream->pipeline_parallel_);
            EXPECT_TRUE(stream->need_release_resource_);
            if (auto earlier = previous_stream.lock()) {
                EXPECT_TRUE(earlier->need_release_resource_);
            } else if (build_calls > 1) {
                ADD_FAILURE() << "earlier build stream was released before all tasks completed";
            }
            previous_stream = stream;
            if (build_calls == fail_on_build_) {
                stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "injected PP system prompt failure");
            } else {
                stream->updateFromPP({torch::tensor({{9}}, torch::kInt32), 1});
            }
            stream->clearPPInflight();
        }
        return absl::OkStatus();
    }

    int build_calls = 0;
    std::vector<int64_t> retired_ids;
    std::weak_ptr<GenerateStream> previous_stream;

private:
    int fail_on_build_;
};

std::shared_ptr<NormalEngine> createDirectBuildEngine(int fail_on_build = 0, bool multi_group = false) {
    auto engine = createFocusedEngine<NormalEngine>(128, 4096, multi_group);
    /** Stop and join the serving loop before substituting the executor. The test drives
     * real build/allocation/commit code synchronously, without constructing PP channels. */
    engine->running_ = false;
    EXPECT_TRUE(engine->scheduler_->stop().ok());
    engine->loop_thread_->join();
    engine->executor_ = std::make_unique<DirectBuildExecutor>(fail_on_build);
    engine->parallelism_config.pp_size = 2;
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
    ASSERT_FALSE(item1.group_block_ids.at("full").empty());
    ASSERT_EQ(item1.prompt_tokens, prompt_1);

    const auto& item2 = result["2"];
    ASSERT_EQ(item2.prompt_tokens.size(), 4);
    ASSERT_FALSE(item2.group_block_ids.at("full").empty());
    ASSERT_EQ(item2.prompt_tokens, prompt_2);
}

TEST_F(SystemPromptConstructorTest, testMultiGroupPromptPreservesEveryTaggedRow) {
    auto engine  = createFocusedEngine<NormalEngine>(128, 4096, /*multi_group=*/true);
    auto manager = engine->resourceContext().cache_manager;
    ASSERT_EQ(manager->cacheConfig().group("first").policy.group_type, CacheGroupType::FULL);
    ASSERT_EQ(manager->cacheConfig().group("second").policy.group_type, CacheGroupType::SWA);
    ASSERT_TRUE(manager->cacheConfig().group("second").policy.enable_prefix_reuse);
    KVCacheConfig config;
    config.multi_task_prompt_tokens = {{"both", {1, 2, 3}}};
    SystemPromptConstructor constructor;
    auto                    result = constructor.construct(config, engine.get(), manager.get(), true);
    ASSERT_TRUE(result.ok()) << result.status();
    const auto& prompt = result->at("both");
    EXPECT_EQ(prompt.prompt_tokens, (std::vector<int>{1, 2, 3}));
    ASSERT_EQ(prompt.group_block_ids.size(), 2u);
    for (const auto& group : manager->cacheConfig().topology().groups()) {
        const auto& blocks = prompt.group_block_ids.at(group.tag);
        ASSERT_FALSE(blocks.empty()) << group.tag;
        const auto& pools = manager->coordinator_manager_->cacheGroups();
        auto        owner = std::find_if(
            pools.begin(), pools.end(), [&](const auto& candidate) { return candidate->tag() == group.tag; });
        ASSERT_NE(owner, pools.end());
        for (auto block : blocks) {
            EXPECT_GT((*owner)->blockPool()->refCount(block), 0u) << group.tag;
        }
    }
}

TEST_F(SystemPromptConstructorTest, testResidentInsertFailureFailsWarmupAndReleasesRequestOwnership) {
    const std::shared_ptr<NormalEngine>   engine         = createFocusedEngine<NormalEngine>();
    const std::shared_ptr<KVCacheManager> engine_manager = engine->resourceContext().cache_manager;
    const DeviceBlockPoolPtr pool = engine_manager->blockTreeCache()->groupSets().front()->devicePools().front();
    const size_t             request_blocks_before = pool->referencedBlocksNum();
    auto                     insert_manager = std::make_shared<KVCacheManager>(engine_manager->cacheConfig(), true);
    auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(insert_manager->cacheConfig());
    insert_manager->coordinator_manager_ = allocator;
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
    EXPECT_EQ(resource->blocksNum(0, "full"), 95);
    EXPECT_EQ(manager->freeBlocksNum(), 4u);

    tokens->setSeqLength(198);
    ASSERT_TRUE(manager->malloc(info).success);
    EXPECT_EQ(resource->blocksNum(0, "full"), 99);
    EXPECT_EQ(manager->freeBlocksNum(), 0u);
    manager->free(FreeInfo{resource, tokens});
    EXPECT_EQ(manager->freeBlocksNum(), 99u);
}

TEST_F(SystemPromptConstructorTest, testNormalEngineWaitsForAllocatorObserverBeforeSystemPromptExecution) {
    auto engine         = createFocusedEngine<NormalEngine>();
    auto manager        = engine->resourceContext().cache_manager;
    auto real_allocator = manager->coordinator_manager_;
    auto context        = CountingReadyContext::create();
    auto mock_allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(manager->config_);

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

    manager->coordinator_manager_ = mock_allocator;
    auto stream_status            = engine->preRun(makeSystemPromptInput(), preRunMode::build_system_prompt);
    ASSERT_TRUE(stream_status.ok()) << stream_status.status();
    EXPECT_EQ(context->waitCalls(), 1u);
    EXPECT_EQ(stream_status.value()->streamCacheResource().allocator_load_context_, nullptr);

    stream_status.value().reset();
    manager->coordinator_manager_ = real_allocator;
}

TEST_F(SystemPromptConstructorTest, ppDirectBuildWaitsForAllocatorAndPreservesTaggedResidentRows) {
    auto engine = createDirectBuildEngine(0, true);
    auto manager = engine->resourceContext().cache_manager;
    auto real_allocator = manager->coordinator_manager_;
    std::vector<std::shared_ptr<CountingReadyContext>> contexts;
    auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(manager->config_);
    ON_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillByDefault(testing::Invoke([&](const MallocInfo& info) {
            auto result = real_allocator->initMallocForCommonLen(info);
            contexts.push_back(CountingReadyContext::create());
            result.async_context = contexts.back();
            return result;
        }));
    ON_CALL(*allocator, incrMalloc(testing::_)).WillByDefault(testing::Invoke([&](const MallocInfo& info) {
        return real_allocator->incrMalloc(info);
    }));
    ON_CALL(*allocator, free(testing::_)).WillByDefault(testing::Invoke([&](const FreeInfo& info) {
        real_allocator->free(info);
    }));
    ON_CALL(*allocator, insertIntoCache(testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const InsertInfo& info, size_t& prefix) {
            EXPECT_TRUE(info.is_resident);
            EXPECT_EQ(info.target_tier, Tier::DEVICE);
            real_allocator->insertIntoCache(info, prefix);
        }));
    ON_CALL(*allocator, seqSizePerBlock()).WillByDefault(testing::Invoke([&] {
        return real_allocator->seqSizePerBlock();
    }));
    manager->coordinator_manager_ = allocator;
    engine->kv_cache_config.multi_task_prompt_tokens = {{"1", {1, 2, 3}}, {"2", {4, 5, 6}}};
    const auto result = engine->buildSystemPromptsDirect();
    manager->coordinator_manager_ = real_allocator;
    ASSERT_TRUE(result.ok()) << result;
    ASSERT_EQ(contexts.size(), 2u);
    for (const auto& context : contexts) {
        EXPECT_EQ(context->waitCalls(), 1u);
    }
    auto* executor = static_cast<DirectBuildExecutor*>(engine->executor_.get());
    EXPECT_EQ(executor->build_calls, 2);
    EXPECT_EQ(executor->retired_ids, (std::vector<int64_t>{1, 2}));
    ASSERT_NE(engine->resource_context_.system_prompt, nullptr);
    for (const auto& [task_id, tokens] : engine->kv_cache_config.multi_task_prompt_tokens) {
        GenerateConfig config;
        config.task_id = task_id;
        const auto prompt = engine->resource_context_.system_prompt->getPromptParams(config);
        EXPECT_EQ(prompt.prompt_tokens, tokens);
        ASSERT_EQ(prompt.group_block_ids.size(), 2u);
        EXPECT_FALSE(prompt.group_block_ids.at("first").empty());
        EXPECT_FALSE(prompt.group_block_ids.at("second").empty());
    }
}

TEST_F(SystemPromptConstructorTest, ppDirectSecondTaskFailureReleasesRequestRefsAndRetiresBothTasks) {
    auto engine = createDirectBuildEngine(2);
    auto manager = engine->resourceContext().cache_manager;
    auto pool = manager->blockTreeCache()->groupSets().front()->devicePools().front();
    engine->kv_cache_config.multi_task_prompt_tokens = {{"1", {1, 2, 3}}, {"2", {1, 2, 4}}};
    const auto result = engine->buildSystemPromptsDirect();
    EXPECT_FALSE(result.ok());
    EXPECT_NE(result.message().find("injected PP system prompt failure"), std::string::npos);
    EXPECT_EQ(engine->resource_context_.system_prompt, nullptr);
    EXPECT_EQ(pool->referencedBlocksNum(), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
    auto* executor = static_cast<DirectBuildExecutor*>(engine->executor_.get());
    EXPECT_EQ(executor->retired_ids, (std::vector<int64_t>{1, 2}));
    EXPECT_TRUE(executor->previous_stream.expired());
}

TEST_F(SystemPromptConstructorTest, ppDirectAllocatorFailureAbortsBeforeSubmissionAndReleasesKv) {
    auto engine = createDirectBuildEngine();
    auto manager = engine->resourceContext().cache_manager;
    auto real_allocator = manager->coordinator_manager_;
    auto pool = manager->blockTreeCache()->groupSets().front()->devicePools().front();
    auto context = CountingReadyContext::create(/*fail_load=*/true);
    auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(manager->config_);
    ON_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillByDefault(testing::Invoke([&](const MallocInfo& info) {
            auto result = real_allocator->initMallocForCommonLen(info);
            result.async_context = context;
            return result;
        }));
    ON_CALL(*allocator, incrMalloc(testing::_)).WillByDefault(testing::Invoke([&](const MallocInfo& info) {
        return real_allocator->incrMalloc(info);
    }));
    ON_CALL(*allocator, free(testing::_)).WillByDefault(testing::Invoke([&](const FreeInfo& info) {
        real_allocator->free(info);
    }));
    manager->coordinator_manager_ = allocator;
    engine->kv_cache_config.multi_task_prompt_tokens = {{"1", {1, 2, 3}}};
    const auto result = engine->buildSystemPromptsDirect();
    manager->coordinator_manager_ = real_allocator;
    EXPECT_FALSE(result.ok());
    EXPECT_NE(result.message().find("allocator load failed"), std::string::npos);
    EXPECT_EQ(context->waitCalls(), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(), 0u);
    auto* executor = static_cast<DirectBuildExecutor*>(engine->executor_.get());
    EXPECT_EQ(executor->build_calls, 0);
    EXPECT_TRUE(executor->retired_ids.empty());
}

TEST_F(SystemPromptConstructorTest, ppReserveCoversTwoWindowsWithoutAddingToMainDSparkReserve) {
    EngineInitParams params;
    for (const auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        params.sp_config.type = type;
        params.sp_config.gen_num_per_cycle = 3;
        params.parallelism_config.pp_size = 1;
        params.pd_sep_config.role_type = RoleType::PDFUSION;
        EXPECT_EQ(NormalEngine::calculateReserveStep(params), type == SP_TYPE_DSPARK ? 9 : 4);
        params.parallelism_config.pp_size = 2;
        EXPECT_EQ(NormalEngine::calculateReserveStep(params), type == SP_TYPE_DSPARK ? 9 : 7);
        params.pd_sep_config.role_type = RoleType::PREFILL;
        EXPECT_EQ(NormalEngine::calculateReserveStep(params), type == SP_TYPE_DSPARK ? 9 : 4);
    }
    params.sp_config.type = SP_TYPE_NONE;
    EXPECT_EQ(NormalEngine::calculateReserveStep(params), 0);
}

}  // namespace rtp_llm
