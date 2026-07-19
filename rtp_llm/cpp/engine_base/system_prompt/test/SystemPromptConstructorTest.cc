#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define private public
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/allocator/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#undef private

using namespace std;

namespace rtp_llm {
namespace {

constexpr char kInjectedExecutorFailure[] = "C003 injected executor failure";
constexpr char kAllocatorLoadFailure[]    = "C004 disk load-back failure";

void setDefaultKvCacheSpecs(ModelConfig& model_config) {
    KVCacheSpecDesc desc;
    desc.tag                = "default";
    desc.cache_type         = CacheType::MHA;
    desc.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    desc.size_per_head      = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    desc.num_kv_heads       = static_cast<uint32_t>(model_config.attn_config.kv_head_num);
    model_config.kv_cache_spec_descs.assign(static_cast<size_t>(model_config.num_layers), {desc});
}

CacheKeyType hashTokens(CacheKeyType seed, std::initializer_list<int32_t> tokens) {
    const std::hash<int32_t> hasher;
    for (const auto token : tokens) {
        seed = hashInt64Func(hasher, seed, token);
    }
    return seed;
}

class ScopedExceptionMode {
public:
    ScopedExceptionMode(): saved_(StaticConfig::user_ft_core_dump_on_exception) {
        StaticConfig::user_ft_core_dump_on_exception = false;
    }

    ~ScopedExceptionMode() {
        StaticConfig::user_ft_core_dump_on_exception = saved_;
    }

private:
    const bool saved_;
};

struct CacheSnapshot {
    size_t                    free_blocks;
    size_t                    active_tree_cached_blocks;
    int64_t                   version;
    std::vector<CacheKeyType> keys;
    std::vector<uint32_t>     refs;
};

DeviceBlockPoolPtr devicePool(const std::shared_ptr<KVCacheManager>& cache_manager) {
    return cache_manager->allocator_->getDeviceBlockPool();
}

CacheSnapshot snapshotCache(const std::shared_ptr<KVCacheManager>& cache_manager) {
    const auto info = cache_manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/true);
    const auto pool = devicePool(cache_manager);
    return {
        cache_manager->freeBlocksNum(),
        cache_manager->allocator_->activeTreeCachedBlocksNum(),
        info.version,
        info.cached_keys,
        pool->refcounts_,
    };
}

void expectSnapshotEqual(const CacheSnapshot& actual, const CacheSnapshot& expected) {
    EXPECT_EQ(actual.free_blocks, expected.free_blocks);
    EXPECT_EQ(actual.active_tree_cached_blocks, expected.active_tree_cached_blocks);
    EXPECT_EQ(actual.version, expected.version);
    EXPECT_EQ(actual.keys, expected.keys);
    EXPECT_EQ(actual.refs, expected.refs);
}

void expectReleased(const DeviceBlockPoolPtr& pool, const BlockIndicesType& blocks) {
    for (const auto block : blocks) {
        ASSERT_GE(block, 0);
        ASSERT_LT(static_cast<size_t>(block), pool->refcounts_.size());
        EXPECT_EQ(pool->refcounts_[block], 0u) << "block=" << block;
        EXPECT_FALSE(pool->isAllocated(block)) << "block=" << block;
    }
}

class CountingMockModel: public MockModel {
public:
    CountingMockModel(size_t vocab_size, size_t* forward_attempts):
        MockModel(vocab_size), forward_attempts_(forward_attempts) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        ++*forward_attempts_;
        return MockModel::forward(inputs);
    }

private:
    size_t* forward_attempts_;
};

class ReadinessEventLog {
public:
    void record(const std::string& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
    }

    std::vector<std::string> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }

private:
    mutable std::mutex       mutex_;
    std::vector<std::string> events_;
};

size_t eventPosition(const std::vector<std::string>& events, const std::string& event) {
    const auto it = std::find(events.begin(), events.end(), event);
    EXPECT_NE(it, events.end()) << "missing event=" << event;
    return static_cast<size_t>(std::distance(events.begin(), it));
}

class PausableAllocatorContext: public AsyncContext {
public:
    PausableAllocatorContext(std::string tier, std::shared_ptr<ReadinessEventLog> events):
        tier_(std::move(tier)), events_(std::move(events)) {}

    void waitDone() override {
        std::unique_lock<std::mutex> lock(mutex_);
        ++wait_calls_;
        waiting_ = true;
        events_->record(tier_ + "_wait_entered");
        cv_.notify_all();
        cv_.wait(lock, [this] { return released_; });
    }

    bool done() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++done_calls_;
        events_->record(tier_ + "_done_observed");
        return done_;
    }

    bool success() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++success_calls_;
        events_->record(tier_ + "_success_observed");
        return success_;
    }

    ErrorInfo errorInfo() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++error_calls_;
        events_->record(tier_ + "_error_observed");
        return error_;
    }

    void waitUntilBlocked() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return waiting_; });
    }

    void complete(bool success, const std::string& error = "") {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_     = true;
            success_  = success;
            error_    = success ? ErrorInfo::OkStatus() : ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error);
            released_ = true;
            events_->record(tier_ + (success ? "_terminal_success" : "_terminal_failure"));
        }
        cv_.notify_all();
    }

    bool pendingForTest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !done_;
    }

    size_t waitCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return wait_calls_;
    }

    size_t doneCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_calls_;
    }

    size_t successCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return success_calls_;
    }

    size_t errorCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_calls_;
    }

private:
    const std::string                        tier_;
    const std::shared_ptr<ReadinessEventLog> events_;
    mutable std::mutex                       mutex_;
    std::condition_variable                  cv_;
    bool                                     waiting_{false};
    bool                                     released_{false};
    bool                                     done_{false};
    bool                                     success_{true};
    ErrorInfo                                error_;
    size_t                                   wait_calls_{0};
    mutable size_t                           done_calls_{0};
    mutable size_t                           success_calls_{0};
    mutable size_t                           error_calls_{0};
};

class ScriptedAllocatorContext: public AsyncContext {
public:
    ScriptedAllocatorContext(bool done, bool success, std::string error = ""):
        done_(done),
        success_(success),
        error_(success ? ErrorInfo::OkStatus() : ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, std::move(error))) {}

    void waitDone() override {
        events_.push_back("wait");
    }

    bool done() const override {
        events_.push_back("done");
        return done_;
    }

    bool success() const override {
        events_.push_back("success");
        return success_;
    }

    ErrorInfo errorInfo() const override {
        events_.push_back("error");
        return error_;
    }

    const std::vector<std::string>& events() const {
        return events_;
    }

private:
    const bool                       done_;
    const bool                       success_;
    const ErrorInfo                  error_;
    mutable std::vector<std::string> events_;
};

class ReadinessCountingMockModel: public MockModel {
public:
    ReadinessCountingMockModel(size_t                             vocab_size,
                               std::atomic<size_t>*               forward_attempts,
                               std::shared_ptr<ReadinessEventLog> events):
        MockModel(vocab_size), forward_attempts_(forward_attempts), events_(std::move(events)) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        forward_attempts_->fetch_add(1, std::memory_order_relaxed);
        events_->record("executor");
        return MockModel::forward(inputs);
    }

private:
    std::atomic<size_t>*               forward_attempts_;
    std::shared_ptr<ReadinessEventLog> events_;
};

class ReadinessTestAllocator: public SingleTypeKVCacheAllocator {
public:
    ReadinessTestAllocator(const CacheConfig&                 config,
                           std::shared_ptr<AsyncContext>      load_context,
                           std::shared_ptr<ReadinessEventLog> events):
        SingleTypeKVCacheAllocator(config), load_context_(std::move(load_context)), events_(std::move(events)) {}

    void insertIntoCache(const InsertInfo& insert_info) override {
        insert_attempts_.fetch_add(1, std::memory_order_relaxed);
        events_->record("insert");
        SingleTypeKVCacheAllocator::insertIntoCache(insert_info);
    }

    size_t insertAttempts() const {
        return insert_attempts_.load(std::memory_order_relaxed);
    }

    BlockIndicesType allocatedBlocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocated_blocks_;
    }

private:
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override {
        auto result = SingleTypeKVCacheAllocator::initMallocForCommonLen(malloc_info);
        if (result.success && load_context_) {
            result.async_context = load_context_;
        }
        return result;
    }

    MallocResult incrMalloc(const MallocInfo& malloc_info) override {
        auto result = SingleTypeKVCacheAllocator::incrMalloc(malloc_info);
        if (result.success) {
            std::lock_guard<std::mutex> lock(mutex_);
            allocated_blocks_ = malloc_info.batch_kv_cache_resource->blocks(/*batch_id=*/0, /*group_id=*/0);
        }
        return result;
    }

    const std::shared_ptr<AsyncContext>      load_context_;
    const std::shared_ptr<ReadinessEventLog> events_;
    std::atomic<size_t>                      insert_attempts_{0};
    mutable std::mutex                       mutex_;
    BlockIndicesType                         allocated_blocks_;
};

template<typename EngineType>
std::shared_ptr<EngineType>
createTestEngine(const CustomConfig& config, size_t test_block_num, size_t* model_forward_attempts = nullptr) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.kv_cache_config.seq_size_per_block           = params.model_config_.attn_config.tokens_per_block;
    params.kv_cache_config.kernel_seq_size_per_block    = params.model_config_.attn_config.tokens_per_block;
    params.kv_cache_config.device_cache_min_free_blocks = 1;
    setDefaultKvCacheSpecs(params.model_config_);
    params.kv_cache_config.test_block_num = test_block_num;

    size_t* attempts                   = model_forward_attempts;
    NormalExecutor::test_model_factory = [vocab = model_config.vocab_size, attempts](const GptModelInitParams&) {
        if (attempts != nullptr) {
            return std::unique_ptr<ModelBase>(new CountingMockModel(vocab, attempts));
        }
        return std::unique_ptr<ModelBase>(new MockModel(vocab));
    };
    auto engine                        = std::make_shared<EngineType>(params, nullptr);
    NormalExecutor::test_model_factory = nullptr;
    return engine;
}

std::shared_ptr<NormalEngine> createReadinessTestEngine(const CustomConfig&  config,
                                                        size_t               test_block_num,
                                                        std::atomic<size_t>* model_forward_attempts,
                                                        const std::shared_ptr<ReadinessEventLog>& events) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.kv_cache_config.seq_size_per_block           = params.model_config_.attn_config.tokens_per_block;
    params.kv_cache_config.kernel_seq_size_per_block    = params.model_config_.attn_config.tokens_per_block;
    params.kv_cache_config.device_cache_min_free_blocks = 1;
    setDefaultKvCacheSpecs(params.model_config_);
    params.kv_cache_config.test_block_num = test_block_num;

    NormalExecutor::test_model_factory =
        [vocab = model_config.vocab_size, model_forward_attempts, events](const GptModelInitParams&) {
            return std::unique_ptr<ModelBase>(new ReadinessCountingMockModel(vocab, model_forward_attempts, events));
        };
    auto engine                        = std::make_shared<NormalEngine>(params, nullptr);
    NormalExecutor::test_model_factory = nullptr;
    return engine;
}

std::shared_ptr<ReadinessTestAllocator> installReadinessAllocator(const std::shared_ptr<KVCacheManager>& cache_manager,
                                                                  const std::shared_ptr<AsyncContext>&   load_context,
                                                                  const std::shared_ptr<ReadinessEventLog>& events) {
    auto allocator = std::make_shared<ReadinessTestAllocator>(cache_manager->config_, load_context, events);
    if (!allocator->init()) {
        return nullptr;
    }
    auto block_tree_cache = createBlockTreeCache(cache_manager->config_, cache_manager->kv_cache_config_, allocator);
    if (!block_tree_cache) {
        return nullptr;
    }
    allocator->setBlockTreeCache(block_tree_cache.get());
    cache_manager->allocator_        = allocator;
    cache_manager->block_tree_cache_ = std::move(block_tree_cache);
    return allocator;
}

GenerateStreamPtr makeUnallocatedStream(const std::shared_ptr<NormalEngine>& engine) {
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    return std::make_shared<NormalGenerateStream>(
        generate_input, engine->model_config_, engine->runtime_config, engine->resource_context_, nullptr);
}

class InjectingNormalEngine: public NormalEngine {
public:
    using NormalEngine::NormalEngine;

    void failExecutorOnCall(size_t call) {
        fail_on_call_ = call;
    }

    const std::vector<BlockIndicesType>& blocksByCall() const {
        return blocks_by_call_;
    }

    const std::vector<std::vector<uint32_t>>& refsAtAllocationByCall() const {
        return refs_at_allocation_by_call_;
    }

    size_t executorAttempts() const {
        return executor_attempts_;
    }

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& input, preRunMode mode) override {
        const size_t call = blocks_by_call_.size();
        if (call != fail_on_call_) {
            auto stream_status = NormalEngine::preRun(input, mode);
            if (!stream_status.ok()) {
                return stream_status.status();
            }
            recordAllocation(stream_status.value());
            return stream_status;
        }

        auto stream = std::make_shared<NormalGenerateStream>(
            input, model_config_, runtime_config, resource_context_, nullptr, 0, mode == preRunMode::prefill_warm_up);
        stream->setReserveStep(reserve_step_);
        if (mode == preRunMode::build_system_prompt) {
            THROW_IF_STATUS_ERROR(stream->initKVBlock());
        }
        recordAllocation(stream);
        ++executor_attempts_;
        THROW_IF_STATUS_ERROR(absl::InternalError(kInjectedExecutorFailure));
        return stream;
    }

private:
    void recordAllocation(const GenerateStreamPtr& stream) {
        const auto blocks = stream->kvCache().blocks(0, 0);
        blocks_by_call_.push_back(blocks);

        const auto            pool = devicePool(resource_context_.cache_manager);
        std::vector<uint32_t> refs;
        refs.reserve(blocks.size());
        for (const auto block : blocks) {
            refs.push_back(pool->refCount(block));
        }
        refs_at_allocation_by_call_.push_back(std::move(refs));
    }

    size_t                             fail_on_call_{std::numeric_limits<size_t>::max()};
    size_t                             executor_attempts_{0};
    std::vector<BlockIndicesType>      blocks_by_call_;
    std::vector<std::vector<uint32_t>> refs_at_allocation_by_call_;
};

std::string runConstructExpectingRTPException(SystemPromptConstructor& constructor,
                                              const KVCacheConfig&     config,
                                              EngineBase*              engine,
                                              KVCacheManager*          cache_manager,
                                              bool                     insert_kv_cache,
                                              bool&                    returned) {
    std::string message;
    try {
        auto result = constructor.construct(config, engine, cache_manager, insert_kv_cache);
        (void)result;
        returned = true;
    } catch (const RTPException& e) {
        message = e.what();
    }
    return message;
}

}  // namespace

class SystemPromptConstructorTest: public DeviceTestBase {};

// C003-T01: exact production exception transport after a real successful allocation.
TEST_F(SystemPromptConstructorTest, testExecutorExceptionReleasesSingleTaskOwnership) {
    SystemPromptConstructor constructor;
    KVCacheConfig           config;
    config.multi_task_prompt_tokens = {{"1", {1, 2, 3}}};

    CustomConfig engine_config;
    engine_config.reuse_cache = true;
    auto engine               = createTestEngine<InjectingNormalEngine>(engine_config, /*test_block_num=*/100);
    engine->failExecutorOnCall(0);

    auto       cache_manager        = engine->resourceContext().cache_manager;
    auto       pool                 = devicePool(cache_manager);
    auto       before               = snapshotCache(cache_manager);
    const bool saved_exception_mode = StaticConfig::user_ft_core_dump_on_exception;

    bool        returned = false;
    std::string exception_message;
    {
        ScopedExceptionMode exception_mode;
        exception_message =
            runConstructExpectingRTPException(constructor, config, engine.get(), cache_manager.get(), true, returned);
    }

    ASSERT_FALSE(returned);
    ASSERT_NE(exception_message.find(kInjectedExecutorFailure), std::string::npos);
    EXPECT_EQ(StaticConfig::user_ft_core_dump_on_exception, saved_exception_mode);
    ASSERT_EQ(engine->executorAttempts(), 1u);
    ASSERT_EQ(engine->blocksByCall().size(), 1u);
    ASSERT_EQ(engine->refsAtAllocationByCall().size(), 1u);
    ASSERT_FALSE(engine->blocksByCall()[0].empty());
    for (const auto ref : engine->refsAtAllocationByCall()[0]) {
        EXPECT_GT(ref, 0u);
    }
    expectReleased(pool, engine->blocksByCall()[0]);
    expectSnapshotEqual(snapshotCache(cache_manager), before);
}

// C003-T02: current NormalEngine initKVBlock failure takes the real THROW_IF_STATUS_ERROR path.
TEST_F(SystemPromptConstructorTest, testNormalEngineAllocationExceptionRestoresState) {
    SystemPromptConstructor constructor;
    KVCacheConfig           config;
    config.multi_task_prompt_tokens = {{"1", {1, 2, 3}}};

    CustomConfig engine_config;
    engine_config.reuse_cache     = true;
    size_t model_forward_attempts = 0;
    auto   engine = createTestEngine<NormalEngine>(engine_config, /*test_block_num=*/2, &model_forward_attempts);

    auto cache_manager = engine->resourceContext().cache_manager;
    auto before        = snapshotCache(cache_manager);
    ASSERT_EQ(before.free_blocks, 1u);
    const bool saved_exception_mode = StaticConfig::user_ft_core_dump_on_exception;

    bool        returned = false;
    std::string exception_message;
    {
        ScopedExceptionMode exception_mode;
        exception_message =
            runConstructExpectingRTPException(constructor, config, engine.get(), cache_manager.get(), true, returned);
    }

    ASSERT_FALSE(returned);
    ASSERT_NE(exception_message.find("malloc failed"), std::string::npos);
    EXPECT_EQ(StaticConfig::user_ft_core_dump_on_exception, saved_exception_mode);
    EXPECT_EQ(model_forward_attempts, 0u);
    expectSnapshotEqual(snapshotCache(cache_manager), before);
}

// C003-T03: successful construction retains cache/tree plus resident-request ownership.
TEST_F(SystemPromptConstructorTest, testMultiTaskPromptConstructRetainsResidentOwnership) {
    SystemPromptConstructor constructor;
    KVCacheConfig           config;
    const vector<int>       prompt_1 = {1, 2, 3};
    const vector<int>       prompt_2 = {4, 5, 6, 7};
    config.multi_task_prompt_tokens  = {{"1", prompt_1}, {"2", prompt_2}};

    CustomConfig engine_config;
    engine_config.reuse_cache = true;
    auto engine               = createTestEngine<NormalEngine>(engine_config, /*test_block_num=*/100);

    auto cache_manager = engine->resourceContext().cache_manager;
    auto pool          = devicePool(cache_manager);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 99u);

    auto result_status = constructor.construct(config, engine.get(), cache_manager.get(), true);
    ASSERT_TRUE(result_status.ok()) << result_status.status();
    const auto& result = result_status.value();
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(cache_manager->freeBlocksNum(), 95u);

    const auto& item1 = result.at("1");
    EXPECT_EQ(item1.prompt_tokens, prompt_1);
    ASSERT_EQ(item1.block_ids.size(), 2u);

    const auto& item2 = result.at("2");
    EXPECT_EQ(item2.prompt_tokens, prompt_2);
    ASSERT_EQ(item2.block_ids.size(), 2u);

    // Three full keyed blocks each have one cache/tree ref and one resident-request ref.
    EXPECT_EQ(pool->refCount(item1.block_ids[0]), 2u);
    EXPECT_EQ(pool->refCount(item2.block_ids[0]), 2u);
    EXPECT_EQ(pool->refCount(item2.block_ids[1]), 2u);
    // Prompt 1's final partial block is deliberately unkeyed but remains resident-request owned.
    EXPECT_EQ(pool->refCount(item1.block_ids[1]), 1u);

    const auto cache_info        = cache_manager->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/true);
    const auto prompt_1_full_key = hashTokens(0, {1, 2});
    const auto prompt_2_key_1    = hashTokens(0, {4, 5});
    const auto prompt_2_key_2    = hashTokens(prompt_2_key_1, {6, 7});
    auto       actual_keys       = cache_info.cached_keys;
    auto       expected_keys     = CacheKeysType{prompt_1_full_key, prompt_2_key_1, prompt_2_key_2};
    std::sort(actual_keys.begin(), actual_keys.end());
    std::sort(expected_keys.begin(), expected_keys.end());
    EXPECT_EQ(actual_keys, expected_keys);
}

// C003-T04: disabling insertion must not suppress ordinary stream cleanup.
TEST_F(SystemPromptConstructorTest, testInsertDisabledRestoresOwnershipAndCapacity) {
    SystemPromptConstructor constructor;
    KVCacheConfig           config;
    config.multi_task_prompt_tokens = {{"1", {1, 2, 3}}};

    CustomConfig engine_config;
    engine_config.reuse_cache = true;
    auto engine               = createTestEngine<InjectingNormalEngine>(engine_config, /*test_block_num=*/100);

    auto cache_manager = engine->resourceContext().cache_manager;
    auto pool          = devicePool(cache_manager);
    auto before        = snapshotCache(cache_manager);

    auto result_status = constructor.construct(config, engine.get(), cache_manager.get(), false);
    ASSERT_TRUE(result_status.ok()) << result_status.status();
    EXPECT_TRUE(result_status.value().empty());

    ASSERT_EQ(engine->blocksByCall().size(), 1u);
    ASSERT_FALSE(engine->blocksByCall()[0].empty());
    expectReleased(pool, engine->blocksByCall()[0]);
    expectSnapshotEqual(snapshotCache(cache_manager), before);
}

// C003-T05: a later exception must not commit resident holds for an earlier inserted task.
TEST_F(SystemPromptConstructorTest, testSecondTaskExceptionRollsBackResidentRequestHolds) {
    SystemPromptConstructor constructor;
    KVCacheConfig           config;
    config.multi_task_prompt_tokens = {
        {"1", {1, 2, 3}},
        {"2", {1, 2, 4}},
    };

    CustomConfig engine_config;
    engine_config.reuse_cache = true;
    auto engine               = createTestEngine<InjectingNormalEngine>(engine_config, /*test_block_num=*/100);
    engine->failExecutorOnCall(1);

    auto       cache_manager        = engine->resourceContext().cache_manager;
    auto       pool                 = devicePool(cache_manager);
    auto       before               = snapshotCache(cache_manager);
    const bool saved_exception_mode = StaticConfig::user_ft_core_dump_on_exception;

    bool        returned = false;
    std::string exception_message;
    {
        ScopedExceptionMode exception_mode;
        exception_message =
            runConstructExpectingRTPException(constructor, config, engine.get(), cache_manager.get(), true, returned);
    }

    ASSERT_FALSE(returned);
    ASSERT_NE(exception_message.find(kInjectedExecutorFailure), std::string::npos);
    EXPECT_EQ(StaticConfig::user_ft_core_dump_on_exception, saved_exception_mode);
    ASSERT_EQ(engine->executorAttempts(), 1u);
    ASSERT_EQ(engine->blocksByCall().size(), 2u);
    ASSERT_EQ(engine->blocksByCall()[0].size(), 2u);
    ASSERT_EQ(engine->blocksByCall()[1].size(), 2u);

    const auto first_full  = engine->blocksByCall()[0][0];
    const auto first_tail  = engine->blocksByCall()[0][1];
    const auto second_full = engine->blocksByCall()[1][0];
    const auto second_tail = engine->blocksByCall()[1][1];
    ASSERT_EQ(second_full, first_full);

    // The shared full block is only cache-owned and therefore evictable; no resident request ref survived.
    EXPECT_EQ(pool->refCount(first_full), 1u);
    EXPECT_EQ(cache_manager->allocator_->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(cache_manager->blockTreeCache()->getStats().device_heap_total_size, 1u);
    expectReleased(pool, BlockIndicesType{first_tail, second_tail});

    const auto after = snapshotCache(cache_manager);
    EXPECT_EQ(after.keys, CacheKeysType({hashTokens(0, {1, 2})}));
    EXPECT_NE(after.version, before.version);
    EXPECT_EQ(after.free_blocks, before.free_blocks - 1);
    EXPECT_EQ(after.active_tree_cached_blocks, before.active_tree_cached_blocks);
}

// C004-T01: a null allocator observer preserves synchronous execution and C-003 residency.
TEST_F(SystemPromptConstructorTest, testNullAllocatorContextPreservesResidentPublication) {
    SystemPromptConstructor constructor;
    KVCacheConfig           config;
    const vector<int>       prompt  = {1, 2, 3};
    config.multi_task_prompt_tokens = {{"1", prompt}};

    auto                events = std::make_shared<ReadinessEventLog>();
    std::atomic<size_t> model_forward_attempts{0};
    CustomConfig        engine_config;
    engine_config.reuse_cache = true;
    auto engine = createReadinessTestEngine(engine_config, /*test_block_num=*/100, &model_forward_attempts, events);
    auto cache_manager = engine->resourceContext().cache_manager;
    auto allocator     = installReadinessAllocator(cache_manager, /*load_context=*/nullptr, events);
    ASSERT_NE(allocator, nullptr);
    auto pool   = devicePool(cache_manager);
    auto before = snapshotCache(cache_manager);

    auto result_status = constructor.construct(config, engine.get(), cache_manager.get(), true);
    ASSERT_TRUE(result_status.ok()) << result_status.status();
    const auto& item = result_status.value().at("1");
    EXPECT_EQ(item.prompt_tokens, prompt);
    ASSERT_EQ(item.block_ids.size(), 2u);
    EXPECT_EQ(model_forward_attempts.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(allocator->insertAttempts(), 1u);
    EXPECT_EQ(pool->refCount(item.block_ids[0]), 2u);
    EXPECT_EQ(pool->refCount(item.block_ids[1]), 1u);
    EXPECT_EQ(cache_manager->freeBlocksNum(), before.free_blocks - 2);

    events->record("resident_commit_observed");
    const auto ordered = events->snapshot();
    EXPECT_EQ(std::count(ordered.begin(), ordered.end(), "executor"), 1);
    EXPECT_LT(eventPosition(ordered, "executor"), eventPosition(ordered, "insert"));
    EXPECT_LT(eventPosition(ordered, "insert"), eventPosition(ordered, "resident_commit_observed"));
    EXPECT_EQ(std::count_if(ordered.begin(),
                            ordered.end(),
                            [](const std::string& event) { return event.find("wait") != std::string::npos; }),
              0);
}

// C004-T02: a real request allocation blocks model/cache publication until terminal HOST success.
TEST_F(SystemPromptConstructorTest, testHostAllocatorLoadWaitsBeforeExecutionAndPublication) {
    SystemPromptConstructor constructor;
    KVCacheConfig           config;
    const vector<int>       prompt  = {1, 2, 3};
    config.multi_task_prompt_tokens = {{"1", prompt}};

    auto                events       = std::make_shared<ReadinessEventLog>();
    auto                load_context = std::make_shared<PausableAllocatorContext>("host", events);
    std::atomic<size_t> model_forward_attempts{0};
    std::atomic<bool>   construct_returned{false};
    CustomConfig        engine_config;
    engine_config.reuse_cache = true;
    auto engine = createReadinessTestEngine(engine_config, /*test_block_num=*/100, &model_forward_attempts, events);
    auto cache_manager = engine->resourceContext().cache_manager;
    auto allocator     = installReadinessAllocator(cache_manager, load_context, events);
    ASSERT_NE(allocator, nullptr);
    auto pool   = devicePool(cache_manager);
    auto before = snapshotCache(cache_manager);

    std::unordered_map<std::string, SystemPromptParams> result;
    std::string                                         worker_error;
    std::thread                                         worker([&] {
        try {
            auto status = constructor.construct(config, engine.get(), cache_manager.get(), true);
            if (status.ok()) {
                result = std::move(status.value());
            } else {
                worker_error = status.status().ToString();
            }
        } catch (const std::exception& e) {
            worker_error = e.what();
        }
        construct_returned.store(true, std::memory_order_release);
    });

    load_context->waitUntilBlocked();
    EXPECT_TRUE(load_context->pendingForTest());
    EXPECT_FALSE(construct_returned.load(std::memory_order_acquire));
    EXPECT_EQ(model_forward_attempts.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(allocator->insertAttempts(), 0u);
    const auto allocated_blocks = allocator->allocatedBlocks();
    EXPECT_EQ(allocated_blocks.size(), 2u);
    for (const auto block : allocated_blocks) {
        EXPECT_GT(pool->refCount(block), 0u);
    }
    const auto pending_snapshot = snapshotCache(cache_manager);
    EXPECT_EQ(pending_snapshot.version, before.version);
    EXPECT_EQ(pending_snapshot.keys, before.keys);

    load_context->complete(/*success=*/true);
    worker.join();

    ASSERT_TRUE(worker_error.empty()) << worker_error;
    ASSERT_TRUE(construct_returned.load(std::memory_order_acquire));
    ASSERT_EQ(result.size(), 1u);
    const auto& item = result.at("1");
    EXPECT_EQ(item.prompt_tokens, prompt);
    ASSERT_EQ(item.block_ids.size(), 2u);
    EXPECT_EQ(model_forward_attempts.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(allocator->insertAttempts(), 1u);
    EXPECT_EQ(load_context->waitCalls(), 1u);
    EXPECT_EQ(load_context->doneCalls(), 1u);
    EXPECT_EQ(load_context->successCalls(), 1u);
    EXPECT_EQ(load_context->errorCalls(), 1u);
    EXPECT_EQ(pool->refCount(item.block_ids[0]), 2u);
    EXPECT_EQ(pool->refCount(item.block_ids[1]), 1u);
    EXPECT_EQ(cache_manager->freeBlocksNum(), before.free_blocks - 2);

    events->record("resident_commit_observed");
    const auto ordered = events->snapshot();
    EXPECT_LT(eventPosition(ordered, "host_terminal_success"), eventPosition(ordered, "host_done_observed"));
    EXPECT_LT(eventPosition(ordered, "host_done_observed"), eventPosition(ordered, "host_success_observed"));
    EXPECT_LT(eventPosition(ordered, "host_success_observed"), eventPosition(ordered, "executor"));
    EXPECT_LT(eventPosition(ordered, "executor"), eventPosition(ordered, "insert"));
    EXPECT_LT(eventPosition(ordered, "insert"), eventPosition(ordered, "resident_commit_observed"));
}

// C004-T03: terminal DISK failure executes neither model nor insert and unwinds request ownership once.
TEST_F(SystemPromptConstructorTest, testDiskAllocatorLoadFailureUnwindsWithoutPublication) {
    SystemPromptConstructor constructor;
    KVCacheConfig           config;
    config.multi_task_prompt_tokens = {{"1", {1, 2, 3}}};

    auto                events       = std::make_shared<ReadinessEventLog>();
    auto                load_context = std::make_shared<PausableAllocatorContext>("disk", events);
    std::atomic<size_t> model_forward_attempts{0};
    std::atomic<bool>   construct_returned{false};
    CustomConfig        engine_config;
    engine_config.reuse_cache = true;
    auto engine = createReadinessTestEngine(engine_config, /*test_block_num=*/100, &model_forward_attempts, events);
    auto cache_manager = engine->resourceContext().cache_manager;
    auto allocator     = installReadinessAllocator(cache_manager, load_context, events);
    ASSERT_NE(allocator, nullptr);
    auto pool   = devicePool(cache_manager);
    auto before = snapshotCache(cache_manager);

    const bool  saved_exception_mode = StaticConfig::user_ft_core_dump_on_exception;
    std::string exception_message;
    {
        ScopedExceptionMode exception_mode;
        std::thread         worker([&] {
            try {
                auto status = constructor.construct(config, engine.get(), cache_manager.get(), true);
                (void)status;
                construct_returned.store(true, std::memory_order_release);
            } catch (const RTPException& e) {
                exception_message = e.what();
            }
        });

        load_context->waitUntilBlocked();
        EXPECT_TRUE(load_context->pendingForTest());
        EXPECT_FALSE(construct_returned.load(std::memory_order_acquire));
        EXPECT_EQ(model_forward_attempts.load(std::memory_order_relaxed), 0u);
        EXPECT_EQ(allocator->insertAttempts(), 0u);
        const auto allocated_blocks = allocator->allocatedBlocks();
        EXPECT_EQ(allocated_blocks.size(), 2u);
        for (const auto block : allocated_blocks) {
            EXPECT_GT(pool->refCount(block), 0u);
        }

        load_context->complete(/*success=*/false, kAllocatorLoadFailure);
        worker.join();

        ASSERT_FALSE(construct_returned.load(std::memory_order_acquire));
        ASSERT_NE(exception_message.find(kAllocatorLoadFailure), std::string::npos);
        EXPECT_EQ(model_forward_attempts.load(std::memory_order_relaxed), 0u);
        EXPECT_EQ(allocator->insertAttempts(), 0u);
        expectReleased(pool, allocated_blocks);
        expectSnapshotEqual(snapshotCache(cache_manager), before);
    }
    EXPECT_EQ(StaticConfig::user_ft_core_dump_on_exception, saved_exception_mode);
    EXPECT_EQ(load_context->waitCalls(), 1u);
    EXPECT_EQ(load_context->doneCalls(), 1u);
    EXPECT_EQ(load_context->successCalls(), 1u);
    EXPECT_EQ(load_context->errorCalls(), 1u);

    const auto ordered = events->snapshot();
    EXPECT_LT(eventPosition(ordered, "disk_terminal_failure"), eventPosition(ordered, "disk_done_observed"));
    EXPECT_LT(eventPosition(ordered, "disk_done_observed"), eventPosition(ordered, "disk_success_observed"));
    EXPECT_LT(eventPosition(ordered, "disk_success_observed"), eventPosition(ordered, "disk_error_observed"));
    EXPECT_EQ(std::find(ordered.begin(), ordered.end(), "executor"), ordered.end());
    EXPECT_EQ(std::find(ordered.begin(), ordered.end(), "insert"), ordered.end());
}

// C004-T04: a nonterminal return is fail-closed and keeps its observer installed for ordinary cleanup.
TEST_F(SystemPromptConstructorTest, testWaitForAllocatorLoadRetainsNonterminalObserver) {
    CustomConfig        engine_config;
    auto                events = std::make_shared<ReadinessEventLog>();
    std::atomic<size_t> model_forward_attempts{0};
    auto  engine   = createReadinessTestEngine(engine_config, /*test_block_num=*/100, &model_forward_attempts, events);
    auto  stream   = makeUnallocatedStream(engine);
    auto& resource = stream->streamCacheResource();
    auto  context  = std::make_shared<ScriptedAllocatorContext>(/*done=*/false, /*success=*/true);
    resource.allocator_load_context_ = context;

    const auto status = resource.waitForAllocatorLoad();
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_NE(status.message().find("non-terminal"), std::string::npos);
    EXPECT_EQ(resource.allocator_load_context_, context);
    EXPECT_EQ(context->events(), std::vector<std::string>({"wait", "done"}));
}

// C004-T04: a terminal failure snapshots done before success/error and resets only after terminal observation.
TEST_F(SystemPromptConstructorTest, testWaitForAllocatorLoadPropagatesTerminalFailureAndResetsObserver) {
    CustomConfig        engine_config;
    auto                events = std::make_shared<ReadinessEventLog>();
    std::atomic<size_t> model_forward_attempts{0};
    auto  engine   = createReadinessTestEngine(engine_config, /*test_block_num=*/100, &model_forward_attempts, events);
    auto  stream   = makeUnallocatedStream(engine);
    auto& resource = stream->streamCacheResource();
    auto  context = std::make_shared<ScriptedAllocatorContext>(/*done=*/true, /*success=*/false, kAllocatorLoadFailure);
    resource.allocator_load_context_ = context;

    const auto status = resource.waitForAllocatorLoad();
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_NE(status.message().find(kAllocatorLoadFailure), std::string::npos);
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(context->events(), std::vector<std::string>({"wait", "done", "success", "error"}));
}
}  // namespace rtp_llm
