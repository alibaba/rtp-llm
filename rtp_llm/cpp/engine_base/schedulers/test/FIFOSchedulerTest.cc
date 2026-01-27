#include <memory>
#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class FIFOSchedulerTest: public DeviceTestBase {
public:
};

TEST_F(FIFOSchedulerTest, testSimple) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    stream->setFinishedWithoutLock();

    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);
}

TEST_F(FIFOSchedulerTest, testInitKVCacheLackMem) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 2, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 0);
    ASSERT_TRUE(stream->stopped());
    ASSERT_EQ(stream->stopReason(), "input len 3 is greater than kv cache max available tokens num 2");

    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);
}

TEST_F(FIFOSchedulerTest, testIncrKVCacheLackMem) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 3, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 1);
    ASSERT_FALSE(stream->stopped());
    ASSERT_EQ(stream->stopReason(), "");
    ASSERT_EQ(cache_manager->freeBlocksNum(), 0);

    stream->setSeqLength(stream->seqLength() + 1);
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_TRUE(stream->stopped());
    ASSERT_EQ(stream->stopReason(), "incrKVBlock failed: LACK MEM");
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);
}

TEST_F(FIFOSchedulerTest, testInitKVCacheRejectedByReserveBlocks) {
    CacheConfig cache_config = makeMhaCacheConfig(/*layer_num=*/1,
                                                  /*block_num=*/11,
                                                  /*local_head_num_kv=*/1,
                                                  /*size_per_head=*/4,
                                                  /*tokens_per_block=*/1,
                                                  rtp_llm::DataType::TYPE_FP16);

    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio = 50;  // reserve = 50% * available(10) = 5 blocks

    std::shared_ptr<KVCacheManager> cache_manager =
        std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/false, nullptr, kv_cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Need 6 blocks. With reserve=5 blocks and available=10 blocks, init malloc should be rejected.
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({6}, {1, 2, 3, 4, 5, 6}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();

    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 0);
    ASSERT_TRUE(stream->stopped());
    ASSERT_EQ(stream->stopReason(), "LACK MEM");
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);
    ASSERT_EQ(cache_manager->availableBlocksNum(), 10);
}

TEST_F(FIFOSchedulerTest, testReserveBlocksOnlyAffectInitMallocNotIncrMalloc) {
    CacheConfig cache_config = makeMhaCacheConfig(/*layer_num=*/1,
                                                  /*block_num=*/11,
                                                  /*local_head_num_kv=*/1,
                                                  /*size_per_head=*/4,
                                                  /*tokens_per_block=*/1,
                                                  rtp_llm::DataType::TYPE_FP16);

    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio = 50;  // reserve = 5 blocks

    std::shared_ptr<KVCacheManager> cache_manager =
        std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/false, nullptr, kv_cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Init need 4 blocks, should pass: 10 >= 4 + 5.
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();

    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto streams_status1 = scheduler.schedule();
    ASSERT_TRUE(streams_status1.ok());
    ASSERT_EQ(streams_status1.value().size(), 1);
    ASSERT_FALSE(stream->stopped());

    stream->setSeqLength(9);
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 1);
    ASSERT_FALSE(stream->stopped());
}

TEST_F(FIFOSchedulerTest, testReuseCache) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({5}, {1, 2, 3, 4, 5}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream1 =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());

    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);

    stream1->setFinishedWithoutLock();
    auto streams_status2 = scheduler.schedule();

    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
    query2->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto streams_status3 = scheduler.schedule();
    ASSERT_TRUE(streams_status3.ok());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 6);

    stream2->setFinishedWithoutLock();
    auto streams_status4 = scheduler.schedule();
    ASSERT_TRUE(streams_status4.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 6);
}

TEST_F(FIFOSchedulerTest, testMaxContextBatchSize) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;

    ModelConfig model_config;
    model_config.max_seq_len = 100;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 1;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size  = 100;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    {
        // test normalcase
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = createBuffer<int32_t>({5}, {1, 2, 3, 4, 5}, AllocationType::HOST);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream1 =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream1).ok());

        auto streams_status = scheduler.schedule();
        ASSERT_TRUE(streams_status.ok());

        stream1->setFinishedWithoutLock();
        auto streams_status2 = scheduler.schedule();

        ASSERT_TRUE(streams_status2.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    }

    {
        // test normal case with tile num
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = createBuffer<int32_t>({5}, {1, 2, 3, 4, 5}, AllocationType::HOST);
        query->generate_config               = make_shared<GenerateConfig>();
        query->generate_config->num_beams    = 2;
        shared_ptr<GenerateStream> stream1 =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream1).ok());

        auto streams_status = scheduler.schedule();
        ASSERT_TRUE(streams_status.ok());

        stream1->setFinishedWithoutLock();
        auto streams_status2 = scheduler.schedule();

        ASSERT_TRUE(streams_status2.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    }

    {
        // test abnormal case with tile num
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
        query2->input_ids                     = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
        query2->generate_config               = make_shared<GenerateConfig>();
        query2->generate_config->num_return_sequences = 20;
        shared_ptr<GenerateStream> stream2 =
            make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream2).ok());

        auto streams_status3 = scheduler.schedule();
        ASSERT_TRUE(streams_status3.ok());
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
        ASSERT_EQ(stream2->stopReason(), "input len [7] * batch size [20] > max_batch_tokens_size [100]");

        stream2->setFinishedWithoutLock();
        auto streams_status4 = scheduler.schedule();
        ASSERT_TRUE(streams_status4.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    }
}

TEST_F(FIFOSchedulerTest, testBatchEnqueue) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    vector<GenerateStreamPtr> streams;
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        streams.push_back(stream);
    }
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        streams.push_back(stream);
    }
    ASSERT_TRUE(scheduler.batchEnqueue(streams).ok());
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

TEST_F(FIFOSchedulerTest, testSchedule_ReturnEmpty_WhenAsyncLoadCacheTrue) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    auto ctx    = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Return(false));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    // Make StreamCacheResource::asyncLoadCache() return true without calling KVCacheManager::asyncLoadCache().
    stream->stream_cache_resource_->load_cache_context_ = ctx;

    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto status = scheduler.schedule();
    ASSERT_TRUE(status.ok());
    // New stream enters loading_cache_streams_ and won't be returned as running in the same schedule cycle.
    ASSERT_EQ(status.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 1);
    ASSERT_FALSE(scheduler.empty());
    ASSERT_EQ(scheduler.onflightStreams(), 1);
    ASSERT_TRUE(stream->loadingCache());
}

TEST_F(FIFOSchedulerTest, testSchedule_ReturnEmpty_WhenLoadCacheNotDone) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    auto ctx    = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Return(false));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    stream->stream_cache_resource_->load_cache_context_ = ctx;

    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    ASSERT_TRUE(scheduler.schedule().ok());
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);

    // Cache load still not done, it should remain in loading list.
    auto status2 = scheduler.schedule();
    ASSERT_TRUE(status2.ok());
    ASSERT_EQ(status2.value().size(), 0);
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testSchedule_ReturnNonEmpty_WhenLoadCacheDone) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;

    auto stream    = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    bool done_flag = false;
    auto ctx       = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Invoke([&done_flag]() { return done_flag; }));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    stream->stream_cache_resource_->load_cache_context_ = ctx;

    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    ASSERT_TRUE(scheduler.schedule().ok());
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);

    done_flag    = true;
    auto status2 = scheduler.schedule();
    ASSERT_TRUE(status2.ok());
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(status2.value().size(), 1);

    // Avoid asyncStoreCache() side effects on eviction by disabling it for this stream.
    stream->generateInput()->generate_config->enable_memory_cache = false;
    stream->setFinishedWithoutLock();
    auto status3 = scheduler.schedule();
    ASSERT_TRUE(status3.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testSchedule_ReturnEmpty_WhenLoadCacheDoneButStreamFinished) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;

    auto stream    = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    bool done_flag = false;
    auto ctx       = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Invoke([&done_flag]() { return done_flag; }));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    stream->stream_cache_resource_->load_cache_context_ = ctx;

    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    ASSERT_TRUE(scheduler.schedule().ok());
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 1);

    // Mark finished while still loading, then mark load done: scheduler should evict it from loading list.
    stream->setFinishedWithoutLock();
    done_flag    = true;
    auto status2 = scheduler.schedule();
    ASSERT_TRUE(status2.ok());
    ASSERT_EQ(status2.value().size(), 0);
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testScheduleNew_ReturnEmpty_WhenAsyncLoadCacheTrue) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;

    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->request_id                           = 201;
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;
    query->generate_config->timeout_ms          = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    auto ctx    = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Return(false));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    stream->stream_cache_resource_->load_cache_context_ = ctx;

    scheduler.waiting_streams_.push_back(stream);
    ASSERT_EQ(scheduler.waiting_streams_.size(), 1);

    auto new_streams = scheduler.scheduleNew(/*reserve_step=*/0);
    ASSERT_TRUE(new_streams.empty());
    ASSERT_EQ(scheduler.waiting_streams_.size(), 0);
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 1);
    ASSERT_TRUE(stream->loadingCache());
}

TEST_F(FIFOSchedulerTest, testScheduleNew_ReturnNonEmpty_WhenAsyncLoadCacheFalseAndSetRunningTrue) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.enable_memory_cache = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;

    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->request_id                           = 202;
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = false;
    query->generate_config->timeout_ms          = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    scheduler.waiting_streams_.push_back(stream);
    auto new_streams = scheduler.scheduleNew(/*reserve_step=*/0);
    ASSERT_EQ(new_streams.size(), 1);
    ASSERT_EQ(new_streams.front().get(), stream.get());
    ASSERT_EQ(scheduler.waiting_streams_.size(), 0);
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 0);
    ASSERT_TRUE(stream->running());
}

TEST_F(FIFOSchedulerTest, testScheduleNew_ReturnEmpty_WhenSetRunningFalse) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.enable_memory_cache = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;

    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->request_id                           = 203;
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = false;
    query->generate_config->timeout_ms          = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setStopWithoutLock(ErrorCode::CANCELLED, "cancel stream");

    scheduler.waiting_streams_.push_back(stream);
    auto new_streams = scheduler.scheduleNew(/*reserve_step=*/0);
    ASSERT_TRUE(new_streams.empty());
    // scheduleNew() does not erase this stream on setRunning() failure.
    ASSERT_EQ(scheduler.waiting_streams_.size(), 1);
    ASSERT_TRUE(stream->stopped());
}

TEST_F(FIFOSchedulerTest, testScheduleNew_ReturnEmpty_WhenEvaluateNewStreamFalseAndInputExceedsMaxAvailableTokens) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 2, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.enable_memory_cache = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;

    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->request_id                    = 204;
    query->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    query->generate_config->timeout_ms   = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    scheduler.waiting_streams_.push_back(stream);
    auto new_streams = scheduler.scheduleNew(/*reserve_step=*/0);
    ASSERT_TRUE(new_streams.empty());
    ASSERT_EQ(scheduler.waiting_streams_.size(), 1);
    ASSERT_TRUE(stream->stopped());
    ASSERT_NE(stream->stopReason().find("input len 3"), std::string::npos);
    ASSERT_NE(stream->stopReason().find("kv cache max available tokens num"), std::string::npos);
}

TEST_F(FIFOSchedulerTest,
       testScheduleNew_ReturnEmpty_WhenEvaluateNewStreamFalseAndBatchTokensExceedMaxBatchTokensSize) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.enable_memory_cache = false;

    ModelConfig model_config;
    // Make evaluateRunningMemory() use the max_batch_tokens_size_ constraint (avoid the early-return path),
    // while still being valid for a 2-token input.
    model_config.max_seq_len = 2;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 1;  // trigger input_len * batch > max

    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->request_id                    = 205;
    query->input_ids                     = createBuffer<int32_t>({2}, {1, 2}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    query->generate_config->timeout_ms   = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    scheduler.waiting_streams_.push_back(stream);
    auto new_streams = scheduler.scheduleNew(/*reserve_step=*/0);
    ASSERT_TRUE(new_streams.empty());
    ASSERT_EQ(scheduler.waiting_streams_.size(), 1);
    ASSERT_TRUE(stream->stopped());
    ASSERT_NE(stream->stopReason().find("max_batch_tokens_size"), std::string::npos);
}

TEST_F(FIFOSchedulerTest, testScheduleNew_ReturnEmpty_WhenEvaluateNewStreamFalse_OtherwiseLackMem) {
    CacheConfig cache_config = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    // Force initKVBlock() to fail (malloc should respect reserved blocks) while still keeping
    // maxAvailableTokensNum() large enough so we hit the "LACK MEM" branch (not "exceeds max len").
    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio           = 100;
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(
        cache_config, device_, /*warmup=*/false, /*metrics_reporter=*/nullptr, kv_cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.enable_memory_cache = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;

    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->request_id                    = 206;
    query->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    query->generate_config->timeout_ms   = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    scheduler.waiting_streams_.push_back(stream);
    auto new_streams = scheduler.scheduleNew(/*reserve_step=*/0);
    ASSERT_TRUE(new_streams.empty());
    ASSERT_EQ(scheduler.waiting_streams_.size(), 1);
    ASSERT_TRUE(stream->stopped());
    ASSERT_EQ(stream->stopReason(), "LACK MEM");
}

TEST_F(FIFOSchedulerTest, testEvaluateLoadingCacheStreams_ReturnEmpty_WhenLoadCacheNotDone) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->request_id                           = 101;
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;
    query->generate_config->timeout_ms          = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    auto ctx    = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Return(false));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    stream->stream_cache_resource_->load_cache_context_ = ctx;

    scheduler.loading_cache_streams_.push_back(stream);
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 1);

    auto done = scheduler.evaluateLoadingCacheStreams();
    ASSERT_TRUE(done.empty());
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 1);
}

TEST_F(FIFOSchedulerTest, testEvaluateLoadingCacheStreams_ReturnEmpty_WhenLoadCacheDoneAndStreamStopped) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->request_id                           = 102;
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;
    query->generate_config->timeout_ms          = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    auto ctx    = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Return(true));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    stream->stream_cache_resource_->load_cache_context_ = ctx;
    stream->setStopWithoutLock(ErrorCode::CANCELLED, "cancel stream");

    scheduler.loading_cache_streams_.push_back(stream);
    auto done = scheduler.evaluateLoadingCacheStreams();
    ASSERT_TRUE(done.empty());
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 0);
    ASSERT_TRUE(stream->stopped());
}

TEST_F(FIFOSchedulerTest, testEvaluateLoadingCacheStreams_ReturnEmpty_WhenLoadCacheDoneAndStreamFinished) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->request_id                           = 103;
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;
    query->generate_config->timeout_ms          = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    auto ctx    = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Return(true));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    stream->stream_cache_resource_->load_cache_context_ = ctx;
    stream->setFinishedWithoutLock();

    scheduler.loading_cache_streams_.push_back(stream);
    auto done = scheduler.evaluateLoadingCacheStreams();
    ASSERT_TRUE(done.empty());
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 0);
    ASSERT_TRUE(stream->finished());
}

TEST_F(FIFOSchedulerTest, testEvaluateLoadingCacheStreams_ReturnNonEmpty_WhenLoadCacheDoneAndSetRunningTrue) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_memory_cache = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->request_id                           = 104;
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->enable_memory_cache = true;
    query->generate_config->reuse_cache         = true;
    query->generate_config->timeout_ms          = 0;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    auto ctx    = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Return(true));
    ON_CALL(*ctx, success()).WillByDefault(testing::Return(true));
    stream->stream_cache_resource_->load_cache_context_ = ctx;
    ASSERT_TRUE(stream->asyncLoadCache());

    scheduler.loading_cache_streams_.push_back(stream);
    auto done = scheduler.evaluateLoadingCacheStreams();
    ASSERT_EQ(done.size(), 1);
    ASSERT_EQ(done.front().get(), stream.get());
    ASSERT_EQ(scheduler.loading_cache_streams_.size(), 0);
    ASSERT_TRUE(stream->running());
}

TEST_F(FIFOSchedulerTest, testEvictDoneStreams_EvictsFinishedStreamInWaitingList) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = false;
    resource_context.enable_memory_cache = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    query->generate_config->reuse_cache  = false;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setFinishedWithoutLock();

    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto status = scheduler.schedule();
    ASSERT_TRUE(status.ok());
    // evictDoneStreams(waiting_streams_) should remove it before scheduling.
    EXPECT_EQ(status.value().size(), 0u);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler.runningStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testEvictDoneStreams_EvictsFinishedStreamInRunningListAndReleasesBlocks) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = false;
    resource_context.enable_memory_cache = false;
    resource_context.enable_device_cache = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->reuse_cache         = false;
    query->generate_config->enable_memory_cache = false;
    query->generate_config->enable_device_cache = false;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    // First schedule: stream should become running and allocate blocks.
    auto status1 = scheduler.schedule();
    ASSERT_TRUE(status1.ok());
    ASSERT_EQ(status1.value().size(), 1u);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_GT(stream->curBlocksNum(), 0);

    // Mark finished, then next schedule should evict it from running_streams_.
    stream->setFinishedWithoutLock();
    auto status2 = scheduler.schedule();
    ASSERT_TRUE(status2.ok());
    EXPECT_EQ(status2.value().size(), 0u);
    EXPECT_EQ(scheduler.runningStreamsSize(), 0);
    // maybeReleaseResource() should have released kv blocks.
    EXPECT_EQ(stream->curBlocksNum(), 0);
}

TEST_F(FIFOSchedulerTest, testEvictDoneStreams_DoesNotEvictWhenNotDone) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = cache_manager;
    resource_context.reuse_cache         = false;
    resource_context.enable_memory_cache = false;
    resource_context.enable_device_cache = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
    query->input_ids                            = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config                      = make_shared<GenerateConfig>();
    query->generate_config->reuse_cache         = false;
    query->generate_config->enable_memory_cache = false;
    query->generate_config->enable_device_cache = false;

    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto status1 = scheduler.schedule();
    ASSERT_TRUE(status1.ok());
    ASSERT_EQ(status1.value().size(), 1u);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    // Not finished/stopped: a second schedule should keep it in running list.
    auto status2 = scheduler.schedule();
    ASSERT_TRUE(status2.ok());
    EXPECT_EQ(scheduler.runningStreamsSize(), 1);
}

}  // namespace rtp_llm
