#include <memory>
#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

using namespace std;

namespace rtp_llm {

class FIFOSchedulerTest: public DeviceTestBase {
public:
    FIFOSchedulerTest() {}
};

TEST_F(FIFOSchedulerTest, testSimple) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
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
    query->input_ids                     = torch::tensor({1}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false without enable_memory_cache)
    // Since no cache loading is needed, stream transitions directly to RUNNING in one schedule call
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    stream->reportEvent(StreamEvents::GenerateDone);

    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);
}

TEST_F(FIFOSchedulerTest, testInitKVCacheLackMem) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 2, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
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
    query->input_ids                     = torch::tensor({1, 2, 3}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    // In the new code, checkInputLength rejects at enqueue time
    ASSERT_FALSE(scheduler.enqueue(stream).ok());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "input len 3 is greater than kv cache max available tokens num 2");
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);
}

TEST_F(FIFOSchedulerTest, testIncrKVCacheLackMem) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 3, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
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
    query->input_ids                     = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false)
    // Since no cache loading is needed, stream transitions directly to RUNNING in one schedule call
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 1);
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "");
    ASSERT_EQ(cache_manager->freeBlocksNum(), 0);

    stream->setSeqLength(stream->seqLength() + 1);
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_TRUE(stream->hasError());
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
        std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr, kv_cache_config);
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
    query->input_ids                     = torch::tensor({1, 2, 3, 4, 5, 6}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();

    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 0);
    ASSERT_TRUE(stream->hasError());
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
        std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr, kv_cache_config);
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
    query->input_ids                     = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();

    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false)
    // Since no cache loading is needed, stream transitions directly to RUNNING in one schedule call
    auto streams_status1 = scheduler.schedule();
    ASSERT_TRUE(streams_status1.ok());
    ASSERT_EQ(streams_status1.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_FALSE(stream->hasError());

    stream->setSeqLength(9);
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 1);
    ASSERT_FALSE(stream->hasError());
}

TEST_F(FIFOSchedulerTest, testReuseCache) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
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
    query->input_ids                     = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream1 =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false without enable_memory_cache)
    // Since no cache loading is needed, stream transitions directly to RUNNING in one schedule call
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    // Stream is already running, no need for second schedule
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);

    stream1->reportEvent(StreamEvents::GenerateDone);
    auto streams_status3 = scheduler.schedule();

    ASSERT_TRUE(streams_status3.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 8);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
    query2->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    // Third schedule for stream2: transitions to RUNNING in single call (no cache loading needed)
    auto streams_status4 = scheduler.schedule();
    ASSERT_TRUE(streams_status4.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 6);

    stream2->reportEvent(StreamEvents::GenerateDone);
    auto streams_status6 = scheduler.schedule();
    ASSERT_TRUE(streams_status6.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);
}

TEST_F(FIFOSchedulerTest, testMaxContextBatchSize) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = true;

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
        query->input_ids                     = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream1 =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream1).ok());

        // Single schedule: transitions to RUNNING (no cache loading needed)
        auto streams_status = scheduler.schedule();
        ASSERT_TRUE(streams_status.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 1);

        stream1->reportEvent(StreamEvents::GenerateDone);
        auto streams_status2 = scheduler.schedule();

        ASSERT_TRUE(streams_status2.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    }

    {
        // test normal case with tile num
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        query->generate_config->num_beams    = 2;
        shared_ptr<GenerateStream> stream1 =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream1).ok());

        // Single schedule: transitions to RUNNING (no cache loading needed)
        auto streams_status = scheduler.schedule();
        ASSERT_TRUE(streams_status.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 1);

        stream1->reportEvent(StreamEvents::GenerateDone);
        auto streams_status2 = scheduler.schedule();

        ASSERT_TRUE(streams_status2.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    }

    {
        // After fix (commit 2238d50cf): checkInputLength no longer rejects when
        // inputLength * currentBatchSize > max_batch_tokens_size. Such requests must
        // be admitted to the waiting queue; the per-round token-budget constraint is
        // enforced later in evaluateRunningMemory using contextLength (not batch_size).
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        std::shared_ptr<GenerateInput> query2         = make_shared<GenerateInput>();
        query2->input_ids                             = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query2->generate_config                       = make_shared<GenerateConfig>();
        query2->generate_config->num_return_sequences = 20;
        shared_ptr<GenerateStream> stream2 =
            make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
        // input_len 7 * batch_size 20 = 140 > max_batch_tokens_size 100, but enqueue still succeeds.
        ASSERT_TRUE(scheduler.enqueue(stream2).ok());
        ASSERT_FALSE(stream2->hasError());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    }
}

// Regression test for commit 2238d50cf: removing the
// `inputLength * currentBatchSize() > max_batch_tokens_size_` rejection in
// FIFOScheduler::checkInputLength. The check was wrong because max_batch_tokens_size
// bounds per-scheduling-round token usage (computed from contextLength in
// evaluateRunningMemory), not the multi-sequence batch fan-out at enqueue.
TEST_F(FIFOSchedulerTest, testCheckInputLengthIgnoresBatchSizeFanOut) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    // max_seq_len must exceed the longest input below (maxAvailableTokensNum + 1) so that
    // stream construction does not throw on its own seq_length check; the scheduler-level
    // checkInputLength is what we actually want to exercise here.
    model_config.max_seq_len = 1024;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 100;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    {
        // num_return_sequences fan-out: input_len 7 * batch 20 = 140 > 100, but accepted.
        std::shared_ptr<GenerateInput> query         = make_shared<GenerateInput>();
        query->input_ids                             = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->num_return_sequences = 20;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        ASSERT_FALSE(stream->hasError());
    }

    {
        // num_beams fan-out: input_len 10 * batch 16 = 160 > 100, but accepted.
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        query->generate_config->num_beams    = 16;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        ASSERT_FALSE(stream->hasError());
    }

    // The KV-cache-bound check still rejects (input_len > maxAvailableTokensNum).
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        // Cache has 20 blocks * 8 tokens/block - 1 reserved tail = 159 max available; pick a length above it.
        std::vector<int32_t> ids(int(cache_manager->maxAvailableTokensNum()) + 1, 1);
        query->input_ids       = torch::tensor(ids, torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_FALSE(scheduler.enqueue(stream).ok());
        ASSERT_TRUE(stream->hasError());
    }
}

TEST_F(FIFOSchedulerTest, testBatchEnqueue) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
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
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        streams.push_back(stream);
    }
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        streams.push_back(stream);
    }
    auto enqueued = scheduler.batchEnqueue(streams);
    ASSERT_EQ(enqueued.size(), streams.size());

    // Single schedule: both streams transition to RUNNING (no cache loading needed)
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

TEST_F(FIFOSchedulerTest, testForceBatchGroupComplete) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
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

    int64_t group_id   = 100;
    int     group_size = 3;

    // Enqueue only 2 of 3 — group incomplete, should not be scheduled
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // First schedule: streams stay in WAITING (group incomplete, cannot run yet)
    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);

    // Enqueue the 3rd — group complete, all 3 should be scheduled together
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // Second schedule: group complete, all 3 streams transition to RUNNING in single call
    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
}

TEST_F(FIFOSchedulerTest, testForceBatchTimeout) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
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

    int64_t group_id   = 200;
    int     group_size = 3;
    int     timeout_ms = 10;
    int64_t past_time  = autil::TimeUtility::currentTimeInMicroSeconds() - (timeout_ms + 100) * 1000;

    // Enqueue only 2 of 3 with begin_time far in the past so timeout has expired
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = timeout_ms;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = past_time;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = timeout_ms;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = past_time;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // Single schedule: timeout expired, streams transition to RUNNING
    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testForceBatchIsolation) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
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

    int64_t group_id   = 300;
    int     group_size = 2;

    // Enqueue: normal stream first, then a complete force batch group
    shared_ptr<GenerateStream> normal_stream;
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        query->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
        normal_stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(normal_stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // Round 1: normal stream transitions to RUNNING (force batch streams skipped due to batch isolation)
    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    // Finish the normal stream
    normal_stream->reportEventWithoutLock(StreamEvents::GenerateDone);

    // Round 2: force batch group transitions to RUNNING
    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

// Two different complete force batch groups: only one group per scheduling round
TEST_F(FIFOSchedulerTest, testTwoForceBatchGroupsIsolation) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
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

    int64_t group_id_a = 500;
    int64_t group_id_b = 600;
    int     group_size = 2;

    // Enqueue group A (2 streams), then group B (2 streams), both complete
    vector<shared_ptr<GenerateStream>> group_a_streams;
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id_a;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        group_a_streams.push_back(stream);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id_b;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // Round 1: group A transitions to RUNNING (group B skipped due to batch isolation)
    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);

    // Finish group A
    for (auto& s : group_a_streams) {
        s->reportEventWithoutLock(StreamEvents::GenerateDone);
    }

    // Round 2: group B transitions to RUNNING
    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

TEST_F(FIFOSchedulerTest, testPrefillRunsBeforeDecodeBatchWhenKVEnough) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> decode_query = make_shared<GenerateInput>();
    decode_query->request_id                      = 1;
    decode_query->input_ids                       = torch::tensor({1}, torch::kInt32);
    decode_query->generate_config                 = make_shared<GenerateConfig>();
    decode_query->generate_config->max_new_tokens = 4;
    auto decode_running =
        make_shared<NormalGenerateStream>(decode_query, model_config, runtime_config, resource_context, nullptr);
    decode_running->setIsContextStream(false);
    ASSERT_TRUE(scheduler.enqueue(decode_running).ok());
    auto first_round = scheduler.schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    std::shared_ptr<GenerateInput> decode_wait_query = make_shared<GenerateInput>();
    decode_wait_query->request_id                      = 2;
    decode_wait_query->input_ids                       = torch::tensor({2}, torch::kInt32);
    decode_wait_query->generate_config                 = make_shared<GenerateConfig>();
    decode_wait_query->generate_config->max_new_tokens = 4;
    auto decode_waiting =
        make_shared<NormalGenerateStream>(decode_wait_query, model_config, runtime_config, resource_context, nullptr);
    decode_waiting->setIsContextStream(false);
    ASSERT_TRUE(scheduler.enqueue(decode_waiting).ok());

    std::shared_ptr<GenerateInput> prefill_query = make_shared<GenerateInput>();
    prefill_query->request_id                       = 3;
    prefill_query->input_ids                         = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    prefill_query->generate_config                   = make_shared<GenerateConfig>();
    prefill_query->generate_config->max_new_tokens   = 4;
    auto prefill_waiting =
        make_shared<NormalGenerateStream>(prefill_query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(prefill_waiting).ok());

    auto second_round = scheduler.schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round.value().size(), 1);
    ASSERT_EQ(second_round.value().front()->streamId(), prefill_waiting->streamId());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);

    prefill_waiting->reportEventWithoutLock(StreamEvents::GenerateDone);
    auto third_round = scheduler.schedule();
    ASSERT_TRUE(third_round.ok());
    ASSERT_EQ(third_round.value().size(), 1);
    std::vector<int64_t> running_ids;
    for (const auto& stream : third_round.value()) {
        running_ids.push_back(stream->streamId());
    }
    ASSERT_EQ(running_ids.size(), 1);
    ASSERT_EQ(running_ids.front(), decode_running->streamId());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, testPrefillWaitsWhenKVInsufficient) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> decode_query = make_shared<GenerateInput>();
    decode_query->request_id                            = 1;
    decode_query->input_ids                             = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    decode_query->generate_config                       = make_shared<GenerateConfig>();
    auto decode_running =
        make_shared<NormalGenerateStream>(decode_query, model_config, runtime_config, resource_context, nullptr);
    decode_running->setIsContextStream(false);
    ASSERT_TRUE(scheduler.enqueue(decode_running).ok());
    auto first_round = scheduler.schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round.value().size(), 1);

    std::shared_ptr<GenerateInput> prefill_query = make_shared<GenerateInput>();
    prefill_query->request_id                    = 2;
    prefill_query->input_ids                     = torch::tensor({5, 6, 7}, torch::kInt32);
    prefill_query->generate_config               = make_shared<GenerateConfig>();
    auto prefill_waiting =
        make_shared<NormalGenerateStream>(prefill_query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(prefill_waiting).ok());

    auto second_round = scheduler.schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round.value().size(), 1);
    ASSERT_EQ(second_round.value().front()->streamId(), decode_running->streamId());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, testPreferPrefillSchedulesMultiplePrefills) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Step 1: put a decode stream into running
    std::shared_ptr<GenerateInput> decode_query = make_shared<GenerateInput>();
    decode_query->request_id                      = 1;
    decode_query->input_ids                       = torch::tensor({1}, torch::kInt32);
    decode_query->generate_config                 = make_shared<GenerateConfig>();
    decode_query->generate_config->max_new_tokens = 4;
    auto decode_running =
        make_shared<NormalGenerateStream>(decode_query, model_config, runtime_config, resource_context, nullptr);
    decode_running->setIsContextStream(false);
    ASSERT_TRUE(scheduler.enqueue(decode_running).ok());
    auto first_round = scheduler.schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round.value().size(), 1);

    int64_t group_id   = 400;
    int     group_size = 2;

    // Step 2: enqueue force_batch prefill #1 from group A
    std::shared_ptr<GenerateInput> fb_query_1        = make_shared<GenerateInput>();
    fb_query_1->request_id                           = 10;
    fb_query_1->input_ids                            = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    fb_query_1->generate_config                      = make_shared<GenerateConfig>();
    fb_query_1->generate_config->max_new_tokens      = 4;
    fb_query_1->generate_config->force_batch         = true;
    fb_query_1->generate_config->batch_group_timeout = 10000;
    fb_query_1->batch_group_id                       = group_id;
    fb_query_1->batch_group_size                     = group_size;
    fb_query_1->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
    auto fb_stream_1 =
        make_shared<NormalGenerateStream>(fb_query_1, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(fb_stream_1).ok());

    // Step 3: enqueue a normal (non-force_batch) prefill in between
    std::shared_ptr<GenerateInput> normal_query = make_shared<GenerateInput>();
    normal_query->request_id                      = 20;
    normal_query->input_ids                       = torch::tensor({5, 6, 7, 8}, torch::kInt32);
    normal_query->generate_config                 = make_shared<GenerateConfig>();
    normal_query->generate_config->max_new_tokens = 4;
    auto normal_stream =
        make_shared<NormalGenerateStream>(normal_query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(normal_stream).ok());

    // Step 4: enqueue force_batch prefill #2 from same group A
    std::shared_ptr<GenerateInput> fb_query_2        = make_shared<GenerateInput>();
    fb_query_2->request_id                           = 11;
    fb_query_2->input_ids                            = torch::tensor({1, 2}, torch::kInt32);
    fb_query_2->generate_config                      = make_shared<GenerateConfig>();
    fb_query_2->generate_config->max_new_tokens      = 4;
    fb_query_2->generate_config->force_batch         = true;
    fb_query_2->generate_config->batch_group_timeout = 10000;
    fb_query_2->batch_group_id                       = group_id;
    fb_query_2->batch_group_size                     = group_size;
    fb_query_2->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
    auto fb_stream_2 =
        make_shared<NormalGenerateStream>(fb_query_2, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(fb_stream_2).ok());

    // Step 5: schedule — both group-A prefills scheduled, normal prefill stays in waiting
    auto second_round = scheduler.schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round.value().size(), 2);
    std::set<int64_t> scheduled_ids;
    for (const auto& s : second_round.value()) {
        scheduled_ids.insert(s->streamId());
    }
    ASSERT_TRUE(scheduled_ids.count(fb_stream_1->streamId()));
    ASSERT_TRUE(scheduled_ids.count(fb_stream_2->streamId()));
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
}

TEST_F(FIFOSchedulerTest, testForceBatchGroupNotSplitByPreferPrefill) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Step 1: put a decode stream into running
    std::shared_ptr<GenerateInput> decode_query = make_shared<GenerateInput>();
    decode_query->request_id                      = 1;
    decode_query->input_ids                       = torch::tensor({1}, torch::kInt32);
    decode_query->generate_config                 = make_shared<GenerateConfig>();
    decode_query->generate_config->max_new_tokens = 4;
    auto decode_running =
        make_shared<NormalGenerateStream>(decode_query, model_config, runtime_config, resource_context, nullptr);
    decode_running->setIsContextStream(false);
    ASSERT_TRUE(scheduler.enqueue(decode_running).ok());
    auto first_round = scheduler.schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    // Step 2: enqueue a complete force_batch group of 3 prefill streams
    int64_t group_id   = 300;
    int     group_size = 3;
    std::vector<shared_ptr<GenerateStream>> prefill_streams;
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->request_id                           = 10 + i;
        query->input_ids                            = torch::tensor({1, 2}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens      = 4;
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10000;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        prefill_streams.push_back(stream);
    }

    // Step 3: schedule — all 3 force_batch prefills must be scheduled in one round
    auto second_round = scheduler.schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round.value().size(), 3);
    for (const auto& s : second_round.value()) {
        ASSERT_TRUE(s->isContextStream());
    }
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 4);
}

TEST_F(FIFOSchedulerTest, testPrefillRejectedWhenRunningDecodeNeedsBlocksAtBoundary) {
    // tokens_per_block=4, block_num=8 → 7 usable, reserve=0 (5% of 7 rounds to 0)
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 8, 1, 4, 4, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Put 3 decode streams into running, each with seqLen=3, max_new_tokens=8
    // Each: peak = ceil((3+8)/4) = 3 blocks, currently holds 1 → estimatePeak = 2
    std::vector<shared_ptr<GenerateStream>> decode_streams;
    for (int i = 0; i < 3; i++) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->request_id                      = i + 1;
        query->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 8;
        auto stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(false);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        decode_streams.push_back(stream);
    }
    auto first_round = scheduler.schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round.value().size(), 3);
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
    // 3 decode streams each hold 1 block → 3 used, 4 free (7 total - 3 used)

    // Enqueue a prefill with seqLen=3, max_new_tokens=4 → peak = ceil((3+4)/4) = 2 blocks
    // Running estimate: 3 × 2 = 6, candidate: 2, total = 8
    // available = 4, reserve = 0, effective = 4
    // 8 > 4 → should be REJECTED
    std::shared_ptr<GenerateInput> prefill_query = make_shared<GenerateInput>();
    prefill_query->request_id                      = 10;
    prefill_query->input_ids                       = torch::tensor({4, 5, 6}, torch::kInt32);
    prefill_query->generate_config                 = make_shared<GenerateConfig>();
    prefill_query->generate_config->max_new_tokens = 4;
    auto prefill_stream =
        make_shared<NormalGenerateStream>(prefill_query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(prefill_stream).ok());

    auto second_round = scheduler.schedule();
    ASSERT_TRUE(second_round.ok());
    // Prefill should NOT be scheduled — KV too tight for running decode's future needs
    ASSERT_EQ(second_round.value().size(), 3);  // only the 3 existing decode streams
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
}

// Same boundary test but with hybrid attention (1 linear group + 1 full group).
// Linear attention's estimatePeakNeedBlocks must use slot-based formula (not steady-state hold)
// so that the scheduler correctly accounts for future block boundary crossings across both groups.
TEST_F(FIFOSchedulerTest, testPrefillRejectedWithHybridAttentionAtBoundary) {
    // Build a hybrid CacheConfig: 4 layers, [0,1]=linear, [2,3]=full, tokens_per_block=4
    CacheConfig config;
    config.dtype                     = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                 = 4;
    config.layer_all_num             = 4;
    config.block_num                 = 12;
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 2;
    config.linear_step               = 2;
    config.group_layer_num           = 2;

    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = config.dtype;
    linear_spec->layer_num          = 2;
    linear_spec->local_num_k_heads  = 1;
    linear_spec->local_num_v_heads  = 1;
    linear_spec->head_k_dim         = 1;
    linear_spec->head_v_dim         = 1;
    linear_spec->conv_kernel_dim    = 2;
    linear_spec->local_head_num_kv  = 1;
    linear_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = config.dtype;
    full_spec->layer_num          = 2;
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    config.layer_ids        = {{0, 1}, {2, 3}};
    config.global_layer_ids = config.layer_ids;
    config.cache_specs      = {linear_spec, full_spec};
    config.linear_group_num = 1;
    config.full_group_num   = 1;
    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes;
    config.layer_to_group_id.assign(static_cast<size_t>(config.layer_num), 0);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int lid : config.layer_ids[gid]) {
            config.layer_to_group_id[static_cast<size_t>(lid)] = static_cast<int>(gid);
        }
    }

    // block_num=12 → 11 usable, reserve=0 (5% of 11 rounds to 0)
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // 2 decode streams, each seqLen=3, max_new_tokens=8
    // Each group: peak = ceil((3+8)/4) = 3 slots, after initMalloc holds 1 slot → estimatePeak = 2 per group
    // Per stream: 2 groups × 2 = 4 incremental blocks
    std::vector<shared_ptr<GenerateStream>> decode_streams;
    for (int i = 0; i < 2; i++) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->request_id                      = i + 1;
        query->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 8;
        auto stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(false);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        decode_streams.push_back(stream);
    }
    auto first_round = scheduler.schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round.value().size(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    // 2 streams × 2 groups × 1 block = 4 used, 7 free (11 - 4)

    // Prefill: seqLen=3, max_new_tokens=8 → 2 groups × 2 = 4 incremental blocks
    // Total: running 2×4=8, candidate 4 = 12
    // available = 7, effective = 7
    // 12 > 7 → REJECTED
    std::shared_ptr<GenerateInput> prefill_query = make_shared<GenerateInput>();
    prefill_query->request_id                      = 10;
    prefill_query->input_ids                       = torch::tensor({4, 5, 6}, torch::kInt32);
    prefill_query->generate_config                 = make_shared<GenerateConfig>();
    prefill_query->generate_config->max_new_tokens = 8;
    auto prefill_stream =
        make_shared<NormalGenerateStream>(prefill_query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(prefill_stream).ok());

    auto second_round = scheduler.schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round.value().size(), 2);  // only existing decode streams
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

// Peak estimation allows a short-lived prefill to be admitted alongside long-running decodes,
// while naive per-stream accumulation would reject it.
TEST_F(FIFOSchedulerTest, testPeakEstimationAllowsMoreConcurrency) {
    // tokens_per_block=4, block_num=12 → 11 usable, reserve=0
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 12, 1, 4, 4, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                              = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size         = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // 2 long-running decode streams: seqLen=4, max_new_tokens=16 (b=16)
    std::vector<shared_ptr<GenerateStream>> decode_streams;
    for (int i = 0; i < 2; i++) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->request_id                      = i + 1;
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 16;
        auto stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(false);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        decode_streams.push_back(stream);
    }
    auto first_round = scheduler.schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round.value().size(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);

    // Short-lived prefill: seqLen=4, max_new_tokens=2 (b=2)
    // Lifetime: peak_tokens=40 → peak_blocks=10 ≤ totalBlocks=12 → ACCEPT
    std::shared_ptr<GenerateInput> prefill_query = make_shared<GenerateInput>();
    prefill_query->request_id                      = 10;
    prefill_query->input_ids                       = torch::tensor({5, 6, 7, 8}, torch::kInt32);
    prefill_query->generate_config                 = make_shared<GenerateConfig>();
    prefill_query->generate_config->max_new_tokens = 2;
    auto prefill_stream =
        make_shared<NormalGenerateStream>(prefill_query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(prefill_stream).ok());

    auto second_round = scheduler.schedule();
    ASSERT_TRUE(second_round.ok());
    // Lifetime estimation accepts the prefill
    ASSERT_EQ(second_round.value().size(), 1);
    ASSERT_EQ(second_round.value().front()->streamId(), prefill_stream->streamId());
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
}

// Even with peak estimation, tight KV capacity correctly rejects new prefill.
TEST_F(FIFOSchedulerTest, testPeakEstimationRejectWhenTight) {
    // tokens_per_block=4, block_num=10 → 9 usable, reserve=0
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 4, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                              = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size         = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // 2 long-running decode streams: seqLen=4, max_new_tokens=16
    std::vector<shared_ptr<GenerateStream>> decode_streams;
    for (int i = 0; i < 2; i++) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->request_id                      = i + 1;
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 16;
        auto stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(false);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        decode_streams.push_back(stream);
    }
    auto first_round = scheduler.schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round.value().size(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    // 2 streams each hold 1 block → 7 available

    // Prefill with seqLen=8, max_new_tokens=8
    // Peak: ab=[(4,16),(4,16),(8,8)] → peak_tokens=40, peak_blocks=10, held=2, incremental=8
    // available=7, 8 > 7 → REJECT
    std::shared_ptr<GenerateInput> prefill_query = make_shared<GenerateInput>();
    prefill_query->request_id                      = 10;
    prefill_query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7, 8}, torch::kInt32);
    prefill_query->generate_config                 = make_shared<GenerateConfig>();
    prefill_query->generate_config->max_new_tokens = 8;
    auto prefill_stream =
        make_shared<NormalGenerateStream>(prefill_query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(prefill_stream).ok());

    auto second_round = scheduler.schedule();
    ASSERT_TRUE(second_round.ok());
    // Should be REJECTED — only existing decode streams returned
    ASSERT_EQ(second_round.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

}  // namespace rtp_llm
