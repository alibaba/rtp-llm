#include <memory>
#include <chrono>
#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "autil/TimeUtility.h"

#define protected public
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class FIFOSchedulerTest: public DeviceTestBase {
public:
    FIFOSchedulerTest() {}
};

static PDSepConfig makePDFusionPDSepConfig() {
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    return pd_sep_config;
}

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

// ---------------------------------------------------------------------------
// Helper used by the prefill-first cadence / KV-gate tests (Tasks 5–7)
// ---------------------------------------------------------------------------

static std::shared_ptr<GenerateStream> makeStream(const std::vector<int>& ids,
                                                  const ModelConfig&      model_config,
                                                  const RuntimeConfig&    runtime_config,
                                                  const ResourceContext&  resource_context) {
    auto query             = std::make_shared<GenerateInput>();
    query->input_ids       = torch::tensor(ids, torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();
    return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
}

static std::shared_ptr<GenerateStream> makeForceBatchStream(const std::vector<int>& ids,
                                                            int64_t                 group_id,
                                                            int                     group_size,
                                                            const ModelConfig&      model_config,
                                                            const RuntimeConfig&    runtime_config,
                                                            const ResourceContext&  resource_context) {
    auto query                                  = std::make_shared<GenerateInput>();
    query->input_ids                            = torch::tensor(ids, torch::kInt32);
    query->generate_config                      = std::make_shared<GenerateConfig>();
    query->generate_config->force_batch         = true;
    query->generate_config->batch_group_timeout = 1000;
    query->batch_group_id                       = group_id;
    query->batch_group_size                     = group_size;
    query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
    return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
}

// ---------------------------------------------------------------------------
// Task 5: cadence tests (strict alternation S=1, decode-heavy S=3, prefill-heavy S=-3)
// ---------------------------------------------------------------------------

TEST_F(FIFOSchedulerTest, testPrefillFirstAlternation) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1";  // strict alternation
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());

    // R1: seed PREFILL (running+pending empty). Admits s1 -> pending; not yet running.
    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    s1->setSeqLength(s1->seqLength() + 1);  // simulate prefill forward

    // Enqueue s2 AFTER the seed so it stays waiting (proves cadence, not just "no work").
    auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    // R2: DECODE (decode_since_prefill_=0 < 1). Promotes s1 into running; s2 stays waiting.
    auto r2 = scheduler.schedule();
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(r2.value().size(), 1);  // pure-decode batch (s1)
    s1->setSeqLength(s1->seqLength() + 1);

    // R3: PREFILL (decode_since_prefill_=1 >= 1). Admits s2 (pure context); s1 held back in running.
    auto r3 = scheduler.schedule();
    ASSERT_TRUE(r3.ok());
    ASSERT_EQ(r3.value().size(), 1);                     // pure-context batch (s2 only)
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);        // s1 still running, held back
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);  // s2 pending
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testDecodeHeavyCadence) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "3";  // 1 prefill : 3 decode
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());

    // R1: seed PREFILL s1.
    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 1);
    s1->setSeqLength(s1->seqLength() + 1);

    // Keep s2 waiting throughout to prove the 3 decode rounds are cadence-forced (not "no work").
    auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    // R2,R3,R4: DECODE (decode_since_prefill_ = 0,1,2 < 3). s2 must stay waiting each round.
    for (int i = 0; i < 3; ++i) {
        auto r = scheduler.schedule();
        ASSERT_TRUE(r.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 1) << "decode round " << i << " must not admit s2";
        s1->setSeqLength(s1->seqLength() + 1);
    }

    // R5: PREFILL (decode_since_prefill_ == 3 >= 3) -> admits s2.
    auto r5 = scheduler.schedule();
    ASSERT_TRUE(r5.ok());
    ASSERT_EQ(r5.value().size(), 1);  // pure-context (s2)
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testDecodeRoundReapsErroredWaitingStream) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "3";  // keep s2 waiting during decode rounds
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());

    auto r1 = scheduler.schedule();  // PREFILL s1
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 1);
    s1->setSeqLength(s1->seqLength() + 1);

    auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    auto r2 = scheduler.schedule();  // DECODE s1, keep s2 waiting
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(r2.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    s1->setSeqLength(s1->seqLength() + 1);

    s2->reportError(ErrorCode::CANCELLED, "cancelled while waiting");

    auto r3 = scheduler.schedule();  // still a DECODE round; should reap cancelled s2 without admitting it
    ASSERT_TRUE(r3.ok());
    ASSERT_EQ(r3.value().size(), 1);
    ASSERT_EQ(r3.value().front().get(), s1.get());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_TRUE(s2->isFinished());
    ASSERT_EQ(s2->stopReason(), "cancelled while waiting");
}

TEST_F(FIFOSchedulerTest, testInvalidDecodePrefillRatioFallsBackToAlternation) {
    const std::vector<std::string> invalid_ratios = {"", "0", "1/0", "-1", "abc", "2/3"};
    for (const auto& ratio : invalid_ratios) {
        CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
        auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
        ASSERT_TRUE(cache_manager->init());
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                     = 100;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
        runtime_config.fifo_scheduler_config.decode_prefill_ratio  = ratio;
        PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
        ParallelismConfig      parallelism_config;
        ModelSpecificConfig    model_specific_config;
        PDFusionRatioScheduler scheduler(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

        auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
        ASSERT_TRUE(scheduler.enqueue(s1).ok()) << ratio;
        auto seed = scheduler.schedule();
        ASSERT_TRUE(seed.ok()) << ratio;
        ASSERT_EQ(seed.value().size(), 1) << ratio;
        s1->setSeqLength(s1->seqLength() + 1);

        auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
        ASSERT_TRUE(scheduler.enqueue(s2).ok()) << ratio;
        auto decode = scheduler.schedule();
        ASSERT_TRUE(decode.ok()) << ratio;
        ASSERT_EQ(decode.value().size(), 1) << ratio;
        ASSERT_EQ(decode.value().front().get(), s1.get()) << ratio;
        s1->setSeqLength(s1->seqLength() + 1);

        auto prefill = scheduler.schedule();
        ASSERT_TRUE(prefill.ok()) << ratio;
        ASSERT_EQ(prefill.value().size(), 1) << ratio;
        ASSERT_EQ(prefill.value().front().get(), s2.get()) << ratio;
    }
}

TEST_F(FIFOSchedulerTest, testDecodeHeavyCadenceSeedsAfterInFlightDrains) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "3";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    auto seed = scheduler.schedule();
    ASSERT_TRUE(seed.ok());
    ASSERT_EQ(seed.value().size(), 1);
    s1->setSeqLength(s1->seqLength() + 1);

    auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s2).ok());
    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode.value().size(), 1);
    ASSERT_EQ(decode.value().front().get(), s1.get());

    s1->reportEventWithoutLock(StreamEvents::GenerateDone);
    auto reseed = scheduler.schedule();
    ASSERT_TRUE(reseed.ok());
    ASSERT_EQ(reseed.value().size(), 1);
    ASSERT_EQ(reseed.value().front().get(), s2.get());
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, testPrefillHeavyCadence) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1/3";  // 3 prefill : 1 decode
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Enqueue ONE stream right before each of the 3 prefill rounds so each admits exactly one,
    // accumulating in the pending pool (multi-admit would otherwise grab all at once).
    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    auto r1 = scheduler.schedule();  // seed PREFILL s1
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 1);
    s1->setSeqLength(s1->seqLength() + 1);

    auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s2).ok());
    auto r2 = scheduler.schedule();  // PREFILL (prefill_since_decode_=1 < 3) -> s2
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(r2.value().size(), 1);
    s2->setSeqLength(s2->seqLength() + 1);

    auto s3 = makeStream({5, 6}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s3).ok());
    auto r3 = scheduler.schedule();  // PREFILL (prefill_since_decode_=2 < 3) -> s3
    ASSERT_TRUE(r3.ok());
    ASSERT_EQ(r3.value().size(), 1);
    s3->setSeqLength(s3->seqLength() + 1);

    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 3);  // s1,s2,s3 all pending
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);

    // Keep s4 waiting to prove the 4th round is cadence-forced DECODE (not "no work").
    auto s4 = makeStream({7, 8}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s4).ok());

    // R4: DECODE (prefill_since_decode_ == 3, not < 3) -> promotes s1,s2,s3; s4 stays waiting.
    auto r4 = scheduler.schedule();
    ASSERT_TRUE(r4.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);  // s4 not admitted: cadence forced decode
}

// ---------------------------------------------------------------------------
// Task 6: legacy-via-large-N, concurrency cap counts pending
// ---------------------------------------------------------------------------

TEST_F(FIFOSchedulerTest, testLargeStepDecodeFirst) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "20000";  // legacy decode-drain
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    auto r1 = scheduler.schedule();  // seed PREFILL s1
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 1);
    s1->setSeqLength(s1->seqLength() + 1);

    // s2 queued but must NOT be admitted while s1 decodes (huge step => always decode).
    auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    for (int i = 0; i < 3; ++i) {
        auto r = scheduler.schedule();  // DECODE rounds (decode_since_prefill_ << 20000)
        ASSERT_TRUE(r.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 1) << "s2 must stay waiting during decode-drain";
        s1->setSeqLength(s1->seqLength() + 1);
    }

    // Finish s1; once running+pending drain, the seed branch admits s2.
    s1->reportEvent(StreamEvents::GenerateDone);
    auto r_after = scheduler.schedule();  // reaps s1 (running empty), then seed PREFILL s2
    ASSERT_TRUE(r_after.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);  // s2 finally admitted
}

TEST_F(FIFOSchedulerTest, testConcurrencyCapCountsPending) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 2;  // cap = 2 in-flight
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1/100";  // very prefill-heavy
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::vector<std::shared_ptr<GenerateStream>> streams;
    for (int i = 0; i < 4; ++i) {
        auto s = makeStream({1, 2}, model_config, runtime_config, resource_context);
        streams.push_back(s);
        ASSERT_TRUE(scheduler.enqueue(s).ok());
    }

    // R1: seed PREFILL. Cap = running(0)+pending(0)+streams+1 > 2 => admits exactly 2, rejects 2.
    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 2);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    streams[0]->setSeqLength(streams[0]->seqLength() + 1);
    streams[1]->setSeqLength(streams[1]->seqLength() + 1);

    // R2: cadence says PREFILL but cap is full (pending=2) => admits nothing => degrades to DECODE,
    // promoting the 2 pending into running. running+pending stays at the cap; the other 2 stay waiting.
    auto r2 = scheduler.schedule();
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(scheduler.runningStreamsSize() + scheduler.pendingDecodeStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
}

TEST_F(FIFOSchedulerTest, testEmptyDegradedPrefillDoesNotAdvanceDecodeCounter) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 0;  // force admission failure
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());

    auto start      = std::chrono::steady_clock::now();
    auto r1         = scheduler.schedule();  // PREFILL selected, admission fails, no decode batch runs.
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 0);
    ASSERT_GE(elapsed_ms.count(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.decodeSincePrefillForTest(), 0);
}

// ---------------------------------------------------------------------------
// Task 7: KV-gated admission, pending promotion, no incrKVBlock on prefill rounds
// ---------------------------------------------------------------------------

TEST_F(FIFOSchedulerTest, testKvGatedAdmission) {
    // Only 2 free KV blocks: two one-block prompts can both prefill. The ratio scheduler should
    // not hold back the second stream with a scheduler-side first-decode headroom estimate.
    CacheConfig cache_config  = makeMhaCacheConfig(1, 3, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1/100";  // prefill-heavy
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    auto r1 = scheduler.schedule();  // seed PREFILL: real initKVBlock admits both prompts
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 2);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_FALSE(s1->hasError());
    ASSERT_FALSE(s2->hasError());
}

TEST_F(FIFOSchedulerTest, testMultiBlockPromptPromotesWithIncrementalKv) {
    // A multi-block prompt uses 4 blocks at prefill. The next decode step needs only one
    // incremental block, so promotion must succeed even when free blocks are fewer than the
    // prompt's total block count.
    CacheConfig cache_config  = makeMhaCacheConfig(1, 8, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto stream = makeStream({1, 2, 3, 4, 5, 6, 7, 8}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto prefill = scheduler.schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill.value().size(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);

    stream->setSeqLength(stream->seqLength() + 1);
    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode.value().size(), 1);
    ASSERT_EQ(decode.value().front().get(), stream.get());
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_FALSE(stream->hasError());
}

TEST_F(FIFOSchedulerTest, testPrefillAdmissionAccountsPromptBlocksAcrossRound) {
    // Two 8-token prompts need 4 blocks each. With 7 free blocks, scheduler-side admission no
    // longer predicts KV. Both streams are tried in the prefill round; real initKVBlock admits the
    // first and fails the second.
    CacheConfig cache_config  = makeMhaCacheConfig(1, 8, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1/100";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto s1 = makeStream({1, 2, 3, 4, 5, 6, 7, 8}, model_config, runtime_config, resource_context);
    auto s2 = makeStream({9, 10, 11, 12, 13, 14, 15, 16}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    auto prefill = scheduler.schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill.value().size(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_FALSE(s1->hasError());
    ASSERT_TRUE(s2->hasError());
}

TEST_F(FIFOSchedulerTest, testPrefillAdmissionUsesRealInitKvBlockWithoutDecodeHeadroom) {
    // A single 8-token prompt needs 4 prompt blocks. With exactly 4 free blocks and no extra
    // first-decode headroom, prefill should still run; decode allocation is decided later by the
    // real state machine.
    CacheConfig cache_config  = makeMhaCacheConfig(1, 5, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 4);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto stream = makeStream({1, 2, 3, 4, 5, 6, 7, 8}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto prefill = scheduler.schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill.value().size(), 1);
    ASSERT_EQ(prefill.value().front().get(), stream.get());
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testPendingDecodePromotionMallocFailureFinishes) {
    // Pending decode promotion uses moveToNext() directly. If incrKVBlock cannot allocate, the
    // state machine reports MALLOC_FAILED and the scheduler removes the finished stream.
    CacheConfig cache_config  = makeMhaCacheConfig(1, 3, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto stream = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto prefill = scheduler.schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill.value().size(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);

    stream->setSeqLength(64);
    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode.value().size(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_TRUE(stream->hasError());
}

TEST_F(FIFOSchedulerTest, testPendingDecodePromotionMallocFailureDoesNotSpin) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 5, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 4);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Seed-prefill s1 (1 block), promote it to running on the next decode round.
    auto s1 = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    auto r1 = scheduler.schedule();  // PREFILL s1
    ASSERT_EQ(r1.value().size(), 1);
    s1->setSeqLength(s1->seqLength() + 1);
    auto r2 = scheduler.schedule();  // DECODE: promote s1 into running
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);

    // Now prefill s2, and push its seq so promoting it needs more blocks than currently free.
    auto s2 = makeStream({5, 6}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s2).ok());
    s1->setSeqLength(s1->seqLength() + 1);
    auto r3 = scheduler.schedule();  // PREFILL s2 (held in pending)
    ASSERT_EQ(r3.value().size(), 1);
    s2->setSeqLength(64);
    s1->setSeqLength(s1->seqLength() + 1);

    auto r4 = scheduler.schedule();  // DECODE: pending decode is consumed, not left spinning.
    ASSERT_TRUE(r4.ok());
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, testNoIncrKvBlockOnPrefillRounds) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1/3";  // P P P D
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // One long-running decode stream that will be held back during the prefill burst.
    auto held = makeStream({1, 2, 3, 4, 5, 6, 7, 8},
                           model_config,
                           runtime_config,
                           resource_context);  // fills a block boundary
    ASSERT_TRUE(scheduler.enqueue(held).ok());
    auto r0 = scheduler.schedule();  // seed PREFILL held
    held->setSeqLength(held->seqLength() + 1);
    auto r1 = scheduler.schedule();  // DECODE: promote held into running
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    // Now queue more work to trigger prefill rounds while `held` is in running (held back).
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(scheduler.enqueue(makeStream({1, 2}, model_config, runtime_config, resource_context)).ok());
    }
    // Drive a prefill round. `held` is at a block boundary, so a wrongful incrKVBlock on it would
    // consume an extra free block. The prefill round must NOT advance running, so the only block(s)
    // consumed are the admitted prefill prompt's — never a decode block for `held`.
    const size_t blocks_before = cache_manager->freeBlocksNum();
    auto         rp            = scheduler.schedule();
    ASSERT_TRUE(rp.ok());
    const size_t blocks_after = cache_manager->freeBlocksNum();

    // `held` is not in the returned (pure-context) batch ...
    for (const auto& s : rp.value()) {
        ASSERT_NE(s.get(), held.get());
    }
    // ... and the prefill consumed at most the admitted prompts' blocks. With a 2-token prompt and
    // block_size 8, one admitted prefill needs exactly 1 block; `held` (held back) must add 0.
    ASSERT_LE(blocks_before - blocks_after, static_cast<size_t>(rp.value().size()));
}

TEST_F(FIFOSchedulerTest, testPrefillRoundDoesNotAccountHeldDecodeAsBatchedWithPrefill) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1/3";  // P P P D
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto held = makeStream({1, 2}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(held).ok());
    auto seed = scheduler.schedule();  // PREFILL held
    ASSERT_TRUE(seed.ok());
    held->setSeqLength(held->seqLength() + 1);
    auto decode = scheduler.schedule();  // DECODE: promote held into running
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    auto prefill = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(prefill).ok());
    held->setSeqLength(held->seqLength() + 1);
    auto prefill_round = scheduler.schedule();  // PREFILL prefill; held stays out of returned batch
    ASSERT_TRUE(prefill_round.ok());
    ASSERT_EQ(prefill_round.value().size(), 1);
    ASSERT_EQ(prefill_round.value().front().get(), prefill.get());
    ASSERT_EQ(held->batch_with_prefill_times_, 0);
    ASSERT_EQ(held->batch_with_prefill_len_, 0);
}

TEST_F(FIFOSchedulerTest, testPrefillFirstForceBatchGroupComplete) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "1";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    const int64_t group_id   = 700;
    const int     group_size = 3;
    auto g1 = makeForceBatchStream({1, 2}, group_id, group_size, model_config, runtime_config, resource_context);
    auto g2 = makeForceBatchStream({3, 4}, group_id, group_size, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(g1).ok());
    ASSERT_TRUE(scheduler.enqueue(g2).ok());

    auto start      = std::chrono::steady_clock::now();
    auto incomplete = scheduler.schedule();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    ASSERT_TRUE(incomplete.ok());
    ASSERT_EQ(incomplete.value().size(), 0);
    ASSERT_GE(elapsed_ms.count(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);

    start      = std::chrono::steady_clock::now();
    incomplete = scheduler.schedule();
    elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    ASSERT_TRUE(incomplete.ok());
    ASSERT_EQ(incomplete.value().size(), 0);
    ASSERT_GE(elapsed_ms.count(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);

    auto g3 = makeForceBatchStream({5, 6}, group_id, group_size, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(g3).ok());
    auto complete = scheduler.schedule();
    ASSERT_TRUE(complete.ok());
    ASSERT_EQ(complete.value().size(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 3);
}

}  // namespace rtp_llm
