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

TEST_F(FIFOSchedulerTest, testMaxInitedKVCacheStreamsBlocksNewInit) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                           = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
    runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = 1;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_stream = [&]() {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        return make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto stream1 = make_stream();
    auto stream2 = make_stream();
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_GT(stream1->curBlocksNum(), 0);
    ASSERT_EQ(stream2->curBlocksNum(), 0);

    stream1->reportEvent(StreamEvents::GenerateDone);
    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_GT(stream2->curBlocksNum(), 0);
}

TEST_F(FIFOSchedulerTest, testRejectInputWithoutSpeculativeReserveSpace) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 32, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 20;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_stream = [&](size_t input_len) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::full({static_cast<int64_t>(input_len)}, 1, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setReserveStep(4);
        return stream;
    };

    auto invalid_stream = make_stream(17);
    ASSERT_FALSE(scheduler.enqueue(invalid_stream).ok());
    ASSERT_TRUE(invalid_stream->hasError());
    ASSERT_EQ(invalid_stream->statusInfo().code(), ErrorCode::LONG_PROMPT_ERROR);
    ASSERT_NE(invalid_stream->stopReason().find("reserve_step 4"), std::string::npos);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);

    auto valid_stream    = make_stream(16);
    auto invalid_stream2 = make_stream(17);
    auto enqueued        = scheduler.enqueueGroup({invalid_stream2, valid_stream});
    ASSERT_EQ(enqueued.size(), 1);
    ASSERT_EQ(enqueued[0], valid_stream);
    ASSERT_TRUE(invalid_stream2->hasError());
    ASSERT_EQ(invalid_stream2->statusInfo().code(), ErrorCode::LONG_PROMPT_ERROR);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
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
    ASSERT_NE(stream->stopReason().find("incrKVBlock(advance) failed: malloc failed"), std::string::npos);
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
    ASSERT_NE(stream->stopReason().find("initKVBlock failed: malloc failed"), std::string::npos);
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
        // test abnormal case with tile num
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        std::shared_ptr<GenerateInput> query2         = make_shared<GenerateInput>();
        query2->input_ids                             = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query2->generate_config                       = make_shared<GenerateConfig>();
        query2->generate_config->num_return_sequences = 20;
        shared_ptr<GenerateStream> stream2 =
            make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
        // In the new code, checkInputLength rejects at enqueue time
        ASSERT_FALSE(scheduler.enqueue(stream2).ok());
        ASSERT_TRUE(stream2->hasError());
        ASSERT_EQ(stream2->stopReason(), "input len [7] * batch size [20] > max_batch_tokens_size [100]");
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    }
}

TEST_F(FIFOSchedulerTest, testEnqueueGroup) {
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
    auto enqueued = scheduler.enqueueGroup(streams);
    ASSERT_EQ(enqueued.size(), streams.size());

    // Single schedule: both streams transition to RUNNING (no cache loading needed)
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

namespace {

std::shared_ptr<GenerateStream> makeGroupedStream(int64_t                group_id,
                                                  int                    group_size,
                                                  const ModelConfig&     model_config,
                                                  const RuntimeConfig&   runtime_config,
                                                  const ResourceContext& resource_context,
                                                  std::vector<int>       tokens = {1, 2, 3}) {
    auto query             = std::make_shared<GenerateInput>();
    query->input_ids       = torch::tensor(tokens, torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();
    query->group_id        = group_id;
    query->group_size      = group_size;
    return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
}

std::shared_ptr<GenerateStream> makeSingleStream(const ModelConfig&     model_config,
                                                 const RuntimeConfig&   runtime_config,
                                                 const ResourceContext& resource_context,
                                                 std::vector<int>       tokens = {1, 2, 3}) {
    auto query             = std::make_shared<GenerateInput>();
    query->input_ids       = torch::tensor(tokens, torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();
    return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
}

}  // namespace

TEST_F(FIFOSchedulerTest, groupIsolation_size2) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    vector<GenerateStreamPtr> streams = {
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    for (const auto& t : scheduler.runningTaskList()) {
        ASSERT_EQ(t.batch_id, 100);
    }
}

TEST_F(FIFOSchedulerTest, groupTokenCapExceeded) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 100;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Each stream has 60 tokens; group total = 120 > max_batch_tokens_size (100)
    std::vector<int>          tokens(60, 1);
    vector<GenerateStreamPtr> streams = {
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context, tokens),
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context, tokens),
    };
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    // Group exceeds token cap, so it should be actively removed from waiting
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    // Streams should have error state
    for (const auto& s : streams) {
        ASSERT_TRUE(s->hasError());
        ASSERT_EQ(s->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    }
}

TEST_F(FIFOSchedulerTest, groupIsolation_size3) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    vector<GenerateStreamPtr> streams;
    for (int i = 0; i < 3; ++i) {
        streams.push_back(makeGroupedStream(100, 3, model_config, runtime_config, resource_context));
    }
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    for (const auto& t : scheduler.runningTaskList()) {
        ASSERT_EQ(t.batch_id, 100);
    }
}

TEST_F(FIFOSchedulerTest, groupIsolation_size4) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 6, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    vector<GenerateStreamPtr> streams;
    for (int i = 0; i < 4; ++i) {
        streams.push_back(makeGroupedStream(100, 4, model_config, runtime_config, resource_context));
    }
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 4);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    for (const auto& t : scheduler.runningTaskList()) {
        ASSERT_EQ(t.batch_id, 100);
    }
}

TEST_F(FIFOSchedulerTest, groupIsolation_groupNotMixedWithSingles_groupFirst) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    vector<GenerateStreamPtr> group_streams = {
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(group_streams);
    scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context));

    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);

    auto running = scheduler.runningTaskList();
    for (const auto& t : running) {
        ASSERT_EQ(t.batch_id, 100);
    }
}

TEST_F(FIFOSchedulerTest, groupIsolation_groupNotMixedWithSingles_singlesFirst) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context));
    scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context));
    vector<GenerateStreamPtr> group_streams = {
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(group_streams);

    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    // Singles admitted first, group waits (cannot mix with already-admitted singles)
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);

    auto running = scheduler.runningTaskList();
    for (const auto& t : running) {
        ASSERT_EQ(t.batch_id, -1);
    }

    // Second schedule: singles still running, group still cannot be admitted
    auto r2 = scheduler.schedule();
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
}

TEST_F(FIFOSchedulerTest, groupIsolation_twoGroupsNotMixed) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    vector<GenerateStreamPtr> group_a = {
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
    };
    vector<GenerateStreamPtr> group_b = {
        makeGroupedStream(200, 3, model_config, runtime_config, resource_context),
        makeGroupedStream(200, 3, model_config, runtime_config, resource_context),
        makeGroupedStream(200, 3, model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(group_a);
    scheduler.enqueueGroup(group_b);

    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 3);

    auto running = scheduler.runningTaskList();
    for (const auto& t : running) {
        ASSERT_EQ(t.batch_id, 100);
    }
}

TEST_F(FIFOSchedulerTest, groupIsolation_singlesCanMix) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context));
    scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context));
    scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);

    auto running = scheduler.runningTaskList();
    for (const auto& t : running) {
        ASSERT_EQ(t.batch_id, -1);
    }
}

TEST_F(FIFOSchedulerTest, groupIsolation_interleavedSinglesAndGroup) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    // Enqueue order: single_A, group(100, 2), single_B
    scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context));

    vector<GenerateStreamPtr> group_streams = {
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
        makeGroupedStream(100, 2, model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(group_streams);
    scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context));

    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    // Two singles admitted together, group skipped (cannot join already-admitted singles)
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);

    auto running = scheduler.runningTaskList();
    for (const auto& t : running) {
        ASSERT_EQ(t.batch_id, -1);
    }
}

TEST_F(FIFOSchedulerTest, testPdDecodePreCanRunStillRespectsMaxGenerateBatchSize) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.role_type     = RoleType::DECODE;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_pd_decode_stream = [&]() {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

        // DecodeRpcServer pre-sets CanRun to drive pre-enqueue KV allocation.
        stream->reportEvent(StreamEvents::CanRun);
        stream->prepare();
        EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::CanRun));
        EXPECT_TRUE(stream->hasEvent(StreamEvents::LoadInitiated));
        stream->setIsContextStream(false);
        return stream;
    };

    auto stream1 = make_pd_decode_stream();
    auto stream2 = make_pd_decode_stream();
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);

    stream1->reportEvent(StreamEvents::GenerateDone);
    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testPdDecodePreCanRunCanTopUpToMaxGenerateBatchSize) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.role_type     = RoleType::DECODE;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 2;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_pd_decode_stream = [&]() {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

        // DecodeRpcServer pre-sets CanRun to drive pre-enqueue KV allocation.
        stream->reportEvent(StreamEvents::CanRun);
        stream->prepare();
        EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::CanRun));
        EXPECT_TRUE(stream->hasEvent(StreamEvents::LoadInitiated));
        stream->setIsContextStream(false);
        return stream;
    };

    auto stream1 = make_pd_decode_stream();
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());

    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);

    auto stream2 = make_pd_decode_stream();
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);

    auto stream3 = make_pd_decode_stream();
    ASSERT_TRUE(scheduler.enqueue(stream3).ok());

    auto result3 = scheduler.schedule();
    ASSERT_TRUE(result3.ok());
    ASSERT_EQ(result3.value().size(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, testMaxInitedKVCacheStreamsAllowsAlreadyInitedStreams) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.role_type     = RoleType::DECODE;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                           = 2;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
    runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = 1;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_preinited_decode_stream = [&]() {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

        stream->reportEvent(StreamEvents::CanRun);
        stream->prepare();
        EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
        EXPECT_GT(stream->curBlocksNum(), 0);
        stream->setIsContextStream(false);
        return stream;
    };

    auto stream1 = make_preinited_decode_stream();
    auto stream2 = make_preinited_decode_stream();
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testMaxInitedKVCacheStreamsMixedGroupUnitBlocked) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.role_type     = RoleType::DECODE;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                           = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
    runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = 1;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // stream1: pre-inited group stream (curBlocksNum() > 0)
    auto stream1 = makeGroupedStream(100, 2, model_config, runtime_config, resource_context);
    stream1->reportEvent(StreamEvents::CanRun);
    stream1->prepare();
    EXPECT_GT(stream1->curBlocksNum(), 0);
    stream1->setIsContextStream(false);

    // stream2: uninitialized group stream (curBlocksNum() == 0)
    auto stream2 = makeGroupedStream(100, 2, model_config, runtime_config, resource_context);
    EXPECT_EQ(stream2->curBlocksNum(), 0);

    scheduler.enqueueGroup({stream1, stream2});

    // max=1: inited_kv_streams(1) + uninited_in_unit(1) = 2 > 1, whole unit skipped
    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
}

TEST_F(FIFOSchedulerTest, testMaxInitedKVCacheStreamsMixedGroupUnitAdmitted) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.role_type     = RoleType::DECODE;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                           = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
    runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = 2;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // stream1: pre-inited group stream (curBlocksNum() > 0)
    auto stream1 = makeGroupedStream(100, 2, model_config, runtime_config, resource_context);
    stream1->reportEvent(StreamEvents::CanRun);
    stream1->prepare();
    EXPECT_GT(stream1->curBlocksNum(), 0);
    stream1->setIsContextStream(false);

    // stream2: uninitialized group stream (curBlocksNum() == 0)
    auto stream2 = makeGroupedStream(100, 2, model_config, runtime_config, resource_context);
    EXPECT_EQ(stream2->curBlocksNum(), 0);

    scheduler.enqueueGroup({stream1, stream2});

    // max=2: inited_kv_streams(1) + uninited_in_unit(1) = 2 <= 2, whole unit admitted
    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testPdDecodePreCanRunWithPendingAsyncStillCountsRunningStream) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.role_type     = RoleType::DECODE;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_pd_decode_stream = [&]() {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

        // DecodeRpcServer pre-sets CanRun to drive pre-enqueue KV allocation.
        stream->reportEvent(StreamEvents::CanRun);
        stream->prepare();
        EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::CanRun));
        EXPECT_TRUE(stream->hasEvent(StreamEvents::LoadInitiated));
        stream->setIsContextStream(false);
        return stream;
    };

    auto stream1 = make_pd_decode_stream();
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());

    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_EQ(stream1->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);

    // Simulate RTP_LLM_STREAM_ASYNC=1 where the output worker still owns this stream.
    stream1->incPendingAsyncBookkeeping();
    ASSERT_TRUE(stream1->hasPendingAsyncBookkeeping());

    auto stream2 = make_pd_decode_stream();
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(stream2->getStatus(), StreamState::WAITING);

    stream1->decPendingAsyncBookkeepingAndMaybeRelease();
    ASSERT_FALSE(stream1->hasPendingAsyncBookkeeping());
    stream1->reportEvent(StreamEvents::GenerateDone);

    auto result3 = scheduler.schedule();
    ASSERT_TRUE(result3.ok());
    ASSERT_EQ(result3.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(stream2->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerTest, testCpForceSinglePrefillConfig) {
    auto schedule_two_prefills = [](bool cp_force_single_prefill) {
        CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
        std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
        EXPECT_TRUE(cache_manager->init());
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                       = 100;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size   = 8192;
        runtime_config.fifo_scheduler_config.cp_force_single_prefill = cp_force_single_prefill;
        PDSepConfig         pd_sep_config;
        ParallelismConfig   parallelism_config;
        ModelSpecificConfig model_specific_config;
        parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
        FIFOScheduler scheduler(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

        vector<GenerateStreamPtr> streams;
        for (size_t i = 0; i < 2; ++i) {
            std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
            query->input_ids                     = torch::tensor({1}, torch::kInt32);
            query->generate_config               = make_shared<GenerateConfig>();
            streams.push_back(
                make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr));
        }
        scheduler.enqueueGroup(streams);
        auto streams_status = scheduler.schedule();
        EXPECT_TRUE(streams_status.ok());
        return streams_status.value().size();
    };

    ASSERT_EQ(schedule_two_prefills(true), 1);
    ASSERT_EQ(schedule_two_prefills(false), 2);
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

    // New architecture: enqueueGroup() accepts a complete group at once.
    // The old "incomplete group" scenario no longer applies — enqueue all 3 together.
    {
        vector<GenerateStreamPtr> group_streams;
        for (int i = 0; i < group_size; ++i) {
            std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
            query->input_ids                      = torch::tensor({1}, torch::kInt32);
            query->generate_config                = make_shared<GenerateConfig>();
            query->generate_config->group_timeout = 10;
            query->group_id                       = group_id;
            query->group_size                     = group_size;
            query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
            group_streams.push_back(
                make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr));
        }
        scheduler.enqueueGroup(group_streams);
    }

    // Schedule: group is complete, all 3 streams transition to RUNNING
    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
}

TEST_F(FIFOSchedulerTest, testForceBatchCompleteGroupSkipsTokenCapAfterTimeout) {
    // New architecture (ScheduleUnit) does not implement force batch group timeout
    // skip-token-cap logic. enqueueGroup() accepts complete groups atomically;
    // there is no "timeout expired, skip token cap" path.
    GTEST_SKIP() << "New architecture does not implement force batch group timeout skip-token-cap logic";
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 2;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    int64_t group_id   = 101;
    int     group_size = 3;
    int     timeout_ms = 10;
    int64_t past_time  = autil::TimeUtility::currentTimeInMicroSeconds() - (timeout_ms + 100) * 1000;

    for (int i = 0; i < group_size; ++i) {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = timeout_ms;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = past_time;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
}

TEST_F(FIFOSchedulerTest, testForceBatchTimeout) {
    // New architecture: enqueueGroup() accepts complete groups atomically.
    // "Incomplete group with timeout" scenario no longer exists.
    GTEST_SKIP() << "Incomplete group + timeout scenario no longer exists in new ScheduleUnit architecture";
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
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = timeout_ms;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = past_time;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = timeout_ms;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = past_time;
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

TEST_F(FIFOSchedulerTest, testIncompleteForceBatchTimeoutUsesNormalTokenCap) {
    // New architecture: enqueueGroup() accepts complete groups atomically.
    // "Incomplete group with timeout + token cap" scenario no longer exists.
    GTEST_SKIP() << "Incomplete group + timeout + token cap scenario no longer exists in new ScheduleUnit architecture";
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 2;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    int64_t group_id   = 201;
    int     group_size = 3;
    int     timeout_ms = 10;
    int64_t past_time  = autil::TimeUtility::currentTimeInMicroSeconds() - (timeout_ms + 100) * 1000;

    for (int i = 0; i < 2; ++i) {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = timeout_ms;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = past_time;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
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
    // Force batch group: enqueue all group members together via enqueueGroup()
    {
        vector<GenerateStreamPtr> group_streams;
        for (int i = 0; i < group_size; ++i) {
            std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
            query->input_ids                      = torch::tensor({1}, torch::kInt32);
            query->generate_config                = make_shared<GenerateConfig>();
            query->generate_config->group_timeout = 10;
            query->group_id                       = group_id;
            query->group_size                     = group_size;
            query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
            group_streams.push_back(
                make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr));
        }
        scheduler.enqueueGroup(group_streams);
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
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id_a;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        group_a_streams.push_back(stream);
    }
    scheduler.enqueueGroup(group_a_streams);

    vector<shared_ptr<GenerateStream>> group_b_streams;
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id_b;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        group_b_streams.push_back(stream);
    }
    scheduler.enqueueGroup(group_b_streams);

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

}  // namespace rtp_llm
