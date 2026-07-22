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

    auto valid_stream                          = make_stream(16);
    auto invalid_stream2                       = make_stream(17);
    auto [enqueue_successes, enqueued_streams] = scheduler.enqueueGroup({invalid_stream2, valid_stream});
    ASSERT_EQ(enqueue_successes, std::vector<bool>({false, true}));
    ASSERT_EQ(enqueued_streams, std::vector<GenerateStreamPtr>({invalid_stream2, valid_stream}));
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
    auto [enqueue_successes, enqueued_streams] = scheduler.enqueueGroup(streams);
    ASSERT_EQ(enqueue_successes, std::vector<bool>(streams.size(), true));
    ASSERT_EQ(enqueued_streams, streams);

    // Single schedule: both streams transition to RUNNING (no cache loading needed)
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

namespace {

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
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, enqueueGroupFallsBackToIndividualStreamsWhenGroupExceedsInitedLimit) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    vector<GenerateStreamPtr> streams = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    auto [enqueue_successes, returned_streams] = scheduler.enqueueGroup(streams);

    EXPECT_EQ(enqueue_successes, std::vector<bool>({true, true}));
    EXPECT_EQ(returned_streams, streams);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 2);
    EXPECT_EQ(scheduler.runningStreamsSize(), 0);
    for (const auto& stream : streams) {
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->curBlocksNum(), 0);
    }

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().size(), 1);
    EXPECT_EQ(scheduler.runningStreamsSize(), 1);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, enqueueGroupFallsBackToIndividualStreamsWhenGroupExceedsBatchLimit) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    vector<GenerateStreamPtr> streams = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    auto [enqueue_successes, returned_streams] = scheduler.enqueueGroup(streams);

    EXPECT_EQ(enqueue_successes, std::vector<bool>({true, true}));
    EXPECT_EQ(returned_streams, streams);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 2);
    EXPECT_EQ(scheduler.runningStreamsSize(), 0);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().size(), 1);
    EXPECT_EQ(scheduler.runningStreamsSize(), 1);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, enqueueGroupIgnoresCurrentlyInitedStreamsWhenGroupFitsLimit) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 5, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                           = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
    runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = 2;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto running_stream = makeSingleStream(model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(running_stream).ok());
    auto first_result = scheduler.schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_GT(running_stream->curBlocksNum(), 0);

    vector<GenerateStreamPtr> group_streams = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    auto [enqueue_successes, returned_streams] = scheduler.enqueueGroup(group_streams);
    EXPECT_EQ(enqueue_successes, std::vector<bool>({true, true}));
    EXPECT_EQ(returned_streams, group_streams);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 2);

    running_stream->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler.schedule();
    ASSERT_TRUE(group_result.ok());
    EXPECT_EQ(group_result.value().size(), 2);
    EXPECT_EQ(scheduler.runningStreamsSize(), 2);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, waitingStreamRunsBeforeGroupAtInitedLimit) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 5, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                           = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
    runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = 2;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    vector<GenerateStreamPtr> group_streams = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    auto waiting_stream = makeSingleStream(model_config, runtime_config, resource_context);
    ASSERT_EQ(scheduler.enqueueGroup(group_streams).first, std::vector<bool>({true, true}));
    ASSERT_TRUE(scheduler.enqueue(waiting_stream).ok());

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().size(), 1);
    EXPECT_EQ(scheduler.runningStreamsSize(), 1);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 2);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::RUNNING);

    waiting_stream->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler.schedule();
    ASSERT_TRUE(group_result.ok());
    EXPECT_EQ(group_result.value().size(), 2);
    EXPECT_EQ(scheduler.runningStreamsSize(), 2);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, groupTokenCapExceededDissolvesGroup) {
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
        makeSingleStream(model_config, runtime_config, resource_context, tokens),
        makeSingleStream(model_config, runtime_config, resource_context, tokens),
    };
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    // The first stream fits. The second one is deferred and the group is dissolved.
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(streams[1]->getStatus(), StreamState::WAITING);
    for (const auto& s : streams) {
        ASSERT_FALSE(s->hasError());
    }
}

TEST_F(FIFOSchedulerTest, groupCacheShortageDissolvesPreparedStreams) {
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
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    vector<GenerateStreamPtr> streams = {
        makeSingleStream(model_config, runtime_config, resource_context, {1, 2, 3}),
        makeSingleStream(model_config, runtime_config, resource_context, {1, 2, 3}),
    };
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    EXPECT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(streams[0]->hasError());
    EXPECT_GT(streams[0]->curBlocksNum(), 0);
    EXPECT_EQ(streams[1]->getStatus(), StreamState::FINISHED);
    EXPECT_TRUE(streams[1]->hasError());
    EXPECT_EQ(streams[1]->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    EXPECT_EQ(streams[1]->curBlocksNum(), 0);
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
        streams.push_back(makeSingleStream(model_config, runtime_config, resource_context));
    }
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
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
        streams.push_back(makeSingleStream(model_config, runtime_config, resource_context));
    }
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 4);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, waitingStreamRunsBeforeGroupWhenGroupWasEnqueuedFirst) {
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
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    auto waiting_stream = makeSingleStream(model_config, runtime_config, resource_context);
    scheduler.enqueueGroup(group_streams);
    scheduler.enqueue(waiting_stream);

    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    ASSERT_EQ(waiting_stream->getStatus(), StreamState::RUNNING);

    waiting_stream->reportEvent(StreamEvents::GenerateDone);
    auto r2 = scheduler.schedule();
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    for (const auto& stream : group_streams) {
        ASSERT_EQ(stream->getStatus(), StreamState::RUNNING);
    }
}

TEST_F(FIFOSchedulerTest, waitingStreamsRunBeforeGroupWhenEnqueuedFirst) {
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

    auto waiting_stream_1 = makeSingleStream(model_config, runtime_config, resource_context);
    auto waiting_stream_2 = makeSingleStream(model_config, runtime_config, resource_context);
    scheduler.enqueue(waiting_stream_1);
    scheduler.enqueue(waiting_stream_2);
    vector<GenerateStreamPtr> group_streams = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(group_streams);

    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);

    waiting_stream_1->reportEvent(StreamEvents::GenerateDone);
    waiting_stream_2->reportEvent(StreamEvents::GenerateDone);
    auto r2 = scheduler.schedule();
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    for (const auto& stream : group_streams) {
        ASSERT_EQ(stream->getStatus(), StreamState::RUNNING);
    }
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
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    vector<GenerateStreamPtr> group_b = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(group_a);
    scheduler.enqueueGroup(group_b);

    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 3);

    for (const auto& stream : group_a) {
        ASSERT_EQ(stream->getStatus(), StreamState::RUNNING);
    }
    for (const auto& stream : group_b) {
        ASSERT_EQ(stream->getStatus(), StreamState::WAITING);
    }
}

TEST_F(FIFOSchedulerTest, waitingFallbackFromFrontGroupPrecedesNextGroup) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 10, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
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

    std::vector<int>          long_prompt(60, 1);
    vector<GenerateStreamPtr> rejected_group = {
        makeSingleStream(model_config, runtime_config, resource_context, long_prompt),
        makeSingleStream(model_config, runtime_config, resource_context, long_prompt),
    };
    vector<GenerateStreamPtr> next_group = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    auto waiting_stream = makeSingleStream(model_config, runtime_config, resource_context);
    scheduler.enqueueGroup(rejected_group);
    scheduler.enqueueGroup(next_group);
    ASSERT_TRUE(scheduler.enqueue(waiting_stream).ok());

    auto first_result = scheduler.schedule();
    ASSERT_TRUE(first_result.ok());
    EXPECT_EQ(first_result.value().size(), 1);
    EXPECT_EQ(scheduler.runningStreamsSize(), 1);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 4);
    EXPECT_EQ(rejected_group[0]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(rejected_group[1]->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(rejected_group[0]->hasError());
    EXPECT_FALSE(rejected_group[1]->hasError());
    EXPECT_FALSE(next_group[0]->hasError());
    EXPECT_FALSE(next_group[1]->hasError());
    EXPECT_EQ(next_group[0]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(next_group[1]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::RUNNING);

    waiting_stream->reportEvent(StreamEvents::GenerateDone);
    auto second_result = scheduler.schedule();
    ASSERT_TRUE(second_result.ok());
    EXPECT_EQ(second_result.value().size(), 1);
    EXPECT_EQ(scheduler.runningStreamsSize(), 1);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 3);
    EXPECT_EQ(rejected_group[0]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(rejected_group[1]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(next_group[0]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(next_group[1]->getStatus(), StreamState::WAITING);

    rejected_group[0]->reportEvent(StreamEvents::GenerateDone);
    auto third_result = scheduler.schedule();
    ASSERT_TRUE(third_result.ok());
    EXPECT_EQ(third_result.value().size(), 1);
    EXPECT_EQ(rejected_group[1]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(next_group[0]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(next_group[1]->getStatus(), StreamState::WAITING);

    rejected_group[1]->reportEvent(StreamEvents::GenerateDone);
    auto fourth_result = scheduler.schedule();
    ASSERT_TRUE(fourth_result.ok());
    EXPECT_EQ(fourth_result.value().size(), 2);
    EXPECT_EQ(next_group[0]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(next_group[1]->getStatus(), StreamState::RUNNING);
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

    // Enqueue order: single_A, enqueueGroup(2 streams), single_B
    auto single_a = makeSingleStream(model_config, runtime_config, resource_context);
    scheduler.enqueue(single_a);

    vector<GenerateStreamPtr> group_streams = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(group_streams);
    auto single_b = makeSingleStream(model_config, runtime_config, resource_context);
    scheduler.enqueue(single_b);

    auto r1 = scheduler.schedule();
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);

    single_a->reportEvent(StreamEvents::GenerateDone);
    single_b->reportEvent(StreamEvents::GenerateDone);
    auto r2 = scheduler.schedule();
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    for (const auto& stream : group_streams) {
        ASSERT_EQ(stream->getStatus(), StreamState::RUNNING);
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
        EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
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
        EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
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
        EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
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
        EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
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
        for (const auto& stream : streams) {
            EXPECT_TRUE(scheduler.enqueue(stream).ok());
        }
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

    // Enqueue only 2 of 3 — group incomplete, should not be scheduled
    {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
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
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
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

TEST_F(FIFOSchedulerTest, enqueueGroupDissolvesWhenOnlyPartFitsTokenCap) {
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

    int64_t                   group_id   = 101;
    int                       group_size = 3;
    vector<GenerateStreamPtr> streams;

    for (int i = 0; i < group_size; ++i) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        query->group_id                      = group_id;
        query->group_size                    = group_size;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        streams.push_back(stream);
    }
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(streams[1]->getStatus(), StreamState::WAITING);
    ASSERT_EQ(streams[2]->getStatus(), StreamState::WAITING);
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
    {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
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
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id_a;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        group_a_streams.push_back(stream);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 10;
        query->group_id                       = group_id_b;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
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

}  // namespace rtp_llm
