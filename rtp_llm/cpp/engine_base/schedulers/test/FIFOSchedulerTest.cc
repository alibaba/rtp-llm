#include <chrono>
#include <iostream>
#include <limits>
#include <algorithm>
#include <memory>
#include <numeric>
#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "autil/TimeUtility.h"

#define protected public
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

namespace {

bool enqueueIndividually(FIFOScheduler& scheduler, const vector<GenerateStreamPtr>& streams) {
    return std::all_of(streams.begin(), streams.end(), [&scheduler](const auto& stream) {
        return scheduler.enqueue(stream).ok();
    });
}

}  // namespace

class FIFOSchedulerTest: public DeviceTestBase {
public:
    FIFOSchedulerTest() {}
};

static std::shared_ptr<GenerateConfig> makeTestGenerateConfig(int max_new_tokens = 1) {
    auto generate_config            = make_shared<GenerateConfig>();
    generate_config->max_new_tokens = max_new_tokens;
    return generate_config;
}

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
    query->generate_config               = makeTestGenerateConfig();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false without enable_host_cache)
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
    query->generate_config               = makeTestGenerateConfig();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    // checkInputLength rejects the oversized input at enqueue time.
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

TEST_F(FIFOSchedulerTest, testRejectSpeculativeTailWithoutReserveSpace) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 128, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 512;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    const size_t gamma        = 3;
    const size_t reserve_step = gamma + 1;
    auto         make_stream  = [&](size_t input_len) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::full({static_cast<int64_t>(input_len)}, 1, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setReserveStep(reserve_step);
        return stream;
    };

    auto valid_stream = make_stream(508);
    ASSERT_TRUE(scheduler.enqueue(valid_stream).ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);

    auto invalid_stream = make_stream(509);
    ASSERT_FALSE(scheduler.enqueue(invalid_stream).ok());
    ASSERT_TRUE(invalid_stream->hasError());
    ASSERT_EQ(invalid_stream->statusInfo().code(), ErrorCode::LONG_PROMPT_ERROR);
    ASSERT_NE(invalid_stream->stopReason().find("reserve_step 4"), std::string::npos);
    ASSERT_NE(invalid_stream->stopReason().find("allowed max input len for speculative decoding is 508"),
              std::string::npos);
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
    query->generate_config               = makeTestGenerateConfig(0);
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
    query->generate_config               = makeTestGenerateConfig();

    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 0);
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "LACK MEM");
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);
    const auto block_tree_cache = cache_manager->blockTreeCache();
    ASSERT_NE(block_tree_cache, nullptr);
    ASSERT_EQ(block_tree_cache->groupSets().size(), 1u);
    ASSERT_EQ(block_tree_cache->groupSets().front()->devicePools().size(), 1u);
    const auto device_pool = block_tree_cache->groupSets().front()->devicePools().front();
    ASSERT_NE(device_pool, nullptr);
    ASSERT_EQ(device_pool->freeBlocksNum(), 10);
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
    query->generate_config               = makeTestGenerateConfig();

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
    query->generate_config               = makeTestGenerateConfig();
    shared_ptr<GenerateStream> stream1 =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false without enable_host_cache)
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
    query2->generate_config               = makeTestGenerateConfig();
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
        query->generate_config               = makeTestGenerateConfig();
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
        query->generate_config               = makeTestGenerateConfig();
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
        query2->generate_config                       = makeTestGenerateConfig();
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
        query->generate_config                       = makeTestGenerateConfig();
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
        query->generate_config               = makeTestGenerateConfig();
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
        query->generate_config = makeTestGenerateConfig();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_FALSE(scheduler.enqueue(stream).ok());
        ASSERT_TRUE(stream->hasError());
    }
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
        runtime_config.max_generate_batch_size                         = 100;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size     = 8192;
        runtime_config.fifo_scheduler_config.cp_force_single_prefill   = cp_force_single_prefill;
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
        EXPECT_TRUE(enqueueIndividually(scheduler, streams));
        auto streams_status = scheduler.schedule();
        EXPECT_TRUE(streams_status.ok());
        return streams_status.value().size();
    };

    ASSERT_EQ(schedule_two_prefills(true), 1);
    ASSERT_EQ(schedule_two_prefills(false), 2);
}

// ---------------------------------------------------------------------------
// Helper used by the prefill-first cadence / KV-gate tests (Tasks 5–7)
// ---------------------------------------------------------------------------

static std::shared_ptr<GenerateStream> makeStream(const std::vector<int>& ids,
                                                  const ModelConfig&      model_config,
                                                  const RuntimeConfig&    runtime_config,
                                                  const ResourceContext&  resource_context,
                                                  int                     max_new_tokens       = 1,
                                                  int                     num_return_sequences = 1,
                                                  const std::vector<int>& variable_num_beams   = {}) {
    auto query                                   = std::make_shared<GenerateInput>();
    query->input_ids                             = torch::tensor(ids, torch::kInt32);
    query->generate_config                       = makeTestGenerateConfig();
    query->generate_config->max_new_tokens       = max_new_tokens;
    query->generate_config->num_return_sequences = num_return_sequences;
    query->generate_config->variable_num_beams   = variable_num_beams;
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
    const std::vector<std::string> invalid_ratios = {"", "1/0", "-1", "abc", "2/3"};
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

TEST_F(FIFOSchedulerTest, testZeroRatioTriesPrefillBeforeDecode) {
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
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "0";
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
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    s1->setSeqLength(s1->seqLength() + 1);

    auto s2 = makeStream({3, 4}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    // decode_prefill_ratio=0 means any waiting stream gets a PREFILL attempt before decode.
    auto r2 = scheduler.schedule();
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(r2.value().size(), 1);
    ASSERT_EQ(r2.value().front().get(), s2.get());
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
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

TEST_F(FIFOSchedulerTest, testZeroRatioFallsBackToDecodeWhenKvAdmissionRejectsAllWaiting) {
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
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "0";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto running = makeStream({1}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(running).ok());
    auto r1 = scheduler.schedule();  // seed PREFILL running
    ASSERT_TRUE(r1.ok());
    ASSERT_EQ(r1.value().size(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    running->setSeqLength(running->seqLength() + 1);

    auto blocked = makeStream({2, 3, 4, 5, 6, 7, 8}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(blocked).ok());

    // decode_prefill_ratio=0 selects a PREFILL round, but the KV admission check rejects the only
    // waiting stream. The scheduler must then fall back to DECODE and promote the pending stream.
    auto r2 = scheduler.schedule();
    ASSERT_TRUE(r2.ok());
    ASSERT_EQ(r2.value().size(), 1);
    ASSERT_EQ(r2.value().front().get(), running.get());
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_FALSE(blocked->hasError());
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
    // Only 2 free KV blocks: with max_new_tokens=1, two 1-token prompts can both prefill and
    // still satisfy the estimated peak check.
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

    auto s1 = makeStream({1}, model_config, runtime_config, resource_context);
    auto s2 = makeStream({3}, model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    auto r1 = scheduler.schedule();  // seed PREFILL: KV admission admits both prompts
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
    // Two block-aligned 8-token prompts need 4 blocks each. With 7
    // free blocks, scheduler-side admission admits only the first and leaves the second waiting.
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
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_FALSE(s1->hasError());
    ASSERT_FALSE(s2->hasError());
}

TEST_F(FIFOSchedulerTest, testPrefillAdmissionAccountsFanOutAtShortLifetimePeak) {
    // The first stream has a short lifetime but high num_return_sequences fan-out. Its 7-token
    // prompt is a partial block, so prefill allocates one physical block per return sequence. Its
    // second generated token is cached by the final decode step in the next block, where fan-out
    // needs another block per sequence. A long single-output stream must not be admitted when that
    // peak exceeds the remaining KV capacity.
    constexpr int kTokensPerBlock = 8;
    CacheConfig   cache_config    = makeMhaCacheConfig(1, 81, 1, 4, kTokensPerBlock, rtp_llm::DataType::TYPE_FP16);
    auto          cache_manager   = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 80);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len                  = 8192;
    model_config.attn_config.tokens_per_block = kTokensPerBlock;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "0";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto high_fanout = makeStream({1, 2, 3, 4, 5, 6, 7},
                                  model_config,
                                  runtime_config,
                                  resource_context,
                                  /*max_new_tokens=*/3,
                                  /*num_return_sequences=*/40);
    ASSERT_TRUE(scheduler.enqueue(high_fanout).ok());

    auto seed = scheduler.schedule();
    ASSERT_TRUE(seed.ok());
    ASSERT_EQ(seed.value().size(), 1);
    ASSERT_EQ(seed.value().front().get(), high_fanout.get());
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 40);

    auto long_remaining = makeStream({2},
                                     model_config,
                                     runtime_config,
                                     resource_context,
                                     /*max_new_tokens=*/20,
                                     /*num_return_sequences=*/1);
    ASSERT_TRUE(scheduler.enqueue(long_remaining).ok());

    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode.value().size(), 1);
    ASSERT_EQ(decode.value().front().get(), high_fanout.get());
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_FALSE(long_remaining->hasError());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 40);
}

TEST_F(FIFOSchedulerTest, testPeakEstimateSharesAlignedPromptAcrossMaximumBatchWidth) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 64, 1, 4, 4, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len                  = 8192;
    model_config.attn_config.tokens_per_block = 4;
    RuntimeConfig runtime_config;

    const std::vector<int> aligned_prompt{1, 2, 3, 4, 5, 6, 7, 8};
    auto                   single_beam  = makeStream(aligned_prompt,
                                  model_config,
                                  runtime_config,
                                  resource_context,
                                  /*max_new_tokens=*/3,
                                  /*num_return_sequences=*/1,
                                  /*variable_num_beams=*/{1});
    auto                   multi_return = makeStream(aligned_prompt,
                                   model_config,
                                   runtime_config,
                                   resource_context,
                                   /*max_new_tokens=*/3,
                                   /*num_return_sequences=*/4);
    auto                   dynamic_beam = makeStream(aligned_prompt,
                                   model_config,
                                   runtime_config,
                                   resource_context,
                                   /*max_new_tokens=*/3,
                                   /*num_return_sequences=*/1,
                                   /*variable_num_beams=*/{1, 4, 2});

    ASSERT_EQ(dynamic_beam->nextBatchSize(), 1);
    ASSERT_EQ(dynamic_beam->maxBatchSize(), 4);
    ASSERT_EQ(single_beam->estimatePeakNeedBlocks(/*remaining_tokens=*/3), 3);
    // Two prompt blocks stay shared; the one independent future block is charged four times.
    ASSERT_EQ(dynamic_beam->estimateInitialNeedBlocks(), 2);
    ASSERT_EQ(multi_return->estimatePeakNeedBlocks(/*remaining_tokens=*/3), 6);
    ASSERT_EQ(dynamic_beam->estimatePeakNeedBlocks(/*remaining_tokens=*/3), 6);

    // Prefill has allocated the two shared blocks at width one. The non-empty estimate must retain
    // those blocks and reserve four private future blocks for the maximum beam width.
    ASSERT_TRUE(dynamic_beam->initKVBlock().ok());
    ASSERT_EQ(dynamic_beam->estimateInitialNeedBlocks(), 0);
    ASSERT_EQ(dynamic_beam->estimatePeakNeedBlocks(/*remaining_tokens=*/3), 4);
    dynamic_beam->releaseResource();
}

TEST_F(FIFOSchedulerTest, testMultiSequenceAdmissionMatchesPhysicalFreeBlockWatermark) {
    constexpr int kTokensPerBlock = 4;
    CacheConfig   cache_config    = makeMhaCacheConfig(1, 8, 1, 4, kTokensPerBlock, rtp_llm::DataType::TYPE_FP16);
    auto          cache_manager   = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len                  = 8192;
    model_config.attn_config.tokens_per_block = kTokensPerBlock;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "0";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto in_flight = makeStream({0},
                                model_config,
                                runtime_config,
                                resource_context,
                                /*max_new_tokens=*/0);
    ASSERT_TRUE(scheduler.enqueue(in_flight).ok());
    auto first_prefill = scheduler.schedule();
    ASSERT_TRUE(first_prefill.ok());
    ASSERT_EQ(first_prefill.value().size(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 6);

    auto         candidate             = makeStream({1, 2, 3, 4, 5, 6, 7, 8},
                                model_config,
                                runtime_config,
                                resource_context,
                                /*max_new_tokens=*/4,
                                /*num_return_sequences=*/4);
    const size_t free_before_admission = cache_manager->freeBlocksNum();
    const int    estimated_peak        = candidate->estimatePeakNeedBlocks(/*remaining_tokens=*/4);
    ASSERT_EQ(estimated_peak, 6);
    ASSERT_EQ(free_before_admission, static_cast<size_t>(estimated_peak));
    ASSERT_TRUE(scheduler.enqueue(candidate).ok());

    auto candidate_prefill = scheduler.schedule();
    ASSERT_TRUE(candidate_prefill.ok());
    ASSERT_EQ(candidate_prefill.value().size(), 1);
    ASSERT_EQ(candidate_prefill.value().front().get(), candidate.get());
    ASSERT_FALSE(candidate->hasError());

    size_t min_free_blocks = cache_manager->freeBlocksNum();
    ASSERT_EQ(min_free_blocks, 4);  // two aligned prompt blocks are physically shared

    auto promote = scheduler.schedule();
    ASSERT_TRUE(promote.ok());
    ASSERT_EQ(promote.value().size(), 2);

    candidate->setSeqLength(candidate->seqLength() + 1);
    auto cross_boundary = scheduler.schedule();
    ASSERT_TRUE(cross_boundary.ok());
    ASSERT_EQ(cross_boundary.value().size(), 2);
    ASSERT_FALSE(candidate->hasError());
    min_free_blocks = std::min(min_free_blocks, cache_manager->freeBlocksNum());

    ASSERT_EQ(min_free_blocks, free_before_admission - static_cast<size_t>(estimated_peak));
}

TEST_F(FIFOSchedulerTest, testHybridAdmissionAllowsUnderestimateAtDifferentBlockPeak) {
    // The logical-token peak is at t=0, where the block estimate is 13. The short stream's physical
    // block peak is 16 at t=9 because Hybrid block boundaries do not necessarily follow token volume.
    // Admission intentionally accepts this underestimate instead of scanning every endpoint.
    auto cache_config = test::makeSimpleHybridMhaCacheConfig(
        /*layer_num=*/4, /*block_num=*/64, /*tokens_per_block=*/1, rtp_llm::DataType::TYPE_FP16);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext no_reuse_context;
    no_reuse_context.cache_manager = cache_manager;
    no_reuse_context.reuse_cache   = false;
    ResourceContext reuse_context  = no_reuse_context;
    reuse_context.reuse_cache      = true;

    ModelConfig model_config;
    model_config.max_seq_len                  = 8192;
    model_config.attn_config.tokens_per_block = 1;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto long_context_short_lifetime = makeStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                                  model_config,
                                                  runtime_config,
                                                  no_reuse_context,
                                                  /*max_new_tokens=*/1);
    auto short_context_long_lifetime = makeStream({11},
                                                  model_config,
                                                  runtime_config,
                                                  reuse_context,
                                                  /*max_new_tokens=*/10);

    ASSERT_EQ(long_context_short_lifetime->estimatePeakNeedBlocks(/*remaining_tokens=*/0), 11);
    ASSERT_EQ(short_context_long_lifetime->estimatePeakNeedBlocks(/*remaining_tokens=*/0), 2);
    ASSERT_EQ(short_context_long_lifetime->estimatePeakNeedBlocks(/*remaining_tokens=*/9), 16);

    scheduler.buildAdmissionPeakState();
    ASSERT_TRUE(scheduler.tryAddToAdmissionPeakState(
        long_context_short_lifetime, /*initial_capacity=*/13, /*lifecycle_capacity=*/13));
    ASSERT_FALSE(scheduler.tryAddToAdmissionPeakState(
        short_context_long_lifetime, /*initial_capacity=*/13, /*lifecycle_capacity=*/12));
    // The rejected attempt must not mutate the committed state.
    ASSERT_TRUE(scheduler.tryAddToAdmissionPeakState(
        short_context_long_lifetime, /*initial_capacity=*/13, /*lifecycle_capacity=*/13));
}

TEST_F(FIFOSchedulerTest, testKVAdmissionCostAcrossStreamCounts) {
    constexpr int kTokensPerBlock = 8;
    auto          cache_config    = test::makeSimpleHybridMhaCacheConfig(
        /*layer_num=*/4, /*block_num=*/64, kTokensPerBlock, rtp_llm::DataType::TYPE_FP16);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = true;

    ModelConfig model_config;
    model_config.max_seq_len                  = 8192;
    model_config.attn_config.tokens_per_block = kTokensPerBlock;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 1024;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 1 << 20;
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::vector<int> prompt(1024);
    std::iota(prompt.begin(), prompt.end(), 1);
    constexpr int kRepeats = 3;

    for (const int stream_count : {8, 32, 64, 128, 256, 512}) {
        scheduler.running_streams_.clear();

        for (int i = 0; i < stream_count; ++i) {
            auto stream = makeStream(prompt,
                                     model_config,
                                     runtime_config,
                                     resource_context,
                                     /*max_new_tokens=*/i + 2);
            scheduler.running_streams_.push_back(stream);
        }
        auto candidate = makeStream(prompt,
                                    model_config,
                                    runtime_config,
                                    resource_context,
                                    /*max_new_tokens=*/stream_count + 128);

        // Warm caches. Distinct max_new_tokens exercise one unique lifetime per stream.
        scheduler.buildAdmissionPeakState();
        ASSERT_TRUE(scheduler.tryAddToAdmissionPeakState(
            candidate, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()));

        int64_t build_total_us     = 0;
        int64_t admission_total_us = 0;
        for (int repeat = 0; repeat < kRepeats; ++repeat) {
            auto begin = std::chrono::steady_clock::now();
            scheduler.buildAdmissionPeakState();
            build_total_us +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();

            begin               = std::chrono::steady_clock::now();
            const bool admitted = scheduler.tryAddToAdmissionPeakState(
                candidate, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max());
            const auto admission_elapsed = std::chrono::steady_clock::now() - begin;
            ASSERT_TRUE(admitted);
            admission_total_us += std::chrono::duration_cast<std::chrono::microseconds>(admission_elapsed).count();
        }

        const int64_t build_avg_us     = build_total_us / kRepeats;
        const int64_t admission_avg_us = admission_total_us / kRepeats;
        std::cout << "[KV_ADMISSION_COST] streams=" << stream_count << " sampled_endpoints=1"
                  << " build_avg_us=" << build_avg_us << " candidate_avg_us=" << admission_avg_us << std::endl;
        RecordProperty("streams_" + std::to_string(stream_count) + "_build_avg_us", build_avg_us);
        RecordProperty("streams_" + std::to_string(stream_count) + "_candidate_avg_us", admission_avg_us);
        if (stream_count == 512) {
            // Keep ample headroom for loaded CI hosts while preventing a regression to the previous ~50 ms rebuild.
            EXPECT_LT(build_avg_us, 10000);
            EXPECT_LT(admission_avg_us, 10000);
        }
    }
}

TEST_F(FIFOSchedulerTest, testScheduleBatchKVAdmissionCostAcrossWaitingCounts) {
    constexpr int kTokensPerBlock = 8;

    for (const int stream_count : {256, 512}) {
        // Block zero is unavailable for allocation, leaving exactly one physical block per stream.
        auto          cache_config = makeMhaCacheConfig(/*layer_num=*/1,
                                               /*block_num=*/stream_count + 1,
                                               /*local_head_num_kv=*/1,
                                               /*size_per_head=*/4,
                                               kTokensPerBlock,
                                               rtp_llm::DataType::TYPE_FP16);
        KVCacheConfig kv_cache_config;
        kv_cache_config.reserve_block_ratio = 0;
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr, kv_cache_config);
        ASSERT_TRUE(cache_manager->init());
        ASSERT_EQ(cache_manager->availableBlocksNum(), stream_count);
        ASSERT_EQ(cache_manager->reserveBlocksNum(), 0);

        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = false;

        ModelConfig model_config;
        model_config.max_seq_len                  = 8192;
        model_config.attn_config.tokens_per_block = kTokensPerBlock;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                     = 1024;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 1 << 20;
        PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
        ParallelismConfig      parallelism_config;
        ModelSpecificConfig    model_specific_config;
        PDFusionRatioScheduler scheduler(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

        for (int i = 0; i < stream_count; ++i) {
            auto stream = makeStream({i + 1},
                                     model_config,
                                     runtime_config,
                                     resource_context,
                                     /*max_new_tokens=*/1);
            ASSERT_TRUE(scheduler.enqueue(stream).ok());
        }
        ASSERT_EQ(scheduler.waitingStreamsSize(), stream_count);

        // Measure the public, lock-holding path: the whole waiting batch is admitted in this schedule call.
        const auto begin     = std::chrono::steady_clock::now();
        auto       scheduled = scheduler.schedule();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

        ASSERT_TRUE(scheduled.ok());
        ASSERT_EQ(scheduled.value().size(), stream_count);
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), stream_count);
        ASSERT_EQ(cache_manager->freeBlocksNum(), 0);

        std::cout << "[KV_BATCH_ADMISSION_COST] waiting_streams=" << stream_count << " schedule_total_ms=" << elapsed_ms
                  << std::endl;
        RecordProperty("waiting_streams_" + std::to_string(stream_count) + "_schedule_total_ms", elapsed_ms);
        // Keep ample headroom for loaded CI hosts while bounding scheduler-lock latency for the largest batch.
        EXPECT_LT(elapsed_ms, 100);
    }
}

TEST_F(FIFOSchedulerTest, testReserveOnlyLimitsInitialAllocationNotLifecycleGrowth) {
    CacheConfig   cache_config = makeMhaCacheConfig(/*layer_num=*/1,
                                                  /*block_num=*/11,
                                                  /*local_head_num_kv=*/1,
                                                  /*size_per_head=*/4,
                                                  /*tokens_per_block=*/1,
                                                  rtp_llm::DataType::TYPE_FP16);
    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio = 20;
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr, kv_cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->availableBlocksNum(), 10);
    ASSERT_EQ(cache_manager->reserveBlocksNum(), 2);

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len                  = 8192;
    model_config.attn_config.tokens_per_block = 1;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "0";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto long_running = makeStream({1, 2},
                                   model_config,
                                   runtime_config,
                                   resource_context,
                                   /*max_new_tokens=*/4);
    ASSERT_TRUE(scheduler.enqueue(long_running).ok());
    auto first_prefill = scheduler.schedule();
    ASSERT_TRUE(first_prefill.ok());
    ASSERT_EQ(first_prefill.value().size(), 1);
    ASSERT_EQ(cache_manager->availableBlocksNum(), 8);

    auto candidate = makeStream({3, 4, 5},
                                model_config,
                                runtime_config,
                                resource_context,
                                /*max_new_tokens=*/3);
    ASSERT_EQ(candidate->estimatePeakNeedBlocks(/*remaining_tokens=*/0), 3);
    ASSERT_LE(3 + cache_manager->reserveBlocksNum(), cache_manager->availableBlocksNum());
    // At the candidate endpoint: existing growth(2) + candidate lifecycle(5) = 7.
    // It exceeds available-reserve(6), but fits the full available capacity(8).
    ASSERT_EQ(long_running->estimatePeakNeedBlocks(/*remaining_tokens=*/2)
                  + candidate->estimatePeakNeedBlocks(/*remaining_tokens=*/2),
              7);
    ASSERT_TRUE(scheduler.enqueue(candidate).ok());

    auto second_prefill = scheduler.schedule();
    ASSERT_TRUE(second_prefill.ok());
    ASSERT_EQ(second_prefill.value().size(), 1);
    ASSERT_EQ(second_prefill.value().front().get(), candidate.get());
    ASSERT_FALSE(candidate->hasError());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 5);

    auto promote = scheduler.schedule();
    ASSERT_TRUE(promote.ok());
    ASSERT_EQ(promote.value().size(), 2);
    for (int step = 0; step < 2; ++step) {
        long_running->setSeqLength(long_running->seqLength() + 1);
        candidate->setSeqLength(candidate->seqLength() + 1);
        auto decode = scheduler.schedule();
        ASSERT_TRUE(decode.ok());
        ASSERT_EQ(decode.value().size(), 2);
        ASSERT_FALSE(long_running->hasError());
        ASSERT_FALSE(candidate->hasError());
    }
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);
}

TEST_F(FIFOSchedulerTest, testMaxNewTokensOneDoesNotReserveFinalTokenKVBlock) {
    // Keep one stream running so KV admission is enforced. The aligned candidate prompt needs two
    // blocks; its only generated token completes the request and is never written into KV cache.
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 4, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len                  = 8192;
    model_config.attn_config.tokens_per_block = 4;
    model_config.vocab_size                   = 32;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "0";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto running = makeStream({0}, model_config, runtime_config, resource_context, /*max_new_tokens=*/2);
    ASSERT_TRUE(scheduler.enqueue(running).ok());
    auto running_prefill = scheduler.schedule();
    ASSERT_TRUE(running_prefill.ok());
    ASSERT_EQ(running_prefill.value().size(), 1);
    auto promote = scheduler.schedule();
    ASSERT_TRUE(promote.ok());
    ASSERT_EQ(promote.value().size(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);

    auto candidate =
        makeStream({1, 2, 3, 4, 5, 6, 7, 8}, model_config, runtime_config, resource_context, /*max_new_tokens=*/1);
    ASSERT_TRUE(scheduler.enqueue(candidate).ok());
    auto prefill = scheduler.schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill.value().size(), 1);
    ASSERT_EQ(prefill.value().front().get(), candidate.get());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 0);

    auto final_token = torch::tensor({9}, torch::kInt32).reshape({1, 1});
    candidate->update({final_token,
                       /*num_new_tokens=*/1,
                       torch::Tensor(),
                       torch::Tensor(),
                       torch::Tensor(),
                       torch::Tensor(),
                       torch::Tensor(),
                       torch::Tensor(),
                       torch::Tensor(),
                       torch::Tensor(),
                       /*update_remote_generate=*/false,
                       /*force_update_info=*/false});
    ASSERT_TRUE(candidate->hasEvent(StreamEvents::GenerateDone));

    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode.value().size(), 1);
    ASSERT_EQ(decode.value().front().get(), running.get());
    ASSERT_FALSE(candidate->hasError());
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);
}

TEST_F(FIFOSchedulerTest, testSinglePrefillDefersReserveCapacityFailureToAllocator) {
    CacheConfig   cache_config = makeMhaCacheConfig(/*layer_num=*/1,
                                                  /*block_num=*/11,
                                                  /*local_head_num_kv=*/1,
                                                  /*size_per_head=*/4,
                                                  /*tokens_per_block=*/1,
                                                  rtp_llm::DataType::TYPE_FP16);
    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio = 50;
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr, kv_cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->totalBlocksNum(), 10);
    ASSERT_EQ(cache_manager->reserveBlocksNum(), 5);

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

    // The input fits the 10-block physical cache but violates the 5-block reserve watermark.
    // The idle scheduler admits it, then the allocator reports the reserve-capacity failure.
    auto stream = makeStream({1, 2, 3, 4, 5, 6},
                             model_config,
                             runtime_config,
                             resource_context,
                             /*max_new_tokens=*/0);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);

    auto prefill = scheduler.schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill.value().size(), 0);
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    ASSERT_EQ(stream->stopReason(), "LACK MEM");
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->availableBlocksNum(), 10);
}

TEST_F(FIFOSchedulerTest, testPrefillAdmissionUsesMaxTokenNumRemainingTokens) {
    // Keep one stream in flight so the single-request fast path does not apply. The candidate asks
    // for far more tokens than model max_seq_len allows; admission should use GenerateStream::maxTokenNum(),
    // not the raw max_new_tokens request value.
    CacheConfig cache_config  = makeMhaCacheConfig(1, 5, 1, 4, 4, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 4);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 10;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    runtime_config.fifo_scheduler_config.decode_prefill_ratio  = "0";
    PDSepConfig            pd_sep_config                       = makePDFusionPDSepConfig();
    ParallelismConfig      parallelism_config;
    ModelSpecificConfig    model_specific_config;
    PDFusionRatioScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto seed = makeStream({0}, model_config, runtime_config, resource_context, /*max_new_tokens=*/0);
    ASSERT_TRUE(scheduler.enqueue(seed).ok());
    auto seed_prefill = scheduler.schedule();
    ASSERT_TRUE(seed_prefill.ok());
    ASSERT_EQ(seed_prefill.value().size(), 1);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);

    auto stream = makeStream({1, 2, 3, 4, 5, 6, 7, 8},
                             model_config,
                             runtime_config,
                             resource_context,
                             /*max_new_tokens=*/1000);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto prefill = scheduler.schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill.value().size(), 1);
    ASSERT_EQ(prefill.value().front().get(), stream.get());
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 2);
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


// ---------------------------------------------------------------------------
// FIFOScheduler per-round token-budget and retryable-admission cases
// ---------------------------------------------------------------------------

TEST_F(FIFOSchedulerTest, permanentlyOversizedStreamDoesNotBlockLaterStream) {
    CacheConfig   cache_config = makeMhaCacheConfig(/*layer_num=*/1,
                                                  /*block_num=*/11,
                                                  /*local_head_num_kv=*/1,
                                                  /*size_per_head=*/4,
                                                  /*tokens_per_block=*/1,
                                                  rtp_llm::DataType::TYPE_FP16);
    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio = 50;
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, false, nullptr, kv_cache_config);
    ASSERT_TRUE(cache_manager->init());

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

    auto make_stream = [&](std::vector<int> tokens) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(tokens, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };
    vector<GenerateStreamPtr> streams = {
        make_stream({1, 2, 3, 4, 5, 6}),
        make_stream({1, 2, 3, 4}),
    };
    ASSERT_EQ(scheduler.enqueueGroup(streams).first.size(), streams.size());

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    EXPECT_TRUE(streams[0]->hasError());
    EXPECT_EQ(streams[0]->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(streams[1]->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(streams[1]->hasError());
}

TEST_F(FIFOSchedulerTest, retryableKVShortageStillAdmitsLaterSmallerStreams) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 6, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 5);

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

    auto make_stream = [&](std::vector<int> tokens) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(tokens, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };
    vector<GenerateStreamPtr> streams = {
        make_stream({1, 2, 3, 4}),
        make_stream({1, 2, 3}),
        make_stream({1}),
        make_stream({1, 2}),
    };
    ASSERT_EQ(scheduler.enqueueGroup(streams).first.size(), streams.size());

    auto first_result = scheduler.schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_EQ(first_result.value().size(), 2);
    EXPECT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(streams[1]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(streams[2]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(streams[3]->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(streams[1]->hasError());
    EXPECT_FALSE(streams[3]->hasError());
    EXPECT_EQ(scheduler.waitingStreamsSize(), 2);

    streams[0]->reportEvent(StreamEvents::GenerateDone);
    streams[2]->reportEvent(StreamEvents::GenerateDone);
    auto second_result = scheduler.schedule();
    ASSERT_TRUE(second_result.ok());
    ASSERT_EQ(second_result.value().size(), 2);
    EXPECT_EQ(streams[1]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(streams[3]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, retryableKVShortageDoesNotConsumeBatchTokenBudget) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 5, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 4);

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PREFILL;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 9;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_stream = [&](std::vector<int> tokens) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(tokens, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto kv_holder = make_stream({9});
    kv_holder->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(kv_holder->moveToNext(), StreamState::RUNNING);
    ASSERT_EQ(kv_holder->curBlocksNum(), 1);
    ASSERT_TRUE(kv_holder->hasEvent(StreamEvents::LoadInitiated));
    kv_holder->generate_status_->status.store(StreamState::WAITING);

    auto blocked_front_1 = make_stream({1, 2, 3, 4});
    auto blocked_front_2 = make_stream({5, 6, 7, 8});
    ASSERT_TRUE(enqueueIndividually(scheduler, {blocked_front_1, blocked_front_2, kv_holder}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    EXPECT_EQ(result.value().front(), kv_holder);
    EXPECT_EQ(blocked_front_1->getStatus(), StreamState::WAITING);
    EXPECT_EQ(blocked_front_2->getStatus(), StreamState::WAITING);
    EXPECT_EQ(kv_holder->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 2);
}

TEST_F(FIFOSchedulerTest, batchTokenQuotaIncludesPostAllocationPrefixLength) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PREFILL;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 10;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_stream = [&](int token_count) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::ones({token_count}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto cached_stream = make_stream(8);
    cached_stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(cached_stream->moveToNext(), StreamState::RUNNING);
    cached_stream->setReuseLength(7);
    cached_stream->generate_status_->status.store(StreamState::WAITING);
    ASSERT_EQ(cached_stream->contextLength(), 1);

    auto new_stream = make_stream(8);
    ASSERT_EQ(new_stream->prefixLength(), 0);
    ASSERT_TRUE(enqueueIndividually(scheduler, {cached_stream, new_stream}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 1);
    EXPECT_EQ(cached_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(new_stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, batchTokenQuotaAccountsForStreamBatchSize) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PREFILL;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_stream = [&](int batch_size) {
        auto query                                   = std::make_shared<GenerateInput>();
        query->input_ids                             = torch::ones({3}, torch::kInt32);
        query->generate_config                       = std::make_shared<GenerateConfig>();
        query->generate_config->num_return_sequences = batch_size;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto batched_stream = make_stream(2);
    auto single_stream  = make_stream(1);
    ASSERT_TRUE(enqueueIndividually(scheduler, {batched_stream, single_stream}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 1);
    EXPECT_EQ(batched_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(single_stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, withoutCacheQuotaUsesPostAllocationContextLengthAndStopsTail) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PREFILL;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                              = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size          = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache = 2;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_stream = [&](int token_count) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::ones({token_count}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto cached_stream = make_stream(8);
    cached_stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(cached_stream->moveToNext(), StreamState::RUNNING);
    cached_stream->setReuseLength(7);
    cached_stream->generate_status_->status.store(StreamState::WAITING);
    ASSERT_EQ(cached_stream->contextLength(), 1);

    auto threshold_crossing_stream = make_stream(1);
    auto tail_stream               = make_stream(1);
    auto errored_tail_stream       = make_stream(1);
    errored_tail_stream->reportError(ErrorCode::CANCELLED, "cancelled before admission");
    ASSERT_TRUE(enqueueIndividually(
        scheduler, {cached_stream, threshold_crossing_stream, tail_stream, errored_tail_stream}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 2);
    EXPECT_EQ(cached_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(threshold_crossing_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(tail_stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(errored_tail_stream->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 1);

    cached_stream->reportEvent(StreamEvents::GenerateDone);
    threshold_crossing_stream->reportEvent(StreamEvents::GenerateDone);
    auto next_result = scheduler.schedule();
    ASSERT_TRUE(next_result.ok());
    ASSERT_EQ(next_result->size(), 1);
    EXPECT_EQ(tail_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, withoutCacheQuotaUsesSharedZigzagPaddingAndStopsTail) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PREFILL;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                              = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size          = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache = 10;
    // This case is about the shared zigzag padding quota, so CP prefill batching must stay on.
    runtime_config.fifo_scheduler_config.cp_force_single_prefill = false;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    parallelism_config.tp_size                  = 2;
    parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    FIFOScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_stream = [&](int token_count) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::ones({token_count}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto first_stream  = make_stream(5);
    auto second_stream = make_stream(1);
    auto tail_stream   = make_stream(1);
    ASSERT_TRUE(enqueueIndividually(scheduler, {first_stream, second_stream, tail_stream}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 2);
    EXPECT_EQ(first_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(second_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(tail_stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, batchTokenQuotaUsesUnpaddedFullInputLength) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PREFILL;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 10;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    parallelism_config.tp_size                  = 4;
    parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    FIFOScheduler scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto query             = std::make_shared<GenerateInput>();
    query->input_ids       = torch::ones({9}, torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();
    auto stream =
        std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    EXPECT_TRUE(scheduler.enqueue(stream).ok());
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(scheduler.waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, withoutCacheQuotaAllowsMetadataResidualToProgress) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PREFILL;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                              = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size          = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache = 2;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_group_stream = [&](int group_size) {
        auto query                            = std::make_shared<GenerateInput>();
        query->input_ids                      = torch::ones({1}, torch::kInt32);
        query->generate_config                = std::make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 60000;
        query->group_id                       = 1001;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto first_stream    = make_group_stream(3);
    auto crossing_stream = make_group_stream(3);
    auto residual_stream = make_group_stream(3);
    ASSERT_TRUE(enqueueIndividually(scheduler, {first_stream, crossing_stream, residual_stream}));

    auto first_result = scheduler.schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_EQ(first_result->size(), 2);
    EXPECT_EQ(first_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(crossing_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(residual_stream->getStatus(), StreamState::WAITING);

    first_stream->reportEvent(StreamEvents::GenerateDone);
    crossing_stream->reportEvent(StreamEvents::GenerateDone);
    auto second_result = scheduler.schedule();
    ASSERT_TRUE(second_result.ok());
    ASSERT_EQ(second_result->size(), 1);
    EXPECT_EQ(residual_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);

    auto metadata_stream = make_group_stream(2);
    ASSERT_TRUE(scheduler.enqueue(metadata_stream).ok());
    residual_stream->reportEvent(StreamEvents::GenerateDone);
    auto third_result = scheduler.schedule();
    ASSERT_TRUE(third_result.ok());
    ASSERT_EQ(third_result->size(), 1);
    EXPECT_EQ(metadata_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, retryableMetadataStreamDoesNotBlockFollowingNormalStream) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 5, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PREFILL;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 9;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto make_stream = [&](std::vector<int> tokens, bool force_batch) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(tokens, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        query->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
        if (force_batch) {
            query->generate_config->group_timeout = 60000;
            query->group_id                       = 1001;
            query->group_size                     = 2;
        }
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    auto kv_holder = make_stream({9}, false);
    kv_holder->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(kv_holder->moveToNext(), StreamState::RUNNING);
    ASSERT_EQ(kv_holder->curBlocksNum(), 1);
    kv_holder->generate_status_->status.store(StreamState::WAITING);

    auto blocked_group_1 = make_stream({1, 2, 3, 4}, true);
    auto blocked_group_2 = make_stream({5, 6, 7, 8}, true);
    ASSERT_TRUE(enqueueIndividually(scheduler, {blocked_group_1, blocked_group_2, kv_holder}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    EXPECT_EQ(result.value().front(), kv_holder);
    EXPECT_EQ(blocked_group_1->getStatus(), StreamState::WAITING);
    EXPECT_EQ(blocked_group_2->getStatus(), StreamState::WAITING);
    EXPECT_EQ(kv_holder->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerTest, explicitGroupResidualPrecedesFollowingExplicitGroup) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 6, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

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

    auto make_group_stream = [&](std::vector<int> tokens, int64_t group_id, int group_size) {
        auto query                            = std::make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor(tokens, torch::kInt32);
        query->generate_config                = std::make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 60000;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    };

    vector<GenerateStreamPtr> first_group = {
        make_group_stream({1, 2, 3, 4}, 1001, 4),
        make_group_stream({1, 2, 3}, 1001, 4),
        make_group_stream({1}, 1001, 4),
        make_group_stream({1, 2}, 1001, 4),
    };
    vector<GenerateStreamPtr> following_group = {make_group_stream({1}, 1002, 1)};
    ASSERT_EQ(scheduler.enqueueGroup(first_group).first.size(), first_group.size());
    ASSERT_EQ(scheduler.enqueueGroup(following_group).first.size(), following_group.size());

    auto first_result = scheduler.schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_EQ(first_result.value().size(), 2);
    EXPECT_EQ(first_group[0]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(first_group[1]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(first_group[2]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(first_group[3]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(following_group[0]->getStatus(), StreamState::WAITING);

    first_group[0]->reportEvent(StreamEvents::GenerateDone);
    first_group[2]->reportEvent(StreamEvents::GenerateDone);
    auto second_result = scheduler.schedule();
    ASSERT_TRUE(second_result.ok());
    ASSERT_EQ(second_result.value().size(), 2);
    EXPECT_EQ(first_group[1]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(first_group[3]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(following_group[0]->getStatus(), StreamState::WAITING);

    first_group[1]->reportEvent(StreamEvents::GenerateDone);
    first_group[3]->reportEvent(StreamEvents::GenerateDone);
    auto third_result = scheduler.schedule();
    ASSERT_TRUE(third_result.ok());
    ASSERT_EQ(third_result.value().size(), 1);
    EXPECT_EQ(following_group[0]->getStatus(), StreamState::RUNNING);
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

TEST_F(FIFOSchedulerTest, prefillShapeLimitAppliesToOrdinaryAndGroupAdmission) {
    auto verify_admission = [](bool grouped) {
        CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
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

        vector<GenerateStreamPtr> streams = {
            makeSingleStream(model_config, runtime_config, resource_context, std::vector<int>(60, 1)),
            makeSingleStream(model_config, runtime_config, resource_context, std::vector<int>(39, 1)),
        };
        if (grouped) {
            ASSERT_EQ(scheduler.enqueueGroup(streams).first, std::vector<bool>({true, true}));
        } else {
            ASSERT_TRUE(enqueueIndividually(scheduler, streams));
        }

        auto result = scheduler.schedule();
        ASSERT_TRUE(result.ok());
        ASSERT_EQ(result->size(), 1);
        EXPECT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
        EXPECT_EQ(streams[1]->getStatus(), StreamState::WAITING);
    };

    verify_admission(false);
    verify_admission(true);
}

TEST_F(FIFOSchedulerTest, prefillShapeIncludesReusedPrefixLength) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.role_type     = RoleType::PREFILL;

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

    auto cached_stream =
        makeSingleStream(model_config, runtime_config, resource_context, std::vector<int>(60, 1));
    cached_stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(cached_stream->moveToNext(), StreamState::RUNNING);
    cached_stream->setReuseLength(59);
    cached_stream->generate_status_->status.store(StreamState::WAITING);
    ASSERT_EQ(cached_stream->contextLength(), 1);
    ASSERT_EQ(cached_stream->prefixLength(), 59);

    auto short_stream =
        makeSingleStream(model_config, runtime_config, resource_context, std::vector<int>(39, 1));
    ASSERT_TRUE(enqueueIndividually(scheduler, {cached_stream, short_stream}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(cached_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(short_stream->getStatus(), StreamState::WAITING);
}

TEST_F(FIFOSchedulerTest, prefillShapeUsesCurrentBatchSizeAsWidth) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
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

    auto long_single =
        makeSingleStream(model_config, runtime_config, resource_context, std::vector<int>(40, 1));
    auto batched_query                                   = std::make_shared<GenerateInput>();
    batched_query->input_ids                             = torch::ones({10}, torch::kInt32);
    batched_query->generate_config                       = std::make_shared<GenerateConfig>();
    batched_query->generate_config->num_return_sequences = 3;
    auto short_triple =
        std::make_shared<NormalGenerateStream>(batched_query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_EQ(short_triple->currentBatchSize(), 3);

    vector<GenerateStreamPtr> streams = {long_single, short_triple};
    ASSERT_EQ(scheduler.enqueueGroup(streams).first, std::vector<bool>({true, true}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(long_single->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(short_triple->getStatus(), StreamState::WAITING);
}

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
    for (const auto& stream : streams) {
        stream->generateInput()->group_id   = 42;
        stream->generateInput()->group_size = static_cast<int>(streams.size());
    }
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler.pending_group_fallback_count_.load(), 0);
    for (const auto& task : scheduler.runningTaskList()) {
        EXPECT_EQ(task.batch_id, 42);
    }
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
    EXPECT_EQ(scheduler.pending_group_fallback_count_.load(), 1);
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
    EXPECT_EQ(scheduler.pending_group_fallback_count_.load(), 1);

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

TEST_F(FIFOSchedulerTest, groupTokenCapExceededKeepsResidualGroup) {
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
    // The first stream fits. The second one remains in the residual group.
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(streams[1]->getStatus(), StreamState::WAITING);
    for (const auto& s : streams) {
        ASSERT_FALSE(s->hasError());
    }
}

TEST_F(FIFOSchedulerTest, groupCacheShortageDefersUnallocatedStreams) {
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
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    EXPECT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(streams[0]->hasError());
    EXPECT_GT(streams[0]->curBlocksNum(), 0);
    EXPECT_EQ(streams[1]->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(streams[1]->hasError());
    EXPECT_EQ(streams[1]->curBlocksNum(), 0);
}

TEST_F(FIFOSchedulerTest, groupCacheShortageStillAdmitsLaterSmallerStreams) {
    // Five usable blocks. The first request consumes four, the second request
    // needs three and must be deferred, the one-block request still fits, and
    // the final two-block request remains in the residual group.
    CacheConfig cache_config  = makeMhaCacheConfig(1, 6, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 5);
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
        makeSingleStream(model_config, runtime_config, resource_context, {1, 2, 3, 4}),
        makeSingleStream(model_config, runtime_config, resource_context, {1, 2, 3}),
        makeSingleStream(model_config, runtime_config, resource_context, {1}),
        makeSingleStream(model_config, runtime_config, resource_context, {1, 2}),
    };
    ASSERT_EQ(scheduler.enqueueGroup(streams).first, std::vector<bool>({true, true, true, true}));

    auto first_result = scheduler.schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_EQ(first_result.value().size(), 2);
    EXPECT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(streams[1]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(streams[2]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(streams[3]->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(streams[1]->hasError());
    EXPECT_FALSE(streams[3]->hasError());
    EXPECT_EQ(scheduler.runningStreamsSize(), 2);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 2);

    streams[0]->reportEvent(StreamEvents::GenerateDone);
    streams[2]->reportEvent(StreamEvents::GenerateDone);
    auto second_result = scheduler.schedule();
    ASSERT_TRUE(second_result.ok());
    ASSERT_EQ(second_result.value().size(), 2);
    EXPECT_EQ(streams[1]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(streams[3]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, residualGroupPrecedesFollowingGroups) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 6, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
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

    vector<GenerateStreamPtr> first_group = {
        makeSingleStream(model_config, runtime_config, resource_context, {1, 2, 3, 4}),
        makeSingleStream(model_config, runtime_config, resource_context, {1, 2, 3}),
        makeSingleStream(model_config, runtime_config, resource_context, {1}),
        makeSingleStream(model_config, runtime_config, resource_context, {1, 2}),
    };
    vector<GenerateStreamPtr> second_group = {
        makeSingleStream(model_config, runtime_config, resource_context, {1}),
    };
    scheduler.enqueueGroup(first_group);
    scheduler.enqueueGroup(second_group);

    auto first_result = scheduler.schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_EQ(first_result.value().size(), 2);
    ASSERT_EQ(first_group[1]->getStatus(), StreamState::WAITING);
    ASSERT_EQ(first_group[3]->getStatus(), StreamState::WAITING);
    ASSERT_EQ(second_group[0]->getStatus(), StreamState::WAITING);

    first_group[0]->reportEvent(StreamEvents::GenerateDone);
    first_group[2]->reportEvent(StreamEvents::GenerateDone);
    auto second_result = scheduler.schedule();
    ASSERT_TRUE(second_result.ok());
    ASSERT_EQ(second_result.value().size(), 2);
    EXPECT_EQ(first_group[1]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(first_group[3]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(second_group[0]->getStatus(), StreamState::WAITING);
}

TEST_F(FIFOSchedulerTest, groupTokenCapStillAdmitsLaterSmallerStreams) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 100, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
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

    vector<GenerateStreamPtr> streams = {
        makeSingleStream(model_config, runtime_config, resource_context, std::vector<int>(40, 1)),
        makeSingleStream(model_config, runtime_config, resource_context, std::vector<int>(80, 1)),
        makeSingleStream(model_config, runtime_config, resource_context, std::vector<int>(40, 1)),
    };
    scheduler.enqueueGroup(streams);

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 2);
    EXPECT_EQ(streams[0]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(streams[1]->getStatus(), StreamState::WAITING);
    EXPECT_EQ(streams[2]->getStatus(), StreamState::RUNNING);
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
    ASSERT_TRUE(scheduler.enqueue(waiting_stream).ok());

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
    ASSERT_TRUE(scheduler.enqueue(waiting_stream_1).ok());
    ASSERT_TRUE(scheduler.enqueue(waiting_stream_2).ok());
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

TEST_F(FIFOSchedulerTest, continuousOrdinaryTrafficCannotStarveExplicitGroup) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 20, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto running_normal = makeSingleStream(model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(running_normal).ok());
    ASSERT_EQ(scheduler.schedule()->size(), 1);

    vector<GenerateStreamPtr> group_streams = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    ASSERT_EQ(scheduler.enqueueGroup(group_streams).first, std::vector<bool>({true, true}));

    auto ordinary_tail_1 = makeSingleStream(model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(ordinary_tail_1).ok());
    auto drain_result = scheduler.schedule();
    ASSERT_TRUE(drain_result.ok());
    ASSERT_EQ(drain_result->size(), 1);
    EXPECT_EQ(ordinary_tail_1->getStatus(), StreamState::WAITING);

    // A second ordinary arrival before the running stream completes must not
    // extend the normal lane indefinitely (Decode normally allows top-up).
    auto ordinary_tail_2 = makeSingleStream(model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(ordinary_tail_2).ok());
    running_normal->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler.schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result->size(), 2);
    for (const auto& stream : group_streams) {
        EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    }
    EXPECT_EQ(ordinary_tail_1->getStatus(), StreamState::WAITING);
    EXPECT_EQ(ordinary_tail_2->getStatus(), StreamState::WAITING);

    for (const auto& stream : group_streams) {
        stream->reportEvent(StreamEvents::GenerateDone);
    }
    auto normal_result = scheduler.schedule();
    ASSERT_TRUE(normal_result.ok());
    ASSERT_EQ(normal_result->size(), 2);
    EXPECT_EQ(ordinary_tail_1->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(ordinary_tail_2->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerTest, blockedNormalLaneLeavesResidualCapacityToExplicitGroup) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 8, 1, 4, 1, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 6;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 6;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // This ordinary stream is permanently too large for the per-round token
    // budget but valid against the Engine's total KV capacity.
    auto blocked_normal = makeSingleStream(
        model_config, runtime_config, resource_context, std::vector<int>(6, 1));
    vector<GenerateStreamPtr> small_group = {
        makeSingleStream(model_config, runtime_config, resource_context, {1}),
        makeSingleStream(model_config, runtime_config, resource_context, {2}),
    };
    ASSERT_TRUE(scheduler.enqueue(blocked_normal).ok());
    ASSERT_EQ(scheduler.enqueueGroup(small_group).first, std::vector<bool>({true, true}));

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 2);
    EXPECT_EQ(blocked_normal->getStatus(), StreamState::WAITING);
    EXPECT_EQ(small_group[0]->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(small_group[1]->getStatus(), StreamState::RUNNING);
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

    ASSERT_TRUE(scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context)).ok());
    ASSERT_TRUE(scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context)).ok());
    ASSERT_TRUE(scheduler.enqueue(makeSingleStream(model_config, runtime_config, resource_context)).ok());

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
    ASSERT_TRUE(scheduler.enqueue(single_a).ok());

    vector<GenerateStreamPtr> group_streams = {
        makeSingleStream(model_config, runtime_config, resource_context),
        makeSingleStream(model_config, runtime_config, resource_context),
    };
    scheduler.enqueueGroup(group_streams);
    auto single_b = makeSingleStream(model_config, runtime_config, resource_context);
    ASSERT_TRUE(scheduler.enqueue(single_b).ok());

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

TEST_F(FIFOSchedulerTest, testCpPrefillBatchesMultipleStreams) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    // This test exercises CP prefill batching, so opt out of the conservative
    // single-prefill default (see testCpForceSinglePrefillConfig).
    runtime_config.fifo_scheduler_config.cp_force_single_prefill = false;
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
    ASSERT_TRUE(enqueueIndividually(scheduler, streams));
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
}

TEST_F(FIFOSchedulerTest, testGroupMetadataDoesNotDelayWaitingStreams) {
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

    for (int i = 0; i < 2; ++i) {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 60000;
        query->group_id                       = 100;
        query->group_size                     = 3;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    for (const auto& task : scheduler.runningTaskList()) {
        EXPECT_EQ(task.batch_id, 100);
    }
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
    EXPECT_EQ(scheduler.pending_group_fallback_count_.load(), 1);
}

TEST_F(FIFOSchedulerTest, testExpiredGroupMetadataDoesNotAffectWaitingStreams) {
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

    // Expired metadata is still reported, but never changes FIFO admission.
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

    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testGroupMetadataDoesNotBypassNormalTokenCap) {
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

TEST_F(FIFOSchedulerTest, testGroupMetadataDoesNotIsolateWaitingStreams) {
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

    // Enqueue a normal stream followed by two streams carrying the same batch metadata.
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        query->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
        auto normal_stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(normal_stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 60000;
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
        query->generate_config->group_timeout = 60000;
        query->group_id                       = group_id;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
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

TEST_F(FIFOSchedulerTest, testDifferentGroupMetadataDoesNotIsolateWaitingStreams) {
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

    // These are ordinary enqueue() calls; batch metadata must not create scheduler groups.
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 60000;
        query->group_id                       = group_id_a;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query  = make_shared<GenerateInput>();
        query->input_ids                      = torch::tensor({1}, torch::kInt32);
        query->generate_config                = make_shared<GenerateConfig>();
        query->generate_config->group_timeout = 60000;
        query->group_id                       = group_id_b;
        query->group_size                     = group_size;
        query->begin_time_us                  = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 4);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 4);
}

}  // namespace rtp_llm
