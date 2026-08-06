#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
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

TEST_F(FIFOSchedulerTest, testWakeUnblocksIdleSchedule) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    auto schedule_future = std::async(std::launch::async, [&scheduler]() { return scheduler.schedule(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.wake();

    ASSERT_EQ(schedule_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = schedule_future.get();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.value().empty());
}

TEST_F(FIFOSchedulerTest, testBatchDecodeWakeUnblocksIdleSchedule) {
    RuntimeConfig runtime_config;
    runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 8;
    BatchDecodeScheduler scheduler(runtime_config, nullptr, nullptr);

    auto schedule_future = std::async(std::launch::async, [&scheduler]() { return scheduler.schedule(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.wake();

    ASSERT_EQ(schedule_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = schedule_future.get();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.value().empty());
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
