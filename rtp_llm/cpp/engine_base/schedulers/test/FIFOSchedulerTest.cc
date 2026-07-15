#include <memory>
#include <chrono>
#include <string>
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

static StreamUpdateInfo makeSingleTokenUpdate(int token_id) {
    auto new_tokens = torch::tensor(std::vector<int32_t>{token_id}, torch::kInt32).reshape({1, 1});
    return {new_tokens,
            1,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            true,
            false};
}

class FIFOSchedulerTest: public DeviceTestBase {
public:
    FIFOSchedulerTest() {}
};

static PDSepConfig makePDFusionPDSepConfig() {
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    return pd_sep_config;
}

static GenerateStreamPtr makeStream(const std::vector<int>& ids,
                                    const ModelConfig&      model_config,
                                    const RuntimeConfig&    runtime_config,
                                    const ResourceContext&  resource_context,
                                    int                     max_new_tokens       = 0,
                                    int                     num_return_sequences = 1) {
    auto query             = std::make_shared<GenerateInput>();
    query->input_ids       = torch::tensor(ids, torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();
    if (max_new_tokens > 0) {
        query->generate_config->max_new_tokens = max_new_tokens;
    }
    query->generate_config->num_return_sequences = num_return_sequences;
    return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
}

static void expireStream(const GenerateStreamPtr& stream) {
    stream->generateConfig()->timeout_ms = 1;
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 100 * 1000);
}

static testing::AssertionResult
expectPrefillBatch(const absl::StatusOr<std::list<GenerateStreamPtr>>& batch_status,
                   const std::vector<GenerateStreamPtr>&                expected_streams,
                   const std::vector<int>&                              expected_chunk_lens) {
    if (!batch_status.ok()) {
        return testing::AssertionFailure() << "schedule failed: " << batch_status.status().ToString();
    }
    const auto& batch = batch_status.value();
    if (batch.size() != expected_streams.size()) {
        return testing::AssertionFailure() << "batch size " << batch.size() << ", expected "
                                           << expected_streams.size();
    }
    if (expected_streams.size() != expected_chunk_lens.size()) {
        return testing::AssertionFailure() << "expected stream/chunk size mismatch: " << expected_streams.size()
                                           << " vs " << expected_chunk_lens.size();
    }

    auto it = batch.begin();
    for (size_t i = 0; i < expected_streams.size(); ++i, ++it) {
        if (it->get() != expected_streams[i].get()) {
            return testing::AssertionFailure() << "stream mismatch at index " << i;
        }
        if ((*it)->currentChunkLen() != expected_chunk_lens[i]) {
            return testing::AssertionFailure() << "chunk length at index " << i << " is "
                                               << (*it)->currentChunkLen() << ", expected "
                                               << expected_chunk_lens[i];
        }
    }
    return testing::AssertionSuccess();
}

struct ChunkSchedulerTestConfig {
    RoleType    role_type                 = RoleType::PREFILL;
    int         block_num                 = 64;
    int         seq_size_per_block        = 4;
    int         max_batch_tokens_size     = 1024;
    int         prefill_chunk_size        = 16;
    std::string decode_prefill_ratio;
};

template<typename SchedulerType>
class ChunkSchedulerTestEnv {
public:
    explicit ChunkSchedulerTestEnv(const ChunkSchedulerTestConfig& config):
        cache_manager(std::make_shared<KVCacheManager>(rtp_llm::test::makeSimpleMhaCacheConfig(
            1, config.block_num, config.seq_size_per_block, rtp_llm::DataType::TYPE_FP16, 1, 4))) {
        resource_context.cache_manager = cache_manager;
        resource_context.role_type     = config.role_type;

        model_config.max_seq_len = 128;
        model_config.vocab_size  = 2048;

        runtime_config.max_generate_batch_size                     = 16;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = config.max_batch_tokens_size;
        runtime_config.fifo_scheduler_config.prefill_chunk_size    = config.prefill_chunk_size;
        if (!config.decode_prefill_ratio.empty()) {
            runtime_config.fifo_scheduler_config.decode_prefill_ratio = config.decode_prefill_ratio;
        }
        pd_sep_config.role_type = config.role_type;
    }

    bool init() {
        if (!cache_manager->init()) {
            return false;
        }
        scheduler_ = std::make_unique<SchedulerType>(runtime_config,
                                                     model_config,
                                                     pd_sep_config,
                                                     parallelism_config,
                                                     model_specific_config,
                                                     cache_manager);
        return true;
    }

    SchedulerType& scheduler() {
        return *scheduler_;
    }

    GenerateStreamPtr makeStream(const std::vector<int>& ids,
                                 int                     max_new_tokens       = 0,
                                 int                     num_return_sequences = 1) const {
        return rtp_llm::makeStream(
            ids, model_config, runtime_config, resource_context, max_new_tokens, num_return_sequences);
    }

    std::shared_ptr<KVCacheManager> cache_manager;
    ResourceContext                 resource_context;
    ModelConfig                     model_config;
    RuntimeConfig                   runtime_config;

private:
    PDSepConfig                   pd_sep_config;
    ParallelismConfig             parallelism_config;
    ModelSpecificConfig           model_specific_config;
    std::unique_ptr<SchedulerType> scheduler_;
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

TEST_F(FIFOSchedulerTest, testChunkedPrefillNeverReturnsMixedContextAndDecodeBatch) {
    ChunkSchedulerTestConfig config;
    config.role_type          = RoleType::PDFUSION;
    config.seq_size_per_block = 2;
    config.prefill_chunk_size = 4;
    ChunkSchedulerTestEnv<FIFOScheduler> env(config);
    ASSERT_TRUE(env.init());
    auto& scheduler = env.scheduler();

    auto short_stream = env.makeStream({1, 2}, 4);
    auto long_stream  = env.makeStream({3, 4, 5, 6, 7, 8, 9, 10}, 4);
    scheduler.batchEnqueue({short_stream, long_stream});

    auto first = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(first, {short_stream, long_stream}, {2, 2}));
    short_stream->update(makeSingleTokenUpdate(101));
    long_stream->update(makeSingleTokenUpdate(102));
    short_stream->setReserveStep(1);
    const size_t short_blocks_before_park  = short_stream->curBlocksNum();
    const size_t long_blocks_before_decode = long_stream->curBlocksNum();
    const size_t free_blocks_before_park   = env.cache_manager->freeBlocksNum();

    auto second = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(second, {long_stream}, {4}));
    ASSERT_EQ(short_stream->curBlocksNum(), short_blocks_before_park);
    ASSERT_EQ(long_stream->curBlocksNum(), long_blocks_before_decode);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), free_blocks_before_park);
    long_stream->update(makeSingleTokenUpdate(103));  // middle chunk -> context
    ASSERT_FALSE(short_stream->isContextStream());
    ASSERT_TRUE(long_stream->isContextStream());

    auto third = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(third, {long_stream}, {2}));
    ASSERT_TRUE(third->front()->isContextStream());
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(scheduler.onflightStreams(), 2);
    const auto running_tasks = scheduler.runningTaskList();
    ASSERT_EQ(running_tasks.size(), 2);
    for (const auto& task : running_tasks) {
        ASSERT_EQ(task.prefix_length, 0);  // initial cache hit, not the current chunk start
    }
    ASSERT_EQ(short_stream->curBlocksNum(), short_blocks_before_park);
    ASSERT_EQ(long_stream->curBlocksNum(), long_blocks_before_decode);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), free_blocks_before_park);

    long_stream->update(makeSingleTokenUpdate(104));
    ASSERT_FALSE(long_stream->isContextStream());
    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode->size(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.onflightStreams(), 2);
    ASSERT_EQ(short_stream->curBlocksNum(), short_blocks_before_park + 1);
    ASSERT_EQ(long_stream->curBlocksNum(), long_blocks_before_decode + 1);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), free_blocks_before_park - 2);
    for (const auto& stream : *decode) {
        ASSERT_FALSE(stream->isContextStream());
    }
}

TEST_F(FIFOSchedulerTest, testChunkedPrefillReapsTerminalStreamsBeforeDeferredDecodePromotion) {
    ChunkSchedulerTestConfig config;
    config.role_type          = RoleType::PDFUSION;
    config.block_num          = 8;
    config.seq_size_per_block = 2;
    config.prefill_chunk_size = 6;
    ChunkSchedulerTestEnv<FIFOScheduler> env(config);
    ASSERT_TRUE(env.init());
    auto& scheduler = env.scheduler();

    auto cancelled_stream = env.makeStream({1, 2}, 4);
    auto parked_stream    = env.makeStream({3, 4}, 4);
    auto final_stream     = env.makeStream({5, 6, 7, 8, 9, 10, 11, 12}, 1);
    scheduler.batchEnqueue({cancelled_stream, parked_stream, final_stream});

    auto first = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(first, {cancelled_stream, parked_stream, final_stream}, {2, 2, 2}));
    cancelled_stream->update(makeSingleTokenUpdate(101));
    parked_stream->update(makeSingleTokenUpdate(102));
    final_stream->update(makeSingleTokenUpdate(103));

    parked_stream->setReserveStep(1);
    const size_t parked_blocks_before_decode = parked_stream->curBlocksNum();
    const size_t free_blocks_while_parked    = env.cache_manager->freeBlocksNum();
    ASSERT_EQ(free_blocks_while_parked, 1);

    auto second = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(second, {final_stream}, {6}));
    ASSERT_EQ(parked_stream->curBlocksNum(), parked_blocks_before_decode);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), free_blocks_while_parked);
    final_stream->update(makeSingleTokenUpdate(104));
    ASSERT_TRUE(final_stream->hasEvent(StreamEvents::GenerateDone));

    cancelled_stream->reportError(ErrorCode::CANCELLED, "cancelled while deferred");
    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode->size(), 1);
    ASSERT_EQ(decode->front().get(), parked_stream.get());
    ASSERT_TRUE(cancelled_stream->isFinished());
    ASSERT_EQ(cancelled_stream->curBlocksNum(), 0);
    ASSERT_TRUE(final_stream->isFinished());
    ASSERT_EQ(final_stream->curBlocksNum(), 0);
    ASSERT_FALSE(parked_stream->hasError());
    ASSERT_EQ(parked_stream->curBlocksNum(), parked_blocks_before_decode + 1);
    ASSERT_GT(env.cache_manager->freeBlocksNum(), free_blocks_while_parked);
}

TEST_F(FIFOSchedulerTest, testChunkedPrefillDeferredDecodeMallocFailureOccursOnPromotion) {
    ChunkSchedulerTestConfig config;
    config.role_type          = RoleType::PDFUSION;
    config.block_num          = 5;
    config.seq_size_per_block = 2;
    config.prefill_chunk_size = 4;
    ChunkSchedulerTestEnv<FIFOScheduler> env(config);
    ASSERT_TRUE(env.init());
    auto& scheduler = env.scheduler();

    auto short_stream = env.makeStream({1, 2}, 4);
    auto long_stream  = env.makeStream({3, 4, 5, 6, 7, 8}, 4);
    scheduler.batchEnqueue({short_stream, long_stream});

    auto first = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(first, {short_stream, long_stream}, {2, 2}));
    short_stream->update(makeSingleTokenUpdate(101));
    long_stream->update(makeSingleTokenUpdate(102));
    short_stream->setReserveStep(1);
    const size_t short_blocks_before_park = short_stream->curBlocksNum();
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), 0);

    auto second = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(second, {long_stream}, {4}));
    ASSERT_FALSE(short_stream->hasError());
    ASSERT_EQ(short_stream->curBlocksNum(), short_blocks_before_park);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), 0);
    long_stream->update(makeSingleTokenUpdate(103));

    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode->size(), 1);
    ASSERT_EQ(decode->front().get(), long_stream.get());
    ASSERT_TRUE(short_stream->isFinished());
    ASSERT_TRUE(short_stream->hasError());
    ASSERT_EQ(short_stream->stopReason(), "incrKVBlock failed: LACK MEM");
    ASSERT_EQ(short_stream->curBlocksNum(), 0);
    ASSERT_FALSE(long_stream->hasError());
    ASSERT_EQ(scheduler.onflightStreams(), 1);
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

TEST_F(FIFOSchedulerTest, testDecodeRoundReapsTimedOutWaitingStream) {
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

    expireStream(s2);

    auto r3 = scheduler.schedule();  // still a DECODE round; reap timed-out s2 without admitting it
    ASSERT_TRUE(r3.ok());
    ASSERT_EQ(r3.value().size(), 1);
    ASSERT_EQ(r3.value().front().get(), s1.get());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_TRUE(s2->isFinished());
    ASSERT_EQ(s2->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
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

TEST_F(FIFOSchedulerTest, testPendingAndRunningDecodeTimeoutsAreReaped) {
    ChunkSchedulerTestConfig config;
    config.role_type            = RoleType::PDFUSION;
    config.decode_prefill_ratio = "1/2";
    ChunkSchedulerTestEnv<PDFusionRatioScheduler> env(config);
    ASSERT_TRUE(env.init());
    auto& scheduler = env.scheduler();

    auto first = env.makeStream({1, 2});
    ASSERT_TRUE(scheduler.enqueue(first).ok());
    auto first_prefill = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(first_prefill, {first}, {2}));
    first->update(makeSingleTokenUpdate(101));

    auto second = env.makeStream({3, 4});
    ASSERT_TRUE(scheduler.enqueue(second).ok());
    auto second_prefill = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(second_prefill, {second}, {2}));
    second->update(makeSingleTokenUpdate(102));
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 2);

    expireStream(first);
    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode->size(), 1);
    ASSERT_EQ(decode->front().get(), second.get());
    ASSERT_TRUE(first->isFinished());
    ASSERT_EQ(first->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    auto next_prefill = env.makeStream({5, 6});
    ASSERT_TRUE(scheduler.enqueue(next_prefill).ok());
    expireStream(second);

    auto prefill_round = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(prefill_round, {next_prefill}, {2}));
    ASSERT_TRUE(second->isFinished());
    ASSERT_EQ(second->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
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

TEST_F(FIFOSchedulerTest, testPDFusionContinuationKeepsPrefillBatchBoundary) {
    ChunkSchedulerTestConfig config;
    config.role_type             = RoleType::PDFUSION;
    config.seq_size_per_block    = 2;
    config.max_batch_tokens_size = 8;
    config.prefill_chunk_size    = 2;
    config.decode_prefill_ratio  = "1/2";
    ChunkSchedulerTestEnv<PDFusionRatioScheduler> env(config);
    ASSERT_TRUE(env.init());
    auto& scheduler = env.scheduler();

    // Establish an active decode stream so the 1/2 cadence can choose consecutive prefill rounds.
    auto decode_stream = env.makeStream({1});
    ASSERT_TRUE(scheduler.enqueue(decode_stream).ok());
    auto seed_prefill = scheduler.schedule();
    ASSERT_TRUE(seed_prefill.ok());
    decode_stream->update(makeSingleTokenUpdate(100));
    auto seed_decode = scheduler.schedule();
    ASSERT_TRUE(seed_decode.ok());
    ASSERT_EQ(seed_decode->size(), 1);
    decode_stream->update(makeSingleTokenUpdate(101));

    auto first_long  = env.makeStream({2, 3, 4, 5, 6, 7});
    auto second_long = env.makeStream({8, 9, 10, 11, 12, 13});
    ASSERT_TRUE(scheduler.enqueue(first_long).ok());
    ASSERT_TRUE(scheduler.enqueue(second_long).ok());

    auto first_prefill = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(first_prefill, {first_long}, {2}));
    first_long->update(makeSingleTokenUpdate(102));
    ASSERT_TRUE(first_long->isMiddleChunk());

    // The next prefill slot must continue the same admitted batch. Admitting second_long here
    // would let both context batches accumulate and later bypass max_batch_tokens_size on promote.
    auto continuation = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(continuation, {first_long}, {2}));
    ASSERT_TRUE(continuation->front()->isContextStream());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    first_long->update(makeSingleTokenUpdate(103));
    ASSERT_TRUE(first_long->isLastChunk());
    auto decode_round = scheduler.schedule();
    ASSERT_TRUE(decode_round.ok());
    ASSERT_EQ(decode_round->size(), 1);
    ASSERT_EQ(decode_round->front().get(), decode_stream.get());
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);

    expireStream(first_long);
    decode_stream->update(makeSingleTokenUpdate(104));
    auto next_prefill = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(next_prefill, {second_long}, {2}));
    ASSERT_TRUE(first_long->isFinished());
    ASSERT_EQ(first_long->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, testPDFusionMixedActiveBatchSeparatesFinalAndMiddleChunks) {
    ChunkSchedulerTestConfig config;
    config.role_type            = RoleType::PDFUSION;
    config.seq_size_per_block   = 2;
    config.prefill_chunk_size   = 4;
    config.decode_prefill_ratio = "1/3";
    ChunkSchedulerTestEnv<PDFusionRatioScheduler> env(config);
    ASSERT_TRUE(env.init());
    auto& scheduler = env.scheduler();

    auto short_stream = env.makeStream({1, 2});
    auto long_stream  = env.makeStream({3, 4, 5, 6, 7, 8, 9, 10});
    ASSERT_TRUE(scheduler.enqueue(short_stream).ok());
    ASSERT_TRUE(scheduler.enqueue(long_stream).ok());

    auto first_prefill = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(first_prefill, {short_stream, long_stream}, {2, 2}));

    short_stream->update(makeSingleTokenUpdate(101));
    long_stream->update(makeSingleTokenUpdate(102));
    ASSERT_FALSE(short_stream->isContextStream());
    ASSERT_TRUE(long_stream->isMiddleChunk());
    ASSERT_EQ(long_stream->prefixLength(), 2);

    short_stream->setReserveStep(1);
    long_stream->setReserveStep(1);
    const size_t short_blocks_before_continuation = short_stream->curBlocksNum();
    const size_t long_blocks_before_continuation  = long_stream->curBlocksNum();
    const size_t free_blocks_before_continuation  = env.cache_manager->freeBlocksNum();

    // The short final prefill waits for a real decode round without growing KV. The long middle
    // chunk remains the only member of the active prefill batch.
    auto second_prefill = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(second_prefill, {long_stream}, {4}));
    ASSERT_TRUE(second_prefill->front()->isContextStream());
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 2);
    ASSERT_EQ(short_stream->curBlocksNum(), short_blocks_before_continuation);
    ASSERT_EQ(long_stream->curBlocksNum(), long_blocks_before_continuation);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), free_blocks_before_continuation);

    long_stream->update(makeSingleTokenUpdate(103));
    ASSERT_TRUE(long_stream->isContextStream());
    ASSERT_TRUE(long_stream->isLastChunk());
    ASSERT_EQ(long_stream->prefixLength(), 6);

    auto final_prefill = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(final_prefill, {long_stream}, {2}));
    ASSERT_EQ(short_stream->curBlocksNum(), short_blocks_before_continuation);
    ASSERT_EQ(long_stream->curBlocksNum(), long_blocks_before_continuation);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), free_blocks_before_continuation);

    long_stream->update(makeSingleTokenUpdate(104));
    ASSERT_FALSE(long_stream->isContextStream());

    auto decode = scheduler.schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode->size(), 2);
    auto decode_it = decode->begin();
    ASSERT_EQ((decode_it++)->get(), short_stream.get());
    ASSERT_EQ(decode_it->get(), long_stream.get());
    ASSERT_FALSE(short_stream->isContextStream());
    ASSERT_FALSE(long_stream->isContextStream());
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(scheduler.pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(short_stream->curBlocksNum(), short_blocks_before_continuation + 1);
    ASSERT_EQ(long_stream->curBlocksNum(), long_blocks_before_continuation + 1);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), free_blocks_before_continuation - 2);
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

template<typename SchedulerType>
static void verifyGlobalChunkBudgetFourRoundPrefix(RoleType role_type, const std::string& decode_prefill_ratio = "") {
    ChunkSchedulerTestConfig config;
    config.role_type              = role_type;
    config.decode_prefill_ratio   = decode_prefill_ratio;
    ChunkSchedulerTestEnv<SchedulerType> env(config);
    ASSERT_TRUE(env.init());
    auto& scheduler = env.scheduler();

    auto s1 = env.makeStream({1});
    auto s2 = env.makeStream({2});
    auto s3 = env.makeStream({3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21});
    auto s4 = env.makeStream(
        {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50});
    s3->setReuseLength(4);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    ASSERT_TRUE(scheduler.enqueue(s2).ok());
    ASSERT_TRUE(scheduler.enqueue(s3).ok());
    ASSERT_TRUE(scheduler.enqueue(s4).ok());

    auto round1 = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(round1, {s1, s2, s3}, {1, 1, 12}));
    if (role_type == RoleType::PREFILL) {
        ASSERT_EQ(scheduler.runningStreamsSize(), 4);
    }
    ASSERT_EQ(s4->reuseLength(), 0);
    s1->update(makeSingleTokenUpdate(101));
    s2->update(makeSingleTokenUpdate(102));
    s3->update(makeSingleTokenUpdate(103));

    auto round2 = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(round2, {s3, s4}, {3, 12}));
    s3->update(makeSingleTokenUpdate(104));
    s4->update(makeSingleTokenUpdate(105));

    auto round3 = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(round3, {s4}, {16}));
    s4->update(makeSingleTokenUpdate(106));

    auto round4 = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(round4, {s4}, {3}));
}

TEST_F(FIFOSchedulerTest, testGlobalChunkBudgetFIFOFourRoundPrefix) {
    verifyGlobalChunkBudgetFourRoundPrefix<FIFOScheduler>(RoleType::PREFILL);
}

TEST_F(FIFOSchedulerTest, testGlobalChunkBudgetPDFusionFourRoundPrefix) {
    verifyGlobalChunkBudgetFourRoundPrefix<PDFusionRatioScheduler>(RoleType::PDFUSION, "1/100");
}

TEST_F(FIFOSchedulerTest, testGlobalChunkBudgetValidatesFanoutReuseAndShortFinal) {
    ChunkSchedulerTestConfig config;
    ChunkSchedulerTestEnv<FIFOScheduler> env(config);
    ASSERT_TRUE(env.init());
    auto& scheduler = env.scheduler();

    const size_t free_blocks_before = env.cache_manager->freeBlocksNum();
    auto         long_stream        = env.makeStream({1, 2, 3, 4, 5}, 0, 8);
    ASSERT_FALSE(scheduler.enqueue(long_stream).ok());
    ASSERT_TRUE(long_stream->hasError());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(env.cache_manager->freeBlocksNum(), free_blocks_before);

    auto unaligned_stream = env.makeStream({6, 7, 8, 9, 10, 11, 12, 13, 14});
    unaligned_stream->setReuseLength(1);
    ASSERT_TRUE(scheduler.enqueue(unaligned_stream).ok());

    auto short_stream = env.makeStream({15, 16}, 0, 8);
    ASSERT_TRUE(scheduler.enqueue(short_stream).ok());
    auto batch = scheduler.schedule();
    ASSERT_TRUE(expectPrefillBatch(batch, {short_stream}, {2}));
    ASSERT_TRUE(unaligned_stream->hasError());
    ASSERT_TRUE(unaligned_stream->isFinished());
    ASSERT_EQ(short_stream->currentBatchSize(), 8);
}

}  // namespace rtp_llm
