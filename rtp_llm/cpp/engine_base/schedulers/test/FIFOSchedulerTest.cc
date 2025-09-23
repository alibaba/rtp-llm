#include <memory>
#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

class FIFOSchedulerTest: public DeviceTestBase {
public:
};

TEST_F(FIFOSchedulerTest, testSimple) {
    KVCacheParam                  param = {1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16};
    CacheConfig                   cache_config(param);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 3);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GptInitParameter config;
    config.max_seq_len_             = 8192;
    config.max_generate_batch_size_ = 100;
    config.max_batch_tokens_size_   = 8192;
    FIFOScheduler                  scheduler(config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream    = make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 1);
    ASSERT_EQ(cache_manager->freeBlockNums(), 2);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    stream->setFinishedWithoutLock();

    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlockNums(), 3);
}

TEST_F(FIFOSchedulerTest, testInitKVCacheLackMem) {
    KVCacheParam                  param = {1, 2, 1, 4, 2, rtp_llm::DataType::TYPE_FP16};
    CacheConfig                   cache_config(param);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 1);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    GptInitParameter config;
    config.max_seq_len_             = 8192;
    config.max_generate_batch_size_ = 100;
    config.max_batch_tokens_size_   = 8192;
    FIFOScheduler                  scheduler(config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream    = make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 0);
    ASSERT_TRUE(stream->stopped());
    ASSERT_EQ(stream->stopReason(), "input len 3 is greater than kv cache max seq len 2");

    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlockNums(), 1);
}

TEST_F(FIFOSchedulerTest, testIncrKVCacheLackMem) {
    KVCacheParam                  param = {1, 3, 1, 4, 2, rtp_llm::DataType::TYPE_FP16};
    CacheConfig                   cache_config(param);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 2);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    GptInitParameter config;
    config.max_seq_len_             = 8192;
    config.max_generate_batch_size_ = 100;
    config.max_batch_tokens_size_   = 8192;
    FIFOScheduler                  scheduler(config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream    = make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 1);
    ASSERT_FALSE(stream->stopped());
    ASSERT_EQ(stream->stopReason(), "");
    ASSERT_EQ(cache_manager->freeBlockNums(), 0);

    stream->setSeqLength(stream->seqLength() + 1);
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_TRUE(stream->stopped());
    ASSERT_EQ(stream->stopReason(), "LACK MEM");
    ASSERT_EQ(cache_manager->freeBlockNums(), 2);

    auto streams_status3 = scheduler.schedule();
    ASSERT_TRUE(streams_status3.ok());
    ASSERT_EQ(streams_status3.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlockNums(), 2);
}

TEST_F(FIFOSchedulerTest, testIncrKVCacheFallBackReleaseAllBlocks) {
    KVCacheParam                  param = {1, 5, 1, 4, 2, rtp_llm::DataType::TYPE_FP16};
    CacheConfig                   cache_config(param);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 4);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;
    GptInitParameter config;
    config.max_seq_len_             = 8192;
    config.max_generate_batch_size_ = 100;
    config.max_batch_tokens_size_   = 8192;
    FIFOScheduler scheduler(config, cache_manager);
    scheduler.enable_partial_fallback_    = false;
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->request_id                    = 1;
    query1->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query1->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream1 = make_shared<NormalGenerateStream>(query1, config, resource_context, nullptr);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->request_id                    = 2;
    query2->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query2->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream2 = make_shared<NormalGenerateStream>(query2, config, resource_context, nullptr);

    ASSERT_TRUE(scheduler.enqueue(stream1).ok());
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
    ASSERT_FALSE(stream1->stopped());
    ASSERT_FALSE(stream2->stopped());
    ASSERT_EQ(stream1->stopReason(), "");
    ASSERT_EQ(stream2->stopReason(), "");
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(cache_manager->freeBlockNums(), 0);

    stream1->setSeqLength(stream1->seqLength() + 1);
    stream2->setSeqLength(stream2->seqLength() + 1);

    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 1);
    ASSERT_FALSE(stream1->stopped());
    ASSERT_FALSE(stream2->stopped());
    ASSERT_EQ(stream1->stopReason(), "");
    ASSERT_EQ(stream2->stopReason(), "");
    // stream2 pause了，release了所有的block
    ASSERT_TRUE(stream2->paused());
    ASSERT_EQ(stream2->maxBlockSize(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlockNums(), 1);

    stream1->setFinishedWithoutLock();
    auto streams_status3 = scheduler.schedule();
    ASSERT_TRUE(streams_status3.ok());
    ASSERT_EQ(streams_status3.value().size(), 1);
    ASSERT_TRUE(stream1->finished());
    ASSERT_FALSE(stream2->stopped());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlockNums(), 1);
}

TEST_F(FIFOSchedulerTest, testIncrKVCacheFallBackReleasePartBlocks) {
    KVCacheParam                  param = {1, 6, 1, 4, 2, rtp_llm::DataType::TYPE_FP16};
    CacheConfig                   cache_config(param);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 5);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    GptInitParameter config;
    config.max_seq_len_             = 8192;
    config.max_generate_batch_size_ = 100;
    config.max_batch_tokens_size_   = 8192;
    FIFOScheduler scheduler(config, cache_manager);
    scheduler.enable_partial_fallback_   = true;
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream1   = make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
    shared_ptr<GenerateStream> stream2   = make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
    ASSERT_FALSE(stream1->stopped());
    ASSERT_FALSE(stream2->stopped());
    ASSERT_EQ(stream1->stopReason(), "");
    ASSERT_EQ(stream2->stopReason(), "");
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
    ASSERT_EQ(cache_manager->freeBlockNums(), 1);

    stream1->setSeqLength(5);
    stream2->setSeqLength(5);

    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 1);
    ASSERT_FALSE(stream1->stopped());
    ASSERT_FALSE(stream2->stopped());
    ASSERT_EQ(stream1->stopReason(), "");
    ASSERT_EQ(stream2->stopReason(), "");
    // stream2 pause，并且进入waiting queue，release了部分block
    ASSERT_TRUE(stream2->paused());
    ASSERT_EQ(stream2->maxBlockSize(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlockNums(), 1);

    // stream1 需要5个block
    stream1->setSeqLength(9);

    auto streams_status3 = scheduler.schedule();
    ASSERT_TRUE(streams_status3.ok());
    ASSERT_EQ(streams_status3.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    // stream2继续回退block
    ASSERT_EQ(stream2->maxBlockSize(), 0);

    stream1->setFinishedWithoutLock();
    auto streams_status4 = scheduler.schedule();
    ASSERT_TRUE(streams_status4.ok());
    ASSERT_EQ(streams_status4.value().size(), 1);
    ASSERT_TRUE(stream1->finished());
    ASSERT_FALSE(stream2->stopped());
    // stream2开始运行
    ASSERT_EQ(stream2->maxBlockSize(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlockNums(), 2);
}

TEST_F(FIFOSchedulerTest, testReuseCache) {
    KVCacheParam                  param = {1, 11, 1, 4, 2, rtp_llm::DataType::TYPE_FP16};
    CacheConfig                   cache_config(param);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 10);
    ResourceContext  resource_context = {cache_manager, nullptr, nullptr, true};
    GptInitParameter config;
    config.max_seq_len_             = 8192;
    config.max_generate_batch_size_ = 100;
    config.max_batch_tokens_size_   = 8192;
    FIFOScheduler scheduler(config, cache_manager);

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = createBuffer<int32_t>({5}, {1, 2, 3, 4, 5}, AllocationType::HOST);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream1   = make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());

    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(cache_manager->freeBlockNums(), 7);

    stream1->setFinishedWithoutLock();
    auto streams_status2 = scheduler.schedule();

    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlockNums(), 8);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
    query2->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream2 = make_shared<NormalGenerateStream>(query2, config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    auto streams_status3 = scheduler.schedule();
    ASSERT_TRUE(streams_status3.ok());
    ASSERT_EQ(cache_manager->freeBlockNums(), 6);

    stream2->setFinishedWithoutLock();
    auto streams_status4 = scheduler.schedule();
    ASSERT_TRUE(streams_status4.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlockNums(), 7);
}

TEST_F(FIFOSchedulerTest, testMaxContextBatchSize) {
    KVCacheParam                  param = {1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16};
    CacheConfig                   cache_config(param);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 20);
    ResourceContext  resource_context = {cache_manager, nullptr, nullptr, true};
    GptInitParameter config;
    config.max_seq_len_            = 100;
    config.max_context_batch_size_ = 1;
    config.max_batch_tokens_size_  = 100;
    FIFOScheduler scheduler(config, cache_manager);

    {
        // test normalcase
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = createBuffer<int32_t>({5}, {1, 2, 3, 4, 5}, AllocationType::HOST);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream1 =
            make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream1).ok());

        auto streams_status = scheduler.schedule();
        ASSERT_TRUE(streams_status.ok());

        stream1->setFinishedWithoutLock();
        auto streams_status2 = scheduler.schedule();

        ASSERT_TRUE(streams_status2.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlockNums(), 20);
    }

    {
        // test normal case with tile num
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = createBuffer<int32_t>({5}, {1, 2, 3, 4, 5}, AllocationType::HOST);
        query->generate_config               = make_shared<GenerateConfig>();
        query->generate_config->num_beams    = 2;
        shared_ptr<GenerateStream> stream1 =
            make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream1).ok());

        auto streams_status = scheduler.schedule();
        ASSERT_TRUE(streams_status.ok());

        stream1->setFinishedWithoutLock();
        auto streams_status2 = scheduler.schedule();

        ASSERT_TRUE(streams_status2.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlockNums(), 20);
    }

    {
        // test abnormal case with tile num
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
        query2->input_ids                     = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
        query2->generate_config               = make_shared<GenerateConfig>();
        query2->generate_config->num_return_sequences = 20;
        shared_ptr<GenerateStream> stream2 =
            make_shared<NormalGenerateStream>(query2, config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream2).ok());

        auto streams_status3 = scheduler.schedule();
        ASSERT_TRUE(streams_status3.ok());
        ASSERT_EQ(cache_manager->freeBlockNums(), 20);
        ASSERT_EQ(stream2->stopReason(), "input len [7] * batch size [20] > max_batch_tokens_size [100]");

        stream2->setFinishedWithoutLock();
        auto streams_status4 = scheduler.schedule();
        ASSERT_TRUE(streams_status4.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlockNums(), 20);
    }
}

TEST_F(FIFOSchedulerTest, testBatchEnqueue) {
    KVCacheParam                  param = {1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16};
    CacheConfig                   cache_config(param);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 3);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GptInitParameter config;
    config.max_seq_len_             = 8192;
    config.max_generate_batch_size_ = 100;
    config.max_batch_tokens_size_   = 8192;
    FIFOScheduler             scheduler(config, cache_manager);
    vector<GenerateStreamPtr> streams;
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream = make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
        streams.push_back(stream);
    }
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream = make_shared<NormalGenerateStream>(query, config, resource_context, nullptr);
        streams.push_back(stream);
    }
    ASSERT_TRUE(scheduler.batchEnqueue(streams).ok());
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
    ASSERT_EQ(cache_manager->freeBlockNums(), 1);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

}  // namespace rtp_llm
