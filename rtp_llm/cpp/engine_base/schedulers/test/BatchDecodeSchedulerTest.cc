#include <chrono>
#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"
#include "autil/TimeUtility.h"

// Access private queues (waiting_streams_/running_streams_/batch_size_) directly so the tests can
// assert on scheduler internals without adding production-only accessors. BUILD also passes
// -fno-access-control, but the defines keep the intent explicit and the file self-contained.
#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class BatchDecodeSchedulerTest: public DeviceTestBase {
public:
    BatchDecodeSchedulerTest() {}

protected:
    GenerateStreamPtr makeStream(const ResourceContext& resource_context,
                                 const ModelConfig&     model_config,
                                 const RuntimeConfig&   runtime_config,
                                 ReturnAllProbsMode     mode = ReturnAllProbsMode::NONE) {
        auto query             = make_shared<GenerateInput>();
        query->input_ids       = torch::tensor({1}, torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        query->generate_config->return_all_probs = mode;
        return make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    }
};

// Regression: batch_size_ > 1 with only a single waiting stream.
// Before the flush-predicate fix, a non-empty waiting queue made the wait predicate return true
// immediately, so the partial-flush branch (gated on the flush timeout) was never taken and the
// lone stream sat in the waiting queue forever. It must now be flushed as a partial batch.
TEST_F(BatchDecodeSchedulerTest, testSingleStreamPartialFlush) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 4;

    BatchDecodeScheduler scheduler(runtime_config, cache_manager, nullptr);
    ASSERT_EQ(scheduler.batch_size_, 4u);

    auto stream = makeStream(resource_context, model_config, runtime_config);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    ASSERT_EQ(scheduler.waiting_streams_.size(), 1u);

    auto status = scheduler.schedule();
    ASSERT_TRUE(status.ok());
    // The single stream is flushed as a partial batch instead of waiting forever for batch_size_.
    ASSERT_EQ(status.value().size(), 1u);
    ASSERT_EQ(scheduler.running_streams_.size(), 1u);
    ASSERT_EQ(scheduler.waiting_streams_.size(), 0u);
}

// A full batch must be scheduled without paying the partial-flush timeout.
TEST_F(BatchDecodeSchedulerTest, testFullBatchSchedulesWithoutFlushWait) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 8, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 2;

    BatchDecodeScheduler scheduler(runtime_config, cache_manager, nullptr);
    auto                 s1 = makeStream(resource_context, model_config, runtime_config);
    auto                 s2 = makeStream(resource_context, model_config, runtime_config);
    ASSERT_TRUE(scheduler.enqueue(s1).ok());
    ASSERT_TRUE(scheduler.enqueue(s2).ok());

    auto status = scheduler.schedule();
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(status.value().size(), 2u);
    ASSERT_EQ(scheduler.running_streams_.size(), 2u);
    ASSERT_EQ(scheduler.waiting_streams_.size(), 0u);
}

// Mixed ReturnAllProbsMode groups below batch_size_ must still make progress: one compatible mode
// group is flushed on the timeout and the other mode stays queued (not lost) for the next round,
// instead of both stranding while the batch never fills.
TEST_F(BatchDecodeSchedulerTest, testMixedReturnAllProbsPartialFlush) {
    CacheConfig cache_config  = makeMhaCacheConfig(1, 8, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    auto        cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 8;

    BatchDecodeScheduler scheduler(runtime_config, cache_manager, nullptr);
    auto s_default  = makeStream(resource_context, model_config, runtime_config, ReturnAllProbsMode::DEFAULT);
    auto s_original = makeStream(resource_context, model_config, runtime_config, ReturnAllProbsMode::ORIGINAL);
    ASSERT_TRUE(scheduler.enqueue(s_default).ok());
    ASSERT_TRUE(scheduler.enqueue(s_original).ok());
    ASSERT_EQ(scheduler.waiting_streams_.size(), 2u);

    auto status = scheduler.schedule();
    ASSERT_TRUE(status.ok());
    // batch_size_(8) is never reached, but the flush timer fires and one mode group runs.
    ASSERT_GE(scheduler.running_streams_.size(), 1u);
    // Nothing is lost: scheduled + still-waiting accounts for both enqueued streams.
    ASSERT_EQ(scheduler.running_streams_.size() + scheduler.waiting_streams_.size(), 2u);
}

}  // namespace rtp_llm
