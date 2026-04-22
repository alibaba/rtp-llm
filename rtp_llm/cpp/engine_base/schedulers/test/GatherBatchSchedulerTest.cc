#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/engine_base/schedulers/GatherBatchScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GatherBatchSchedulerTest: public DeviceTestBase {
public:
    GatherBatchSchedulerTest() {}

protected:
    std::shared_ptr<GatherBatchScheduler> makeScheduler() {
        cache_config_  = makeMhaCacheConfig(1, 32, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        EXPECT_TRUE(cache_manager_->init());
        resource_context_.cache_manager = cache_manager_;

        model_config_.max_seq_len                                   = 8192;
        runtime_config_.max_generate_batch_size                     = 100;
        runtime_config_.fifo_scheduler_config.max_batch_tokens_size = 8192;

        return std::make_shared<GatherBatchScheduler>(runtime_config_,
                                                      model_config_,
                                                      pd_sep_config_,
                                                      parallelism_config_,
                                                      model_specific_config_,
                                                      cache_manager_,
                                                      /*metrics_reporter=*/nullptr);
    }

    std::shared_ptr<GenerateStream> makeStream() {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        return std::make_shared<NormalGenerateStream>(
            query, model_config_, runtime_config_, resource_context_, nullptr);
    }

    void setGatherBatchSize(GatherBatchScheduler& scheduler, int batch_size) {
        scheduler.updateSchedulerInfo(R"({"batch_size":)" + std::to_string(batch_size) + "}");
    }

    CacheConfig                     cache_config_;
    std::shared_ptr<KVCacheManager> cache_manager_;
    ResourceContext                 resource_context_;
    ModelConfig                     model_config_;
    RuntimeConfig                   runtime_config_;
    PDSepConfig                     pd_sep_config_;
    ParallelismConfig               parallelism_config_;
    ModelSpecificConfig             model_specific_config_;
};

TEST_F(GatherBatchSchedulerTest, testSingleStreamGather) {
    auto scheduler = makeScheduler();
    auto stream    = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    auto streams_status = scheduler->schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

TEST_F(GatherBatchSchedulerTest, testGatherBatchAccumulatesBeforeRunning) {
    auto scheduler = makeScheduler();
    setGatherBatchSize(*scheduler, 2);

    auto stream1 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());

    auto streams_status1 = scheduler->schedule();
    ASSERT_TRUE(streams_status1.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);

    auto stream2 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    auto streams_status2 = scheduler->schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 2);
}

TEST_F(GatherBatchSchedulerTest, testAllowsGatherWhileRunning) {
    auto scheduler = makeScheduler();

    auto stream1 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    auto streams_status1 = scheduler->schedule();
    ASSERT_TRUE(streams_status1.ok());
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);

    auto stream2 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    auto streams_status2 = scheduler->schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 2);
}

TEST_F(GatherBatchSchedulerTest, testBatchGatherAllAtOnce) {
    auto scheduler = makeScheduler();
    setGatherBatchSize(*scheduler, 3);

    auto stream1 = makeStream();
    auto stream2 = makeStream();
    auto stream3 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());
    ASSERT_TRUE(scheduler->enqueue(stream3).ok());

    auto streams_status = scheduler->schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 3);
}

}  // namespace rtp_llm
