#include <chrono>
#include <memory>
#include <thread>

#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "autil/TimeUtility.h"
#include "torch/all.h"

#define protected public
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

namespace {

std::shared_ptr<GenerateConfig> makeCoalesceTestGenerateConfig(int max_new_tokens = 1) {
    auto generate_config            = std::make_shared<GenerateConfig>();
    generate_config->max_new_tokens = max_new_tokens;
    return generate_config;
}

std::shared_ptr<GenerateStream> makeCoalesceStream(const std::vector<int>& ids,
                                                   const ModelConfig&      model_config,
                                                   const RuntimeConfig&    runtime_config,
                                                   const ResourceContext&  resource_context,
                                                   int                     max_new_tokens = 1) {
    auto query                             = std::make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor(ids, torch::kInt32);
    query->generate_config                 = makeCoalesceTestGenerateConfig(max_new_tokens);
    query->generate_config->num_return_sequences = 1;
    return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
}

struct CoalesceFixture {
    CacheConfig                     cache_config  = rtp_llm::test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/21, /*tokens_per_block=*/1, rtp_llm::DataType::TYPE_FP16, 1, 4);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ResourceContext                 resource_context;
    ModelConfig                     model_config;
    RuntimeConfig                   runtime_config;
    PDSepConfig                     pd_sep_config;
    ParallelismConfig               parallelism_config;
    ModelSpecificConfig             model_specific_config;
    std::unique_ptr<FIFOScheduler>  scheduler;

    CoalesceFixture(int64_t window_ms, int64_t target_streams) {
        EXPECT_TRUE(cache_manager->init());
        resource_context.cache_manager                     = cache_manager;
        model_config.max_seq_len                            = 8192;
        runtime_config.max_generate_batch_size              = 16;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
        runtime_config.fifo_scheduler_config.context_coalesce_window_ms       = window_ms;
        runtime_config.fifo_scheduler_config.context_coalesce_target_streams = target_streams;
        scheduler = std::make_unique<FIFOScheduler>(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    }

    long scheduleMs() {
        const auto t0    = std::chrono::steady_clock::now();
        auto       round = scheduler->schedule();
        EXPECT_TRUE(round.ok());
        last_round_size = round->size();
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    }

    size_t last_round_size = 0;
};

}  // namespace

TEST(FIFOSchedulerCoalesceTest, HoldsSoloRoundForConcurrentArrival) {
    CoalesceFixture fixture(/*window_ms=*/500, /*target_streams=*/2);
    auto            stream1 = makeCoalesceStream({1, 2}, fixture.model_config, fixture.runtime_config, fixture.resource_context);
    ASSERT_TRUE(fixture.scheduler->enqueue(stream1).ok());

    auto stream2  = makeCoalesceStream({3, 4}, fixture.model_config, fixture.runtime_config, fixture.resource_context);
    auto enqueue2 = [&fixture, stream2]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        ASSERT_TRUE(fixture.scheduler->enqueue(stream2).ok());
    };
    std::thread late_arrival(enqueue2);

    const long hold_ms = fixture.scheduleMs();
    late_arrival.join();

    EXPECT_EQ(fixture.last_round_size, 2) << "the held round must carry the concurrent arrival";
    EXPECT_GE(hold_ms, 50) << "the solo round must have been held for the arrival";
    EXPECT_LT(hold_ms, 1000);
    EXPECT_FALSE(stream1->hasError());
    EXPECT_FALSE(stream2->hasError());
}

TEST(FIFOSchedulerCoalesceTest, WindowExpiresSoloRound) {
    CoalesceFixture fixture(/*window_ms=*/200, /*target_streams=*/2);
    auto            stream1 = makeCoalesceStream({1, 2}, fixture.model_config, fixture.runtime_config, fixture.resource_context);
    ASSERT_TRUE(fixture.scheduler->enqueue(stream1).ok());

    const long hold_ms = fixture.scheduleMs();

    EXPECT_EQ(fixture.last_round_size, 1) << "no arrival: the window expires and the solo round dispatches";
    EXPECT_GE(hold_ms, 150) << "the round must have been held for (roughly) the full window";
    EXPECT_LT(hold_ms, 2000);
    EXPECT_FALSE(stream1->hasError());
}

TEST(FIFOSchedulerCoalesceTest, DisabledDispatchesImmediately) {
    CoalesceFixture fixture(/*window_ms=*/0, /*target_streams=*/2);
    auto            stream1 = makeCoalesceStream({1, 2}, fixture.model_config, fixture.runtime_config, fixture.resource_context);
    ASSERT_TRUE(fixture.scheduler->enqueue(stream1).ok());

    const long hold_ms = fixture.scheduleMs();

    EXPECT_EQ(fixture.last_round_size, 1);
    EXPECT_LT(hold_ms, 100) << "window=0 must keep the legacy immediate dispatch";
    EXPECT_FALSE(stream1->hasError());
}

TEST(FIFOSchedulerCoalesceTest, SkipsHoldWhenTargetAlreadyQueued) {
    CoalesceFixture fixture(/*window_ms=*/500, /*target_streams=*/2);
    auto            stream1 = makeCoalesceStream({1, 2}, fixture.model_config, fixture.runtime_config, fixture.resource_context);
    auto            stream2 = makeCoalesceStream({3, 4}, fixture.model_config, fixture.runtime_config, fixture.resource_context);
    ASSERT_TRUE(fixture.scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(fixture.scheduler->enqueue(stream2).ok());

    const long hold_ms = fixture.scheduleMs();

    EXPECT_EQ(fixture.last_round_size, 2) << "both queued streams join one round via the normal admission path";
    EXPECT_LT(hold_ms, 100) << "no hold is needed when the queue already reaches the target";
    EXPECT_FALSE(stream1->hasError());
    EXPECT_FALSE(stream2->hasError());
}

}  // namespace rtp_llm
