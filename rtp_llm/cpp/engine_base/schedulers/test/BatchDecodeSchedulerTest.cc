#include <cstdlib>
#include <memory>
#include <string>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {
namespace {
class ScopedPerfLocalEnv {
public:
    explicit ScopedPerfLocalEnv(const char* value) {
        const auto* old = std::getenv("RTP_LLM_PERF_LOCAL_REQUESTS");
        present_ = old != nullptr;
        if (old) {
            previous_ = old;
        }
        setenv("RTP_LLM_PERF_LOCAL_REQUESTS", value, 1);
    }
    ~ScopedPerfLocalEnv() {
        if (present_) {
            setenv("RTP_LLM_PERF_LOCAL_REQUESTS", previous_.c_str(), 1);
        } else {
            unsetenv("RTP_LLM_PERF_LOCAL_REQUESTS");
        }
    }
private:
    bool present_;
    std::string previous_;
};
}  // namespace

class BatchDecodeSchedulerTest: public DeviceTestBase {};

TEST_F(BatchDecodeSchedulerTest, StartupWarmupModePreservesContextAndAllowsOverride) {
    ScopedPerfLocalEnv env("0");
    for (const int warmup_type : {0, 1}) {
        for (const bool switch_mode : {false, true}) {
            SCOPED_TRACE(warmup_type);
            SCOPED_TRACE(switch_mode);
            auto config = makeMhaCacheConfig(1, 32, 1, 4, 8, DataType::TYPE_FP16);
            auto cache = std::make_shared<KVCacheManager>(config);
            ASSERT_TRUE(cache->init());
            ResourceContext resources;
            resources.cache_manager = cache;
            resources.role_type = RoleType::PREFILL;
            ModelConfig model;
            model.max_seq_len = 8192;
            RuntimeConfig runtime;
            runtime.use_batch_decode_scheduler = true;
            runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
            runtime.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type = warmup_type;
            BatchDecodeScheduler scheduler(runtime, cache, nullptr);
            const bool expected_prefill = (warmup_type != 0) != switch_mode;
            if (switch_mode) {
                scheduler.updateSchedulerInfo(expected_prefill ? "{\"batch_size\":1,\"mode\":\"prefill\"}" :
                                                                 "{\"batch_size\":1,\"mode\":\"decode\"}");
            }
            auto input = std::make_shared<GenerateInput>();
            input->input_ids = torch::zeros({2}, torch::kInt32);
            input->generate_config = std::make_shared<GenerateConfig>();
            input->generate_config->max_new_tokens = 1;
            auto stream = std::make_shared<NormalGenerateStream>(input, model, runtime, resources, nullptr);
            ASSERT_TRUE(scheduler.enqueue(stream).ok());

            const auto result = scheduler.schedule();

            ASSERT_TRUE(result.ok());
            ASSERT_EQ(result->size(), 1);
            EXPECT_EQ(stream->isContextStream(), expected_prefill);
            ASSERT_TRUE(scheduler.stop().ok());
        }
    }
}

TEST_F(BatchDecodeSchedulerTest, PerfLocalDecodeAllocatesAndReleasesRealBlocks) {
    ScopedPerfLocalEnv env("1");
    auto config = makeMhaCacheConfig(1, 32, 1, 4, 8, DataType::TYPE_FP16);
    auto cache = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(cache->init());
    for (auto& buffer : cache->gpuCacheTensors()) {
        buffer.fill_(17);
    }
    const auto free_before = cache->freeBlocksNum();
    ResourceContext resources;
    resources.cache_manager = cache;
    resources.role_type = RoleType::DECODE;
    ModelConfig model;
    model.max_seq_len = 8192;
    RuntimeConfig runtime;
    runtime.use_batch_decode_scheduler = true;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
    BatchDecodeScheduler scheduler(runtime, cache, nullptr);
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::zeros({12}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 30;
    auto stream = std::make_shared<NormalGenerateStream>(input, model, runtime, resources, nullptr);
    stream->setReserveStep(4);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(stream->isContextStream());
    EXPECT_TRUE(stream->isPerfTest());
    EXPECT_FALSE(stream->streamCacheResource().fake_inited_);
    EXPECT_LT(cache->freeBlocksNum(), free_before);
    for (const auto& buffer : cache->gpuCacheTensors()) {
        EXPECT_EQ(buffer.count_nonzero().item<int64_t>(), 0);
    }
    // Follow-up running-state allocation must not fail as fakeInitKVBlock does.
    ASSERT_TRUE(scheduler.schedule().ok());
    EXPECT_FALSE(stream->hasError());
    ASSERT_TRUE(scheduler.stop().ok());
    EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(cache->freeBlocksNum(), free_before);
}

TEST_F(BatchDecodeSchedulerTest, PerfGateOffAndOtherRolesKeepCacheContents) {
    for (const auto role : {RoleType::DECODE, RoleType::PDFUSION}) {
        ScopedPerfLocalEnv env(role == RoleType::DECODE ? "0" : "1");
        auto config = makeMhaCacheConfig(1, 32, 1, 4, 8, DataType::TYPE_FP16);
        auto cache = std::make_shared<KVCacheManager>(config);
        ASSERT_TRUE(cache->init());
        for (auto& buffer : cache->gpuCacheTensors()) {
            buffer.fill_(17);
        }
        ResourceContext resources;
        resources.cache_manager = cache;
        resources.role_type = role;
        ModelConfig model;
        model.max_seq_len = 8192;
        RuntimeConfig runtime;
        runtime.use_batch_decode_scheduler = true;
        runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
        BatchDecodeScheduler scheduler(runtime, cache, nullptr);
        auto input = std::make_shared<GenerateInput>();
        input->input_ids = torch::zeros({12}, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        auto stream = std::make_shared<NormalGenerateStream>(input, model, runtime, resources, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        ASSERT_TRUE(scheduler.schedule().ok());
        EXPECT_FALSE(scheduler.perf_cache_initialized_);
        for (const auto& buffer : cache->gpuCacheTensors()) {
            EXPECT_EQ(buffer.flatten()[0].item<int>(), 17);
        }
        ASSERT_TRUE(scheduler.stop().ok());
    }
}

TEST_F(BatchDecodeSchedulerTest, PerfLocalDecodeAllocationFailureTerminates) {
    ScopedPerfLocalEnv env("1");
    auto config = makeMhaCacheConfig(1, 2, 1, 4, 8, DataType::TYPE_FP16);
    auto cache = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(cache->init());
    ResourceContext resources;
    resources.cache_manager = cache;
    resources.role_type = RoleType::DECODE;
    ModelConfig model;
    model.max_seq_len = 8192;
    RuntimeConfig runtime;
    runtime.use_batch_decode_scheduler = true;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
    BatchDecodeScheduler scheduler(runtime, cache, nullptr);
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::zeros({64}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    auto stream = std::make_shared<NormalGenerateStream>(input, model, runtime, resources, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    const auto result = scheduler.schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
    ASSERT_TRUE(scheduler.stop().ok());
}
}  // namespace rtp_llm
