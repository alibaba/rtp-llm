#include <memory>
#include <future>
#include <thread>
#include <vector>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionCoordinatedScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {

struct GdnResources {
    ModelConfig                     model;
    RuntimeConfig                   runtime;
    ResourceContext                 resource;
    std::shared_ptr<KVCacheManager> cache;
    PDSepConfig                     pd;

    explicit GdnResources(int tokens, int sequences = 8, bool gdn = true) {
        cache =
            std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(1, 8192, 16, DataType::TYPE_FP16, 1, 4));
        if (!cache->init()) {
            throw std::runtime_error("test cache initialization failed");
        }
        resource.cache_manager             = cache;
        resource.role_type                 = RoleType::PDFUSION;
        pd.role_type                       = RoleType::PDFUSION;
        model.max_seq_len                  = 131072;
        model.attn_config.tokens_per_block = 16;
        model.hidden_size                  = 4096;
        if (gdn) {
            model.linear_attention_config.linear_conv_kernel_dim = 4;
            model.linear_attention_config.linear_num_key_heads   = 16;
            model.linear_attention_config.linear_num_value_heads = 64;
            model.linear_attention_config.linear_key_head_dim    = 128;
            model.linear_attention_config.linear_value_head_dim  = 128;
        }
        runtime.max_generate_batch_size                       = sequences;
        runtime.fifo_scheduler_config.max_batch_tokens_size   = tokens;
        runtime.fifo_scheduler_config.max_context_batch_size  = 1;
        runtime.fifo_scheduler_config.decode_prefill_ratio    = "0";
        runtime.fifo_scheduler_config.pdfusion_schedule_trace = true;
        runtime.fifo_scheduler_config.pdfusion_coord_group_id = "gdn-memory-test";
    }

    GenerateStreamPtr stream(int length, int width = 1) {
        auto input                                   = std::make_shared<GenerateInput>();
        input->input_ids                             = torch::ones({length}, torch::kInt32);
        input->generate_config                       = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens       = 1;
        input->generate_config->num_return_sequences = width;
        return std::make_shared<NormalGenerateStream>(input, model, runtime, resource, nullptr);
    }

    std::unique_ptr<PDFusionCoordinatedScheduler> coordinated() {
        PDFusionCadenceConfig cadence;
        cadence.adaptive = true;
        return std::make_unique<PDFusionCoordinatedScheduler>(
            runtime, model, pd, ParallelismConfig{}, ModelSpecificConfig{}, cache, nullptr, 0, cadence);
    }

    std::unique_ptr<PDFusionRatioScheduler> ratio() {
        return std::make_unique<PDFusionRatioScheduler>(
            runtime, model, pd, ParallelismConfig{}, ModelSpecificConfig{}, cache);
    }
};

class PendingGdnLoad: public AsyncContext {
public:
    void waitDone() override {
        throw std::logic_error("must not wait under scheduler lock");
    }
    bool done() const override {
        return false;
    }
    bool success() const override {
        return false;
    }
};

}  // namespace

class GdnPrefillSchedulerTest: public DeviceTestBase {};

namespace {
void markPendingMM(const GenerateStreamPtr& stream,
                   const std::shared_ptr<GenerateInput::PrepareMultimodalBatch>& prepare) {
    stream->generateInput()->multimodal_token_layout = MultimodalTokenLayout{{{0, stream->inputLength()}}, true};
    stream->generateInput()->prepare_multimodal_batch = prepare;
}

std::vector<ErrorInfo> fillMM(std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    for (auto& input : inputs) {
        input->input_ids = torch::full_like(input->input_ids, 7);
        input->multimodal_features = std::vector<torch::Tensor>{torch::zeros({input->inputLength(), 1})};
    }
    return std::vector<ErrorInfo>(inputs.size(), ErrorInfo::OkStatus());
}
}  // namespace

TEST_F(GdnPrefillSchedulerTest, LateMMCompletionIsPublishedOnlyOnSchedulerThread) {
    GdnResources f(32);
    auto stream = f.stream(3);
    markPendingMM(stream, std::make_shared<GenerateInput::PrepareMultimodalBatch>(fillMM));
    auto completion = GenerateStream::prepareLateMultimodalBatchDeferred({stream});
    EXPECT_TRUE(stream->hasPendingMultimodal());
    EXPECT_FALSE(stream->generateInput()->multimodal_features.has_value());
    EXPECT_EQ(stream->generateInput()->input_ids[0].item<int>(), 1);
    completion();
    EXPECT_FALSE(stream->hasPendingMultimodal());
    EXPECT_EQ(stream->generateInput()->input_ids[0].item<int>(), 7);
}

TEST_F(GdnPrefillSchedulerTest, RatioContinuesDecodeWhileLateMMWaits) {
    GdnResources f(32);
    auto scheduler = f.ratio();
    auto decode = f.stream(2), pending = f.stream(3), next = f.stream(3);
    auto batch = std::make_shared<GenerateInput::PrepareMultimodalBatch>(fillMM);
    std::promise<void> release;
    auto gate = release.get_future().share();
    auto slow = std::make_shared<GenerateInput::PrepareMultimodalBatch>([gate](auto& inputs) {
        gate.wait_for(std::chrono::seconds(5));
        return fillMM(inputs);
    });
    markPendingMM(pending, slow);
    markPendingMM(next, batch);
    ASSERT_TRUE(scheduler->enqueue(decode).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    decode->setIsContextStream(false);
    ASSERT_TRUE(scheduler->enqueue(pending).ok());
    ASSERT_TRUE(scheduler->enqueue(next).ok());
    const auto started = std::chrono::steady_clock::now();
    auto result = scheduler->schedule();
    const auto elapsed = std::chrono::steady_clock::now() - started;
    release.set_value();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), decode);
    EXPECT_LT(elapsed, std::chrono::seconds(1));
    EXPECT_EQ(pending->curBlocksNum(), 0);
    EXPECT_EQ(next->curBlocksNum(), 0);
    EXPECT_TRUE(next->hasPendingMultimodal());
    for (int attempt = 0; attempt < 1000 && pending->hasPendingMultimodal(); ++attempt) {
        ASSERT_TRUE(scheduler->schedule().ok());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(pending->hasPendingMultimodal());
    EXPECT_TRUE(next->hasPendingMultimodal());
}

TEST_F(GdnPrefillSchedulerTest, CadencePreparesWithoutWaitingAndIsolatesFailedRequests) {
    for (bool adaptive : {false, true}) {
        GdnResources f(32);
        f.runtime.fifo_scheduler_config.decode_prefill_ratio = "7";
        PDFusionCadenceConfig cadence;
        cadence.adaptive = adaptive;
        PDFusionCoordinatedScheduler scheduler(
            f.runtime, f.model, f.pd, ParallelismConfig{}, ModelSpecificConfig{}, f.cache, nullptr, 0, cadence);
        auto failed = f.stream(3), good = f.stream(3);
        std::promise<void> release;
        auto gate = release.get_future().share();
        markPendingMM(failed, std::make_shared<GenerateInput::PrepareMultimodalBatch>([gate](auto& inputs) {
            gate.wait_for(std::chrono::seconds(5));
            return std::vector<ErrorInfo>(inputs.size(), ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "media failed"));
        }));
        markPendingMM(good, std::make_shared<GenerateInput::PrepareMultimodalBatch>(fillMM));
        ASSERT_TRUE(scheduler.enqueue(failed).ok());
        ASSERT_TRUE(scheduler.enqueue(good).ok());
        const auto started = std::chrono::steady_clock::now();
        auto state = scheduler.prepareLocal(1);
        const auto elapsed = std::chrono::steady_clock::now() - started;
        release.set_value();
        EXPECT_LT(elapsed, std::chrono::seconds(1));
        EXPECT_EQ(state.ready_prefill, 0);
        EXPECT_EQ(failed->curBlocksNum(), 0);
        EXPECT_EQ(good->curBlocksNum(), 0);
        scheduler.commitLocal(1, PDFusionPlan::DECODE);
        scheduler.completeModelStep();
        for (int epoch = 2; epoch < 1000; ++epoch) {
            state = scheduler.prepareLocal(epoch);
            auto batch = scheduler.commitLocal(epoch, state.ready_prefill ? PDFusionPlan::PREFILL : PDFusionPlan::DECODE);
            scheduler.completeModelStep();
            if (state.ready_prefill) {
                ASSERT_EQ(batch.size(), 1);
                EXPECT_EQ(batch.front(), good);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_TRUE(failed->hasError());
        EXPECT_FALSE(good->hasError());
        EXPECT_FALSE(good->hasPendingMultimodal());
        EXPECT_GT(good->curBlocksNum(), 0);
    }
}

TEST_F(GdnPrefillSchedulerTest, Splits65245TokenPairWithoutTruncation) {
    GdnResources f(32768);
    auto         scheduler = f.coordinated();
    auto         first = f.stream(32600), second = f.stream(32645);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    auto state = scheduler->prepareLocal(1);
    EXPECT_EQ(state.ready_prefill, 1);
    EXPECT_EQ(state.input_tokens, 32600);
    EXPECT_EQ(state.waiting, 1);
    EXPECT_FALSE(second->hasError());
    EXPECT_EQ(second->curBlocksNum(), 0);
    ASSERT_EQ(scheduler->commitLocal(1, PDFusionPlan::PREFILL).front(), first);
    first->reportEvent(StreamEvents::GenerateDone);
    scheduler->completeModelStep();
    state = scheduler->prepareLocal(2);
    EXPECT_EQ(state.ready_prefill, 1);
    EXPECT_EQ(state.input_tokens, 32645);
    EXPECT_EQ(first->inputLength(), 32600);
    EXPECT_EQ(second->inputLength(), 32645);
    ASSERT_EQ(scheduler->commitLocal(2, PDFusionPlan::PREFILL).front(), second);
    scheduler->completeModelStep();
}

TEST_F(GdnPrefillSchedulerTest, OversizedAndBoundaryRequestsFailWithoutBlockingGroup) {
    GdnResources f(8);
    auto         scheduler = f.coordinated();
    auto         boundary = f.stream(8), oversized = f.stream(9), short_request = f.stream(2);
    auto         result = scheduler->enqueueGroup({boundary, oversized, short_request});
    ASSERT_EQ(result.first.size(), 3);
    auto state = scheduler->prepareLocal(1);
    EXPECT_EQ(state.ready_prefill, 1);
    EXPECT_EQ(state.waiting, 0);
    EXPECT_TRUE(boundary->hasError());
    EXPECT_TRUE(oversized->hasError());
    EXPECT_NE(oversized->stopReason().find("GDN prefill"), std::string::npos);
    EXPECT_EQ(oversized->inputLength(), 9);
    EXPECT_EQ(boundary->curBlocksNum(), 0);
    EXPECT_EQ(oversized->curBlocksNum(), 0);
    ASSERT_EQ(scheduler->commitLocal(1, PDFusionPlan::PREFILL).front(), short_request);
    scheduler->completeModelStep();
}

TEST_F(GdnPrefillSchedulerTest, ExpandedSequenceCountCannotExceedPlannedStateMemory) {
    GdnResources f(100, 2);
    auto         scheduler = f.coordinated();
    auto         oversized = f.stream(2, 3), normal = f.stream(2);
    ASSERT_EQ(oversized->currentBatchSize(), 3);
    ASSERT_TRUE(scheduler->enqueue(oversized).ok());
    ASSERT_TRUE(scheduler->enqueue(normal).ok());
    auto state = scheduler->prepareLocal(1);
    EXPECT_EQ(state.ready_prefill, 1);
    EXPECT_TRUE(oversized->hasError());
    EXPECT_FALSE(normal->hasError());
    ASSERT_EQ(scheduler->commitLocal(1, PDFusionPlan::PREFILL).front(), normal);
    scheduler->completeModelStep();
}

TEST_F(GdnPrefillSchedulerTest, ReadyAndLoadingReservationsAreChargedOnce) {
    GdnResources f(7);
    auto         scheduler = f.coordinated();
    auto         loading = f.stream(3), ready = f.stream(3), excess = f.stream(2);
    ASSERT_TRUE(loading->initKVBlock().ok());
    loading->reportEvent(StreamEvents::CanRun);
    loading->reportEvent(StreamEvents::LoadInitiated);
    loading->generate_status_->status                  = StreamState::LOADING_CACHE;
    loading->streamCacheResource().load_cache_context_ = std::make_shared<PendingGdnLoad>();
    scheduler->loading_cache_streams_.push_back(loading);
    ASSERT_TRUE(scheduler->enqueue(ready).ok());
    ASSERT_TRUE(scheduler->enqueue(excess).ok());
    auto state = scheduler->prepareLocal(1);
    EXPECT_EQ(state.ready_prefill, 1);
    EXPECT_EQ(state.loading, 1);
    EXPECT_EQ(state.waiting, 1);
    EXPECT_FALSE(excess->hasError());
    scheduler->commitLocal(1, PDFusionPlan::DECODE);
    scheduler->completeModelStep();
    loading->streamCacheResource().load_cache_context_.reset();
    state = scheduler->prepareLocal(2);
    EXPECT_EQ(state.ready_prefill, 2);
    EXPECT_EQ(state.loading, 0);
    EXPECT_EQ(state.waiting, 1);
    EXPECT_EQ(excess->curBlocksNum(), 0);
    EXPECT_EQ(scheduler->commitLocal(2, PDFusionPlan::PREFILL).size(), 2);
    scheduler->completeModelStep();
}

TEST_F(GdnPrefillSchedulerTest, RatioHasNoStandaloneBypassAndSkipsUnfitRequest) {
    GdnResources f(8);
    auto         scheduler = f.ratio();
    auto         oversized = f.stream(8), valid = f.stream(7);
    ASSERT_TRUE(scheduler->enqueue(oversized).ok());
    ASSERT_TRUE(scheduler->enqueue(valid).ok());
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), valid);
    EXPECT_TRUE(oversized->hasError());
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
}

TEST_F(GdnPrefillSchedulerTest, NonGdnKeepsLegacySingleRequestBehavior) {
    GdnResources f(1, 8, false);
    auto         scheduler = f.coordinated();
    auto         stream    = f.stream(2);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    EXPECT_EQ(scheduler->prepareLocal(1).ready_prefill, 1);
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(scheduler->commitLocal(1, PDFusionPlan::PREFILL).size(), 1);
    scheduler->completeModelStep();
}

TEST_F(GdnPrefillSchedulerTest, PlannerUsesActivationDtypeAndConfiguredTokens) {
    GdnResources f(65536, 64);
    EXPECT_TRUE(MemoryEvaluationHelper::hasGdnPrefill(f.model));
    const auto full = MemoryEvaluationHelper::getGdnPrefillMemorySize(f.runtime, ParallelismConfig{}, f.model);
    f.model.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    EXPECT_EQ(MemoryEvaluationHelper::getGdnPrefillMemorySize(f.runtime, ParallelismConfig{}, f.model), full);
    f.runtime.fifo_scheduler_config.max_batch_tokens_size = 32768;
    EXPECT_LT(MemoryEvaluationHelper::getGdnPrefillMemorySize(f.runtime, ParallelismConfig{}, f.model), full);
    f.model.linear_attention_config = {};
    EXPECT_EQ(MemoryEvaluationHelper::getGdnPrefillMemorySize(f.runtime, ParallelismConfig{}, f.model), 0);
}

}  // namespace rtp_llm
