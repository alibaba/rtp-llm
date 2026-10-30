#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/test/mock/MockCoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/utils/math_utils.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

namespace rtp_llm {
namespace {

template<class Config, class = void>
struct ExecutionCapacitySetter {
    static void set(Config&, int64_t) {}
};

template<class Config>
struct ExecutionCapacitySetter<Config, std::void_t<decltype(std::declval<Config&>().max_execution_tokens)>> {
    static void set(Config& config, int64_t tokens) {
        config.max_execution_tokens = tokens;
    }
};

// The existing mock model only bootstraps a stopped TP1 engine. Scheduler
// selection, streams, cache allocation and input gathering below are real.
// This test exercises the workspace contract, not K3 kernels or TP collectives.
class SchedulerPhysicalCapacityIntegrationTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        CustomConfig bootstrap;
        bootstrap.max_context_batch_size = 1;
        bootstrap.tokens_per_block       = 4;
        engine_                          = createMockEngine(bootstrap);
        ASSERT_TRUE(engine_->stop().ok());
    }

    void TearDown() override {
        if (engine_) {
            engine_->scheduler_.reset();
        }
        engine_.reset();
        DeviceTestBase::TearDown();
    }

    void initializeScheduler(bool fixed_decode, int tp, int max_seq_len, int token_budget, int requests) {
        auto& parallelism   = engine_->parallelism_config;
        parallelism.tp_size = parallelism.ep_size = parallelism.ffn_tp_size = tp;
        parallelism.world_size = parallelism.local_world_size = tp;
        parallelism.dp_size = parallelism.pp_size      = 1;
        parallelism.role_type                          = RoleType::PDFUSION;
        parallelism.prefill_cp_config                  = PrefillCPConfig{};
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        ASSERT_FALSE(parallelism.prefill_cp_config.is_enabled());

        auto& model                = engine_->model_config_;
        model.max_seq_len          = max_seq_len;
        model.gen_num_per_cycle    = 1;
        model.attn_config.head_num = model.attn_config.kv_head_num = tp;
        model.attn_config.size_per_head                            = 16;
        model.attn_config.is_causal                                = true;
        model.attn_config.tokens_per_block = model.attn_config.kernel_tokens_per_block = 4;
        model.local_kv_cache_layout = LocalKVCacheLayoutPolicy::PAGE_RR;

        auto& runtime                      = engine_->runtime_config;
        runtime.use_batch_decode_scheduler = fixed_decode;
        runtime.max_generate_batch_size    = requests;
        runtime.warm_up = runtime.model_warm_up                                 = false;
        runtime.fifo_scheduler_config.max_context_batch_size                    = 1;
        runtime.fifo_scheduler_config.max_batch_tokens_size                     = token_budget;
        runtime.fifo_scheduler_config.prefill_chunk_size                        = 0;
        runtime.fifo_scheduler_config.pdfusion_scheduler_mode                   = fixed_decode ? "" : "ratio";
        runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = requests;
        engine_->pd_sep_config.role_type                                        = RoleType::PDFUSION;
        engine_->sp_config.type                                                 = SP_TYPE_NONE;
        engine_->sp_config.gen_num_per_cycle                                    = model.gen_num_per_cycle;

        // Match K3's non-CP startup contract: the larger Prefill/Decode extent
        // is rounded to TP. Do not enlarge the test budget for request fanout.
        size_t decode_rows = 0;
        for (const size_t width : {size_t{1}, size_t{2}}) {
            decode_rows = std::max(decode_rows, padSequenceParallelRequests(requests, width, tp) * width);
        }
        model.max_prefill_tokens = pad(std::max<size_t>({static_cast<size_t>(max_seq_len),
                                                         static_cast<size_t>(token_budget),
                                                         decode_rows,
                                                         static_cast<size_t>(tp)}),
                                       tp);
        // The same fixture compiles on the baseline and the fixed schema. A
        // declaring model publishes this exact allocated extent, never a larger
        // test-only budget. Older schemas have no execution declaration.
        ExecutionCapacitySetter<ModelConfig>::set(model, model.max_prefill_tokens);

        KVCacheConfig cache_options;
        cache_options.seq_size_per_block = cache_options.kernel_seq_size_per_block = 4;
        cache_options.enable_device_cache                                          = false;
        auto cache_config = CacheConfigCreator::createConfig(model, parallelism, cache_options);
        cache_config      = test::finalizeCacheConfig(std::move(cache_config), 128);
        auto manager      = std::make_shared<KVCacheManager>(cache_config, false, nullptr, cache_options, parallelism);
        ASSERT_TRUE(manager->init());
        ASSERT_TRUE(manager->cacheConfig().local_kv_cache_sharded);
        ASSERT_FALSE(manager->cacheConfig().linear_replay.enabled());
        engine_->resource_context_.cache_manager = std::move(manager);
        engine_->resource_context_.role_type     = RoleType::PDFUSION;
        engine_->resource_context_.reuse_cache   = false;

        // Select the scheduler through the production NormalEngine entry.
        engine_->initScheduler();
        if (fixed_decode) {
            ASSERT_NE(dynamic_cast<BatchDecodeScheduler*>(engine_->scheduler_.get()), nullptr);
            engine_->scheduler_->updateSchedulerInfo("{\"batch_size\":" + std::to_string(requests)
                                                     + ",\"mode\":\"decode\"}");
        } else {
            ASSERT_NE(dynamic_cast<PDFusionRatioScheduler*>(engine_->scheduler_.get()), nullptr);
        }
    }

    GenerateStreamPtr enqueueStream(int length, int fanout, int request_id, int max_new_tokens = 1) {
        auto input                                   = std::make_shared<GenerateInput>();
        input->request_id                            = request_id;
        input->begin_time_us                         = autil::TimeUtility::currentTimeInMicroSeconds();
        input->input_ids                             = torch::arange(1, length + 1, torch::kInt32);
        input->generate_config                       = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens       = max_new_tokens;
        input->generate_config->num_return_sequences = fanout;
        input->generate_config->top_k                = 1;
        auto stream                                  = engine_->makeStream(input);
        EXPECT_FALSE(stream->hasError()) << stream->statusInfo().ToString();
        EXPECT_FALSE(stream->isFakeStream());
        EXPECT_TRUE(stream->isContextStream());
        EXPECT_EQ(stream->currentBatchSize(), fanout);
        EXPECT_TRUE(engine_->scheduler_->enqueue(stream).ok());
        return stream;
    }

    // Delegate actual allocation/free/cache accounting to the real coordinator.
    // The only controlled boundary is publication of two allocator-load completions.
    std::vector<std::shared_ptr<LoadAsyncContext>>
    setupPendingCacheLoads(const std::shared_ptr<KVCacheManager>& cache_manager) {
        auto coordinator = std::make_shared<LoadContextCoordinator>(
            [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
        std::vector<std::shared_ptr<LoadAsyncContext>> contexts;
        for (int i = 0; i < 2; ++i) {
            auto context = coordinator->create({}, {}, /*matched_blocks=*/0);
            context->setSettlementReadyCallback([](const std::shared_ptr<LoadAsyncContext>&) {});
            EXPECT_TRUE(coordinator->registerContext(context));
            contexts.push_back(std::move(context));
        }
        auto cursor = std::make_shared<size_t>(0);

        const auto real_allocator = cache_manager->coordinator_manager_;
        auto       allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager->config_);
        // Unmocked accounting and failure classification need the delegated allocator's topology.
        allocator->group_block_pools_ = real_allocator->groupBlockPools();
        allocator->kv_cache_groups_   = real_allocator->cacheGroups();
        allocator->setReserveBlocksNum(real_allocator->reserveBlocksNum());
        EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
            .WillRepeatedly(testing::Invoke([real_allocator, contexts, cursor](const MallocInfo& info) {
                auto result = real_allocator->initMallocForCommonLen(info);
                if (result.success && *cursor < contexts.size()) {
                    result.async_context = contexts[(*cursor)++];
                }
                return result;
            }));
        ON_CALL(*allocator, incrMalloc(testing::_))
            .WillByDefault(
                testing::Invoke([real_allocator](const MallocInfo& info) { return real_allocator->incrMalloc(info); }));
        ON_CALL(*allocator, free(testing::_)).WillByDefault(testing::Invoke([real_allocator](const FreeInfo& info) {
            real_allocator->free(info);
        }));
        ON_CALL(*allocator, insertIntoCache(testing::_, testing::_))
            .WillByDefault(testing::Invoke([real_allocator](const InsertInfo& info, size_t& resident_prefix_length) {
                real_allocator->insertIntoCache(info, resident_prefix_length);
            }));
        ON_CALL(*allocator, availableBlocksNum()).WillByDefault(testing::Invoke([real_allocator] {
            return real_allocator->availableBlocksNum();
        }));
        ON_CALL(*allocator, totalBlocksNum()).WillByDefault(testing::Invoke([real_allocator] {
            return real_allocator->totalBlocksNum();
        }));
        ON_CALL(*allocator, maxAvailableTokensNum()).WillByDefault(testing::Invoke([real_allocator] {
            return real_allocator->maxAvailableTokensNum();
        }));
        ON_CALL(*allocator, seqSizePerBlock()).WillByDefault(testing::Invoke([real_allocator] {
            return real_allocator->seqSizePerBlock();
        }));
        ON_CALL(*allocator, getNeedBlocks(testing::_))
            .WillByDefault(testing::Invoke(
                [real_allocator](const MallocInfo& info) { return real_allocator->getNeedBlocks(info); }));
        ON_CALL(*allocator, singleBatchNeedBlocks(testing::_, testing::_, testing::_))
            .WillByDefault(testing::Invoke(
                [real_allocator](const BatchKVCacheResourcePtr& resource, int seq_len, int reserve_step) {
                    return real_allocator->singleBatchNeedBlocks(resource, seq_len, reserve_step);
                }));
        ON_CALL(*allocator, estimatePeakNeedBlocks(testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillByDefault(testing::Invoke([real_allocator](const KVCacheResource& resource,
                                                            int                    seq_len,
                                                            int                    remaining_tokens,
                                                            int                    reserve_step,
                                                            bool                   reuse_cache) {
                return real_allocator->estimatePeakNeedBlocks(
                    resource, seq_len, remaining_tokens, reserve_step, reuse_cache);
            }));
        ON_CALL(
            *allocator,
            estimateInitialBatchPeakNeedBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillByDefault(testing::Invoke([real_allocator](int  seq_len,
                                                            int  common_seq_len,
                                                            int  remaining_tokens,
                                                            int  reserve_step,
                                                            bool reuse_cache,
                                                            int  target_batch_size) {
                return real_allocator->estimateInitialBatchPeakNeedBlocks(
                    seq_len, common_seq_len, remaining_tokens, reserve_step, reuse_cache, target_batch_size);
            }));
        cache_manager->coordinator_manager_ = allocator;
        return contexts;
    }

    void initializeQueryCpWithNativeDraft() {
        ASSERT_NO_FATAL_FAILURE(initializeScheduler(true, 8, 32, 1, 2));
        auto& parallelism                                                     = engine_->parallelism_config;
        parallelism.role_type                                                 = RoleType::PREFILL;
        auto& cp                                                              = parallelism.prefill_cp_config;
        cp.method                                                             = CPRotateMethod::ALL_GATHER;
        cp.prefill_cp_size                                                    = 8;
        cp.kv_cache_sharded                                                   = true;
        cp.preserve_tp_layout                                                 = true;
        engine_->pd_sep_config.role_type                                      = RoleType::PREFILL;
        engine_->resource_context_.role_type                                  = RoleType::PREFILL;
        engine_->sp_config.type                                               = SP_TYPE_MTP;
        engine_->sp_config.gen_num_per_cycle                                  = 3;
        engine_->model_config_.gen_num_per_cycle                              = 3;
        engine_->runtime_config.fifo_scheduler_config.max_context_batch_size  = 2;
        engine_->runtime_config.fifo_scheduler_config.cp_force_single_prefill = false;
        // Actual K3 producer: target Query-CP ceil8(32 + min(32,2)*15)=64;
        // the private CP-off native draft keeps max(Prefill32, Decode8)=32.
        engine_->model_config_.max_prefill_tokens = 64;
        ExecutionCapacitySetter<ModelConfig>::set(engine_->model_config_, 64);
        auto draft                              = std::make_unique<EngineInitParams>();
        draft->model_config_                    = engine_->model_config_;
        draft->model_config_.max_prefill_tokens = 0;
        ExecutionCapacitySetter<ModelConfig>::set(draft->model_config_, 32);
        draft->parallelism_config                   = parallelism;
        draft->parallelism_config.prefill_cp_config = PrefillCPConfig{};
        auto drafts = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        drafts->push_back(std::move(draft));
        engine_->propose_params_ = std::make_unique<ProposeModelEngineInitParams>(SP_TYPE_MTP, 3, std::move(drafts));
        KVCacheConfig cache_options;
        cache_options.seq_size_per_block = cache_options.kernel_seq_size_per_block = 4;
        cache_options.enable_device_cache                                          = false;
        auto cache_config = CacheConfigCreator::createConfig(engine_->model_config_, parallelism, cache_options);
        cache_config      = test::finalizeCacheConfig(std::move(cache_config), 128);
        auto manager      = std::make_shared<KVCacheManager>(cache_config, false, nullptr, cache_options, parallelism);
        ASSERT_TRUE(manager->init());
        engine_->resource_context_.cache_manager = std::move(manager);
        engine_->initScheduler();
        engine_->scheduler_->updateSchedulerInfo("{\"batch_size\":2,\"mode\":\"prefill\"}");
    }

    void initializeDecodeScheduler() {
        ASSERT_NO_FATAL_FAILURE(initializeScheduler(false, 2, 3, 1, 2));
        engine_->parallelism_config.role_type                                 = RoleType::DECODE;
        engine_->pd_sep_config.role_type                                      = RoleType::DECODE;
        engine_->resource_context_.role_type                                  = RoleType::DECODE;
        engine_->runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode = "";
        engine_->initScheduler();
        ASSERT_NE(dynamic_cast<FIFOScheduler*>(engine_->scheduler_.get()), nullptr);
    }

    GenerateStreamPtr handoffDecode(int id) {
        auto input                                   = std::make_shared<GenerateInput>();
        input->request_id                            = id;
        input->begin_time_us                         = autil::TimeUtility::currentTimeInMicroSeconds();
        input->input_ids                             = torch::tensor({1}, torch::kInt32);
        input->generate_config                       = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens       = 2;
        input->generate_config->num_return_sequences = 1;
        auto stream                                  = engine_->makeStream(input);
        // The actual DecodeRpcServer handoff uses this state-machine/allocation
        // and first-token update sequence before scheduler enqueue. This is a
        // stream/cache/scheduler integration check, without an RPC server.
        stream->reportEvent(StreamEvents::CanRun);
        EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
        EXPECT_GT(stream->curBlocksNum(), 0);
        stream->setIsContextStream(false);
        stream->step();
        stream->incLastOutputPos();
        advance(stream);
        EXPECT_FALSE(stream->hasError());
        EXPECT_TRUE(engine_->scheduler_->enqueue(stream).ok());
        return stream;
    }

    void advance(const GenerateStreamPtr& stream) {
        auto tokens = torch::full({stream->currentBatchSize(), 1}, 99, torch::kInt32);
        stream->update({tokens, 1, {}, {}, {}, {}, {}, {}, {}, {}, true, false});
    }

    void checkGatheredWorkspace(const std::list<GenerateStreamPtr>& streams, bool decode) {
        ASSERT_FALSE(streams.empty());
        StreamGroups               groups(streams);
        NormalBatchStreamProcessor processor(engine_->model_config_,
                                             engine_->pd_sep_config,
                                             ProfilingDebugLoggingConfig{},
                                             engine_->getCacheManager()->cacheConfig(),
                                             false);
        TensorHolder               holder;
        auto                       gathered = processor.gatherModelInput(groups, holder);
        ASSERT_TRUE(gathered.ok()) << gathered.status();
        const auto& input = gathered.value();
        ASSERT_FALSE(input.warmup);
        ASSERT_FALSE(input.is_fake_stream);
        int64_t expected_rows      = 0;
        int64_t expected_sequences = 0;
        for (const auto& stream : streams) {
            EXPECT_EQ(stream->isContextStream(), !decode);
            EXPECT_FALSE(stream->isFakeStream());
            expected_sequences += stream->currentBatchSize();
            expected_rows += decode ? stream->currentBatchSize() : stream->currentExecuteTokenSize();
        }
        EXPECT_EQ(input.combo_tokens.numel(), expected_rows);
        EXPECT_EQ(input.input_lengths.numel(), expected_sequences);
        EXPECT_EQ(input.sequence_lengths.numel(), decode ? expected_sequences : 0);
        // This is the real gatherer's extent, before K3's TP padding. Exceeding
        // the initialized workspace here cannot be repaired by later padding.
        EXPECT_LE(input.combo_tokens.numel(), engine_->model_config_.max_prefill_tokens)
            << "scheduler returned expanded rows beyond the initialized model workspace";
    }

    autil::EnvGuard               device_input_{"RTP_LLM_DEVICE_INPUT", "0"};
    autil::EnvGuard               perf_test_{"PERF_TEST", "0"};
    std::shared_ptr<NormalEngine> engine_;
};

TEST_F(SchedulerPhysicalCapacityIntegrationTest, RatioAdmitsSafeFanoutWithoutClamping) {
    ASSERT_NO_FATAL_FAILURE(initializeScheduler(false, 8, 18, 32, 2));
    ASSERT_EQ(engine_->model_config_.max_prefill_tokens, 32);
    auto stream = enqueueStream(9, 2, 1);
    auto output = engine_->scheduler_->schedule();
    ASSERT_TRUE(output.ok()) << output.status();
    ASSERT_EQ(output->streams.size(), 1);
    EXPECT_EQ(stream->currentBatchSize(), 2);
    EXPECT_FALSE(stream->hasError());
    checkGatheredWorkspace(output->streams, false);
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, RatioAccountsForAggregateReturnSequenceRows) {
    ASSERT_NO_FATAL_FAILURE(initializeScheduler(false, 8, 18, 32, 2));
    ASSERT_EQ(engine_->model_config_.max_prefill_tokens, 32);
    // Each stream needs 9*2=18 rows; the legacy 9*2-stream rectangle fits
    // token_budget=32, but their actual combined extent is 36 rows.
    auto first  = enqueueStream(9, 2, 1);
    auto second = enqueueStream(9, 2, 2);
    auto output = engine_->scheduler_->schedule();
    ASSERT_TRUE(output.ok()) << output.status();
    if (!output->streams.empty()) {
        checkGatheredWorkspace(output->streams, false);
    }
    EXPECT_EQ(output->streams.size(), 1);
    EXPECT_EQ(static_cast<PDFusionRatioScheduler*>(engine_->scheduler_.get())->waitingStreamsSize(), 1);
    EXPECT_FALSE(first->hasError());
    EXPECT_FALSE(second->hasError());
    EXPECT_EQ(first->currentBatchSize(), 2);
    EXPECT_EQ(second->currentBatchSize(), 2);
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, DirectDecodeAdmitsSafeFanoutWithoutClamping) {
    ASSERT_NO_FATAL_FAILURE(initializeScheduler(true, 2, 2, 1, 1));
    ASSERT_EQ(engine_->model_config_.max_prefill_tokens, 2);
    auto stream = enqueueStream(1, 2, 1);
    auto output = engine_->scheduler_->schedule();
    ASSERT_TRUE(output.ok()) << output.status();
    ASSERT_EQ(output->streams.size(), 1);
    EXPECT_FALSE(stream->hasError());
    EXPECT_TRUE(stream->isPerfTest());  // Explicit direct-Decode benchmark mode.
    EXPECT_FALSE(stream->isContextStream());
    EXPECT_EQ(stream->currentBatchSize(), 2);
    checkGatheredWorkspace(output->streams, true);
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, DirectDecodeRejectsFanoutBeyondWorkspace) {
    ASSERT_NO_FATAL_FAILURE(initializeScheduler(true, 2, 2, 1, 1));
    ASSERT_EQ(engine_->model_config_.max_prefill_tokens, 2);
    auto stream = enqueueStream(1, 3, 1);
    auto output = engine_->scheduler_->schedule();
    ASSERT_TRUE(output.ok()) << output.status();
    if (!output->streams.empty()) {
        EXPECT_TRUE(stream->isPerfTest());
        checkGatheredWorkspace(output->streams, true);
    }
    EXPECT_TRUE(output->streams.empty());
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(stream->currentBatchSize(), 3);
    EXPECT_FALSE(stream->isFakeStream());
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, RatioReservesDecodeRowsAcrossPrefillRounds) {
    ASSERT_NO_FATAL_FAILURE(initializeScheduler(false, 2, 3, 1, 2));
    ASSERT_EQ(engine_->model_config_.max_prefill_tokens, 4);
    engine_->runtime_config.fifo_scheduler_config.decode_prefill_ratio = "0";
    engine_->initScheduler();
    auto first  = enqueueStream(1, 3, 1, 2);
    auto second = enqueueStream(1, 3, 2, 2);
    // Each Prefill fits 3 physical rows (4 after TP padding). Reserving future
    // Decode slots prevents two separately admitted Prefills from later forming
    // a six-row Decode batch in a four-row workspace.
    for (int round = 0; round < 3; ++round) {
        auto output = engine_->scheduler_->schedule();
        ASSERT_TRUE(output.ok()) << output.status();
        ASSERT_FALSE(output->streams.empty());
        const bool decode = !output->streams.front()->isContextStream();
        checkGatheredWorkspace(output->streams, decode);
        if (round == 1) {
            EXPECT_TRUE(decode);
            EXPECT_EQ(output->streams.front(), first);
            EXPECT_EQ(second->getStatus(), StreamState::WAITING);
            EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
            EXPECT_EQ(second->curBlocksNum(), 0);
        }
        for (const auto& stream : output->streams) {
            advance(stream);
        }
    }
    EXPECT_FALSE(first->hasError());
    EXPECT_FALSE(second->hasError());
    EXPECT_EQ(first->maxBatchSize(), 3);
    EXPECT_EQ(second->maxBatchSize(), 3);
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, UnspecifiedExecutionCapacityPreservesDirectDecode) {
    ASSERT_NO_FATAL_FAILURE(initializeScheduler(true, 2, 2, 1, 1));
    ASSERT_EQ(engine_->model_config_.max_prefill_tokens, 2);
    ExecutionCapacitySetter<ModelConfig>::set(engine_->model_config_, 0);
    engine_->initScheduler();
    auto stream = enqueueStream(1, 3, 1);
    auto output = engine_->scheduler_->schedule();
    ASSERT_TRUE(output.ok()) << output.status();
    ASSERT_EQ(output->streams.size(), 1);
    EXPECT_FALSE(stream->hasError());
    EXPECT_FALSE(stream->isContextStream());
    EXPECT_EQ(stream->currentExecuteTokenSize(), 3);
    // max_prefill_tokens remains two; it must not become a Decode limit.
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, RatioKeepsExpandedChunkBudgetAndContinuation) {
    ASSERT_NO_FATAL_FAILURE(initializeScheduler(false, 2, 18, 1, 2));
    auto& model                                        = engine_->model_config_;
    auto& runtime                                      = engine_->runtime_config;
    runtime.fifo_scheduler_config.prefill_chunk_size   = 16;
    runtime.fifo_scheduler_config.decode_prefill_ratio = "0";
    // K3's chunked producer replaces the full-prompt reserve with chunk rows.
    // Page-RR's native cache stripe is eight tokens for this B4/TP2 topology.
    model.max_prefill_tokens = 16;
    ExecutionCapacitySetter<ModelConfig>::set(model, 16);
    engine_->initScheduler();
    auto first  = enqueueStream(9, 2, 1, 2);
    auto second = enqueueStream(9, 2, 2, 2);
    auto middle = engine_->scheduler_->schedule();
    ASSERT_TRUE(middle.ok()) << middle.status();
    ASSERT_EQ(middle->streams.size(), 1);
    ASSERT_EQ(middle->streams.front(), first);
    EXPECT_EQ(first->contextLength(), 8);
    EXPECT_TRUE(first->isMiddleChunk());
    checkGatheredWorkspace(middle->streams, false);
    advance(first);
    EXPECT_EQ(first->reuseLength(), 8);
    EXPECT_TRUE(first->isContextStream());
    auto tail = engine_->scheduler_->schedule();
    ASSERT_TRUE(tail.ok()) << tail.status();
    ASSERT_EQ(tail->streams.size(), 1);
    EXPECT_EQ(tail->streams.front(), first);
    EXPECT_EQ(first->contextLength(), 1);
    checkGatheredWorkspace(tail->streams, false);
    advance(first);
    auto next = engine_->scheduler_->schedule();
    ASSERT_TRUE(next.ok()) << next.status();
    ASSERT_EQ(next->streams.size(), 1);
    EXPECT_EQ(next->streams.front(), second);
    EXPECT_EQ(second->contextLength(), 8);
    checkGatheredWorkspace(next->streams, false);
    EXPECT_FALSE(first->hasError());
    EXPECT_FALSE(second->hasError());
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, RatioReservesAllPendingLoadCohorts) {
    ASSERT_NO_FATAL_FAILURE(initializeScheduler(false, 8, 18, 32, 2));
    auto loads   = setupPendingCacheLoads(engine_->getCacheManager());
    auto first   = enqueueStream(9, 2, 1);
    auto second  = enqueueStream(9, 2, 2);
    auto loading = engine_->scheduler_->schedule();
    ASSERT_TRUE(loading.ok()) << loading.status();
    EXPECT_TRUE(loading->streams.empty());
    EXPECT_EQ(first->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(second->curBlocksNum(), 0);
    // Repeated rounds must retain the first load's 18-row reservation even
    // though each round creates a fresh local admission list.
    auto pending = engine_->scheduler_->schedule();
    ASSERT_TRUE(pending.ok()) << pending.status();
    EXPECT_TRUE(pending->streams.empty());
    EXPECT_EQ(second->curBlocksNum(), 0);
    ASSERT_TRUE(loads[0]->settle(true));
    auto loaded = engine_->scheduler_->schedule();
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    ASSERT_FALSE(loaded->streams.empty());
    checkGatheredWorkspace(loaded->streams, false);
    EXPECT_EQ(loaded->streams.size(), 1);
    EXPECT_EQ(loaded->streams.front(), first);
    EXPECT_FALSE(first->hasError());
    EXPECT_FALSE(second->hasError());
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(second->curBlocksNum(), 0);
    EXPECT_FALSE(loads[1]->done());

    // The second load has not started: registering its context cannot make it
    // settlement-ready. Finish the actual first Prefill before admitting it.
    advance(first);
    ASSERT_TRUE(first->hasEvent(StreamEvents::GenerateDone));
    auto second_loading = engine_->scheduler_->schedule();
    ASSERT_TRUE(second_loading.ok()) << second_loading.status();
    EXPECT_TRUE(second_loading->streams.empty());
    EXPECT_TRUE(first->isFinished());
    ASSERT_EQ(second->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_TRUE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_TRUE(second->hasEvent(StreamEvents::LoadInitiated));
    EXPECT_GT(second->curBlocksNum(), 0);
    ASSERT_EQ(second->streamCacheResource().allocator_load_context_.get(), loads[1].get());
    EXPECT_FALSE(loads[1]->done());
    ASSERT_TRUE(loads[1]->settle(true));

    auto second_loaded = engine_->scheduler_->schedule();
    ASSERT_TRUE(second_loaded.ok()) << second_loaded.status();
    ASSERT_EQ(second_loaded->streams.size(), 1);
    EXPECT_EQ(second_loaded->streams.front(), second);
    EXPECT_EQ(second->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(second->hasError());
    EXPECT_EQ(second->currentBatchSize(), 2);
    checkGatheredWorkspace(second_loaded->streams, false);
    advance(second);
    ASSERT_TRUE(second->hasEvent(StreamEvents::GenerateDone));
    auto finished = engine_->scheduler_->schedule();
    ASSERT_TRUE(finished.ok()) << finished.status();
    EXPECT_TRUE(finished->streams.empty());
    EXPECT_TRUE(first->isFinished());
    EXPECT_TRUE(second->isFinished());
    EXPECT_FALSE(first->hasError());
    EXPECT_FALSE(second->hasError());
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, FifoRetainsUnitWidthDecodeTopUp) {
    ASSERT_NO_FATAL_FAILURE(initializeDecodeScheduler());
    auto first   = handoffDecode(1);
    auto initial = engine_->scheduler_->schedule();
    ASSERT_TRUE(initial.ok()) << initial.status();
    ASSERT_EQ(initial->streams.size(), 1);
    checkGatheredWorkspace(initial->streams, true);
    auto second    = handoffDecode(2);
    auto topped_up = engine_->scheduler_->schedule();
    ASSERT_TRUE(topped_up.ok()) << topped_up.status();
    ASSERT_EQ(topped_up->streams.size(), 2);
    checkGatheredWorkspace(topped_up->streams, true);
    EXPECT_FALSE(first->hasError());
    EXPECT_FALSE(second->hasError());
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, FifoPreallocatedWaitersDoNotOwnExecutionSlots) {
    ASSERT_NO_FATAL_FAILURE(initializeDecodeScheduler());
    std::vector<GenerateStreamPtr> streams;
    // Five F1 preallocated RPC arrivals exceed physical Ncap4. None has been
    // scheduler-admitted, so an empty engine must admit a legal prefix of two.
    for (int id = 1; id <= 5; ++id) {
        streams.push_back(handoffDecode(id));
    }
    for (const int expected : {2, 2, 1}) {
        auto output = engine_->scheduler_->schedule();
        ASSERT_TRUE(output.ok()) << output.status();
        ASSERT_EQ(output->streams.size(), expected);
        checkGatheredWorkspace(output->streams, true);
        for (const auto& stream : output->streams) {
            advance(stream);
        }
    }
    for (const auto& stream : streams) {
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->maxBatchSize(), 1);
    }
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, FixedQueryCpPrefillHonorsNativeDraftNaturalRows) {
    ASSERT_NO_FATAL_FAILURE(initializeQueryCpWithNativeDraft());
    auto first  = enqueueStream(17, 1, 1);
    auto second = enqueueStream(17, 1, 2);
    auto output = engine_->scheduler_->schedule();
    ASSERT_TRUE(output.ok()) << output.status();
    if (!output->streams.empty()) {
        checkGatheredWorkspace(output->streams, false);
        EXPECT_EQ(StreamGroups(output->streams).modelExecuteTokenSize(), 34);
    }
    // Target CP consumes 32+32=64 and fits, but natural draft rows34
    // become40 after SP and exceed the draft's own32-row workspace.
    EXPECT_TRUE(output->streams.empty());
    EXPECT_TRUE(first->hasError());
    EXPECT_TRUE(second->hasError());
    EXPECT_EQ(first->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(second->statusInfo().code(), ErrorCode::INVALID_PARAMS);
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, FixedQueryCpPrefillAdmitsNativeDraftBoundary) {
    ASSERT_NO_FATAL_FAILURE(initializeQueryCpWithNativeDraft());
    auto first  = enqueueStream(16, 1, 1);
    auto second = enqueueStream(16, 1, 2);
    auto output = engine_->scheduler_->schedule();
    ASSERT_TRUE(output.ok()) << output.status();
    ASSERT_EQ(output->streams.size(), 2);
    checkGatheredWorkspace(output->streams, false);
    EXPECT_EQ(StreamGroups(output->streams).modelExecuteTokenSize(), 32);
    EXPECT_FALSE(first->hasError());
    EXPECT_FALSE(second->hasError());
}

TEST_F(SchedulerPhysicalCapacityIntegrationTest, FixedQueryCpTargetOnlySkipsUnusedDraftBudget) {
    ASSERT_NO_FATAL_FAILURE(initializeQueryCpWithNativeDraft());
    auto first  = enqueueStream(17, 1, 1, 0);
    auto second = enqueueStream(17, 1, 2, 0);
    auto output = engine_->scheduler_->schedule();
    ASSERT_TRUE(output.ok()) << output.status();
    ASSERT_EQ(output->streams.size(), 2);
    checkGatheredWorkspace(output->streams, false);
    EXPECT_FALSE(first->hasError());
    EXPECT_FALSE(second->hasError());
    // Native MTP DP1 target-only takes the production early return before
    // draft forward, so only target64 is relevant, not draft32.
}

}  // namespace
}  // namespace rtp_llm
