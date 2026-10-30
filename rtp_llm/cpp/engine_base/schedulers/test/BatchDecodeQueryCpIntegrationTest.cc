#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "autil/EnvUtil.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

namespace rtp_llm {
namespace {

// One physical CUDA device exercises all eight logical CP ranks. Only input
// planning runs: this test does not call a collective or a transformer forward.
class BatchDecodeQueryCpIntegrationTest: public DeviceTestBase {
protected:
    static constexpr int64_t kCpSize   = 8;
    static constexpr int64_t kCapacity = 48;

    void SetUp() override {
        DeviceTestBase::SetUp();
        CustomConfig bootstrap;
        bootstrap.max_context_batch_size = 1;
        bootstrap.tokens_per_block       = 4;
        engine_                          = createMockEngine(bootstrap);
        ASSERT_TRUE(engine_->stop().ok());

        // Start/stop the existing fixture at TP1, then use its production
        // initScheduler entry without launching a TP8 engine loop/followers.
        auto& parallelism   = engine_->parallelism_config;
        parallelism.tp_size = parallelism.ep_size = parallelism.ffn_tp_size = kCpSize;
        parallelism.world_size = parallelism.local_world_size = kCpSize;
        parallelism.dp_size = parallelism.pp_size        = 1;
        parallelism.role_type                            = RoleType::PREFILL;
        parallelism.prefill_cp_config.method             = CPRotateMethod::ALL_GATHER;
        parallelism.prefill_cp_config.prefill_cp_size    = kCpSize;
        parallelism.prefill_cp_config.kv_cache_sharded   = true;
        parallelism.prefill_cp_config.preserve_tp_layout = true;

        auto& model                = engine_->model_config_;
        model.max_seq_len          = 18;
        model.attn_config.head_num = model.attn_config.kv_head_num = kCpSize;
        model.attn_config.size_per_head                            = 16;
        model.attn_config.is_causal                                = true;
        model.attn_config.tokens_per_block = model.attn_config.kernel_tokens_per_block = 4;
        model.use_page_rr_kv_cache                                                     = true;

        auto& runtime                                        = engine_->runtime_config;
        runtime.use_batch_decode_scheduler                   = true;
        runtime.max_generate_batch_size                      = 2;
        runtime.fifo_scheduler_config.max_context_batch_size = 1;
        runtime.fifo_scheduler_config.max_batch_tokens_size  = 1;
        runtime.fifo_scheduler_config.prefill_chunk_size     = 0;
        engine_->pd_sep_config.role_type                     = RoleType::PREFILL;
        engine_->sp_config.type                              = SP_TYPE_NONE;

        // Non-chunk base=max(max_seq_len, token budget)=18; gamma=1 and
        // request bound=2. Use the production CP capacity helper, not a
        // larger test-only buffer: ceil8(18 + min(18, 2) * 15) = 48 rows.
        const auto token_budget =
            std::max<int64_t>(model.max_seq_len, runtime.fifo_scheduler_config.max_batch_tokens_size);
        model.max_execution_tokens =
            getZigzagPaddedTokenCapacity(token_budget, runtime.max_generate_batch_size, kCpSize);
        ASSERT_EQ(model.max_execution_tokens, kCapacity);

        ASSERT_NO_FATAL_FAILURE(initializePageRrCache());
    }

    void initializePageRrCache() {
        KVCacheConfig cache_options;
        cache_options.seq_size_per_block = cache_options.kernel_seq_size_per_block = 4;
        cache_options.enable_device_cache                                          = false;
        auto cache_config =
            CacheConfigCreator::createConfig(engine_->model_config_, engine_->parallelism_config, cache_options);
        cache_config = test::finalizeCacheConfig(std::move(cache_config), 128);
        ASSERT_TRUE(cache_config.local_kv_cache_sharded);
        auto manager =
            std::make_shared<KVCacheManager>(cache_config, false, nullptr, cache_options, engine_->parallelism_config);
        ASSERT_TRUE(manager->init());
        ASSERT_NE(manager->cpSlotMapper(), nullptr);
        ASSERT_EQ(manager->cpSlotMapper()->cpSize(), kCpSize);
        engine_->resource_context_.cache_manager = std::move(manager);
        engine_->resource_context_.role_type     = RoleType::PREFILL;
        engine_->resource_context_.reuse_cache   = false;
    }

    void TearDown() override {
        // No TP8 loop was started. Destroy the scheduler's real stream/cache
        // references before destroying the already-stopped bootstrap engine.
        engine_->scheduler_.reset();
        engine_.reset();
        DeviceTestBase::TearDown();
    }

    void checkScheduledMetadata(const std::list<GenerateStreamPtr>& streams,
                                int64_t                             raw_tokens,
                                int64_t                             request_rows,
                                int64_t                             expected_cp_rows) {
        NormalBatchStreamProcessor processor(engine_->model_config_,
                                             engine_->pd_sep_config,
                                             ProfilingDebugLoggingConfig{},
                                             engine_->getCacheManager()->cacheConfig(),
                                             false);
        StreamGroups               groups(streams);
        ASSERT_EQ(groups.totalContextBatchSize(), request_rows);
        TensorHolder holder;
        auto         gathered = processor.gatherModelInput(groups, holder);
        ASSERT_TRUE(gathered.ok()) << gathered.status();
        const auto& global = gathered.value();
        ASSERT_FALSE(global.warmup);
        ASSERT_FALSE(global.is_fake_stream);
        ASSERT_EQ(global.combo_tokens.numel(), raw_tokens);
        ASSERT_EQ(global.input_lengths.numel(), request_rows);

        if (!engine_->parallelism_config.prefill_cp_config.is_enabled()) {
            // Page-RR alone changes KV ownership, not query rows. Stop at
            // the real gather boundary; no CP planner or SP forward runs.
            EXPECT_EQ(expected_cp_rows, 0);
            EXPECT_EQ(global.input_lengths.sum().item<int64_t>(), raw_tokens);
            EXPECT_LE(raw_tokens, engine_->model_config_.max_execution_tokens);
            return;
        }

        int64_t total_rank_local_rows = 0;
        for (int rank = 0; rank < kCpSize; ++rank) {
            SCOPED_TRACE(rank);
            auto parallelism    = engine_->parallelism_config;
            parallelism.tp_rank = parallelism.ep_rank = parallelism.ffn_tp_rank = parallelism.world_rank = rank;
            // Keep every logical rank on the owned physical CUDA device.
            parallelism.local_rank                         = 0;
            auto                               rank_inputs = global;
            torch_ext::PyContextParallelParams metadata;
            ZigZagProcessor                    processor(parallelism);
            processor.handleInputs(rank_inputs, metadata);

            const auto rows = metadata.prefill_qkv_restore_indice_cpu.numel();
            EXPECT_EQ(rows, expected_cp_rows);
            EXPECT_EQ(metadata.prefill_qkv_padding_mask_cpu.sum().item<int64_t>(), raw_tokens);
            EXPECT_EQ(metadata.prefill_actual_input_lengths_cpu.numel(), request_rows);
            EXPECT_EQ(rank_inputs.combo_tokens.numel() * kCpSize, rows);
            EXPECT_EQ(rank_inputs.input_lengths.sum().item<int64_t>(), rank_inputs.combo_tokens.numel());
            total_rank_local_rows += rank_inputs.combo_tokens.numel();
            // On the red baseline, this reports the actual CP producer's
            // 64-row extent before the admission assertions below also fail.
            EXPECT_LE(rows, engine_->model_config_.max_execution_tokens)
                << "BDS admitted a batch exceeding the initialized CP token capacity";
        }
        EXPECT_EQ(total_rank_local_rows, expected_cp_rows);
    }

    void checkAdmission(int length, int fanout, int requests, int64_t expected_cp_rows, bool admitted) {
        const int batch                                                                         = fanout * requests;
        engine_->runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = batch;
        engine_->initScheduler();
        ASSERT_NE(dynamic_cast<BatchDecodeScheduler*>(engine_->scheduler_.get()), nullptr);
        engine_->scheduler_->updateSchedulerInfo("{\"batch_size\":" + std::to_string(batch) + ",\"mode\":\"prefill\"}");

        std::vector<GenerateStreamPtr> inputs;
        for (int request = 0; request < requests; ++request) {
            auto input                                   = std::make_shared<GenerateInput>();
            input->request_id                            = request + 1;
            input->input_ids                             = torch::arange(1, length + 1, torch::kInt32);
            input->generate_config                       = std::make_shared<GenerateConfig>();
            input->generate_config->max_new_tokens       = 1;
            input->generate_config->num_return_sequences = fanout;
            input->generate_config->top_k                = 1;
            auto stream                                  = engine_->makeStream(input);
            ASSERT_FALSE(stream->hasError()) << stream->statusInfo().ToString();
            ASSERT_FALSE(stream->isFakeStream());
            ASSERT_TRUE(stream->isContextStream());
            ASSERT_EQ(stream->currentBatchSize(), fanout);
            ASSERT_TRUE(engine_->scheduler_->enqueue(stream).ok());
            inputs.push_back(std::move(stream));
        }

        auto result = engine_->scheduler_->schedule();
        ASSERT_TRUE(result.ok()) << result.status();
        // BDS calls setPerfTest(true), but currentBatchSize must still expand
        // num_return_sequences. This is neither a fake stream nor a beam case.
        for (const auto& stream : inputs) {
            EXPECT_EQ(stream->currentBatchSize(), fanout);
            EXPECT_FALSE(stream->isFakeStream());
        }
        if (!result->streams.empty()) {
            ASSERT_EQ(result->streams.size(), requests);
            checkScheduledMetadata(result->streams, length * fanout * requests, fanout * requests, expected_cp_rows);
        }
        if (admitted) {
            EXPECT_EQ(result->streams.size(), requests);
            for (const auto& stream : inputs) {
                EXPECT_FALSE(stream->hasError()) << stream->statusInfo().ToString();
            }
        } else {
            EXPECT_TRUE(result->streams.empty()) << "unsafe CP cohort must be rejected before model execution";
            for (const auto& stream : inputs) {
                EXPECT_TRUE(stream->hasError());
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            }
        }
    }

    autil::EnvGuard               device_input_{"RTP_LLM_DEVICE_INPUT", "0"};
    autil::EnvGuard               perf_test_{"PERF_TEST", "0"};
    std::shared_ptr<NormalEngine> engine_;
};

TEST_F(BatchDecodeQueryCpIntegrationTest, AdmitsOneSeventeenTokenRequest) {
    checkAdmission(17, 1, 1, 32, true);
}

TEST_F(BatchDecodeQueryCpIntegrationTest, RejectsFanoutWhoseCpRowsExceedCapacity) {
    checkAdmission(17, 2, 1, 64, false);
}

TEST_F(BatchDecodeQueryCpIntegrationTest, RejectsAggregateCpRowsAcrossFixedBatch) {
    checkAdmission(17, 1, 2, 64, false);
}

TEST_F(BatchDecodeQueryCpIntegrationTest, AdmitsShortFanoutWithinPhysicalCapacity) {
    checkAdmission(1, 2, 1, 32, true);
}

TEST_F(BatchDecodeQueryCpIntegrationTest, PageRrWithoutQueryCpAdmitsRawFanoutRows) {
    auto& parallelism             = engine_->parallelism_config;
    parallelism.prefill_cp_config = PrefillCPConfig{};
    ASSERT_FALSE(parallelism.prefill_cp_config.is_enabled());
    // Page-RR still requires its transport/storage declaration. These fields
    // do not enable Query-CP: method and preserve_tp_layout retain defaults.
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    parallelism.prefill_cp_config.prefill_cp_size  = kCpSize;
    ASSERT_FALSE(parallelism.prefill_cp_config.is_enabled());
    ASSERT_EQ(parallelism.tp_size, kCpSize);
    ASSERT_EQ(parallelism.ep_size, kCpSize);
    ASSERT_TRUE(engine_->model_config_.use_page_rr_kv_cache);

    engine_->model_config_.max_seq_len = kCapacity;
    const auto base                    = std::max<int64_t>(engine_->model_config_.max_seq_len,
                                        engine_->runtime_config.fifo_scheduler_config.max_batch_tokens_size);
    ASSERT_EQ((base + kCpSize - 1) / kCpSize * kCpSize, engine_->model_config_.max_execution_tokens);
    ASSERT_EQ(engine_->model_config_.max_execution_tokens, kCapacity);
    ASSERT_NO_FATAL_FAILURE(initializePageRrCache());
    checkAdmission(17, 2, 1, /*expected_cp_rows=*/0, true);
}

}  // namespace
}  // namespace rtp_llm
