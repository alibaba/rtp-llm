#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {
namespace {

class ScopedModelFactory {
public:
    explicit ScopedModelFactory(NormalExecutor::ModelFactory factory):
        previous_(std::move(NormalExecutor::test_model_factory)) {
        NormalExecutor::test_model_factory = std::move(factory);
    }
    ~ScopedModelFactory() {
        NormalExecutor::test_model_factory = std::move(previous_);
    }

private:
    NormalExecutor::ModelFactory previous_;
};

struct ModelInitSnapshot {
    size_t               tokens;
    size_t               kernel_tokens;
    int32_t              groups;
    std::vector<int32_t> layer_to_group;
    bool                 has_cache;
    bool                 graph_enabled;
};

class NormalExecutorWarmupTest: public DeviceTestBase {
protected:
    EngineInitParams makeParams(int physical_tokens = 128, int kernel_tokens = 0) {
        ModelConfig   model;
        RuntimeConfig runtime;
        KVCacheConfig kv;
        auto          params                              = createEngineInitParams(CustomConfig{}, model, runtime, kv);
        params.model_config_.attn_config.tokens_per_block = physical_tokens;
        params.kv_cache_config.seq_size_per_block         = physical_tokens;
        params.kv_cache_config.kernel_seq_size_per_block  = kernel_tokens;
        params.kv_cache_config.test_block_num             = 5;
        params.runtime_config.warm_up                     = true;
        params.runtime_config.model_warm_up               = true;
        params.runtime_config.max_generate_batch_size     = 1;
        params.runtime_config.fifo_scheduler_config.max_context_batch_size = 1;
        params.hw_kernel_config.enable_cuda_graph                          = true;
        return params;
    }

    CacheConfig basicConfig(const EngineInitParams& params) {
        return CacheConfigCreator::createBasicConfig(
            params.model_config_, params.parallelism_config, params.kv_cache_config, false, 0);
    }

    std::unique_ptr<NormalExecutor> prefillExecutor(const EngineInitParams& params,
                                                    const CacheConfig&      config,
                                                    bool                    warm_up    = true,
                                                    bool                    is_propose = false,
                                                    int                     groups     = -1) {
        return std::make_unique<NormalExecutor>(params,
                                                nullptr,
                                                warm_up,
                                                is_propose,
                                                0,
                                                MlaOpsType::AUTO,
                                                groups < 0 ? config.groupNums() : groups,
                                                config.layer_to_group_id,
                                                nullptr,
                                                nullptr,
                                                &config);
    }

    NormalExecutor::ModelFactory recordModels(std::vector<ModelInitSnapshot>& snapshots, size_t vocab_size) {
        return [&snapshots, vocab_size](const GptModelInitParams& params) {
            snapshots.push_back({params.tokens_per_block,
                                 params.kernel_tokens_per_block,
                                 params.kv_cache_group_num,
                                 params.kv_cache_layer_to_group,
                                 params.kv_cache_layer_layout.has_value(),
                                 params.hw_kernel_config.enable_cuda_graph});
            return std::make_unique<MockModel>(vocab_size);
        };
    }
};

TEST_F(NormalExecutorWarmupTest, BasicAndAllocatedCacheResolveTheSameGeometry) {
    for (int kernel_tokens : {0, 1, 64, 128}) {
        SCOPED_TRACE(kernel_tokens);
        auto       params    = makeParams(128, kernel_tokens);
        const auto basic     = basicConfig(params);
        const auto allocated = CacheConfigCreator::createConfig(
            params.model_config_, params.parallelism_config, params.runtime_config, params.kv_cache_config);
        EXPECT_EQ(basic.seq_size_per_block, 128u);
        EXPECT_EQ(basic.kernel_seq_size_per_block, kernel_tokens == 0 ? 128u : kernel_tokens);
        EXPECT_EQ(basic.seq_size_per_block, allocated.seq_size_per_block);
        EXPECT_EQ(basic.kernel_seq_size_per_block, allocated.kernel_seq_size_per_block);
        EXPECT_EQ(basic.kernelBlocksPerKvBlock(), allocated.kernelBlocksPerKvBlock());
        EXPECT_EQ(basic.block_num, 0u);
        EXPECT_EQ(allocated.block_num, 5u);
    }
}

TEST_F(NormalExecutorWarmupTest, BasicConfigRejectsInvalidSplitBeforeWarmup) {
    auto params = makeParams(128, 96);
    EXPECT_THROW(basicConfig(params), std::exception);
    // Validation must use the physical layout returned by the creator.
    params.kv_cache_config.seq_size_per_block        = 256;
    params.kv_cache_config.kernel_seq_size_per_block = 256;
    EXPECT_THROW(basicConfig(params), std::exception);
}

TEST_F(NormalExecutorWarmupTest, SpeculativeSubConfigUsesTheSameKernelResolution) {
    for (int kernel_tokens : {0, 64}) {
        SCOPED_TRACE(kernel_tokens);
        auto params                  = makeParams(128, kernel_tokens);
        auto sp                      = params.sp_config;
        sp.gen_num_per_cycle         = 1;
        const auto   config          = CacheConfigCreator::createSpConfig(params.model_config_,
                                                               params.model_config_,
                                                               params.parallelism_config,
                                                               params.runtime_config,
                                                               params.kv_cache_config,
                                                               sp,
                                                               std::nullopt,
                                                               false,
                                                               false);
        const size_t expected_kernel = kernel_tokens == 0 ? 128 : kernel_tokens;
        EXPECT_EQ(config.seq_size_per_block, 128u);
        EXPECT_EQ(config.kernel_seq_size_per_block, expected_kernel);
        ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
        EXPECT_EQ(config.mtp_sub_configs[0]->seq_size_per_block, 128u);
        EXPECT_EQ(config.mtp_sub_configs[0]->kernel_seq_size_per_block, expected_kernel);
    }
}

TEST_F(NormalExecutorWarmupTest, PrefillWarmupAndRealExecutorReceiveTheSameGeometryWithGraphEnabled) {
    for (int kernel_tokens : {0, 64}) {
        SCOPED_TRACE(kernel_tokens);
        auto                           params = makeParams(128, kernel_tokens);
        std::vector<ModelInitSnapshot> snapshots;
        ScopedModelFactory             factory(recordModels(snapshots, params.model_config_.vocab_size));
        { NormalEngine engine(params, nullptr); }
        // The first executor performs real framework prefill warmup before
        // cache allocation. The model hook observes the Python-model boundary.
        ASSERT_EQ(snapshots.size(), 2u);
        EXPECT_FALSE(snapshots[0].has_cache);
        EXPECT_TRUE(snapshots[1].has_cache);
        EXPECT_EQ(snapshots[0].tokens, 128u);
        EXPECT_EQ(snapshots[0].kernel_tokens, kernel_tokens == 0 ? 128u : kernel_tokens);
        EXPECT_EQ(snapshots[0].tokens, snapshots[1].tokens);
        EXPECT_EQ(snapshots[0].kernel_tokens, snapshots[1].kernel_tokens);
        EXPECT_EQ(snapshots[0].groups, snapshots[1].groups);
        EXPECT_EQ(snapshots[0].layer_to_group, snapshots[1].layer_to_group);
        EXPECT_TRUE(snapshots[0].graph_enabled);
        EXPECT_TRUE(snapshots[1].graph_enabled);
    }
}

TEST_F(NormalExecutorWarmupTest, TypedCacheGeometryReachesModelAndGathererWithoutStorage) {
    auto  params                            = makeParams(256, 128);
    auto& model                             = params.model_config_;
    model.attn_config.tokens_per_block      = 64;
    model.attn_config.kv_head_num           = 1;
    model.attn_config.size_per_head         = 512;
    model.attn_config.rope_head_dim         = 64;
    model.attn_config.sliding_window        = 128;
    model.attn_config.indexer_head_dim      = 128;
    model.attn_config.indexer_head_num      = 64;
    model.attn_config.indexer_topk          = 512;
    model.attn_config.layer_compress_ratios = {4, 128};
    const auto config                       = basicConfig(params);
    ASSERT_EQ(config.seq_size_per_block, 256u);
    ASSERT_EQ(config.kernel_seq_size_per_block, 128u);
    ASSERT_GT(config.groupNums(), 1);
    std::vector<ModelInitSnapshot> snapshots;
    ScopedModelFactory             factory(recordModels(snapshots, model.vocab_size));
    const auto                     executor = prefillExecutor(params, config);
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_FALSE(snapshots[0].has_cache);
    EXPECT_TRUE(snapshots[0].graph_enabled);
    EXPECT_EQ(snapshots[0].tokens, 256u);
    EXPECT_EQ(snapshots[0].kernel_tokens, 128u);
    EXPECT_EQ(snapshots[0].groups, config.groupNums());
    EXPECT_EQ(snapshots[0].layer_to_group, config.layer_to_group_id);
    const auto& gather = executor->batch_stream_processor_->model_input_gatherer_config_;
    EXPECT_EQ(gather.seq_size_per_block, 256u);
    EXPECT_EQ(gather.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(gather.kernel_blocks_per_kv_block, 2u);
    EXPECT_EQ(gather.kv_cache_group_nums, config.groupNums());
    EXPECT_EQ(gather.layer_to_kv_cache_group_id, config.layer_to_group_id);
}

TEST_F(NormalExecutorWarmupTest, PrefillConfigRejectsMisuseAndInvalidGeometry) {
    auto params = makeParams();
    auto config = basicConfig(params);
    EXPECT_THROW(prefillExecutor(params, config, false), std::exception);
    EXPECT_THROW(prefillExecutor(params, config, true, true), std::exception);
    EXPECT_THROW(prefillExecutor(params, config, true, false, 0), std::exception);
    config.kernel_seq_size_per_block = 0;
    EXPECT_THROW(prefillExecutor(params, config), std::exception);
    config.kernel_seq_size_per_block = 96;
    EXPECT_THROW(prefillExecutor(params, config), std::exception);
}

}  // namespace
}  // namespace rtp_llm
