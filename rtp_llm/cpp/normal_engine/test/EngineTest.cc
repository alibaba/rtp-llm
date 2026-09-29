#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <algorithm>
#include <cstdlib>

#define private public
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#undef private

#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <memory>

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

namespace {

/** Exercise real speculative initialization without loading a checkpoint or submitting model work. */
std::unique_ptr<ProposeModelEngineInitParams> makeLocalProposalParams(const EngineInitParams& params) {
    auto modules    = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    auto draft      = std::make_unique<EngineInitParams>(params);
    draft->model_id = params.model_id + 1;
    if (params.sp_config.type == SP_TYPE_DSPARK) {
        const auto options                = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        draft->gpt_weights.dspark_markov_w1 = torch::zeros({params.model_config_.vocab_size, 1}, options);
        draft->gpt_weights.dspark_markov_w2 = torch::zeros({params.model_config_.vocab_size, 1}, options);
    }
    modules->push_back(std::move(draft));
    return std::make_unique<ProposeModelEngineInitParams>(
        params.sp_config.type, params.sp_config.gen_num_per_cycle, std::move(modules));
}

}

#if USING_CUDA
namespace {

struct PPWarmupObserved {};
struct UnexpectedWarmupExecutor {};

/** Stop inside the real constructor's warmup before cache negotiation or the engine loop starts. */
class PPWarmupProbeModel: public ModelBase {
public:
    using Observer = std::function<void(const GptModelInputs&)>;

    PPWarmupProbeModel(int stage, Observer observer): stage_(stage), observer_(std::move(observer)) {}

    GptModelOutputs forward(const GptModelInputs& input) override {
        EXPECT_EQ(input.pp_intermediates.empty(), stage_ == 0);
        if (stage_ != 0) {
            EXPECT_TRUE(torch::equal(input.pp_intermediates.at("hidden_states"),
                                    torch::zeros({input.combo_tokens.numel(), 4})));
        }
        observer_(input);
        throw PPWarmupObserved{};
    }

    PPIntermediateTensors makePPWarmUpInputTensors(const GptModelInputs& input, bool) override {
        return {{{"hidden_states", torch::zeros({input.combo_tokens.numel(), 4})}}};
    }

private:
    int      stage_;
    Observer observer_;
};

struct PPWarmupFactoryGuard {
    NormalExecutor::ModelFactory normal_factory = std::move(NormalExecutor::test_model_factory);
    PPExecutor::ModelFactory     pp_factory     = std::move(PPExecutor::test_model_factory);

    PPWarmupFactoryGuard() {
        NormalExecutor::test_model_factory = [](const GptModelInitParams&) -> std::unique_ptr<ModelBase> {
            ADD_FAILURE() << "NormalEngine selected NormalExecutor for PP warmup";
            throw UnexpectedWarmupExecutor{};
        };
    }

    ~PPWarmupFactoryGuard() {
        NormalExecutor::test_model_factory = std::move(normal_factory);
        PPExecutor::test_model_factory = std::move(pp_factory);
        /** These fixtures initialize runtime tracing as false; the sentinel interrupts warmup's normal reset. */
        setTraceMemory(false);
    }
};

}
#endif

TEST_F(NormalEngineTest, testExecutorDecodesCopyRowsUsingPayloadTags) {
    class CopyPayloadProcessor: public NormalBatchStreamProcessor {
    public:
        CopyPayloadProcessor(const ModelConfig& model, const CacheConfig& config, torch::Tensor mapping):
            NormalBatchStreamProcessor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config, false),
            mapping_(std::move(mapping)) {}

        absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& groups,
                                                        TensorHolder&       holder) const override {
            auto result = NormalBatchStreamProcessor::gatherModelInput(groups, holder);
            if (!result.ok()) {
                return result.status();
            }
            auto& inputs = result.value();
            std::reverse(inputs.kv_cache_group_tags.begin(), inputs.kv_cache_group_tags.end());
            inputs.kv_cache_block_id        = inputs.kv_cache_block_id.flip({0});
            inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id.flip({0});
            inputs.kv_cache_group_types     = inputs.kv_cache_group_types.flip({0});
            inputs.kv_cache_update_mapping  = mapping_;
            return result;
        }

    private:
        torch::Tensor mapping_;
    };
    struct StopBeforeSampling {};

    ModelConfig model;
    model.num_layers                          = 2;
    model.max_seq_len                         = 128;
    model.vocab_size                          = 16;
    model.input_vocab_size                    = 16;
    model.hidden_size                         = 4;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 4;
    model.attn_config.tokens_per_block        = 2;
    model.attn_config.kernel_tokens_per_block = 2;
    CacheConfig config;
    config.layer_num          = 2;
    config.seq_size_per_block = 2;
    config.dtype              = DataType::TYPE_FP16;
    config.fromGroupedSpecs({test::makeResolvedMhaSpec(config.dtype, 1, 4, 2, "first"),
                             test::makeResolvedMhaSpec(config.dtype, 1, 4, 2, "second")},
                            {{0}, {1}},
                            {CacheGroupType::FULL, CacheGroupType::FULL});
    config.finalizeBlockNums(4, RuntimeConfig{});
    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());

    auto blockBytes = [&](int layer, const std::string& tag, int block) {
        auto addr = manager->convertIndexToAddr(layer, tag, block);
        return torch::from_blob(addr.kv_addr,
                                {static_cast<int64_t>(config.group(tag).kvBlockStrideBytes())},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    };
    auto first_src  = blockBytes(0, "first", 1);
    auto first_dst  = blockBytes(0, "first", 2);
    auto second_src = blockBytes(1, "second", 1);
    auto second_dst = blockBytes(1, "second", 2);
    first_src.fill_(11);
    second_src.fill_(22);

    EngineInitParams params;
    params.model_id      = 0;
    params.model_config_ = model;
    params.py_model      = py::none();
    NormalExecutor executor(params, manager, false);
    bool           reached_model = false;
    executor.setModel(std::make_unique<MockModel>(model.vocab_size, [&](const GptModelInputs& inputs) {
        reached_model = true;
        EXPECT_EQ(inputs.kv_cache_group_tags, (std::vector<std::string>{"second", "first"}));
        throw StopBeforeSampling{};
    }));

    for (const bool invalid_row : {false, true}) {
        SCOPED_TRACE(invalid_row);
        first_dst.fill_(33);
        second_dst.fill_(44);
        reached_model = false;
        auto mapping  = invalid_row ? torch::tensor({0, 1, 2, 2, 1, 2}, torch::kInt32).reshape({2, 3}) :
                                      torch::tensor({0, 1, 2}, torch::kInt32).reshape({1, 3});
        executor.setBatchProcessor(std::make_unique<CopyPayloadProcessor>(model, config, mapping));
        auto query                   = std::make_shared<GenerateInput>();
        query->input_ids             = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config       = std::make_shared<GenerateConfig>();
        query->need_release_resource = false;
        auto stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, ResourceContext{}, nullptr);
        BatchKVCacheResource resource;
        resource.resetBatchSize(1);
        resource.initGroups(config.topologyPtr());
        resource.mutableBlockIds(0, "first").assign({1, 2});
        resource.mutableBlockIds(0, "second").assign({1, 2});
        stream->setKVCache(resource);
        if (invalid_row) {
            EXPECT_ANY_THROW((void)executor.process(ScheduleOutput{{stream}}));
            EXPECT_FALSE(reached_model);
        } else {
            EXPECT_THROW((void)executor.process(ScheduleOutput{{stream}}), StopBeforeSampling);
            EXPECT_TRUE(reached_model);
        }
        runtimeSyncAndCheck();
        EXPECT_TRUE(first_src.cpu().eq(11).all().item<bool>());
        EXPECT_TRUE(second_src.cpu().eq(22).all().item<bool>());
        EXPECT_TRUE(first_dst.cpu().eq(33).all().item<bool>());
        EXPECT_TRUE(second_dst.cpu().eq(invalid_row ? 44 : 22).all().item<bool>());
    }
}

TEST_F(NormalEngineTest, testDecodeWarmupReserveTokensAreConvertedToBlocksAfterAddition) {
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/7, /*reserve_tokens=*/1, /*tokens_per_block=*/8), 1u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/8, /*reserve_tokens=*/1, /*tokens_per_block=*/8), 2u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/7, /*reserve_tokens=*/9, /*tokens_per_block=*/8), 2u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/9, /*reserve_tokens=*/8, /*tokens_per_block=*/8), 3u);
    EXPECT_ANY_THROW(
        NormalEngine::warmUpReservedBlockCount(/*seq_len=*/1, /*reserve_tokens=*/1, /*tokens_per_block=*/0));
}

TEST_F(NormalEngineTest, testWarmUpInputLengthAccountsForReserve) {
    EXPECT_EQ(warmUpInputLength(64, 0), 63u);
    EXPECT_EQ(warmUpInputLength(64, 17), 47u);
    EXPECT_ANY_THROW((void)warmUpInputLength(1, 0));
    EXPECT_ANY_THROW((void)warmUpInputLength(17, 17));
}

TEST_F(NormalEngineTest, testDecodeWarmupUsesWarmupCacheTopology) {
    CustomConfig config;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    runtime_config.warm_up         = true;
    auto params                    = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.pd_sep_config.role_type = RoleType::DECODE;

    bool saw_decode_warmup             = false;
    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<MockModel>(model_config.vocab_size, [&](const GptModelInputs& inputs) {
            if (!inputs.warmup) {
                return;
            }
            saw_decode_warmup = true;
            EXPECT_EQ(inputs.kv_cache_group_tags, (std::vector<std::string>{"full"}));
            ASSERT_TRUE(inputs.kv_cache_block_id.defined());
            ASSERT_TRUE(inputs.kv_cache_kernel_block_id.defined());
            EXPECT_EQ(inputs.kv_cache_block_id.size(0), 1);
            EXPECT_EQ(inputs.kv_cache_kernel_block_id.size(0), 1);
        });
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    auto engine = std::make_shared<NormalEngine>(params, nullptr);

    EXPECT_TRUE(saw_decode_warmup);
}

TEST_F(NormalEngineTest, testCacheManagerInitFailureDoesNotPublishPartialState) {
    ModelConfig model;
    model.num_layers                   = 1;
    model.attn_config.head_num         = 1;
    model.attn_config.kv_head_num      = 1;
    model.attn_config.size_per_head    = 1;
    model.attn_config.tokens_per_block = 1;
    model.kv_cache_spec_descs          = {{{"default", KVCacheSpecType::MultiHeadAttention}}};
    const auto config                  = CacheConfigCreator::createWarmupConfig(model, {});

    auto            previous  = std::make_shared<KVCacheManager>(config, /*warmup=*/true);
    auto            candidate = std::make_shared<KVCacheManager>(config, /*warmup=*/true);
    ResourceContext resource_context;
    resource_context.cache_manager = previous;
    resource_context.role_type     = RoleType::PREFILL;
    int group_num                  = 17;

    EXPECT_ANY_THROW(NormalEngine::initializeAndPublishCacheManager(
        resource_context, group_num, RoleType::DECODE, candidate, [](KVCacheManager&) { return false; }));
    EXPECT_EQ(resource_context.cache_manager, previous);
    EXPECT_EQ(resource_context.role_type, RoleType::PREFILL);
    EXPECT_EQ(group_num, 17);
}

TEST_F(NormalEngineTest, testRejectGenerationPrefillWithSpeculativeBeforeRunnerCreation) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.hw_kernel_config.enable_cuda_graph                        = true;
    params.hw_kernel_config.generation_prefill_capture_token_buckets = {8, 16};
    params.parallelism_config.role_type                              = RoleType::PDFUSION;
    params.pd_sep_config.role_type                                   = RoleType::PDFUSION;

    // Exercise the real constructor, not just the wrapper ownership predicate.
    // Both the configured execution mode and a supplied propose model must
    // reject the combination before warmup, KV allocation or runner creation.
    for (const auto speculative_type : {SP_TYPE_MTP, SP_TYPE_DSPARK, SP_TYPE_NONE}) {
        for (const bool has_propose_model : {false, true}) {
            if (speculative_type == SP_TYPE_NONE && !has_propose_model) {
                continue;
            }
            SCOPED_TRACE(::testing::Message() << "speculative_type=" << static_cast<int>(speculative_type)
                                              << " has_propose_model=" << has_propose_model);
            params.sp_config.type = speculative_type;
            auto propose_params =
                has_propose_model ? std::make_unique<ProposeModelEngineInitParams>(SP_TYPE_MTP, 2) : nullptr;
            try {
                NormalEngine engine(params, std::move(propose_params));
                FAIL() << "explicit generation-prefill/speculative combination must fail initialization";
            } catch (const std::exception& error) {
                EXPECT_NE(std::string(error.what())
                              .find("GENERATION_PREFILL_CAPTURE_CONFIG does not support speculative execution"),
                          std::string::npos)
                    << error.what();
            }
        }
    }
}

TEST_F(NormalEngineTest, testRejectMismatchedSpeculativeProposalBeforeWarmup) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.runtime_config.warm_up = false;

    const auto check_error = [&](SpeculativeType config_type,
                                 int64_t         config_gamma,
                                 SpeculativeType proposal_type,
                                 size_t          proposal_gamma,
                                 const char*     expected) {
        params.sp_config.type              = config_type;
        params.sp_config.gen_num_per_cycle = config_gamma;
        try {
            NormalEngine engine(params, std::make_unique<ProposeModelEngineInitParams>(proposal_type, proposal_gamma));
            FAIL() << "mismatched speculative parameters must fail initialization";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find(expected), std::string::npos) << error.what();
        }
    };
    check_error(SP_TYPE_MTP, 2, SP_TYPE_EAGLE, 2, "speculative type mismatch");
    check_error(SP_TYPE_MTP, 2, SP_TYPE_MTP, 3, "speculative gamma mismatch");
    check_error(SP_TYPE_MTP, -1, SP_TYPE_MTP, 2, "speculative gen_num_per_cycle must be non-negative");
}

TEST_F(NormalEngineTest, testPdRolesIgnoreGenerationPrefillWithOrWithoutSpeculativeConfig) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.hw_kernel_config.enable_cuda_graph                        = true;
    params.hw_kernel_config.generation_prefill_capture_token_buckets = {8, 16};
    params.runtime_config.warm_up                                    = false;
    params.sp_config.gen_num_per_cycle                              = 3;
    params.sp_config.sp_dspark_mask_token_id                         = model_config.vocab_size - 1;

    /** Keep the real constructor and complete local proposal parameters; no requests are submitted. */
    NormalExecutor::test_model_factory = [vocab = model_config.vocab_size](const GptModelInitParams&) {
        return std::make_unique<MockModel>(vocab);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    for (const auto role_type : {RoleType::PREFILL, RoleType::DECODE}) {
        for (const auto speculative_type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_DSPARK}) {
            SCOPED_TRACE(::testing::Message() << "role_type=" << static_cast<int>(role_type)
                                              << " speculative_type=" << static_cast<int>(speculative_type));
            params.parallelism_config.role_type = role_type;
            params.pd_sep_config.role_type      = role_type;
            params.sp_config.type               = speculative_type;
            auto proposal = speculative_type == SP_TYPE_NONE ? nullptr : makeLocalProposalParams(params);
            NormalEngine engine(params, std::move(proposal));
            if (speculative_type == SP_TYPE_NONE) {
                ASSERT_NE(dynamic_cast<NormalExecutor*>(engine.executor_.get()), nullptr);
            } else {
                ASSERT_NE(dynamic_cast<MtpExecutor*>(engine.executor_.get()), nullptr);
            }
            const int expected_reserve =
                speculative_type == SP_TYPE_NONE ? 0 : speculative_type == SP_TYPE_DSPARK ? 9 : 4;
            EXPECT_EQ(engine.reserve_step_, expected_reserve);

            std::list<GenerateStreamPtr> streams;
            engine.mayAddFakeStream(streams);
            ASSERT_EQ(streams.size(), 1u);
            EXPECT_TRUE(streams.front()->isFakeStream());
            EXPECT_EQ(streams.front()->isContextStream(), role_type == RoleType::PREFILL);
            if (role_type == RoleType::DECODE && speculative_type != SP_TYPE_NONE) {
                const auto& buffer = streams.front()->getSPOutputBuffer();
                ASSERT_NE(buffer, nullptr);
                EXPECT_EQ(buffer->propose_step, 3);
            }
        }
    }
}

TEST_F(NormalEngineTest, testPrefillWarmUpUsesCachelessSingleInput) {
    CustomConfig config;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    runtime_config.warm_up = true;
    auto params            = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);

    const KVCacheSpecDesc default_desc{"default", KVCacheSpecType::MultiHeadAttention};
    KVCacheSpecDesc       indexer_desc{"indexer_kv", KVCacheSpecType::OpaqueKV};
    indexer_desc.entry_dtype       = DataType::TYPE_UINT8;
    indexer_desc.entry_elems       = 132;  // 128 indexer bytes and one FP32 scale per token.
    indexer_desc.entry_count_mode  = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    indexer_desc.compression_ratio = 1;
    params.model_config_.kv_cache_spec_descs.assign(static_cast<size_t>(params.model_config_.num_layers),
                                                    {default_desc, indexer_desc});

    bool saw_cacheless_warmup          = false;
    NormalExecutor::test_model_factory = [&](const GptModelInitParams& init_params) {
        if (init_params.cache_manager == nullptr) {
            EXPECT_FALSE(init_params.kv_cache_layer_layout.has_value());
            return std::make_unique<MockModel>(model_config.vocab_size, [&](const GptModelInputs& inputs) {
                EXPECT_FALSE(saw_cacheless_warmup);
                saw_cacheless_warmup = true;
                EXPECT_TRUE(inputs.warmup);
                EXPECT_TRUE(inputs.kv_cache_group_tags.empty());
                EXPECT_FALSE(inputs.kv_cache_block_id.defined());
                EXPECT_FALSE(inputs.kv_cache_kernel_block_id.defined());
            });
        }
        return std::make_unique<MockModel>(model_config.vocab_size);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    auto engine = std::make_shared<NormalEngine>(params, nullptr);

    EXPECT_TRUE(saw_cacheless_warmup);
}

#if USING_CUDA
TEST_F(NormalEngineTest, testPpPrefillWarmupStaysCachelessWithGenerationGraphBuckets) {
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
    autil::EnvGuard device_input("RTP_LLM_DEVICE_INPUT", "0");
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.runtime_config.warm_up                                    = true;
    params.runtime_config.fifo_scheduler_config.max_context_batch_size = 2;
    params.parallelism_config.pp_size                               = 2;
    params.parallelism_config.world_size                            = 2;
    params.parallelism_config.pp_stage_layer_counts                 = {1, 1};
    params.parallelism_config.role_type                             = RoleType::PDFUSION;
    params.pd_sep_config.role_type                                  = RoleType::PDFUSION;
    params.hw_kernel_config.enable_cuda_graph                       = true;
    params.hw_kernel_config.generation_prefill_capture_token_buckets = {8, 16};

    for (int stage : {0, 1}) {
        SCOPED_TRACE(stage);
        params.parallelism_config.pp_rank    = stage;
        params.parallelism_config.world_rank = stage;
        PPWarmupFactoryGuard factories;
        bool                 saw_warmup = false;
        PPExecutor::test_model_factory = [&](const GptModelInitParams& init) {
            EXPECT_EQ(init.cache_manager, nullptr);
            EXPECT_FALSE(init.kv_cache_layer_layout.has_value());
            return std::make_unique<PPWarmupProbeModel>(stage, [&](const GptModelInputs& input) {
                saw_warmup = true;
                EXPECT_TRUE(input.warmup);
                EXPECT_TRUE(input.kv_cache_group_tags.empty());
                EXPECT_FALSE(input.kv_cache_block_id.defined());
                EXPECT_FALSE(input.kv_cache_kernel_block_id.defined());
                EXPECT_EQ(input.sequence_lengths.numel(), 0);
                EXPECT_TRUE(input.input_lengths.eq(19).all().item<bool>());
            });
        };
        EXPECT_THROW((void)std::make_unique<NormalEngine>(params, nullptr), PPWarmupObserved);
        EXPECT_TRUE(saw_warmup);
        runtimeSyncAndCheck();
    }
}

TEST_F(NormalEngineTest, testPpDecodeWarmupUsesStageCacheAndSpeculativeHeadroom) {
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
    autil::EnvGuard device_input("RTP_LLM_DEVICE_INPUT", "0");
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.runtime_config.warm_up                 = true;
    params.runtime_config.max_generate_batch_size = 2;
    params.model_config_.max_seq_len              = 64;
    params.model_config_.kv_cache_spec_descs = {
        {{"first", KVCacheSpecType::MultiHeadAttention}},
        {{"last", KVCacheSpecType::MultiHeadAttention}},
    };
    params.parallelism_config.pp_size              = 2;
    params.parallelism_config.world_size           = 2;
    params.parallelism_config.pp_stage_layer_counts = {1, 1};
    params.parallelism_config.role_type             = RoleType::DECODE;
    params.pd_sep_config.role_type                  = RoleType::DECODE;
    params.sp_config.gen_num_per_cycle              = 3;
    params.sp_config.sp_dspark_mask_token_id         = 99;

    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (int stage : {0, 1}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << ", stage=" << stage);
            params.sp_config.type                 = type;
            params.parallelism_config.pp_rank    = stage;
            params.parallelism_config.world_rank = stage;
            const std::string tag                  = stage == 0 ? "first" : "last";
            const int         warmup_length        = type == SP_TYPE_NONE ? 63 : type == SP_TYPE_DSPARK ? 55 : 57;
            std::unique_ptr<ProposeModelEngineInitParams> proposal;
            if (stage == 1 && type != SP_TYPE_NONE) {
                auto modules = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
                modules->push_back(std::make_unique<EngineInitParams>(params));
                proposal = std::make_unique<ProposeModelEngineInitParams>(type, 3, std::move(modules));
            }

            PPWarmupFactoryGuard factories;
            bool                 saw_warmup = false;
            PPExecutor::test_model_factory = [&](const GptModelInitParams& init) {
                EXPECT_NE(init.cache_manager, nullptr);
                EXPECT_TRUE(init.kv_cache_layer_layout.has_value());
                if (init.cache_manager) {
                    const auto& cache = init.cache_manager->cacheConfig();
                    EXPECT_EQ(cache.layer_num, 1u);
                    EXPECT_EQ(cache.global_layer_begin, static_cast<uint32_t>(stage));
                    EXPECT_EQ(cache.groupTags(), (std::vector<std::string>{tag}));
                    EXPECT_EQ(cache.group(tag).block_num, 2u);
                }
                return std::make_unique<PPWarmupProbeModel>(stage, [&](const GptModelInputs& input) {
                    saw_warmup = true;
                    EXPECT_TRUE(input.warmup);
                    EXPECT_EQ(input.kv_cache_group_tags, (std::vector<std::string>{tag}));
                    ASSERT_GT(input.sequence_lengths.numel(), 0);
                    EXPECT_TRUE(input.input_lengths.eq(warmup_length).all().item<bool>());
                    EXPECT_TRUE(input.sequence_lengths.eq(warmup_length - 1).all().item<bool>());
                    ASSERT_TRUE(input.kv_cache_block_id.defined());
                    ASSERT_TRUE(input.kv_cache_kernel_block_id.defined());
                    EXPECT_EQ(input.kv_cache_block_id.sizes().vec(),
                              (std::vector<int64_t>{1, input.input_lengths.numel(), 32}));
                    EXPECT_EQ(input.kv_cache_kernel_block_id.sizes().vec(), input.kv_cache_block_id.sizes().vec());
                    EXPECT_TRUE(input.kv_cache_block_id.eq(0).all().item<bool>());
                    EXPECT_TRUE(input.kv_cache_kernel_block_id.eq(0).all().item<bool>());
                });
            };
            EXPECT_THROW((void)std::make_unique<NormalEngine>(params, std::move(proposal)), PPWarmupObserved);
            EXPECT_TRUE(saw_warmup);
            runtimeSyncAndCheck();
        }
    }
}
#endif

TEST_F(NormalEngineTest, testFp8KVCache) {
    CustomConfig config;
    config.kv_cache_data_type = DataType::TYPE_FP8_E4M3;
    auto engine               = createMockEngine(config);

    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 5;
    query->generate_config->is_streaming   = false;

    shared_ptr<GenerateStream> stream = engine->enqueue(query);

    ASSERT_TRUE(stream != nullptr);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

    ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    auto output2 = stream->nextOutput();
    ASSERT_TRUE(!output2.ok());
}

TEST_F(NormalEngineTest, testSimple) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_FALSE(engine->resourceContext().system_prompt);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);

    // test streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 3;
        query->generate_config->is_streaming   = true;
        query->generate_config->gen_timeline   = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.iter_count, 1);

        auto output2 = stream->nextOutput();
        ASSERT_TRUE(output2.ok());
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.output_len, 2);
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.iter_count, 2);

        auto output3 = stream->nextOutput();
        ASSERT_TRUE(output3.ok());
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.output_len, 3);
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.iter_count, 3);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output4 = stream->nextOutput();
        ASSERT_TRUE(!output4.ok());
    }

    // test non-streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 5;
        query->generate_config->is_streaming   = false;

        shared_ptr<GenerateStream> stream = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testParallelDispatchMultipleRequests) {
    CustomConfig config;
    config.output_dispatcher_worker_count = 2;
    auto engine                           = createMockEngine(config);

    auto make_query = [](int input_token) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({input_token}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 2;
        query->generate_config->is_streaming   = false;
        return query;
    };

    auto [enqueue_successes, streams] = engine->enqueueMultiple({make_query(1), make_query(2)});
    ASSERT_EQ(enqueue_successes, (std::vector<bool>{true, true}));
    ASSERT_EQ(streams.size(), 2u);
    for (auto& stream : streams) {
        ASSERT_NE(stream, nullptr);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs.size(), 1u);
        EXPECT_EQ(output.value().generate_outputs[0].aux_info.input_len, 1);
        EXPECT_EQ(output.value().generate_outputs[0].aux_info.output_len, 2);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    }

    engine.reset();
}

TEST_F(NormalEngineTest, testSystemPrompt) {
    CustomConfig config;
    vector<int>  prompt_1           = {1, 2, 3};
    vector<int>  prompt_2           = {4, 5, 6, 7, 8, 9};
    config.multi_task_prompt_tokens = {{"1", prompt_1}, {"2", prompt_2}};
    config.reuse_cache              = false;
    config.enable_device_cache      = false;
    auto engine                     = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_TRUE(engine->resourceContext().system_prompt);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    ASSERT_TRUE(engine->resourceContext().enable_device_cache);
    const auto block_tree_cache = engine->resourceContext().cache_manager->blockTreeCache();
    ASSERT_NE(block_tree_cache, nullptr);
    ASSERT_TRUE(block_tree_cache->config().enable_device_cache);

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 2);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({10, 20, 30, 40, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({10, 20, 30, 40, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->task_id        = "2";
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testReuseCacheOption) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    config.reuse_cache = false;
    auto engine2       = createMockEngine(config);
    ASSERT_FALSE(engine2->resourceContext().reuse_cache);
}

TEST_F(NormalEngineTest, testReuseCache) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testQueryReuseCacheWhenSwitchIsOn) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    // First query with reuse_cache = true
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Second query with reuse_cache = false (should not reuse cache)
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = false;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Third query with reuse_cache = true (should reuse cache)
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);  // Should be 4 because reuse_cache = true
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testQueryReuseCacheWhenSwitchIsOff) {
    // Test with engine-level reuse_cache = false (master switch off)
    CustomConfig config;
    config.reuse_cache = false;
    auto engine        = createMockEngine(config);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);

    // Query with reuse_cache = true, but should be ignored because engine-level is false
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;  // This should be ignored
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because engine-level reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Query with reuse_cache = false, should also result in no cache reuse
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = false;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because engine-level reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithPrefillCP) {
    CustomConfig config;
    config.output_vocab_ids   = {0, 2, 7};
    config.prefill_cp_enabled = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithDeviceInput) {
    CustomConfig config;
    config.output_vocab_ids = {0, 2, 7};
    setenv("RTP_LLM_DEVICE_INPUT", "1", 1);
    EXPECT_THROW(createMockEngine(config), std::exception);
    unsetenv("RTP_LLM_DEVICE_INPUT");
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithSpeculative) {
    CustomConfig config;
    config.output_vocab_ids    = {0, 2, 7};
    config.speculative_enabled = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithWarmUpWithLoss) {
    CustomConfig config;
    config.output_vocab_ids  = {0, 2, 7};
    config.warm_up_with_loss = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectInvalidOutputVocabIds) {
    CustomConfig unsorted;
    unsorted.output_vocab_ids = {7, 2, 0};
    EXPECT_THROW(createMockEngine(unsorted), std::exception);

    CustomConfig duplicated;
    duplicated.output_vocab_ids = {0, 2, 2, 7};
    EXPECT_THROW(createMockEngine(duplicated), std::exception);

    CustomConfig out_of_range;
    out_of_range.output_vocab_ids = {0, 2, 100};  // vocab_size is 100
    EXPECT_THROW(createMockEngine(out_of_range), std::exception);
}

TEST_F(NormalEngineTest, testAllowsUnsupportedCombosWithoutOutputVocab) {
    CustomConfig config;
    config.speculative_enabled = true;
    config.prefill_cp_enabled = true;
    config.warm_up_with_loss  = true;
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto params           = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.sp_config.type = SP_TYPE_MTP;
    auto proposal         = makeLocalProposalParams(params);
    NormalEngine engine(params, std::move(proposal));
    EXPECT_NE(dynamic_cast<MtpExecutor*>(engine.executor_.get()), nullptr);
}

}  // namespace rtp_llm
