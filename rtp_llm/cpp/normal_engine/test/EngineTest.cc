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
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <limits>
#include <memory>
#include <pybind11/embed.h>

namespace py = pybind11;

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

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
            EXPECT_ANY_THROW((void)executor.process({stream}));
            EXPECT_FALSE(reached_model);
        } else {
            EXPECT_THROW((void)executor.process({stream}), StopBeforeSampling);
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
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/0, /*reserve_tokens=*/0, /*tokens_per_block=*/8), 0u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/8, /*reserve_tokens=*/0, /*tokens_per_block=*/8), 1u);
    EXPECT_ANY_THROW(
        NormalEngine::warmUpReservedBlockCount(/*seq_len=*/1, /*reserve_tokens=*/1, /*tokens_per_block=*/0));
    EXPECT_ANY_THROW(NormalEngine::warmUpReservedBlockCount(
        std::numeric_limits<size_t>::max(), /*reserve_tokens=*/1, /*tokens_per_block=*/8));
}

TEST(NormalEnginePolicyTest, testBatchDecodeSchedulerCannotSuppressMultiDpMtpFakePrefill) {
    EXPECT_FALSE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/true, /*use_batch_decode_scheduler=*/true, /*dp_size=*/2));
    EXPECT_FALSE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/true, /*dp_size=*/1));
    EXPECT_TRUE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/true, /*dp_size=*/2));
    EXPECT_TRUE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/false, /*dp_size=*/1));
}

TEST(NormalEnginePolicyTest, testPrefillOnlyStreamDoesNotReserveSpeculativeSteps) {
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/4, /*is_prefill_only=*/true), 0);
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/4, /*is_prefill_only=*/false), 4);
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/0, /*is_prefill_only=*/true), 0);
}

TEST_F(NormalEngineTest, testMakeStreamReservesKVBeforeEnqueue) {
    struct ReserveCase {
        const char* name;
        int         engine_reserve;
        int         prompt_length;
        bool        prefill_only;
        size_t      expected_blocks;
    };
    const ReserveCase cases[] = {
        {"none_boundary", 0, 8, false, 1},
        {"none_cross_boundary", 0, 9, false, 2},
        {"MTP_boundary", 5, 3, false, 1},
        {"MTP_cross_boundary", 5, 4, false, 2},
        {"DSpARK", 12, 7, false, 3},
        {"prefill_only_boundary", 12, 8, true, 1},
        {"prefill_only_cross_boundary", 12, 9, true, 2},
    };

    for (bool prefill_only : {false, true}) {
        SCOPED_TRACE(testing::Message() << "prefill_only=" << prefill_only);
        CustomConfig config;
        config.role_type        = prefill_only ? RoleType::PREFILL : RoleType::DECODE;
        config.tokens_per_block = 8;
        auto engine             = createMockEngine(config);
        ASSERT_TRUE(engine->stop().ok());
        ASSERT_EQ(engine->getCacheManager()->cacheConfig().seq_size_per_block, 8u);

        for (const auto& test_case : cases) {
            if (test_case.prefill_only != prefill_only) {
                continue;
            }
            SCOPED_TRACE(test_case.name);
            // Inject the constructor's reserve into the real engine after joining its loop;
            // no speculative executor is needed to exercise pre-enqueue KV allocation.
            engine->reserve_step_                  = test_case.engine_reserve;
            auto query                             = make_shared<GenerateInput>();
            query->input_ids                       = torch::arange(1, test_case.prompt_length + 1, torch::kInt32);
            query->generate_config                 = make_shared<GenerateConfig>();
            query->generate_config->max_new_tokens = prefill_only ? 0 : 1;

            auto stream = engine->makeStream(query);
            ASSERT_NE(stream, nullptr);
            ASSERT_FALSE(stream->hasError());
            ASSERT_EQ(stream->seqLength(), test_case.prompt_length);
            ASSERT_EQ(stream->curBlocksNum(), 0u);
            const size_t expected_reserve = prefill_only ? 0 : test_case.engine_reserve;
            EXPECT_EQ(stream->reserveStep(), expected_reserve);
            EXPECT_EQ(stream->completeTokenIdsPtr()->getReserveStep(), expected_reserve);
            EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

            // Match DecodeRpcServer's allocation before P/D handoff and enqueue.
            const auto status = stream->streamCacheResource().initKVBlock();
            ASSERT_TRUE(status.ok()) << status.ToString();
            EXPECT_EQ(stream->curBlocksNum(), test_case.expected_blocks);
            EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
        }
    }
}

TEST(NormalEnginePolicyTest, testEmptyBatchEarlyReturnPreservesRequiredAlignment) {
    EXPECT_TRUE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/1, /*enable_ffn_disaggregate=*/false));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/1, /*enable_ffn_disaggregate=*/true));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/2, /*enable_ffn_disaggregate=*/false));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/false, /*tp_size=*/1, /*enable_ffn_disaggregate=*/false));
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
    model.attn_config.tokens_per_block        = 1;
    model.attn_config.kernel_tokens_per_block = 1;
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

    // Keep the real NormalEngine constructor and its configuration checks.
    // Only model execution is mocked; no draft model is needed to exercise the
    // retained speculative configuration that previously failed at startup.
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
            EXPECT_NO_THROW({ NormalEngine engine(params, nullptr); });
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

TEST_F(NormalEngineTest, testPrefillOnlyReturnsOneEmptyOutput) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto query                             = make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 0;

    auto       stream       = engine->enqueue(query);
    const auto input_length = stream->seqLength();
    auto       output       = stream->nextOutput();

    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 1);
    EXPECT_TRUE(output.value().generate_outputs[0].finished);
    EXPECT_EQ(output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
    EXPECT_EQ(stream->seqLength(), input_length);
    EXPECT_EQ(stream->outputTokenLen(), 0);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
}

TEST_F(NormalEngineTest, testDecodeRoleRejectsPrefillOnlyBeforeScheduling) {
    CustomConfig config;
    config.role_type = RoleType::DECODE;
    auto engine      = createMockEngine(config);

    auto zero_query                             = make_shared<GenerateInput>();
    zero_query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    zero_query->generate_config                 = make_shared<GenerateConfig>();
    zero_query->generate_config->max_new_tokens = 0;

    auto zero_stream = engine->enqueue(zero_query);
    ASSERT_NE(zero_stream, nullptr);
    EXPECT_TRUE(zero_stream->hasError());
    EXPECT_EQ(zero_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(zero_stream->curBlocksNum(), 0);
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

    auto positive_query                             = make_shared<GenerateInput>();
    positive_query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    positive_query->generate_config                 = make_shared<GenerateConfig>();
    positive_query->generate_config->max_new_tokens = 1;

    auto positive_stream = engine->makeStream(positive_query);
    ASSERT_NE(positive_stream, nullptr);
    EXPECT_FALSE(positive_stream->hasError());
    EXPECT_EQ(positive_stream->curBlocksNum(), 0);
}

TEST_F(NormalEngineTest, testDirectEngineEntriesConvertPrefillValidationErrorsToStreamErrors) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto make_invalid_query = [] {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 0;
        query->generate_config->min_new_tokens = 1;
        return query;
    };

    std::shared_ptr<GenerateStream> stream;
    EXPECT_NO_THROW(stream = engine->enqueue(make_invalid_query()));
    ASSERT_NE(stream, nullptr);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

    auto make_stream_query = make_invalid_query();
    EXPECT_NO_THROW(stream = engine->makeStream(make_stream_query));
    ASSERT_NE(stream, nullptr);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);

    std::vector<std::shared_ptr<GenerateInput>>                  inputs{make_invalid_query(), make_invalid_query()};
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>> result;
    EXPECT_NO_THROW(result = engine->enqueueMultiple(inputs));
    EXPECT_EQ(result.first, std::vector<bool>({false, false}));
    ASSERT_EQ(result.second.size(), 2);
    for (const auto& invalid_stream : result.second) {
        ASSERT_NE(invalid_stream, nullptr);
        EXPECT_TRUE(invalid_stream->hasError());
        EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    }
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
}

TEST_F(NormalEngineTest, testEnqueueMultipleRejectsMixedModesBeforeRoleFiltering) {
    CustomConfig config;
    config.role_type = RoleType::DECODE;
    auto engine      = createMockEngine(config);

    auto make_query = [](int64_t request_id, int64_t group_id, int max_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->request_id                      = request_id;
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        query->group_id                        = group_id;
        query->group_size                      = 2;
        return query;
    };

    for (bool generation_first : {false, true}) {
        const int64_t group_id = 100 + generation_first;
        auto          prefill  = make_query(10 + generation_first, group_id, 0);
        auto          generate = make_query(20 + generation_first, group_id, 1);
        auto          inputs   = generation_first ? std::vector<std::shared_ptr<GenerateInput>>{generate, prefill} :
                                                    std::vector<std::shared_ptr<GenerateInput>>{prefill, generate};

        auto [enqueue_successes, streams] = engine->enqueueMultiple(inputs);

        EXPECT_EQ(enqueue_successes, std::vector<bool>({false, false}));
        ASSERT_EQ(streams.size(), 2);
        for (const auto& stream : streams) {
            EXPECT_TRUE(stream->hasError());
            EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(stream->stopReason(), kMixedForceBatchGroupError);
        }
        EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
    }
}

TEST_F(NormalEngineTest, testEnqueueMultipleIgnoresInvalidStreamForModeCheck) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto make_query = [](int max_new_tokens, int min_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        query->generate_config->min_new_tokens = min_new_tokens;
        return query;
    };

    auto [enqueue_successes, streams] = engine->enqueueMultiple({make_query(0, 1), make_query(1, 0)});

    ASSERT_EQ(streams.size(), 2);
    EXPECT_EQ(enqueue_successes, std::vector<bool>({false, true}));
    EXPECT_TRUE(streams[0]->hasError());
    EXPECT_EQ(streams[0]->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(streams[1]->hasError());
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 1);
}

class NormalEngineCustomOutputTest: public NormalEngineTest {
protected:
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
        py::module_::import("torch");
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    static std::shared_ptr<GenerateInput> makeQuery(int max_new_tokens, int first_token = 1) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({first_token, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        return query;
    }

    // Keep torch and pybind interpreter state alive across all selector cases.
    inline static std::unique_ptr<py::scoped_interpreter> interpreter_;
};

TEST_F(NormalEngineCustomOutputTest, testCustomOutputSelectedOnceBeforeSystemPrefix) {
    CustomConfig config;
    config.multi_task_prompt_tokens = {{"1", {8, 9, 10}}};
    auto engine                     = createMockEngine(config);
    // Join the loop to inspect entry/scheduler state without executing the mock model's logits-only output.
    ASSERT_TRUE(engine->stop().ok());
    engine->reserve_step_           = 4;
    int calls                       = 0;
    engine->custom_output_selector_ = py::cpp_function([&](const torch::Tensor& ids, py::object mask) {
        ++calls;
        EXPECT_EQ(ids.numel(), 4);
        EXPECT_EQ(ids[0].item<int>(), 1);
        EXPECT_TRUE(mask.is_none());
        return 2;
    });
    py::gil_scoped_release release;
    auto                   make_query = [&] {
        auto query                      = makeQuery(2);
        query->generate_config->task_id = "1";
        return query;
    };

    auto stream = engine->makeStream(make_query());
    EXPECT_EQ(calls, 1);
    engine->enqueue(stream);
    EXPECT_EQ(calls, 1);
    auto single = engine->enqueue(make_query());
    EXPECT_EQ(calls, 2);
    auto [successes, batch] = engine->enqueueMultiple({make_query(), make_query()});
    EXPECT_EQ(successes, std::vector<bool>({true, true}));
    ASSERT_EQ(batch.size(), 2);
    EXPECT_EQ(calls, 4);
    for (const auto& selected : {stream, single, batch[0], batch[1]}) {
        ASSERT_FALSE(selected->hasError());
        EXPECT_EQ(selected->reserveStep(), 4u);
        EXPECT_EQ(selected->inputLength(), 7);
        EXPECT_EQ(selected->generateInput()->custom_output_token_position, 5);
        EXPECT_EQ(selected->generateInput()->input_ids[5].item<int>(), 3);
    }
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 4);

    auto fake        = make_query();
    fake->fake_query = true;
    EXPECT_FALSE(engine->makeStream(fake)->hasError());
    EXPECT_EQ(fake->custom_output_token_position, -1);
    EXPECT_EQ(calls, 4);
}

TEST_F(NormalEngineCustomOutputTest, testPrefillOnlySkipsCustomSelectorAndCompletes) {
    auto engine                     = createMockEngine(CustomConfig{});
    int  calls                      = 0;
    engine->custom_output_selector_ = py::cpp_function([&](const torch::Tensor& ids, py::object) {
        ++calls;
        if (ids[0].item<int>() != 7) {
            throw py::value_error("generation prompt requires scoring token");
        }
        return 0;
    });
    py::gil_scoped_release release;

    auto generation = engine->makeStream(makeQuery(1, 7));
    ASSERT_FALSE(generation->hasError());
    EXPECT_EQ(generation->generateInput()->custom_output_token_position, 0);
    auto rejected = engine->enqueue(makeQuery(1));
    ASSERT_TRUE(rejected->hasError());
    EXPECT_EQ(rejected->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
    EXPECT_EQ(calls, 2);

    auto expect_finished = [&](const GenerateStreamPtr& stream) {
        auto output = stream->nextOutput(5000);
        ASSERT_TRUE(output.ok()) << output.status().ToString();
        ASSERT_EQ(output.value().generate_outputs.size(), 1);
        const auto& result = output.value().generate_outputs[0];
        EXPECT_TRUE(result.finished);
        EXPECT_EQ(result.output_ids.sizes(), (torch::IntArrayRef{1, 0}));
        EXPECT_FALSE(result.custom_output.has_value());
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->generateInput()->custom_output_token_position, -1);
        EXPECT_EQ(stream->reserveStep(), 0u);
        EXPECT_EQ(stream->seqLength(), 4);
        EXPECT_EQ(stream->outputTokenLen(), 0);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        EXPECT_FALSE(stream->nextOutput(100).ok());
    };
    expect_finished(engine->enqueue(makeQuery(0)));
    // Failed generation must be filtered before the mixed-mode check, without reordering results.
    const std::vector<std::shared_ptr<GenerateInput>> inputs{makeQuery(0), makeQuery(1), makeQuery(0)};
    auto [successes, batch] = engine->enqueueMultiple(inputs);
    EXPECT_EQ(successes, std::vector<bool>({true, false, true}));
    ASSERT_EQ(batch.size(), inputs.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        EXPECT_EQ(batch[i]->generateInput(), inputs[i]);
        if (i == 1) {
            EXPECT_TRUE(batch[i]->hasError());
            EXPECT_EQ(batch[i]->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(batch[i]->curBlocksNum(), 0);
        } else {
            expect_finished(batch[i]);
        }
    }
    auto invalid                             = makeQuery(0);
    invalid->generate_config->min_new_tokens = 1;
    auto invalid_stream                      = engine->enqueue(invalid);
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(calls, 3);
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
    config.prefill_cp_enabled  = true;
    config.speculative_enabled = true;
    config.warm_up_with_loss   = true;
    EXPECT_NO_THROW(createMockEngine(config));
}

}  // namespace rtp_llm
