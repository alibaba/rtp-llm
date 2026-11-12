// Coverage for the SM120 decode startup guards.
//
// These cases cover the sm120_a2a_decode eager-only CUDA-graph rejection and the
// missing-phase MTP fused padding, exercised in a sibling target with the
// file-local fixture, leaving EngineTest.cc unchanged:
//   * NormalEngineTest.testSm120DecodeRejectsCudaGraphBeforeWarmup
//   * NormalEngineTest.testSm120DecodeRejectsNonMtpFusedProfileBeforeWarmup
//   * NormalEngineTest.testMtpFusedStepPadsMissingPhases
//
// The fixture (NormalEngineTest, createMockEngine) is file-local in
// EngineTest.cc, so this translation unit includes it unchanged.

#include "EngineTest.cc"

namespace rtp_llm {

TEST_F(NormalEngineTest, testSm120DecodeRejectsCudaGraphBeforeWarmup) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);

    // DECODE role: a resolved sm120_a2a_decode backend with CUDA graphs enabled
    // must fail in the constructor, before warmup or graph capture.
    params.model_config_.model_type           = "deepseek_v4";
    params.moe_config.moe_strategy            = "sm120_a2a_decode";
    params.parallelism_config.role_type       = RoleType::DECODE;
    params.pd_sep_config.role_type            = RoleType::DECODE;
    params.parallelism_config.pp_size         = 1;
    params.parallelism_config.tp_size         = 1;
    params.parallelism_config.dp_size         = 4;
    params.parallelism_config.ep_size         = 4;
    params.parallelism_config.world_size      = 4;
    params.hw_kernel_config.enable_cuda_graph = true;
    params.runtime_config.warm_up             = true;

    try {
        NormalEngine engine(params, nullptr);
        FAIL() << "sm120_a2a_decode with CUDA graphs must fail initialization";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("sm120_a2a_decode"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("eager execution"), std::string::npos) << error.what();
    }

    // Speculative (MTP) PDFUSION passes the phase-policy check and must still
    // be rejected by the eager-only check before warmup.
    params.parallelism_config.role_type = RoleType::PDFUSION;
    params.pd_sep_config.role_type      = RoleType::PDFUSION;
    params.sp_config.type               = SP_TYPE_MTP;

    try {
        NormalEngine engine(params, nullptr);
        FAIL() << "speculative sm120_a2a_decode PDFUSION with CUDA graphs must fail initialization";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("sm120_a2a_decode"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("eager execution"), std::string::npos) << error.what();
    }
}

TEST_F(NormalEngineTest, testSm120DecodeRejectsNonMtpFusedProfileBeforeWarmup) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);

    // The exact retired profile: resolved sm120_a2a_decode, non-speculative
    // PDFUSION with PP1/TP1 and DP=EP=world>1, no local CP, no FFN
    // disaggregation. Rejection must happen in the constructor, before
    // warmup or any CUDA-graph decision.
    params.model_config_.model_type      = "deepseek_v4";
    params.moe_config.moe_strategy       = "sm120_a2a_decode";
    params.parallelism_config.role_type  = RoleType::PDFUSION;
    params.pd_sep_config.role_type       = RoleType::PDFUSION;
    params.parallelism_config.pp_size    = 1;
    params.parallelism_config.tp_size    = 1;
    params.parallelism_config.dp_size    = 4;
    params.parallelism_config.ep_size    = 4;
    params.parallelism_config.world_size = 4;
    params.sp_config.type                = SP_TYPE_NONE;
    params.runtime_config.warm_up        = true;

    for (const bool enable_cuda_graph : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "enable_cuda_graph=" << enable_cuda_graph);
        params.hw_kernel_config.enable_cuda_graph = enable_cuda_graph;
        try {
            NormalEngine engine(params, nullptr);
            FAIL() << "non-speculative sm120_a2a_decode PDFUSION must fail initialization";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("sm120_a2a_decode"), std::string::npos) << error.what();
        }
    }
}

TEST_F(NormalEngineTest, testMtpFusedStepPadsMissingPhases) {
    auto engine = createMockEngine(CustomConfig{});
    // Stop joins the engine and releases the executor while retaining the
    // resource context; makeStream/mayAddFakeStream only need the retained
    // cache manager and configs.
    ASSERT_TRUE(engine->stop().ok());
    ASSERT_FALSE(engine->running_);
    ASSERT_EQ(engine->executor_, nullptr);
    ASSERT_NE(engine->resource_context_.cache_manager, nullptr);

    // Install an active-MTP proposal only inside this test. The guard restores
    // the original proposal and configs before the engine is destroyed.
    struct EngineStateGuard {
        NormalEngine*                                 engine;
        std::unique_ptr<ProposeModelEngineInitParams> saved_propose;
        SpeculativeExecutionConfig                    saved_sp;
        ParallelismConfig                             saved_parallelism;
        PDSepConfig                                   saved_pd_sep;
        RuntimeConfig                                 saved_runtime;
        ~EngineStateGuard() {
            engine->sp_config          = saved_sp;
            engine->parallelism_config = saved_parallelism;
            engine->pd_sep_config      = saved_pd_sep;
            engine->runtime_config     = saved_runtime;
            engine->propose_params_    = std::move(saved_propose);
        }
    } guard{engine.get(),
            std::move(engine->propose_params_),
            engine->sp_config,
            engine->parallelism_config,
            engine->pd_sep_config,
            engine->runtime_config};

    ModelConfig   draft_model_config;
    RuntimeConfig draft_runtime_config;
    KVCacheConfig draft_kv_cache_config;
    auto          draft_params =
        createEngineInitParams(CustomConfig{}, draft_model_config, draft_runtime_config, draft_kv_cache_config);
    auto mtp_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_params->push_back(std::make_unique<EngineInitParams>(std::move(draft_params)));
    engine->propose_params_ = std::make_unique<ProposeModelEngineInitParams>(SP_TYPE_MTP, 2, std::move(mtp_params));
    engine->sp_config.type  = SP_TYPE_MTP;
    engine->sp_config.gen_num_per_cycle = 2;
    engine->pd_sep_config.role_type     = RoleType::PDFUSION;

    const int64_t hidden_width = engine->model_config_.hidden_size * engine->model_config_.hc_mult;

    auto makeRealStream = [&](bool context) {
        auto input                             = make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
        input->generate_config                 = make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 8;
        auto stream                            = engine->makeStream(input);
        EXPECT_NE(stream, nullptr);
        EXPECT_FALSE(stream->hasError());
        stream->setIsContextStream(context);
        return stream;
    };

    for (const int64_t dp_size : {int64_t{1}, int64_t{4}}) {
        for (const bool bds : {false, true}) {
            engine->parallelism_config.dp_size                = dp_size;
            engine->runtime_config.use_batch_decode_scheduler = bds;
            for (const bool has_prefill : {false, true}) {
                for (const bool has_decode : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << "dp_size=" << dp_size << " bds=" << bds << " has_prefill="
                                                      << has_prefill << " has_decode=" << has_decode);
                    std::list<GenerateStreamPtr>               streams;
                    std::vector<GenerateStreamPtr>             originals;
                    std::vector<std::weak_ptr<GenerateStream>> weak_refs;
                    if (has_prefill) {
                        originals.push_back(makeRealStream(true));
                    }
                    if (has_decode) {
                        originals.push_back(makeRealStream(false));
                    }
                    for (const auto& stream : originals) {
                        streams.push_back(stream);
                    }

                    engine->mayAddFakeStream(streams);

                    // Original streams keep their order at the head of the list.
                    auto it = streams.begin();
                    for (const auto& original : originals) {
                        ASSERT_NE(it, streams.end());
                        EXPECT_EQ(it->get(), original.get());
                        ++it;
                    }
                    const size_t expect_prefill_fake = (!has_prefill && (!bds || dp_size > 1)) ? 1 : 0;
                    const size_t expect_decode_fake  = has_decode ? 0 : 1;
                    ASSERT_EQ(std::distance(it, streams.end()),
                              static_cast<long>(expect_prefill_fake + expect_decode_fake));

                    bool saw_prefill_fake = false;
                    bool saw_decode_fake  = false;
                    for (; it != streams.end(); ++it) {
                        const auto& fake = *it;
                        ASSERT_TRUE(fake->isFakeStream());
                        weak_refs.push_back(fake);
                        if (fake->isContextStream()) {
                            saw_prefill_fake = true;
                            EXPECT_EQ(fake->generateInput()->generate_config->max_new_tokens, 1);
                        } else {
                            saw_decode_fake = true;
                            EXPECT_EQ(fake->generateInput()->generate_config->max_new_tokens,
                                      engine->sp_config.gen_num_per_cycle);
                            const auto& mtp_state = fake->getMtpAsyncDeviceState();
                            ASSERT_TRUE(mtp_state.last_hidden_states_gpu.defined());
                            EXPECT_EQ(mtp_state.last_hidden_states_gpu.size(1), hidden_width);
                        }
                    }
                    EXPECT_EQ(saw_prefill_fake, expect_prefill_fake == 1);
                    EXPECT_EQ(saw_decode_fake, expect_decode_fake == 1);
                    if (dp_size == 4 && !has_prefill && !has_decode) {
                        EXPECT_TRUE(saw_prefill_fake);
                        EXPECT_TRUE(saw_decode_fake);
                    }

                    for (const auto& original : originals) {
                        weak_refs.push_back(original);
                    }
                    streams.clear();
                    originals.clear();
                    for (const auto& ref : weak_refs) {
                        EXPECT_TRUE(ref.expired());
                    }
                    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
                }
            }
        }
    }
}

}  // namespace rtp_llm
