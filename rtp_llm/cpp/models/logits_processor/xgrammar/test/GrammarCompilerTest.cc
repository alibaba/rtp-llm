#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "torch/all.h"
#include "gtest/gtest.h"

#include <xgrammar/tokenizer_info.h>

// White-box access to GrammarCompiler / GrammarLogitsProcessor privates is
// provided by the -fno-access-control copt in this test's BUILD target (matching
// XGrammarBackendTest). Do NOT add `#define private public` here.
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarCompiler.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

namespace {

// 128-char ASCII vocab — enough for xgrammar to build the trie and compile a
// JSON grammar. "$$ANY$$" maps to the builtin JSON grammar (see
// XGrammarBackendTest), which compiles reliably against this vocab.
std::string makeAsciiTokenizerInfoJson() {
    std::vector<std::string> vocab;
    vocab.reserve(128);
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    xgrammar::TokenizerInfo info(vocab,
                                 xgrammar::VocabType::RAW,
                                 /*vocab_size=*/128,
                                 /*stop_token_ids=*/std::vector<int32_t>{0});
    return info.SerializeJSON();
}

XGrammarBackendOptions backendOptionsForTest() {
    XGrammarBackendOptions opts;
    opts.any_whitespace        = true;
    opts.strict_mode           = true;
    opts.max_compiler_threads  = 1;
    opts.enable_compiler_cache = true;
    opts.compiler_cache_bytes  = -1;
    return opts;
}

}  // namespace

class GrammarCompilerTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        cache_config_ = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/8, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(cache_manager_->init());
        GrammarCompiler::resetForTest();
    }

    void TearDown() override {
        GrammarCompiler::resetForTest();
        DeviceTestBase::TearDown();
    }

    GenerateStreamPtr createStream() {
        ResourceContext ctx;
        ctx.cache_manager = cache_manager_;
        ModelConfig   model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        auto          query    = std::make_shared<GenerateInput>();
        auto          cfg      = std::make_shared<GenerateConfig>();
        query->input_ids       = torch::tensor(std::vector<int>{1, 2, 3}, torch::kInt32);
        query->generate_config = cfg;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, ctx, nullptr);
    }

    static std::shared_ptr<GenerateInput> makeInput(const std::optional<std::string>& json_schema, int num_beams = 1) {
        auto query             = std::make_shared<GenerateInput>();
        auto cfg               = std::make_shared<GenerateConfig>();
        cfg->json_schema       = json_schema;
        cfg->num_beams         = num_beams;
        query->input_ids       = torch::tensor(std::vector<int>{1, 2, 3}, torch::kInt32);
        query->generate_config = cfg;
        return query;
    }

    GrammarConfig enabledConfig() {
        GrammarConfig cfg;
        cfg.grammar_backend     = "xgrammar";
        cfg.tokenizer_info_json = makeAsciiTokenizerInfoJson();
        cfg.num_workers         = 1;
        return cfg;
    }

    CacheConfig                     cache_config_;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

// ── GrammarCompiler (worker pool + singleflight + cache) ────────────────────

TEST_F(GrammarCompilerTest, SubmitCompilesValidSchemaThenServesFromCache) {
    auto            backend = std::make_shared<XGrammarBackend>(makeAsciiTokenizerInfoJson(), backendOptionsForTest());
    GrammarCompiler compiler(backend, GrammarConfig{});
    ASSERT_TRUE(compiler.enabled());

    GrammarKeyCpp key{"json", "$$ANY$$"};
    auto          payload = compiler.submit(key).get();
    EXPECT_TRUE(payload.compiled);
    EXPECT_FALSE(payload.is_invalid);
    EXPECT_FALSE(payload.cache_hit) << "first compile is not a cache hit";

    // After the worker eagerly caches, a resubmit is served from cache.
    auto cached = compiler.submit(key).get();
    EXPECT_TRUE(cached.compiled);
    EXPECT_TRUE(cached.cache_hit);
    EXPECT_EQ(cached.compile_time_us, 0);
}

TEST_F(GrammarCompilerTest, SubmitMalformedSchemaIsInvalidAndCached) {
    auto            backend = std::make_shared<XGrammarBackend>(makeAsciiTokenizerInfoJson(), backendOptionsForTest());
    GrammarCompiler compiler(backend, GrammarConfig{});

    GrammarKeyCpp key{"json", "{this is not json at all"};
    auto          payload = compiler.submit(key).get();
    EXPECT_FALSE(payload.compiled);
    EXPECT_TRUE(payload.is_invalid);

    // Cached as invalid -> resubmit short-circuits to a ready invalid future.
    auto again = compiler.submit(key).get();
    EXPECT_TRUE(again.is_invalid);
    EXPECT_FALSE(again.compiled);
    EXPECT_FALSE(backend->getCachedInvalid(key).empty());
}

TEST_F(GrammarCompilerTest, DisabledCompilerHasNoBackend) {
    GrammarCompiler compiler(nullptr, GrammarConfig{});
    EXPECT_FALSE(compiler.enabled());
}

TEST_F(GrammarCompilerTest, InstanceBeforeInitializeDoesNotDisableLaterInitialize) {
    EXPECT_FALSE(GrammarCompiler::instance().enabled());

    GrammarCompiler::initialize(enabledConfig());

    EXPECT_TRUE(GrammarCompiler::instance().enabled());
}

TEST_F(GrammarCompilerTest, ReInitSameConfigIsIdempotent) {
    GrammarCompiler::initialize(enabledConfig());
    ASSERT_TRUE(GrammarCompiler::instance().enabled());

    GrammarCompiler::initialize(enabledConfig());

    EXPECT_TRUE(GrammarCompiler::instance().enabled());
}

TEST_F(GrammarCompilerTest, ReInitDifferentConfigKeepsFirst) {
    GrammarCompiler::initialize(enabledConfig());
    ASSERT_TRUE(GrammarCompiler::instance().enabled());

    GrammarConfig other;
    other.grammar_backend     = "none";
    other.tokenizer_info_json = makeAsciiTokenizerInfoJson();
    GrammarCompiler::initialize(other);

    EXPECT_TRUE(GrammarCompiler::instance().enabled())
        << "a later initialize() with a different config must not replace or disable the first backend";
}

// ── GrammarLogitsProcessor pending factory ──────────────────────────────────

TEST_F(GrammarCompilerTest, TryCreateReturnsNullWhenNoStructuredRequest) {
    GrammarCompiler::initialize(enabledConfig());
    auto prep = GrammarLogitsProcessor::tryCreatePending(makeInput(std::nullopt));
    EXPECT_EQ(prep, nullptr) << "non-structured request needs no preparation";
}

TEST_F(GrammarCompilerTest, DisabledBackendFailsStructuredRequestBeforePrefill) {
    GrammarConfig disabled;
    disabled.grammar_backend = "none";  // disabled
    GrammarCompiler::initialize(disabled);

    auto prep = GrammarLogitsProcessor::tryCreatePending(makeInput(R"({"type":"object"})"));
    ASSERT_NE(prep, nullptr);

    auto stream = createStream();
    EXPECT_EQ(prep->prepare(*stream), PrepareState::Failed);
    EXPECT_TRUE(stream->hasError());
}

TEST_F(GrammarCompilerTest, BeamSearchPlusGrammarRejectedBeforePrefill) {
    GrammarCompiler::initialize(enabledConfig());

    auto prep = GrammarLogitsProcessor::tryCreatePending(makeInput(R"({"type":"object"})", /*num_beams=*/2));
    ASSERT_NE(prep, nullptr);

    auto stream = createStream();
    EXPECT_EQ(prep->prepare(*stream), PrepareState::Failed);
    EXPECT_TRUE(stream->hasError());
    EXPECT_NE(stream->stopReason().find("beam search"), std::string::npos);
}

TEST_F(GrammarCompilerTest, ValidSchemaReadyInstallsMatcher) {
    GrammarCompiler::initialize(enabledConfig());

    auto prep = GrammarLogitsProcessor::tryCreatePending(makeInput("$$ANY$$", /*num_beams=*/1));
    ASSERT_NE(prep, nullptr);

    // Mirror production: the factory appends the (pending) processor to the
    // stream's processor list; the stream resolves it via prepare().
    auto stream = createStream();
    stream->addLogitsProcessor(prep);

    PrepareState state    = PrepareState::Pending;
    const auto   deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (state == PrepareState::Pending && std::chrono::steady_clock::now() < deadline) {
        state = prep->prepare(*stream);
        if (state == PrepareState::Pending) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    EXPECT_EQ(state, PrepareState::Ready);
    EXPECT_FALSE(stream->hasError());

    auto* gp = stream->findProcessor<GrammarLogitsProcessor>();
    ASSERT_NE(gp, nullptr) << "the grammar processor stays on the stream";
    EXPECT_NE(gp->grammarMatcher(), nullptr) << "Ready must install the matcher";
    EXPECT_FALSE(gp->needsPreparation()) << "a resolved processor no longer needs preparation";
}

// Regression: a grammar wait timeout must surface as GENERATE_TIMEOUT (NOT an
// invalid-schema rejection) so the next identical request can retry.
TEST_F(GrammarCompilerTest, TimeoutReportsTimeoutNotInvalid) {
    GrammarCompiler::initialize(enabledConfig());

    // Manually build a compile-backed processor whose future never resolves and
    // whose deadline is already in the past. `never_promise` is kept alive for
    // the duration of the test so the future stays unfulfilled (a destroyed
    // promise would otherwise make the future ready with broken_promise).
    std::promise<GrammarReadyPayload> never_promise;

    auto prep                = std::shared_ptr<GrammarLogitsProcessor>(new GrammarLogitsProcessor());
    prep->kind_              = GrammarLogitsProcessor::Kind::Compile;
    prep->key_    = GrammarKeyCpp{"json", "$$ANY$$"};
    prep->future_            = never_promise.get_future().share();
    prep->deadline_          = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);

    auto stream = createStream();
    EXPECT_EQ(prep->prepare(*stream), PrepareState::Failed);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->stopReason(), "Grammar preprocessing timed out");
}

}  // namespace rtp_llm
