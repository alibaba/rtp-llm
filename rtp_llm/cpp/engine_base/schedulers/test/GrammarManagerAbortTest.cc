#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "torch/all.h"
#include "gtest/gtest.h"

#include <xgrammar/tokenizer_info.h>

#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/engine_base/schedulers/GrammarManager.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class GrammarManagerAbortTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        cache_config_ = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/8, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(cache_manager_->init());
    }

    GenerateStreamPtr createStream() {
        ResourceContext ctx;
        ctx.cache_manager = cache_manager_;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        auto query             = std::make_shared<GenerateInput>();
        auto generate_config   = std::make_shared<GenerateConfig>();
        query->input_ids       = torch::tensor(std::vector<int>{1, 2, 3}, torch::kInt32);
        query->generate_config = generate_config;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, ctx, nullptr);
    }

    // Push a fully-formed GrammarEntry into the queue. Mimics what
    // processReqWithGrammar would do, minus the backend round-trip — we run
    // the manager in disabled mode (backend == nullptr) so no worker thread
    // is spawned and no real compile is attempted.
    void injectEntry(GrammarManager& mgr, const GenerateStreamPtr& stream, const std::string& kid_suffix = "x") {
        GrammarManager::GrammarEntry entry;
        entry.stream            = stream;
        entry.key               = {"json", "{\"k\":\"" + kid_suffix + "\"}"};
        entry.require_reasoning = false;
        entry.deadline          = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        // Leave entry.future invalid — abortAll never reads it.

        std::lock_guard<std::mutex> lock(mgr.queue_mutex_);
        // Match the ref_count invariant maintained by processReqWithGrammar
        // so future readers of the manager's internal state see something
        // self-consistent — even though abortAll itself just clears.
        auto& slot = mgr.in_flight_[entry.key.id()];
        slot.ref_count++;
        mgr.grammar_queue_.emplace_back(std::move(entry));
    }

protected:
    CacheConfig                     cache_config_;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

TEST_F(GrammarManagerAbortTest, AbortAllDrainsQueueAndReportsCancelled) {
    GrammarManager mgr;  // disabled mode — no workers, no backend

    std::vector<GenerateStreamPtr> streams;
    for (int i = 0; i < 3; ++i) {
        auto s = createStream();
        ASSERT_FALSE(s->hasError());
        streams.push_back(s);
        injectEntry(mgr, s, std::to_string(i));
    }
    ASSERT_EQ(mgr.size(), 3u);

    mgr.abortAll();

    EXPECT_EQ(mgr.size(), 0u);
    {
        std::lock_guard<std::mutex> lock(mgr.queue_mutex_);
        EXPECT_TRUE(mgr.compile_tasks_.empty());
        EXPECT_TRUE(mgr.in_flight_.empty());
    }
    for (auto& s : streams) {
        EXPECT_TRUE(s->hasError());
        EXPECT_EQ(s->stopReason(), "scheduler stopped");
    }
}

TEST_F(GrammarManagerAbortTest, AbortAllOnEmptyQueueIsNoOp) {
    GrammarManager mgr;
    ASSERT_EQ(mgr.size(), 0u);
    mgr.abortAll();  // must not crash
    EXPECT_EQ(mgr.size(), 0u);
}

TEST_F(GrammarManagerAbortTest, AbortAllClearsCompileTasksAndInFlight) {
    GrammarManager mgr;
    auto           s = createStream();
    injectEntry(mgr, s, "only");

    // Also seed a synthetic CompileTask so we can verify it's evicted. Its
    // promise is never set — that's fine; abortAll just clears the deque.
    {
        std::lock_guard<std::mutex> lock(mgr.queue_mutex_);
        GrammarManager::CompileTask task;
        task.key               = {"json", "{\"k\":\"only\"}"};
        task.require_reasoning = false;
        mgr.compile_tasks_.emplace_back(std::move(task));
    }

    mgr.abortAll();

    std::lock_guard<std::mutex> lock(mgr.queue_mutex_);
    EXPECT_TRUE(mgr.grammar_queue_.empty());
    EXPECT_TRUE(mgr.compile_tasks_.empty());
    EXPECT_TRUE(mgr.in_flight_.empty());
    EXPECT_TRUE(s->hasError());
}

TEST_F(GrammarManagerAbortTest, AbortAllSkipsNullStreamEntry) {
    GrammarManager mgr;
    {
        std::lock_guard<std::mutex>  lock(mgr.queue_mutex_);
        GrammarManager::GrammarEntry entry;
        entry.stream   = nullptr;  // would never happen via the public API
        entry.key      = {"json", "{}"};
        entry.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        mgr.grammar_queue_.emplace_back(std::move(entry));
    }
    mgr.abortAll();  // must not deref the null stream
    EXPECT_EQ(mgr.size(), 0u);
}

// Minimal XGrammarBackendCpp built on a 128-char ASCII vocab — same trick
// XGrammarBackendCppTest uses. We never need the backend to actually compile,
// just to exist so getCachedInvalid() is callable from the test thread.
namespace {

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

class GrammarManagerTimeoutNoPoisonTest: public GrammarManagerAbortTest {
protected:
    void SetUp() override {
        GrammarManagerAbortTest::SetUp();
        backend_ = std::make_shared<XGrammarBackendCpp>(makeAsciiTokenizerInfoJson(),
                                                        backendOptionsForTest());
    }

    // Push a GrammarEntry whose deadline is already in the past and whose
    // future will never resolve. pollAndDrainLocked must classify it as
    // "failed (timeout)" on the next poll.
    void injectTimedOutEntry(GrammarManager&          mgr,
                             const GenerateStreamPtr& stream,
                             const GrammarKeyCpp&     key) {
        GrammarManager::GrammarEntry entry;
        entry.stream            = stream;
        entry.key               = key;
        entry.require_reasoning = false;
        // Deadline already passed → next pollAndDrainLocked routes to `failed`.
        entry.deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);

        std::lock_guard<std::mutex> lock(mgr.queue_mutex_);
        auto&                       slot = mgr.in_flight_[key.id()];
        slot.ref_count++;
        mgr.grammar_queue_.emplace_back(std::move(entry));
    }

    std::shared_ptr<XGrammarBackendCpp> backend_;
};

// Regression: a grammar timeout must NOT poison the invalid_cache. The next
// request with the same key must still be allowed to attempt a fresh compile.
TEST_F(GrammarManagerTimeoutNoPoisonTest, TimeoutDoesNotPoisonInvalidCache) {
    GrammarManager mgr(backend_, GrammarConfig{});
    auto           stream = createStream();
    GrammarKeyCpp  key{"json", "{\"type\":\"object\"}"};

    // Sanity: nothing in the invalid cache yet.
    ASSERT_TRUE(backend_->getCachedInvalid(key).empty());

    injectTimedOutEntry(mgr, stream, key);
    auto returned = mgr.getReadyGrammarRequests();

    // The timed-out stream is surfaced once, with GENERATE_TIMEOUT.
    ASSERT_EQ(returned.size(), 1u);
    EXPECT_EQ(returned.front().get(), stream.get());
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->stopReason(), "Grammar preprocessing timed out");

    // The cardinal assertion: the cache must remain unpoisoned so the next
    // identical request can retry instead of being permanently rejected.
    EXPECT_TRUE(backend_->getCachedInvalid(key).empty());
}

}  // namespace rtp_llm
