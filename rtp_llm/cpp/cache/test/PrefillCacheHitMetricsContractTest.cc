#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kmonitor/client/core/MetricsRecord.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/metrics/ServiceStatus.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::test {
namespace {

CompleteTokenIdsPtr makeMetricTokens(int seq_len) {
    auto  ids    = std::make_shared<CompleteTokenIds>(1, 1, seq_len + 8, 4);
    auto  tensor = torch::empty({seq_len}, torch::kInt32);
    auto* data   = tensor.data_ptr<int32_t>();
    for (int i = 0; i < seq_len; ++i) {
        data[i] = i + 1;
    }
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = tensor;
    input->generate_config = std::make_shared<GenerateConfig>();
    ids->init(input);
    return ids;
}

std::optional<double> readGauge(kmonitor::MutableMetric& gauge, const kmonitor::MetricsReporter& reporter) {
    const auto tags = kmonTagsWithServiceStatus(&reporter.getTags());
    auto* metric = gauge.DeclareMetric(&tags);
    EXPECT_NE(metric, nullptr);
    if (metric == nullptr) {
        return std::nullopt;
    }
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    metric->Snapshot(&record, 1000);
    gauge.UndeclareMetric(metric);
    if (record.Values().empty()) {
        return std::nullopt;
    }
    EXPECT_EQ(record.Values().size(), 1u);
    return std::stod(record.Values().front()->Value());
}

class PrefillCacheHitMetricsContractTest: public testing::Test {
protected:
    void SetUp() override {
        startKmonServiceStatus(nullptr);
        setKmonServiceServing(true);
    }

    void TearDown() override {
        stopKmonServiceStatus();
    }
};

}  // namespace

TEST(KVCacheHashUtilTest, ExactEndpointUsesBaseSegmentsSeedAndImageIdentity) {
    auto make_tokens = [](char digest_char) {
        auto input               = std::make_shared<GenerateInput>();
        input->input_ids         = torch::arange(16, torch::kInt32);
        input->generate_config   = std::make_shared<GenerateConfig>();
        auto          multimodal = std::make_shared<V41RequestInputs>();
        V41ImageInput image;
        image.start              = 4;
        image.types              = torch::zeros({8}, torch::kInt32);
        image.n_vit_h            = 2;
        image.n_vit_w            = 2;
        image.content_sha256     = std::string(64, digest_char);
        image.processor_identity = std::string(64, 'p');
        multimodal->images.push_back(image);
        input->v41_inputs = multimodal;
        auto tokens       = std::make_shared<CompleteTokenIds>(1, 1, 32, 4);
        tokens->init(input);
        return tokens;
    };
    auto tokens   = make_tokens('a');
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    auto              config = makeSimpleMhaCacheConfig(1, 16, 4, DataType::TYPE_FP16);
    ParallelismConfig parallelism;
    config.finalizeStorageLayout(parallelism);
    resource->initGroups(config.topologyPtr());
    constexpr int64_t seed = -77;
    initCacheKeys(resource, tokens, 4, seed);
    ASSERT_EQ(resource->cacheKeys(0).size(), 4u);
    for (int endpoint : {4, 8, 12, 16}) {
        EXPECT_EQ(cacheKeyAtTokenEnd(*tokens, 0, 4, endpoint, seed), resource->cacheKeys(0)[endpoint / 4 - 1]);
    }
    for (int endpoint : {0, 3, 4, 7, 8, 10, 12, 16}) {
        EXPECT_EQ(cacheKeyAtTokenEnd(*tokens, 0, 4, endpoint, resource->cacheKeys(0), seed),
                  cacheKeyAtTokenEnd(*tokens, 0, 4, endpoint, seed));
    }
    int64_t    expected = hashInt64Array(resource->cacheKeys(0)[1], tokens->data(0) + 8, tokens->data(0) + 10);
    const auto identity = tokens->imageCacheIdentity(8, 2);
    expected            = hashInt64Array(expected, identity.data(), identity.data() + identity.size());
    EXPECT_EQ(cacheKeyAtTokenEnd(*tokens, 0, 4, 10, seed), expected);
    EXPECT_NE(cacheKeyAtTokenEnd(*tokens, 0, 4, 10, seed), cacheKeyAtTokenEnd(*make_tokens('b'), 0, 4, 10, seed));
    const auto             all_tokens = tokens->completeTokenIdsVec(0);
    const std::vector<int> source_segment(all_tokens.begin() + 8, all_tokens.end());
    EXPECT_EQ(cacheKeyForSourceSegment(*tokens, source_segment, 8, 4, resource->cacheKeys(0)[1]),
              resource->cacheKeys(0).back());
    EXPECT_NE(cacheKeyForSourceSegment(*make_tokens('b'), source_segment, 8, 4, resource->cacheKeys(0)[1]),
              resource->cacheKeys(0).back());
    for (int begin : {0, 4, 8}) {
        const std::vector<int> source(all_tokens.begin() + begin, all_tokens.end());
        EXPECT_EQ(cacheKeyForSourceSegment(*tokens, source, begin, 4, resource->cacheKeys(0), seed),
                  resource->cacheKeys(0).back());
    }
    tokens->data(0)[9] = 999;
    EXPECT_EQ(cacheKeyForSourceSegment(*tokens, source_segment, 8, 4, resource->cacheKeys(0), seed),
              resource->cacheKeys(0).back());
    EXPECT_NE(cacheKeyAtTokenEnd(*tokens, 0, 4, 16, seed), resource->cacheKeys(0).back());
    EXPECT_EQ(tokens->maxValidReuseLength(10), 4);
    EXPECT_EQ(tokens->maxValidReuseLength(16, 6), 4);
    EXPECT_EQ(tokens->maxValidReuseLength(16), 16);
    EXPECT_EQ(tokens->maxValidReuseLength(-1), 0);
}

TEST(KVCacheHashUtilTest, InteriorReuseBoundaryHandlesOverlappingAndMissingImageIdentity) {
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::arange(32, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    auto multimodal        = std::make_shared<V41RequestInputs>();
    for (const int start : {4, 8}) {
        V41ImageInput image;
        image.start              = start;
        image.types              = torch::zeros({8}, torch::kInt32);
        image.content_sha256     = std::string(64, 'a');
        image.processor_identity = std::string(64, 'p');
        multimodal->images.push_back(image);
    }
    input->v41_inputs = multimodal;
    CompleteTokenIds tokens(1, 1, 32, 4);
    tokens.init(input);
    EXPECT_EQ(tokens.maxValidReuseLength(14), 4);
    EXPECT_TRUE(tokens.isValidReuseLength(tokens.maxValidReuseLength(14)));
    EXPECT_EQ(tokens.maxValidReuseLength(16), 16);
    multimodal->images[1].content_sha256.clear();
    EXPECT_EQ(tokens.maxValidReuseLength(32), 4);
    EXPECT_TRUE(tokens.isValidReuseLength(tokens.maxValidReuseLength(32)));
}

TEST(KVCacheHashUtilTest, IncrementalPrefillKeepsSeventhPartialBaseKeyForPD) {
    constexpr int     base_tokens = 256;
    constexpr int64_t seed        = -77;
    auto              input       = std::make_shared<GenerateInput>();
    input->input_ids              = torch::arange(1300, torch::kInt32);
    input->generate_config        = std::make_shared<GenerateConfig>();
    auto tokens                   = std::make_shared<CompleteTokenIds>(1, 1, 2048, base_tokens);
    tokens->init(input);

    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    auto              config = makeSimpleMhaCacheConfig(1, 32, base_tokens, DataType::TYPE_FP16);
    ParallelismConfig parallelism;
    config.finalizeStorageLayout(parallelism);
    resource->initGroups(config.topologyPtr());
    initCacheKeys(resource, tokens, base_tokens, seed);
    ASSERT_EQ(resource->cacheKeys(0).size(), 6u);
    EXPECT_FALSE(resource->lastBlockAligned());

    int previous_end = 1300;
    for (int token_end : {1301, 1320, 1536, 1544}) {
        for (int pos = previous_end; pos < token_end; ++pos) {
            tokens->data(0)[pos] = pos;
        }
        tokens->setSeqLength(token_end);
        updateCacheKeys(resource, tokens, base_tokens, seed);
        const auto& keys = resource->cacheKeys(0);
        ASSERT_EQ(keys.size(), static_cast<size_t>((token_end + base_tokens - 1) / base_tokens));
        EXPECT_EQ(resource->lastBlockAligned(), token_end % base_tokens == 0);
        for (size_t index = 0; index < keys.size(); ++index) {
            const int endpoint = std::min(token_end, static_cast<int>((index + 1) * base_tokens));
            EXPECT_EQ(keys[index], cacheKeyAtTokenEnd(*tokens, 0, base_tokens, endpoint, seed));
        }
        previous_end = token_end;
    }
    ASSERT_EQ(resource->cacheKeys(0).size(), 7u);
    EXPECT_EQ(resource->cacheKeys(0)[6], cacheKeyAtTokenEnd(*tokens, 0, base_tokens, 1544, seed));
    dropLastPartialBlock(resource);
    EXPECT_TRUE(resource->lastBlockAligned());
    EXPECT_EQ(resource->cacheKeys(0).size(), 6u);
}

TEST_F(PrefillCacheHitMetricsContractTest, FixedTheoryOutputAndFirstAttemptRoleGate) {
    initLogger();
    createDevice();

    for (int cp_size : {1, 2, 4}) {
        auto              config = makeSimpleMhaCacheConfig(1, 20, 4, DataType::TYPE_FP16);
        ParallelismConfig parallelism;
        parallelism.tp_size                            = cp_size;
        parallelism.tp_rank                            = 0;
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        parallelism.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
        config.finalizeStorageLayout(parallelism);
        BatchKVCacheResource resource;
        resource.resetBatchSize(1);
        resource.initGroups(config.topologyPtr());
        resource.setBatchCacheKeys(0, {10, 11, 12, 13, 14});
        resource.setLastBlockAligned(false);
        CacheKeysType full_theory_keys = resource.cacheKeys(0);
        if (!resource.lastBlockAligned() && !full_theory_keys.empty()) {
            full_theory_keys.pop_back();
        }
        const auto   canonical_theory_keys = config.canonicalCacheKeys(full_theory_keys);
        const size_t key_count             = cp_size == 1 ? 4u : (cp_size == 2 ? 2u : 1u);
        ASSERT_EQ(canonical_theory_keys.size(), key_count);

        auto monitor = std::make_shared<kmonitor::MetricsReporter>(
            "", "prefill_cp" + std::to_string(cp_size), kmonitor::MetricsTags{});
        auto* gauges = monitor->getMetricsGroup<PrefillRecentCacheKeyMetrics>();
        ASSERT_NE(gauges, nullptr);
        PrefillCacheHitMetricsReporter reporter(monitor);
        const int64_t first_id = 100 + cp_size;
        reporter.record(resource, first_id, 20, 4, &canonical_theory_keys, config.canonicalSizePerBlock());
        EXPECT_EQ(readGauge(*gauges->hit_count_metric, *monitor), 0.0);
        EXPECT_EQ(readGauge(*gauges->total_count_metric, *monitor), 20.0);
        EXPECT_EQ(readGauge(*gauges->theory_all_hit_count_metric, *monitor), 0.0);
        EXPECT_EQ(readGauge(*gauges->theory_all_total_count_metric, *monitor), 20.0);

        reporter.record(resource, first_id + 100, 20, 4, &canonical_theory_keys, config.canonicalSizePerBlock());
        EXPECT_EQ(readGauge(*gauges->hit_count_metric, *monitor), 16.0);
        EXPECT_EQ(readGauge(*gauges->total_count_metric, *monitor), 20.0);
        EXPECT_EQ(readGauge(*gauges->theory_all_hit_count_metric, *monitor), 16.0);
        EXPECT_EQ(readGauge(*gauges->theory_all_total_count_metric, *monitor), 40.0);
    }

    auto              config = makeSimpleMhaCacheConfig(1, 10, 4, DataType::TYPE_FP16);
    ParallelismConfig parallelism;
    config.finalizeStorageLayout(parallelism);
    PDSepConfig prefill;
    prefill.role_type = RoleType::PREFILL;
    PDSepConfig decode;
    decode.role_type     = RoleType::DECODE;
    auto monitor = std::make_shared<kmonitor::MetricsReporter>("", "prefill_role", kmonitor::MetricsTags{});
    auto* gauges = monitor->getMetricsGroup<PrefillRecentCacheKeyMetrics>();
    ASSERT_NE(gauges, nullptr);
    auto prefill_manager = std::make_shared<KVCacheManager>(
        config, true, monitor, KVCacheConfig{}, parallelism, RuntimeConfig{}, SpeculativeExecutionConfig{}, prefill);
    auto decode_manager = std::make_shared<KVCacheManager>(
        config, true, monitor, KVCacheConfig{}, parallelism, RuntimeConfig{}, SpeculativeExecutionConfig{}, decode);
    ASSERT_TRUE(prefill_manager->needsPrefillHitSnapshot());
    ASSERT_FALSE(decode_manager->needsPrefillHitSnapshot());
    ASSERT_TRUE(prefill_manager->init());
    ASSERT_TRUE(decode_manager->init());

    auto holder = std::make_shared<BatchKVCacheResource>();
    holder->resetBatchSize(1);
    holder->initGroups(config.topologyPtr());
    auto       holder_tokens = makeMetricTokens(20);
    MallocInfo holder_info{holder, holder_tokens};
    holder_info.reuse_cache         = false;
    holder_info.enable_cache_lookup = false;
    ASSERT_TRUE(prefill_manager->malloc(holder_info).success);
    ASSERT_EQ(holder->curBlocksNum(), 5);
    EXPECT_EQ(readGauge(*gauges->theory_all_total_count_metric, *monitor), 20.0);

    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(config.topologyPtr());
    auto       tokens = makeMetricTokens(20);
    MallocInfo info{resource, tokens};
    info.request_id   = 999;
    const auto first  = prefill_manager->malloc(info);
    EXPECT_FALSE(first.success);
    EXPECT_EQ(first.status, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(resource->curBlocksNum(), 0);
    EXPECT_TRUE(resource->cacheKeysInitialized());
    EXPECT_EQ(readGauge(*gauges->total_count_metric, *monitor), 20.0);
    EXPECT_EQ(readGauge(*gauges->theory_all_total_count_metric, *monitor), 40.0);

    prefill_manager->free(FreeInfo{holder, holder_tokens});
    const auto retry = prefill_manager->malloc(info);
    EXPECT_TRUE(retry.success);
    EXPECT_EQ(retry.status, MallocStatus::NONE);
    EXPECT_EQ(readGauge(*gauges->theory_all_total_count_metric, *monitor), std::nullopt);
    auto decode_resource = std::make_shared<BatchKVCacheResource>();
    decode_resource->resetBatchSize(1);
    decode_resource->initGroups(config.topologyPtr());
    MallocInfo decode_info{decode_resource, tokens};
    EXPECT_TRUE(decode_manager->malloc(decode_info).success);
    EXPECT_TRUE(decode_resource->cacheKeysInitialized());
    EXPECT_EQ(readGauge(*gauges->theory_all_total_count_metric, *monitor), std::nullopt);
}

TEST_F(PrefillCacheHitMetricsContractTest, ShardedMetricRequiresResolvedTheoryWindow) {
    initLogger();
    createDevice();
    auto              config = makeSimpleMhaCacheConfig(1, 20, 4, DataType::TYPE_FP16);
    ParallelismConfig parallelism;
    parallelism.tp_size                            = 2;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    parallelism.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    config.finalizeStorageLayout(parallelism);
    PDSepConfig prefill;
    prefill.role_type = RoleType::PREFILL;
    auto manager      = std::make_shared<KVCacheManager>(
        config, true, nullptr, KVCacheConfig{}, parallelism, RuntimeConfig{}, SpeculativeExecutionConfig{}, prefill);
    ASSERT_TRUE(manager->init());
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(config.topologyPtr());
    auto       tokens = makeMetricTokens(8);
    MallocInfo info{resource, tokens};
    EXPECT_TRUE(manager->malloc(info).success);
}

}  // namespace rtp_llm::test
