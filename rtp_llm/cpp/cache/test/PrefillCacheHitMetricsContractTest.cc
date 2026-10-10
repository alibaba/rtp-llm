#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::test {
namespace {

CompleteTokenIdsPtr makeMetricTokens(int seq_len) {
    auto ids = std::make_shared<CompleteTokenIds>(1, 1, seq_len + 8, 4);
    auto tensor = torch::empty({seq_len}, torch::kInt32);
    auto* data = tensor.data_ptr<int32_t>();
    for (int i = 0; i < seq_len; ++i) {
        data[i] = i + 1;
    }
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = tensor;
    input->generate_config = std::make_shared<GenerateConfig>();
    ids->init(input);
    return ids;
}

std::vector<std::string> readMetricLines(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}

void expectField(const std::string& line, const std::string& field) {
    EXPECT_NE(line.find(field), std::string::npos) << line;
}

}  // namespace

TEST(PrefillCacheHitMetricsContractTest, FixedTheoryOutputAndFirstAttemptRoleGate) {
    initLogger();
    createDevice();
    const std::string log_path = "/tmp/cp07_prefill_metric_" + std::to_string(getpid()) + ".log";
    std::remove(log_path.c_str());
    setenv("PREFILL_CACHE_HIT_METRIC_ENABLE", "1", 1);
    setenv("PREFILL_THEORY_HIT_LOG_ENABLE", "1", 1);
    setenv("PREFILL_THEORY_HIT_LOG_PATH", log_path.c_str(), 1);

    for (int cp_size : {1, 2, 4}) {
        auto config = makeSimpleMhaCacheConfig(1, 20, 4, DataType::TYPE_FP16);
        ParallelismConfig parallelism;
        parallelism.tp_size = cp_size;
        parallelism.tp_rank = 0;
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        config.finalizeStorageLayout(parallelism);
        BatchKVCacheResource resource;
        resource.resetBatchSize(1);
        resource.initGroups(config.topologyPtr());
        resource.setBatchCacheKeys(0, {10, 11, 12, 13, 14});
        resource.setLastBlockAligned(false);
        auto tokens = makeMetricTokens(20);
        const auto resolved = snapshotGlobalCacheRequest(resource, *tokens, -1, true);
        ASSERT_TRUE(resolved->theory_window_keys.has_value());
        ASSERT_EQ(resolved->theory_window_keys->size(), 1u);
        const auto   canonical_theory_keys = config.canonicalCacheKeys(resolved->theory_window_keys->front());
        const size_t key_count = cp_size == 1 ? 4u : (cp_size == 2 ? 2u : 1u);
        ASSERT_EQ(canonical_theory_keys.size(), key_count);

        PrefillCacheHitMetricsReporter reporter(nullptr);
        const int64_t first_id = 100 + cp_size;
        reporter.record(resource, first_id, 20, 4, &canonical_theory_keys, *config.cache_info_block_tokens);
        auto lines = readMetricLines(log_path);
        ASSERT_FALSE(lines.empty());
        expectField(lines.back(), "request_id=" + std::to_string(first_id));
        expectField(lines.back(), "request_hit_tokens=0");
        expectField(lines.back(), "request_input_tokens=20");
        expectField(lines.back(), "all_hit_tokens=0");
        expectField(lines.back(), "all_input_tokens=20");

        reporter.record(resource, first_id + 100, 20, 4, &canonical_theory_keys, *config.cache_info_block_tokens);
        lines = readMetricLines(log_path);
        expectField(lines.back(), "request_id=" + std::to_string(first_id + 100));
        expectField(lines.back(), "request_hit_tokens=16");
        expectField(lines.back(), "request_input_tokens=20");
        expectField(lines.back(), "all_hit_tokens=16");
        expectField(lines.back(), "all_input_tokens=40");
    }

    auto config = makeSimpleMhaCacheConfig(1, 10, 4, DataType::TYPE_FP16);
    ParallelismConfig parallelism;
    config.finalizeStorageLayout(parallelism);
    PDSepConfig prefill;
    prefill.role_type = RoleType::PREFILL;
    PDSepConfig decode;
    decode.role_type = RoleType::DECODE;
    auto prefill_manager = std::make_shared<KVCacheManager>(
        config, true, nullptr, KVCacheConfig{}, parallelism, RuntimeConfig{}, SpeculativeExecutionConfig{}, prefill);
    auto decode_manager = std::make_shared<KVCacheManager>(
        config, true, nullptr, KVCacheConfig{}, parallelism, RuntimeConfig{}, SpeculativeExecutionConfig{}, decode);
    ASSERT_TRUE(prefill_manager->needsPrefillHitSnapshot());
    ASSERT_FALSE(decode_manager->needsPrefillHitSnapshot());
    ASSERT_TRUE(prefill_manager->init());
    ASSERT_TRUE(decode_manager->init());

    auto holder = std::make_shared<BatchKVCacheResource>();
    holder->resetBatchSize(1);
    holder->initGroups(config.topologyPtr());
    auto holder_tokens = makeMetricTokens(20);
    MallocInfo holder_info{holder, holder_tokens};
    holder_info.cache_keys_prepared = true;
    holder_info.reuse_cache = false;
    holder_info.enable_cache_lookup = false;
    ASSERT_TRUE(prefill_manager->malloc(holder_info).success);
    ASSERT_EQ(holder->curBlocksNum(), 5);

    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(config.topologyPtr());
    auto tokens = makeMetricTokens(20);
    initCacheKeys(resource, tokens, 4);
    ASSERT_TRUE(resource->cacheKeysInitialized());
    MallocInfo info{resource, tokens};
    info.request_id = 999;
    info.cache_keys_prepared = true;
    info.cache_keys_initialized_now = true;
    info.request_layout             = snapshotGlobalCacheRequest(*resource, *tokens, -1, true);
    const auto before = readMetricLines(log_path).size();
    const auto first = prefill_manager->malloc(info);
    EXPECT_FALSE(first.success);
    EXPECT_EQ(first.status, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(resource->curBlocksNum(), 0);
    EXPECT_TRUE(resource->cacheKeysInitialized());
    auto lines = readMetricLines(log_path);
    ASSERT_EQ(lines.size(), before + 1);
    expectField(lines.back(), "request_id=999");
    expectField(lines.back(), "request_input_tokens=20");

    prefill_manager->free(FreeInfo{holder, holder_tokens});
    info.cache_keys_initialized_now = false;  // retry of the same initialized request
    const auto retry = prefill_manager->malloc(info);
    EXPECT_TRUE(retry.success);
    EXPECT_EQ(retry.status, MallocStatus::NONE);
    EXPECT_EQ(readMetricLines(log_path).size(), before + 1);
    auto decode_resource = std::make_shared<BatchKVCacheResource>();
    decode_resource->resetBatchSize(1);
    decode_resource->initGroups(config.topologyPtr());
    MallocInfo decode_info{decode_resource, tokens};
    EXPECT_TRUE(decode_manager->malloc(decode_info).success);
    EXPECT_TRUE(decode_resource->cacheKeysInitialized());
    EXPECT_EQ(readMetricLines(log_path).size(), before + 1);
    std::remove(log_path.c_str());
}

TEST(PrefillCacheHitMetricsContractTest, ShardedMetricRequiresResolvedTheoryWindow) {
    initLogger();
    createDevice();
    setenv("PREFILL_CACHE_HIT_METRIC_ENABLE", "1", 1);
    auto config = makeSimpleMhaCacheConfig(1, 20, 4, DataType::TYPE_FP16);
    ParallelismConfig parallelism;
    parallelism.tp_size = 2;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    config.finalizeStorageLayout(parallelism);
    PDSepConfig prefill;
    prefill.role_type = RoleType::PREFILL;
    auto manager = std::make_shared<KVCacheManager>(
        config, true, nullptr, KVCacheConfig{}, parallelism, RuntimeConfig{}, SpeculativeExecutionConfig{}, prefill);
    ASSERT_TRUE(manager->init());
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(config.topologyPtr());
    auto tokens = makeMetricTokens(8);
    MallocInfo info{resource, tokens};
    info.cache_keys_prepared = true;
    info.cache_keys_initialized_now = true;
    EXPECT_THROW((void)manager->malloc(info), std::exception);
}

}  // namespace rtp_llm::test
