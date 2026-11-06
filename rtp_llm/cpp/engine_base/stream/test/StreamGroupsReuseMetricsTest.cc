#include "gtest/gtest.h"

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

#include <list>
#include <memory>
#include <vector>

using namespace std;

namespace rtp_llm {

namespace {

GenerateStreamPtr makeReuseContextStream(const std::vector<int32_t>& input_ids, int reuse_length) {
    auto generate_input       = std::make_shared<GenerateInput>();
    generate_input->input_ids = torch::tensor(input_ids, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 1024;

    ResourceContext resource_context;
    auto stream = std::make_shared<NormalGenerateStream>(
        generate_input, model_config, runtime_config, resource_context, nullptr);
    // Simulates the allocator-published prefix cache hit for this request
    // (StreamCacheResource::publishReuseLengths sets all reuse lengths at once).
    stream->setReuseLength(reuse_length);
    stream->setInitialReuseLength(reuse_length);
    return stream;
}

}  // namespace

// Regression test for per-request reuse accounting in wall TPS with cache:
// a chunked (multi-step) context request must contribute its KV-reused prefix
// exactly ONCE to context_execute_token_size_with_cache_ and to the per-priority
// context_with_cache accounting, not once per executor step.
class StreamGroupsReuseMetricsTest: public DeviceTestBase {
protected:
    std::list<GenerateStreamPtr> oneStreamGroup(GenerateStreamPtr stream) {
        return {stream};
    }
};

TEST_F(StreamGroupsReuseMetricsTest, MultiStepRequestCountsReuseOnceInWallMetrics) {
    const std::vector<int32_t> input_ids(200, 1);
    const int                  reuse_length = 100;
    auto                       stream       = makeReuseContextStream(input_ids, reuse_length);

    // Chunked execution: the same stream is re-grouped on consecutive steps.
    StreamGroups step1(oneStreamGroup(stream));
    StreamGroups step2(oneStreamGroup(stream));
    StreamGroups step3(oneStreamGroup(stream));

    const auto execute1 = step1.contextExecuteTokenSize();
    const auto execute2 = step2.contextExecuteTokenSize();
    const auto execute3 = step3.contextExecuteTokenSize();
    const auto batch    = stream->currentBatchSize();

    const auto with_cache1 = step1.contextExecuteTokenSizeWithCache();
    const auto with_cache2 = step2.contextExecuteTokenSizeWithCache();
    const auto with_cache3 = step3.contextExecuteTokenSizeWithCache();
    const auto total       = with_cache1 + with_cache2 + with_cache3;

    // The reused prefix must be counted exactly once across all steps.
    EXPECT_EQ(total, execute1 + execute2 + execute3 + static_cast<size_t>(reuse_length) * batch);
    // Only the first step carries the reuse contribution.
    EXPECT_EQ(with_cache1, execute1 + static_cast<size_t>(reuse_length) * batch);
    EXPECT_EQ(with_cache2, execute2);
    EXPECT_EQ(with_cache3, execute3);
}

TEST_F(StreamGroupsReuseMetricsTest, MultiStepRequestCountsReuseOnceInPriorityMetrics) {
    const std::vector<int32_t> input_ids(200, 1);
    const int                  reuse_length = 100;
    auto                       stream       = makeReuseContextStream(input_ids, reuse_length);

    StreamGroups step1(oneStreamGroup(stream));
    StreamGroups step2(oneStreamGroup(stream));

    const auto counts1 = step1.tokenCountsByPriority();
    const auto counts2 = step2.tokenCountsByPriority();
    ASSERT_EQ(counts1.size(), 1u);
    ASSERT_EQ(counts2.size(), 1u);
    const auto& c1 = counts1.begin()->second;
    const auto& c2 = counts2.begin()->second;

    const auto batch = stream->currentBatchSize();
    EXPECT_EQ(c1.context_with_cache, c1.context + static_cast<int64_t>(reuse_length) * batch);
    EXPECT_EQ(c2.context_with_cache, c2.context);
    // Priority accounting stays independent from the aggregate-path guard.
    StreamGroups step3(oneStreamGroup(stream));
    const auto  counts3 = step3.tokenCountsByPriority();
    EXPECT_EQ(counts3.begin()->second.context_with_cache, counts3.begin()->second.context);
}

TEST_F(StreamGroupsReuseMetricsTest, SingleStepRequestKeepsExactWithCacheTokens) {
    const std::vector<int32_t> input_ids(64, 1);
    const int                  reuse_length = 48;
    auto                       stream       = makeReuseContextStream(input_ids, reuse_length);

    StreamGroups step1(oneStreamGroup(stream));
    const auto   execute = step1.contextExecuteTokenSize();
    const auto   batch   = stream->currentBatchSize();
    EXPECT_EQ(step1.contextExecuteTokenSizeWithCache(),
              execute + static_cast<size_t>(reuse_length) * batch);
}

TEST_F(StreamGroupsReuseMetricsTest, GuardResetReArmsReuseCountingForRetries) {
    const std::vector<int32_t> input_ids(200, 1);
    const int                  reuse_length = 100;
    auto                       stream       = makeReuseContextStream(input_ids, reuse_length);

    StreamGroups step1(oneStreamGroup(stream));
    EXPECT_EQ(step1.contextExecuteTokenSizeWithCache(),
              step1.contextExecuteTokenSize() + static_cast<size_t>(reuse_length) * stream->currentBatchSize());

    // After step 1 the reuse is already counted...
    StreamGroups step2(oneStreamGroup(stream));
    EXPECT_EQ(step2.contextExecuteTokenSizeWithCache(), step2.contextExecuteTokenSize());

    // ...and a retry (clearCacheReuseState resets the guards together with the
    // reuse lengths) counts the re-published prefix exactly once again.
    stream->resetReuseMetricsGuards();
    StreamGroups retried(oneStreamGroup(stream));
    EXPECT_EQ(retried.contextExecuteTokenSizeWithCache(),
              retried.contextExecuteTokenSize() + static_cast<size_t>(reuse_length) * stream->currentBatchSize());
}

TEST_F(StreamGroupsReuseMetricsTest, ZeroReuseStreamNeverAddsCacheTokens) {
    const std::vector<int32_t> input_ids(200, 1);
    auto                       stream = makeReuseContextStream(input_ids, 0);

    StreamGroups step1(oneStreamGroup(stream));
    StreamGroups step2(oneStreamGroup(stream));
    EXPECT_EQ(step1.contextExecuteTokenSizeWithCache(), step1.contextExecuteTokenSize());
    EXPECT_EQ(step2.contextExecuteTokenSizeWithCache(), step2.contextExecuteTokenSize());
    const auto counts = step1.tokenCountsByPriority();
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.begin()->second.context_with_cache, counts.begin()->second.context);
}

}  // namespace rtp_llm
