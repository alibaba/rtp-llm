#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/RequestPrefixResource.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {

std::shared_ptr<const CacheTopology>
makeTopology(size_t first_span, size_t second_span, std::string first_tag, std::string second_tag, bool reverse) {
    auto      first_spec  = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, first_span, first_tag);
    auto      second_spec = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, second_span, second_tag);
    auto      first       = makeTestGroupBase(first_spec, defaultCacheGroupPolicy(CacheGroupType::FULL), {0});
    auto      second      = makeTestGroupBase(second_spec, defaultCacheGroupPolicy(CacheGroupType::FULL), {0});
    LayerBase layer{
        0, reverse ? std::vector<std::string>{second_tag, first_tag} : std::vector<std::string>{first_tag, second_tag}};
    return CacheTopology::create(
        reverse ? std::vector<GroupBase>{second, first} : std::vector<GroupBase>{first, second}, {layer});
}

TEST(RequestPrefixResourceTest, CanonicalHashesIgnoreTagsAndTopologyOrder) {
    std::vector<int32_t> tokens(37);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<int32_t>(i * 7 + 3);
    }
    RequestPrefixResource lhs;
    lhs.configure(*makeTopology(4, 6, "full-a", "full-b", false));
    lhs.rebuild(tokens.data(), tokens.size());
    RequestPrefixResource rhs;
    rhs.configure(*makeTopology(4, 6, "renamed-x", "renamed-y", true));
    rhs.rebuild(tokens.data(), tokens.size());

    EXPECT_EQ(lhs.matchSpanTokens(), 12u);
    EXPECT_EQ(lhs.keys(), rhs.keys());
    EXPECT_EQ(lhs.tokenExtent(), 37u);
    EXPECT_EQ(lhs.matchLimitTokens(), 36u);
    EXPECT_EQ(lhs.writeLimitTokens(), 36u);
    ASSERT_EQ(lhs.keys().size(), 4u);
    const auto keys_with_partial = lhs.keys();

    tokens.push_back(999);
    lhs.rebuild(tokens.data(), tokens.size());
    EXPECT_EQ(lhs.keys().size(), 4u);
    EXPECT_EQ(lhs.tokenExtent(), 38u);
    EXPECT_EQ(std::vector<RequestPrefixKey>(lhs.keys().begin(), lhs.keys().begin() + 3),
              std::vector<RequestPrefixKey>(keys_with_partial.begin(), keys_with_partial.begin() + 3));
    EXPECT_NE(lhs.keys().back(), keys_with_partial.back());
}

TEST(RequestPrefixResourceTest, TierReuseTokensAreAlignedAndBoundedByMatchLimit) {
    std::vector<int32_t>  tokens(37, 1);
    RequestPrefixResource prefix;
    prefix.configure(*makeTopology(4, 6, "full-a", "full-b", false));
    prefix.rebuild(tokens.data(), tokens.size());

    prefix.setDeviceReuseTokens(12);
    prefix.setMemoryReuseTokens(12);
    prefix.setRemoteReuseTokens(12);
    EXPECT_EQ(prefix.reuseTokens(), 36u);

    EXPECT_ANY_THROW(prefix.setDeviceReuseTokens(13));
    EXPECT_ANY_THROW(prefix.setMemoryReuseTokens(24));
    EXPECT_EQ(prefix.deviceReuseTokens(), 12u);
    EXPECT_EQ(prefix.memoryReuseTokens(), 12u);
    EXPECT_EQ(prefix.remoteReuseTokens(), 12u);

    prefix.setMemoryReuseTokens(0);
    EXPECT_EQ(prefix.reuseTokens(), 24u);
}

TEST(RequestPrefixResourceTest, TierUpdatesAndCopiesShareOneSynchronizedSnapshot) {
    auto                  topology = makeTopology(4, 4, "full-a", "full-b", false);
    std::vector<int32_t>  tokens(13, 1);
    RequestPrefixResource prefix;
    prefix.configure(*topology);
    prefix.rebuild(tokens.data(), tokens.size());
    prefix.setDeviceReuseTokens(4);

    std::atomic<bool> start{false};
    std::thread       memory_writer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int i = 0; i < 1000; ++i) {
            prefix.setMemoryReuseTokens(4);
        }
    });
    std::thread       remote_writer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int i = 0; i < 1000; ++i) {
            prefix.setRemoteReuseTokens(4);
        }
    });
    start.store(true, std::memory_order_release);
    memory_writer.join();
    remote_writer.join();

    RequestPrefixResource copy(prefix);
    EXPECT_EQ(copy.keys(), prefix.keys());
    EXPECT_EQ(copy.deviceReuseTokens(), 4u);
    EXPECT_EQ(copy.memoryReuseTokens(), 4u);
    EXPECT_EQ(copy.remoteReuseTokens(), 4u);
    EXPECT_EQ(copy.reuseTokens(), 12u);
}

}  // namespace
}  // namespace rtp_llm::test
