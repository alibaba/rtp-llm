#include "rtp_llm/cpp/cache/test/KVCacheManagerWithTierCacheTestBase.h"

namespace rtp_llm::test {
using namespace tier_cache_test_detail;

INSTANTIATE_TEST_SUITE_P(TierLayouts,
                         KVCacheManagerWithTierCacheTest,
                         ::testing::Values(TierLayout::HOST_ONLY, TierLayout::HOST_DISK),
                         [](const ::testing::TestParamInfo<TierLayout>& info) { return layoutName(info.param); });

}  // namespace rtp_llm::test
