#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/model_rpc/DraftCacheTransferWindow.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {
namespace test {

TEST(DraftCacheTransferWindowTest, RequiresMatchingPeersAndNoDecodeReuse) {
    EXPECT_EQ(negotiateDraftCacheTransferWindow(4096, 4096, false), 4096u);
    EXPECT_EQ(negotiateDraftCacheTransferWindow(0, 4096, false), 0u);
    EXPECT_EQ(negotiateDraftCacheTransferWindow(4096, 0, false), 0u);
    EXPECT_EQ(negotiateDraftCacheTransferWindow(4096, 8192, false), 0u);
    EXPECT_EQ(negotiateDraftCacheTransferWindow(4096, 4096, true), 0u);
}

TEST(DraftCacheTransferWindowTest, LegacyWireDefaultsKeepFullHistory) {
    GenerateRequestPB request;
    request.set_request_id(123);
    GenerateRequestPB decoded;
    ASSERT_TRUE(decoded.ParseFromString(request.SerializeAsString()));
    EXPECT_EQ(decoded.draft_cache_window_tokens(), 0u);
    GenerateOutputsPB response;
    EXPECT_EQ(response.draft_cache_window_tokens(), 0u);
    BroadcastLoadRequestPB load;
    EXPECT_EQ(load.draft_cache_window_tokens(), 0u);
    EXPECT_EQ(load.draft_cache_context_tokens(), 0u);
}

TEST(DraftCacheTransferWindowTest, WireAgreementAndBroadcastPreserveAbsoluteWindow) {
    GenerateRequestPB offered;
    offered.set_draft_cache_window_tokens(4096);
    GenerateRequestPB decoded;
    ASSERT_TRUE(decoded.ParseFromString(offered.SerializeAsString()));
    GenerateOutputsPB response;
    response.set_draft_cache_window_tokens(
        negotiateDraftCacheTransferWindow(decoded.draft_cache_window_tokens(), 4096, false));
    GenerateOutputsPB echoed;
    ASSERT_TRUE(echoed.ParseFromString(response.SerializeAsString()));
    ASSERT_EQ(echoed.draft_cache_window_tokens(), offered.draft_cache_window_tokens());
    for (size_t context : {1u, 127u, 128u, 129u, 4095u, 4096u, 4097u, 51000u, 74011u, 100000u}) {
        BroadcastLoadRequestPB load;
        load.set_draft_cache_window_tokens(echoed.draft_cache_window_tokens());
        load.set_draft_cache_context_tokens(context);
        BroadcastLoadRequestPB remote;
        ASSERT_TRUE(remote.ParseFromString(load.SerializeAsString()));
        const auto range =
            cacheTransferPageRange(remote.draft_cache_context_tokens(), remote.draft_cache_window_tokens(), 128);
        std::vector<int> counts(range.end - range.begin, 0);
        for (int rank = 0; rank < 4; ++rank) {
            for (const auto& pair : buildFullCacheStoreBlockPlanForWindow(context, 4096, 128, rank, 4)) {
                ASSERT_GE(pair.key_index, static_cast<int>(range.begin));
                ASSERT_LT(pair.key_index, static_cast<int>(range.end));
                EXPECT_EQ(pair.offset_index, pair.key_index / 4);
                ++counts[pair.key_index - range.begin];
            }
        }
        for (int count : counts)
            EXPECT_EQ(count, 1);
    }
}

}  // namespace test
}  // namespace rtp_llm
