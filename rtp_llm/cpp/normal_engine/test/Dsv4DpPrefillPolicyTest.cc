#include "rtp_llm/cpp/normal_engine/Dsv4DpPrefillPolicy.h"

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

struct Stream {
    int  id;
    bool context;
    bool fake;
    bool isContextStream() const {
        return context;
    }
};
using StreamPtr = std::shared_ptr<Stream>;
using Streams   = std::list<StreamPtr>;

StreamPtr makeStream(int id, bool context, bool fake = false) {
    return std::make_shared<Stream>(Stream{id, context, fake});
}

TEST(Dsv4DpPrefillPolicyTest, ProfileGating) {
    for (int world : {2, 4, 8}) {
        SCOPED_TRACE(testing::Message() << "world=" << world);
        EXPECT_TRUE(dsv4DpPrefillPhaseSyncEnabled(
            "deepseek_v4", "sm120_decode", RoleType::PDFUSION, 1, 1, world, world, world, false, false, false));
        for (RoleType role : {RoleType::PREFILL, RoleType::DECODE}) {
            SCOPED_TRACE(testing::Message() << "role=" << role);
            EXPECT_FALSE(dsv4DpPrefillPhaseSyncEnabled(
                "deepseek_v4", "sm120_decode", role, 1, 1, world, world, world, false, false, false));
        }
    }
    auto enabled = [](const char* model,
                      const char* strategy,
                      int         pp,
                      int         tp,
                      int         dp,
                      int         ep,
                      int         world,
                      bool        speculative = false,
                      bool        cp          = false,
                      bool        ffn         = false) {
        return dsv4DpPrefillPhaseSyncEnabled(
            model, strategy, RoleType::PDFUSION, pp, tp, dp, ep, world, speculative, cp, ffn);
    };
    EXPECT_FALSE(enabled("other", "sm120_decode", 1, 1, 4, 4, 4)) << "other models unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", nullptr, 1, 1, 4, 4, 4)) << "unset strategy unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "other", 1, 1, 4, 4, 4)) << "other strategies unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 2, 2, 1, 2, 4)) << "CEP2PP2 unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 2, 4, 1, 4, 8)) << "CEP4PP2 unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 1, 2, 2, 4, 4)) << "TP split unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 1, 1, 1, 1, 1)) << "single rank unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 1, 1, 4, 2, 4)) << "partial EP group unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 1, 1, 4, 4, 8)) << "world mismatch rejected";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 1, 1, 4, 4, 4, true)) << "speculation unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 1, 1, 4, 4, 4, false, true)) << "local CP unchanged";
    EXPECT_FALSE(enabled("deepseek_v4", "sm120_decode", 1, 1, 4, 4, 4, false, false, true))
        << "FFN disaggregation unchanged";
}

class Dsv4DpPrefillPhaseTest: public testing::TestWithParam<int> {};

TEST_P(Dsv4DpPrefillPhaseTest, AlignsEveryContextPlacement) {
    const int world = GetParam();
    SCOPED_TRACE(testing::Message() << "world=" << world);
    // Enumerate empty, decode-only, context-only and mixed scheduler outputs.
    int configurations = 1;
    for (int rank = 0; rank < world; ++rank) {
        configurations *= 4;
    }
    for (int mask = 0; mask < configurations; ++mask) {
        SCOPED_TRACE(testing::Message() << "mask=" << mask);
        std::vector<Streams>   scheduled(world);
        std::vector<StreamPtr> decodes(world);
        std::vector<bool>      local_context(world);
        bool                   any_context = false;
        int                    code        = mask;
        for (int rank = 0; rank < world; ++rank, code /= 4) {
            const int state = code % 4;
            if (state & 1) {
                decodes[rank] = makeStream(rank * 10, false);
                scheduled[rank].push_back(decodes[rank]);
            }
            if (state & 2) {
                scheduled[rank].push_back(makeStream(rank * 10 + 1, true));
                local_context[rank] = true;
                any_context         = true;
            }
        }
        for (int rank = 0; rank < world; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank=" << rank);
            auto       work       = scheduled[rank];  // Scheduler ownership must survive deferral.
            int        reductions = 0;
            int        fakes      = 0;
            const bool selected   = alignDsv4DpPrefillPhase(
                work,
                [&](bool local) {
                    ++reductions;
                    EXPECT_EQ(local, local_context[rank]) << "reducer sees rank-local context presence";
                    return any_context;
                },
                [&] {
                    ++fakes;
                    return makeStream(-1, true, true);
                });
            EXPECT_EQ(reductions, 1) << "all ranks vote once, including empty ranks";
            EXPECT_EQ(selected, any_context) << "all ranks select the same phase";
            if (any_context) {
                ASSERT_EQ(work.size(), 1u) << "context-only batch or one fake prefill";
                EXPECT_TRUE(work.front()->isContextStream()) << "no idle/decode rank replays a graph during prefill";
                EXPECT_EQ(fakes, !local_context[rank]) << "fake only where context is absent";
            } else {
                EXPECT_EQ(work, scheduled[rank]) << "pure decode/idle batch left unchanged";
                EXPECT_EQ(fakes, 0) << "legacy decode placeholder creation remains with caller";
            }
            if (decodes[rank]) {
                EXPECT_FALSE(decodes[rank]->context || decodes[rank]->fake) << "real decode request not mutated";
                Streams next{decodes[rank]};
                EXPECT_FALSE(alignDsv4DpPrefillPhase(
                    next,
                    [](bool local) { return local; },
                    [] {
                        ADD_FAILURE() << "deferred decode must not create a fake prefill";
                        return makeStream(-1, true, true);
                    }))
                    << "deferred decode can execute in next decode phase";
                ASSERT_FALSE(next.empty());
                EXPECT_EQ(next.front(), decodes[rank]) << "deferred request identity retained";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(WorldSizes,
                         Dsv4DpPrefillPhaseTest,
                         testing::Values(2, 4),
                         [](const testing::TestParamInfo<int>& info) { return "World" + std::to_string(info.param); });

TEST(Dsv4DpPrefillPolicyTest, PreservesMultiplePrefillsInOrder) {
    auto    p0 = makeStream(1, true);
    auto    p1 = makeStream(2, true);
    auto    d0 = makeStream(3, false);
    Streams work{p0, d0, p1};
    alignDsv4DpPrefillPhase(work, [](bool local) { return local; }, [] { return makeStream(-1, true, true); });
    EXPECT_EQ(work, Streams({p0, p1})) << "all real prefills retain original order";
}

}  // namespace
}  // namespace rtp_llm
