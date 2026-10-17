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
    bool async_bookkeeping = false;
    int  sequences         = 1;
    bool isContextStream() const {
        return context;
    }
    int currentBatchSize() const {
        if (async_bookkeeping) {
            throw std::runtime_error("token history belongs to async worker");
        }
        return 1;
    }
    int maxBatchSize() const {
        return sequences;
    }
    bool hasNumBeams() const {
        return false;
    }
};
using StreamPtr = std::shared_ptr<Stream>;
using Streams   = std::list<StreamPtr>;

StreamPtr makeStream(int id, bool context, bool fake = false) {
    return std::make_shared<Stream>(Stream{id, context, fake});
}

TEST(Dsv4DpPrefillPolicyTest, ResolvedStrategyUsesExplicitConfigBeforeAlias) {
    EXPECT_EQ(resolveDsv4MoeStrategy("sm120_decode", nullptr), "sm120_decode");
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", "sm120_decode"), "sm120_decode");
    EXPECT_EQ(resolveDsv4MoeStrategy("sm120_fused_moe", "sm120_decode"), "sm120_fused_moe");
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", "unknown"), "auto");
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", nullptr), "auto");
    EXPECT_EQ(resolveDsv4MoeStrategy(" auto\t", " sm120_decode\n"), "sm120_decode");
    EXPECT_EQ(resolveDsv4MoeStrategy("\tsm120_fused_moe ", "sm120_decode"), "sm120_fused_moe");
    EXPECT_EQ(resolveDsv4MoeStrategy(" \n", "sm120_decode"), "sm120_decode");
}

TEST(Dsv4DpPrefillPolicyTest, EnvironmentAliasIsScopedToDsv4Descriptors) {
    for (const auto* model : {"deepseek_v4", "deepseek_v4_mtp", "deepseek_v4_dspark"}) {
        EXPECT_EQ(resolveModelMoeStrategy(model, "auto", "sm120_decode"), "sm120_decode");
    }
    for (const auto* model : {"qwen_3", "deepseek_v3", "deepseek_v41", "deepseek_v41_dspark", ""}) {
        EXPECT_EQ(resolveModelMoeStrategy(model, "auto", "sm120_decode"), "auto");
        EXPECT_EQ(resolveModelMoeStrategy(model, "sm120_fused_moe", "sm120_decode"), "sm120_fused_moe");
        EXPECT_EQ(resolveModelMoeStrategy(model, "sm120_decode", nullptr), "sm120_decode");
    }
}

TEST(Dsv4DpPrefillPolicyTest, AlignsUnevenAndIdleDecodeRanks) {
    for (const auto& counts : std::vector<std::vector<int>>{{0, 0}, {1, 2}, {1, 8}, {0, 1, 8, 2}}) {
        const int target = std::max(1, *std::max_element(counts.begin(), counts.end()));
        for (int count : counts) {
            Streams streams;
            for (int i = 0; i < count; ++i) {
                streams.push_back(makeStream(i, false));
            }
            const auto original   = streams;
            int        reductions = 0;
            alignDsv4DpDecodeBatch(
                streams,
                [&](int64_t local) {
                    ++reductions;
                    EXPECT_EQ(local, std::max(1, count));
                    return target;
                },
                [] { return makeStream(-1, false, true); });
            EXPECT_EQ(reductions, 1);
            EXPECT_EQ(streams.size(), target);
            EXPECT_TRUE(std::equal(original.begin(), original.end(), streams.begin()));
            EXPECT_EQ(
                std::count_if(streams.begin(), streams.end(), [](const StreamPtr& stream) { return stream->fake; }),
                target - count);
        }
    }
}

TEST(Dsv4DpPrefillPolicyTest, RejectsOverflowOnEveryRankAfterReduction) {
    Streams idle;
    EXPECT_THROW(alignDsv4DpDecodeBatch(
                     idle, [](int64_t) { return 8; }, [] { return makeStream(-1, false, true); }, 4),
                 std::runtime_error);
    EXPECT_TRUE(idle.empty());
}

TEST(Dsv4DpPrefillPolicyTest, CancelledDecodeDoesNotPadOrCheckCapacity) {
    for (int count : {0, 1, 8}) {
        Streams streams;
        for (int i = 0; i < count; ++i) {
            streams.push_back(makeStream(i, false));
        }
        const auto original   = streams;
        int        reductions = 0;
        alignDsv4DpDecodeBatch(
            streams,
            [&](int64_t) -> std::optional<int64_t> {
                ++reductions;
                return std::nullopt;
            },
            [] {
                ADD_FAILURE() << "a coordinated stop must not create fake decode work";
                return makeStream(-1, false, true);
            },
            4);
        EXPECT_EQ(reductions, 1);
        EXPECT_EQ(streams, original);
    }
}

TEST(Dsv4DpPrefillPolicyTest, CancelledPrefillPreservesMixedBatch) {
    Streams    streams{makeStream(1, true), makeStream(2, false)};
    const auto original   = streams;
    int        reductions = 0;
    EXPECT_FALSE(alignDsv4DpPrefillPhase(
        streams,
        [&](bool local_prefill) -> std::optional<bool> {
            EXPECT_TRUE(local_prefill);
            ++reductions;
            return std::nullopt;
        },
        [] {
            ADD_FAILURE() << "a coordinated stop must not create fake prefill work";
            return makeStream(-1, true, true);
        }));
    EXPECT_EQ(reductions, 1);
    EXPECT_EQ(streams, original);
}

TEST(Dsv4DpPrefillPolicyTest, SpeculativeAlignmentDoesNotReadWorkerOwnedHistory) {
    auto stream               = makeStream(0, false);
    stream->async_bookkeeping = true;
    Streams streams{stream};
    alignDsv4DpDecodeBatch(
        streams,
        [](int64_t local) {
            EXPECT_EQ(local, 1);
            return 2;
        },
        [] { return makeStream(-1, false, true); },
        0,
        true);
    EXPECT_EQ(streams.size(), 2);
    EXPECT_EQ(streams.front(), stream);
}

TEST(Dsv4DpPrefillPolicyTest, OrdinaryAlignmentUsesImmutableSequenceCount) {
    auto stream               = makeStream(0, false);
    stream->async_bookkeeping = true;
    stream->sequences         = 3;
    Streams streams{stream};
    alignDsv4DpDecodeBatch(
        streams,
        [](int64_t local) {
            EXPECT_EQ(local, 3);
            return 4;
        },
        [] { return makeStream(-1, false, true); });
    EXPECT_EQ(streams.size(), 2);
    EXPECT_EQ(streams.front(), stream);
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
