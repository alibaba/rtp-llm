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

TEST(Dsv4DpPrefillPolicyTest, SustainedPrefillCannotStarveDecode) {
    for (const auto& ranks : std::vector<std::vector<int>>{{1, 2, 0}, {3, 0}, {1, 3, 2, 0}}) {
        bool previous_prefill = false;
        for (int step = 0; step < 20; ++step) {
            int64_t vote = 0;
            for (int work : ranks) {
                vote = std::max(vote, dsv4DpPhaseVote(work & 1, work & 2, previous_prefill));
            }
            const bool prefill = dsv4DpPhaseIsPrefill(vote, previous_prefill);
            EXPECT_EQ(prefill, step % 2 == 0);
            previous_prefill = prefill;
        }
    }
}

TEST(Dsv4DpPrefillPolicyTest, SinglePhaseTrafficNeverAddsAnEmptyOppositePhase) {
    for (bool initial_prefill : {false, true}) {
        for (int work : {0, 1, 2}) {
            bool previous_prefill = initial_prefill;
            for (int step = 0; step < 20; ++step) {
                const auto vote    = dsv4DpPhaseVote(work & 1, work & 2, previous_prefill);
                const bool prefill = dsv4DpPhaseIsPrefill(vote, previous_prefill);
                EXPECT_EQ(prefill, work == 1);
                previous_prefill = prefill;
            }
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

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeDpGraphGuardAcceptsSupportedProfiles) {
    // Synchronized DP decode stays allowed for both roles that run decode.
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned("sm120_decode", RoleType::DECODE, 2, true, true));
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned("sm120_decode", RoleType::PDFUSION, 8, true, true));
    // PREFILL roles never decode, and single-DP engines share one batch.
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned("sm120_decode", RoleType::PREFILL, 4, true, false));
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned("sm120_decode", RoleType::DECODE, 1, true, false));
    // Eager decode coordinates its rounds inside the executor.
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned("sm120_decode", RoleType::DECODE, 4, false, false));
    // Other strategies are unaffected.
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned("sm120_fused_moe", RoleType::DECODE, 4, true, false));
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned("auto", RoleType::PDFUSION, 4, true, false));
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned(nullptr, RoleType::DECODE, 4, true, false));
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeDpGraphGuardRejectsUnsynchronizedTopologies) {
    for (RoleType role : {RoleType::DECODE, RoleType::PDFUSION}) {
        SCOPED_TRACE(testing::Message() << "role=" << role);
        // Rank-local graph buckets would mismatch the WORLD all-gather.
        EXPECT_TRUE(sm120DecodeDpGraphMisaligned("sm120_decode", role, 2, true, false));
        EXPECT_TRUE(sm120DecodeDpGraphMisaligned("sm120_decode", role, 4, true, false));
    }
}

TEST(Dsv4DpPrefillPolicyTest, Sm120FusedMoeDpDecodeGuardAcceptsSupportedProfiles) {
    // PREFILL-only roles never join decode-phase collectives.
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", RoleType::PREFILL, 4));
    // TP-only and single-DP engines keep every rank in the same phase.
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", RoleType::DECODE, 1));
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", RoleType::PDFUSION, 1));
    // Other strategies, including unresolved "auto", are unaffected here.
    // Python strategy selection decides whether auto may pick sm120_fused_moe.
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_decode", RoleType::DECODE, 4));
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("auto", RoleType::PDFUSION, 4));
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned(nullptr, RoleType::DECODE, 4));
}

TEST(Dsv4DpPrefillPolicyTest, Sm120FusedMoeDpDecodeGuardRejectsDecodeCapableTopologies) {
    for (RoleType role : {RoleType::DECODE, RoleType::PDFUSION}) {
        SCOPED_TRACE(testing::Message() << "role=" << role);
        // Decode-capable roles would issue WORLD collectives out of phase.
        EXPECT_TRUE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", role, 2));
        EXPECT_TRUE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", role, 8));
        EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("auto", role, 2));
        EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("auto", role, 8));
    }
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodePrefillLocalDecodeGuard) {
    // Construction-time PREFILL exemption is unchanged: true PD prefill does
    // not decode on this engine.
    EXPECT_FALSE(sm120DecodeDpGraphMisaligned("sm120_decode", RoleType::PREFILL, 4, true, false));
    // The helper names the PREFILL + sm120_decode + CUDA graph + DP>1 profile.
    // Request-level filtering (PD separation, isPrefillOnly, max_new_tokens,
    // fake_query) lives in NormalEngine::validateStreamForScheduling.
    EXPECT_TRUE(sm120DecodePrefillLocalDecodeMisaligned("sm120_decode", RoleType::PREFILL, 4, true));
    EXPECT_TRUE(sm120DecodePrefillLocalDecodeMisaligned("sm120_decode", RoleType::PREFILL, 2, true));
    EXPECT_FALSE(sm120DecodePrefillLocalDecodeMisaligned("sm120_decode", RoleType::PREFILL, 1, true));
    EXPECT_FALSE(sm120DecodePrefillLocalDecodeMisaligned("sm120_decode", RoleType::PREFILL, 4, false));
    EXPECT_FALSE(sm120DecodePrefillLocalDecodeMisaligned("sm120_decode", RoleType::DECODE, 4, true));
    EXPECT_FALSE(sm120DecodePrefillLocalDecodeMisaligned("sm120_decode", RoleType::PDFUSION, 4, true));
    EXPECT_FALSE(sm120DecodePrefillLocalDecodeMisaligned("sm120_fused_moe", RoleType::PREFILL, 4, true));
    EXPECT_FALSE(sm120DecodePrefillLocalDecodeMisaligned("auto", RoleType::PREFILL, 4, true));
    EXPECT_FALSE(sm120DecodePrefillLocalDecodeMisaligned(nullptr, RoleType::PREFILL, 4, true));
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeGraphBatchRequirementFollowsAlignmentMode) {
    // Only engine-formed bounds count: synchronized steps pad up to
    // RTP_LLM_DECODE_FIXED_BS, and the batch-decode scheduler gates rounds by
    // its own batch size (a 0 scheduler batch size means the batch-decode
    // scheduler is disabled).
    EXPECT_EQ(sm120DecodeGraphBatchRequirement(0, 0), 0);
    EXPECT_EQ(sm120DecodeGraphBatchRequirement(0, -1), 0);
    EXPECT_EQ(sm120DecodeGraphBatchRequirement(8, 0), 8);
    EXPECT_EQ(sm120DecodeGraphBatchRequirement(8, 4), 8);
    EXPECT_EQ(sm120DecodeGraphBatchRequirement(4, 16), 16);
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeGraphCoverageRejectsOnlyEngineControlledGaps) {
    // Mirrors the ctor decision: reject when the largest captured bucket (the
    // default buckets end at the concurrency limit, 8 here) falls below the
    // hard requirement; max_generate_batch_size shortfalls are rejected too.
    auto rejected = [](const std::vector<int>& capture_sizes, int64_t scheduler_batch_size, int64_t fixed_batch_size) {
        const int64_t captured = largestDecodeCaptureBatchSize(capture_sizes, 8);
        const int64_t required = sm120DecodeGraphBatchRequirement(scheduler_batch_size, fixed_batch_size);
        return captured < required;
    };
    // Buckets ending below max_generate_batch_size are rejected, whether they
    // are the defaults (capped by the concurrency limit) or an explicit list.
    EXPECT_FALSE(rejected({}, 0, 0));
    EXPECT_FALSE(rejected({1, 4}, 0, 0));
    EXPECT_TRUE(sm120DecodeCapturedBelowMaxGenerate(16, 8));
    EXPECT_TRUE(sm120DecodeCapturedBelowMaxGenerate(16, 4));
    EXPECT_FALSE(sm120DecodeCapturedBelowMaxGenerate(8, 8));
    EXPECT_FALSE(sm120DecodeCapturedBelowMaxGenerate(4, 8));
    // RTP_LLM_DECODE_FIXED_BS pads every synchronized step up to itself.
    EXPECT_TRUE(rejected({}, 0, 16));
    EXPECT_TRUE(rejected({1, 4}, 0, 16));
    // A batch-decode scheduler round can fill its whole batch size.
    EXPECT_TRUE(rejected({}, 16, 0));
    EXPECT_TRUE(rejected({1, 4}, 16, 0));
    // Aligned configurations stay accepted.
    EXPECT_FALSE(rejected({}, 0, 8));
    EXPECT_FALSE(rejected({1, 4, 8}, 4, 0));
}

TEST(Dsv4DpPrefillPolicyTest, LargestDecodeCaptureBatchSizeMirrorsRunnerBuckets) {
    // Default buckets end at the concurrency limit.
    EXPECT_EQ(largestDecodeCaptureBatchSize({}, 32), 32);
    EXPECT_EQ(largestDecodeCaptureBatchSize({}, 1), 1);
    // An explicit list replaces the default buckets, even above the limit.
    EXPECT_EQ(largestDecodeCaptureBatchSize({1, 8}, 32), 8);
    EXPECT_EQ(largestDecodeCaptureBatchSize({8, 1, 4}, 32), 8);
    EXPECT_EQ(largestDecodeCaptureBatchSize({64}, 8), 64);
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeRequestFanOutMustFitLargestCapturedBucket) {
    EXPECT_FALSE(sm120DecodeRequestExceedsGraph(1, 1));
    EXPECT_FALSE(sm120DecodeRequestExceedsGraph(2, 4));
    EXPECT_TRUE(sm120DecodeRequestExceedsGraph(2, 1));
    // An explicit graph list is sized by its largest bucket, not the limit.
    EXPECT_FALSE(sm120DecodeRequestExceedsGraph(2, largestDecodeCaptureBatchSize({1, 2}, 8)));
    EXPECT_TRUE(sm120DecodeRequestExceedsGraph(4, largestDecodeCaptureBatchSize({1, 2}, 8)));
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeAdmissionRowCapPrefersFixedBatchSize) {
    EXPECT_EQ(sm120DecodeAdmissionRowCap(/*captured=*/32, /*fixed_bs=*/0), 32);
    EXPECT_EQ(sm120DecodeAdmissionRowCap(/*captured=*/32, /*fixed_bs=*/8), 8);
    EXPECT_EQ(sm120DecodeAdmissionRowCap(/*captured=*/4, /*fixed_bs=*/4), 4);
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeMaxFanOutForAdmissionIsRowCapOverStreamBound) {
    EXPECT_EQ(sm120DecodeMaxFanOutForAdmission(/*row_cap=*/8, /*stream_bound=*/8), 1);
    EXPECT_EQ(sm120DecodeMaxFanOutForAdmission(/*row_cap=*/32, /*stream_bound=*/8), 4);
    EXPECT_EQ(sm120DecodeMaxFanOutForAdmission(/*row_cap=*/8, /*stream_bound=*/0), 8);  // bound clamped to 1
    EXPECT_EQ(sm120DecodeMaxFanOutForAdmission(/*row_cap=*/1, /*stream_bound=*/8), 1);
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeRejectsSpeculativeFanOutMatchesAlignPredicate) {
    EXPECT_FALSE(sm120DecodeRejectsSpeculativeFanOut(/*max_batch_size=*/1, /*has_beams=*/false));
    EXPECT_TRUE(sm120DecodeRejectsSpeculativeFanOut(/*max_batch_size=*/2, /*has_beams=*/false));
    EXPECT_TRUE(sm120DecodeRejectsSpeculativeFanOut(/*max_batch_size=*/1, /*has_beams=*/true));
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
