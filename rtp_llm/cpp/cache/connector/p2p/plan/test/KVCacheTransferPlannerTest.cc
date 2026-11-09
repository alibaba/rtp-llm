#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/PipelineLayoutCodec.h"

#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"

#include <gtest/gtest.h>
#include <tuple>

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace rtp_llm {

TEST(KVCacheTransferPlannerTest, RejectsUnboundedPeerRankExpansion) {
    ShardLayout prefill;
    ShardLayout decode;
    prefill.pc.tp_size = std::numeric_limits<int>::max();
    decode.pc.tp_size  = 1;

    const auto result = KVCacheTransferPlanner::plan(prefill, decode, {"full"});
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.plan.routes.empty());
    EXPECT_NE(result.error.ToString().find("rank count exceeds"), std::string::npos);
}

// ---------------------------------------------------------------------------
// 公共基线（设计文档 §9.0）
//
// 8 个 KV head、head 维 128、fp16（无 scale）、每块 64 token
//   -> kv_block_stride_bytes = 64 * 128 * 2 (fp16) * 2 (K+V) * local_kv_heads
//                            = 32768 * local_kv_heads
//   -> local_kv_heads = 8 / get_attn_tp_size()
//   -> 全局块大小恒为 262144
// MLA 基线：单 latent，不随 TP 切分，块大小恒为 65536。
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kGlobalMhaBlockBytes = 262144;
constexpr size_t kMlaBlockBytes       = 65536;
constexpr size_t kSeqSizePerBlock     = 64;

constexpr const char* kFullTag  = "full";
constexpr const char* kSwaTag   = "swa";
constexpr const char* kFixedTag = "fixed";

ParallelismConfig makePc(int tp_size, CPRotateMethod method, bool kv_cache_sharded) {
    ParallelismConfig pc;
    pc.tp_size                              = tp_size;
    pc.prefill_cp_config.method             = method;
    pc.prefill_cp_config.kv_cache_sharded   = kv_cache_sharded;
    if (kv_cache_sharded) {
        pc.prefill_cp_config.prefill_cp_size = tp_size;
    }
    return pc;
}

ShardLayout::GroupLayout makeMhaFullGroup() {
    ShardLayout::GroupLayout g;
    g.policy             = defaultCacheGroupPolicy(CacheGroupType::FULL);  // cp_mapping = BLOCK_ROUND_ROBIN
    g.spec_type          = KVCacheSpecType::MultiHeadAttention;
    g.seq_size_per_block = kSeqSizePerBlock;
    return g;
}

ShardLayout::GroupLayout makeMlaFullGroup() {
    ShardLayout::GroupLayout g;
    g.policy             = defaultCacheGroupPolicy(CacheGroupType::FULL);
    g.spec_type          = KVCacheSpecType::MultiHeadLatentAttention;
    g.seq_size_per_block = kSeqSizePerBlock;
    return g;
}

ShardLayout::GroupLayout makeSwaGroup() {
    ShardLayout::GroupLayout g;
    g.policy             = defaultCacheGroupPolicy(CacheGroupType::SWA);  // COMPACT_LAST_RANK, active_tail_blocks = 2
    g.policy.cp_slice    = CpBlockSliceMode::NONE;
    g.spec_type          = KVCacheSpecType::MultiHeadAttention;
    g.seq_size_per_block = 4 * kSeqSizePerBlock;
    return g;
}

/// 依据 pc 与 spec 类型补齐 head_shard_count 与 stride，使全局块大小落在基线上。
void finalize(ShardLayout& layout) {
    layout.deriveHeadShardCounts();
    for (auto& [tag, g] : layout.groups) {
        const size_t global = g.spec_type == KVCacheSpecType::MultiHeadLatentAttention ? kMlaBlockBytes :
                                                                                        kGlobalMhaBlockBytes;
        size_t divisor = static_cast<size_t>(std::max(1, layout.headShardCount(tag)));
        if (g.pre_sliced) {
            divisor *= static_cast<size_t>(std::max(1, layout.cpSize()));
        }
        g.kv_block_stride_bytes = global / divisor;
    }
}

ShardLayout makeLayout(int                                                              tp_size,
                       CPRotateMethod                                                   method,
                       bool                                                             kv_cache_sharded,
                       const std::vector<std::pair<std::string, ShardLayout::GroupLayout>>& groups) {
    ShardLayout layout;
    layout.pc = makePc(tp_size, method, kv_cache_sharded);
    layout.cache_key_tokens_per_block = kSeqSizePerBlock;
    for (const auto& [tag, g] : groups) {
        layout.groups[tag] = g;
    }
    finalize(layout);
    return layout;
}

ShardLayout mhaLayout(int tp_size, CPRotateMethod method = CPRotateMethod::DISABLED, bool sharded = false) {
    return makeLayout(tp_size, method, sharded, {{kFullTag, makeMhaFullGroup()}});
}

ShardLayout mlaLayout(int tp_size, CPRotateMethod method = CPRotateMethod::DISABLED, bool sharded = false) {
    return makeLayout(tp_size, method, sharded, {{kFullTag, makeMlaFullGroup()}});
}

// ---- 断言辅助 ----

std::set<std::pair<int, int>> rankPairs(const TransferPlan& plan, const std::string& tag) {
    std::set<std::pair<int, int>> out;
    for (const auto* r : plan.forTag(tag)) {
        out.insert({r->src_rank, r->dst_rank});
    }
    return out;
}

void expectAllRouteBytesMatch(const TransferPlan& plan) {
    for (const auto& r : plan.routes) {
        EXPECT_EQ(r.src_bytes, r.dst_bytes) << "route " << r.route_id << " src_rank=" << r.src_rank
                                            << " dst_rank=" << r.dst_rank << " tag=" << r.cache_tag;
        EXPECT_GT(r.src_bytes, 0u);
    }
}

void expectUniqueRouteIds(const TransferPlan& plan) {
    std::set<int> ids;
    for (const auto& r : plan.routes) {
        EXPECT_TRUE(ids.insert(r.route_id).second) << "duplicate route_id " << r.route_id;
    }
}

void expectNoSlicing(const TransferPlan& plan) {
    for (const auto& r : plan.routes) {
        EXPECT_EQ(r.src_slice.mode, CpBlockSliceMode::NONE);
        EXPECT_EQ(r.dst_slice.mode, CpBlockSliceMode::NONE);
    }
}

void expectNoHeadPartitioning(const TransferPlan& plan) {
    for (const auto& r : plan.routes) {
        EXPECT_EQ(r.src_partition, (PartitionSpec{1, 0})) << "route " << r.route_id;
        EXPECT_EQ(r.dst_partition, (PartitionSpec{1, 0})) << "route " << r.route_id;
    }
}

PipelineTransferLayout pipeline(const std::vector<int>& cuts, const ShardLayout& shards) {
    PipelineTransferLayout result;
    int                    layer_begin = 0;
    for (int count : cuts) {
        auto& stage            = result.stages.emplace_back();
        stage.main_layer_begin = layer_begin;
        stage.main_layer_count = count;
        stage.main_shards      = shards;
        stage.layer_tags.resize(count, {kFullTag});
        for (int lane = 0; lane < shards.rankCount(); ++lane) {
            stage.worker_indices.push_back(static_cast<int>(result.worker_count++));
        }
        layer_begin += count;
    }
    return result;
}

}  // namespace

TEST(PlannerPipeline, IndexerEncodingSurvivesSpecAndWireLayout) {
    auto             pc = makePc(1, CPRotateMethod::DISABLED, false);
    SpecBuildContext ctx;
    ctx.seq_size_per_block = 128;
    ctx.parallelism_config = &pc;
    KVCacheSpecDesc desc;
    desc.tag                    = kFullTag;
    desc.cache_type             = KVCacheSpecType::OpaqueKV;
    desc.entry_dtype            = DataType::TYPE_UINT8;
    desc.entry_elems            = 132;
    ctx.kernel_tokens_per_block = 128;
    desc.entry_count_mode       = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    std::set<std::string> fingerprints;
    for (auto format : {IndexerScaleFormat::FP32, IndexerScaleFormat::UE8M0}) {
        desc.indexer_scale_format = format;
        auto spec                 = SpecBuilder::build(desc, ctx);
        fingerprints.insert(spec->fingerprint());
        GroupBase group;
        group.tag       = desc.tag;
        group.spec      = spec;
        group.policy    = defaultCacheGroupPolicy(CacheGroupType::FULL);
        group.block_num = 8;
        auto topology   = CacheTopology::create({group}, {{0, {kFullTag}}});
        auto layout     = ShardLayoutFactory::fromTopology(*topology, pc, RoleType::PREFILL, 128);
        EXPECT_EQ(layout.group(kFullTag).indexer_scale_format, format);
        P2PShardLayoutPB wire;
        encodeP2PShardLayout(layout, &wire);
        ShardLayout received;
        ASSERT_TRUE(decodeP2PShardLayout(wire, RoleType::PREFILL, received));
        EXPECT_EQ(received.group(kFullTag).indexer_scale_format, format);
        EXPECT_EQ(spec->k_block_payload_bytes(), 128u * 132);
        const auto inferred = ShardLayout::forPeer(layout, pc, false);
        EXPECT_FALSE(KVCacheTransferPlanner::plan(layout, inferred, {kFullTag}).ok());
        EXPECT_FALSE(KVCacheTransferPlanner::plan(inferred, layout, {kFullTag}).ok());
        wire.mutable_groups(0)->set_indexer_scale_format(3);
        ShardLayout invalid;
        EXPECT_FALSE(decodeP2PShardLayout(wire, RoleType::PREFILL, invalid));
    }
    EXPECT_EQ(fingerprints.size(), 2u);
}

TEST(PlannerPipeline, PreservesLinearAndIndexerMetadataInOneWireLayout) {
    auto  layout                  = mlaLayout(1);
    auto& indexer                 = layout.groups.at(kFullTag);
    indexer.spec_type             = KVCacheSpecType::OpaqueKV;
    indexer.indexer_scale_format  = IndexerScaleFormat::UE8M0;
    indexer.kv_block_stride_bytes = kSeqSizePerBlock * 132;
    indexer.k_block_payload_bytes = indexer.kv_block_stride_bytes;
    indexer.kv_scale_stride_bytes = 0;
    auto& linear                  = layout.groups[kFixedTag];
    linear.policy                 = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    linear.spec_type              = KVCacheSpecType::LinearAttention;
    linear.seq_size_per_block     = kSeqSizePerBlock;
    linear.k_block_payload_bytes  = 64;
    linear.linear_q_row_bytes     = 8;
    linear.linear_k_row_bytes     = 8;
    linear.linear_v_row_bytes     = 16;
    linear.linear_history_rows    = 3;
    linear.kv_block_stride_bytes  = 64 + 3 * (8 + 8 + 16);
    layout.deriveHeadShardCounts();

    P2PShardLayoutPB wire;
    encodeP2PShardLayout(layout, &wire);
    ShardLayout received;
    ASSERT_TRUE(decodeP2PShardLayout(wire, RoleType::PREFILL, received));
    EXPECT_EQ(received.group(kFullTag).indexer_scale_format, IndexerScaleFormat::UE8M0);
    const auto& restored = received.group(kFixedTag);
    EXPECT_EQ(restored.indexer_scale_format, IndexerScaleFormat::NONE);
    EXPECT_EQ(restored.linear_q_row_bytes, 8u);
    EXPECT_EQ(restored.linear_k_row_bytes, 8u);
    EXPECT_EQ(restored.linear_v_row_bytes, 16u);
    EXPECT_EQ(restored.linear_history_rows, 3u);
}

TEST(PlannerPipeline, RequiresMatchingIndexerEncodingForMainAndDraft) {
    auto  shards                = mlaLayout(1);
    auto& group                 = shards.groups.at(kFullTag);
    group.spec_type             = KVCacheSpecType::OpaqueKV;
    group.kv_block_stride_bytes = group.k_block_payload_bytes = kSeqSizePerBlock * 132;
    const auto native =
        KVCacheTransferPlanner::planStages(pipeline({3, 5}, shards), pipeline({2, 6}, shards), "request");
    ASSERT_TRUE(native.ok()) << native.error.ToString();
    std::set<uint64_t> digests;
    for (auto format : {IndexerScaleFormat::FP32, IndexerScaleFormat::UE8M0}) {
        group.indexer_scale_format = format;
        auto src                   = pipeline({3, 5}, shards);
        auto dst                   = pipeline({2, 6}, shards);
        for (auto* side : {&src, &dst}) {
            side->stages.back().draft_shards = shards;
            side->stages.back().layer_tags.push_back({kFullTag});
        }
        const auto matched = KVCacheTransferPlanner::planStages(src, dst, "request");
        ASSERT_TRUE(matched.ok()) << matched.error.ToString();
        digests.insert(matched.plan.digest());
        expectAllRouteBytesMatch(matched.plan);
        // The format marker must not alter the native copy geometry.
        auto main_only =
            KVCacheTransferPlanner::planStages(pipeline({3, 5}, shards), pipeline({2, 6}, shards), "request");
        ASSERT_TRUE(main_only.ok());
        ASSERT_EQ(main_only.plan.routes.size(), native.plan.routes.size());
        for (size_t i = 0; i < native.plan.routes.size(); ++i) {
            EXPECT_EQ(main_only.plan.routes[i].src_bytes, native.plan.routes[i].src_bytes);
            EXPECT_EQ(main_only.plan.routes[i].dst_bytes, native.plan.routes[i].dst_bytes);
        }
        for (auto other : {IndexerScaleFormat::NONE, IndexerScaleFormat::FP32, IndexerScaleFormat::UE8M0}) {
            if (other == format)
                continue;
            // Check both transfer directions, with a mismatch only in a later PP stage
            // or only in draft, so matching main layers cannot conceal an invalid draft.
            for (bool draft : {false, true}) {
                auto  mismatch                                  = dst;
                auto& stage                                     = mismatch.stages.back();
                auto& target                                    = draft ? stage.draft_shards : stage.main_shards;
                target.groups.at(kFullTag).indexer_scale_format = other;
                for (const auto& result : {KVCacheTransferPlanner::planStages(src, mismatch, "request"),
                                           KVCacheTransferPlanner::planStages(mismatch, src, "request")}) {
                    EXPECT_FALSE(result.ok());
                    EXPECT_NE(result.error.ToString().find("indexer scale format mismatch"), std::string::npos);
                    EXPECT_TRUE(result.plan.routes.empty());
                }
            }
        }
    }
    EXPECT_EQ(digests.size(), 2u);
}

TEST(PlannerPipeline, CoversDifferentStageCutsInBothDirections) {
    const std::vector<std::pair<std::vector<int>, std::vector<int>>> cuts = {
        {{8}, {3, 5}}, {{3, 5}, {8}}, {{3, 5}, {2, 6}}, {{3, 5}, {2, 4, 2}}};
    for (const auto& [src_cuts, dst_cuts] : cuts) {
        auto       src    = pipeline(src_cuts, mhaLayout(1));
        auto       dst    = pipeline(dst_cuts, mhaLayout(2));
        const auto result = KVCacheTransferPlanner::planStages(src, dst, "request");
        ASSERT_TRUE(result.ok()) << result.error.ToString();
        expectUniqueRouteIds(result.plan);
        expectAllRouteBytesMatch(result.plan);
        std::set<std::pair<int, int>> seen;
        for (const auto& route : result.plan.routes) {
            const auto& source_stage = src.stages.at(route.src_rank);
            const auto& target_stage = dst.stages.at(route.dst_rank / 2);
            EXPECT_EQ(route.src_partition.count, 2);
            ASSERT_FALSE(route.layers.empty());
            for (const auto& layer : route.layers) {
                EXPECT_EQ(layer.global_id, source_stage.main_layer_begin + layer.src_local_id);
                EXPECT_EQ(layer.global_id, target_stage.main_layer_begin + layer.dst_local_id);
                EXPECT_LT(layer.src_local_id, source_stage.main_layer_count);
                EXPECT_LT(layer.dst_local_id, target_stage.main_layer_count);
                EXPECT_TRUE(seen.emplace(layer.global_id, route.dst_rank % 2).second);
            }
        }
        EXPECT_EQ(seen.size(), 16u);
        EXPECT_EQ(result.plan.digest(), KVCacheTransferPlanner::planStages(src, dst, "request").plan.digest());
    }
}

TEST(PlannerPipeline, PreservesCpKeyRulesAndUsesPublishedDraftGeometry) {
    auto src                       = pipeline({3, 5}, mlaLayout(2, CPRotateMethod::ALL_GATHER, true));
    auto dst                       = pipeline({2, 4, 2}, mlaLayout(1));
    src.stages.back().draft_shards = src.stages.back().main_shards;
    dst.stages.back().draft_shards = dst.stages.back().main_shards;
    for (auto* side : {&src, &dst}) {
        auto& stage = side->stages.back();
        stage.draft_shards.groups.at(kFullTag).kv_block_stride_bytes *= 2;
        stage.layer_tags.push_back({kFullTag});
    }
    auto& mixed_src                       = src.stages.back();
    auto& mixed_dst                       = dst.stages[1];
    for (auto* side : {&src, &dst}) {
        auto& root                       = side->stages.front();
        root.main_shards.groups["mixed"] = root.main_shards.groups.at(kFullTag);
        root.layer_tags.front().push_back("mixed");
    }
    mixed_src.main_shards.groups["mixed"] = mixed_src.main_shards.groups.at(kFullTag);
    mixed_dst.main_shards.groups["mixed"] = mixed_dst.main_shards.groups.at(kFullTag);
    mixed_src.layer_tags[2].push_back("mixed");
    mixed_dst.layer_tags[3].push_back("mixed");
    const auto result = KVCacheTransferPlanner::planStages(src, dst, "request");
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    std::set<size_t> draft_positions;
    int              mixed_routes = 0;
    for (const auto& route : result.plan.routes) {
        ASSERT_FALSE(route.layers.empty());
        if (route.cache_tag == "mixed") {
            ASSERT_EQ(route.layers.size(), 1u);
            const auto& layer = route.layers.front();
            if (layer.global_id == 5) {
                EXPECT_EQ(layer.src_local_id, 2);
                EXPECT_EQ(layer.dst_local_id, 3);
            } else {
                EXPECT_EQ(layer.global_id, 0);
                EXPECT_EQ(layer.src_local_id, 0);
                EXPECT_EQ(layer.dst_local_id, 0);
            }
            ++mixed_routes;
        }
        for (const auto& layer : route.layers) {
            EXPECT_LE(layer.global_id, 8);
            if (layer.global_id == 8) {
                EXPECT_EQ(route.src_bytes, 2 * kMlaBlockBytes);
                EXPECT_EQ(route.dst_rank, 2);
                EXPECT_EQ(layer.src_local_id, 5);
                EXPECT_EQ(layer.dst_local_id, 2);
                for (auto pos : KVCacheTransferPlanner::resolveKeys(route.src_keys, 5)) {
                    if (pos >= 2) {  // A Decode hit on the first two keys leaves [2, 5) to copy.
                        EXPECT_TRUE(draft_positions.insert(pos).second);
                    }
                }
            }
        }
    }
    EXPECT_EQ(mixed_routes, 4);
    EXPECT_EQ(draft_positions, (std::set<size_t>{2, 3, 4}));
}

TEST(PlannerPipeline, RejectsInvalidCoverageAndWorkerMapping) {
    const auto src                     = pipeline({3, 5}, mlaLayout(1));
    const auto dst                     = pipeline({2, 4, 2}, mlaLayout(1));
    auto       overlap                 = dst;
    overlap.stages[1].main_layer_begin = 1;
    EXPECT_FALSE(KVCacheTransferPlanner::planStages(src, overlap, "request").ok());

    auto duplicate_worker                        = dst;
    duplicate_worker.stages[1].worker_indices[0] = 0;
    EXPECT_FALSE(KVCacheTransferPlanner::planStages(src, duplicate_worker, "request").ok());

    auto missing_src_tag = src;
    missing_src_tag.stages[1].layer_tags[2].clear();
    const auto missing = KVCacheTransferPlanner::planStages(missing_src_tag, dst, "request");
    EXPECT_FALSE(missing.ok());
    EXPECT_TRUE(missing.plan.routes.empty());
    EXPECT_NE(missing.error.ToString().find("PP stage 1 local_layer=2"), std::string::npos);

    auto missing_dst_tag = dst;
    missing_dst_tag.stages[1].layer_tags[3].clear();
    const auto missing_destination = KVCacheTransferPlanner::planStages(src, missing_dst_tag, "request");
    EXPECT_FALSE(missing_destination.ok());
    EXPECT_TRUE(missing_destination.plan.routes.empty());
    EXPECT_NE(missing_destination.error.ToString().find("empty or duplicate layer tags"), std::string::npos);
}

TEST(PlannerPipeline, RejectsMissingDecodeDraftLayers) {
    auto src = pipeline({8}, mlaLayout(1));
    auto dst = pipeline({3, 5}, mlaLayout(1));
    for (auto* side : {&src, &dst}) {
        auto& tail        = side->stages.back();
        tail.draft_shards = tail.main_shards;
    }
    auto& src_layers = src.stages.back().layer_tags;
    dst.stages.back().layer_tags.resize(7, {kFullTag});
    for (int published = 0; published < 2; ++published) {
        src_layers.resize(8 + published, {kFullTag});
        const auto result = KVCacheTransferPlanner::planStages(src, dst, "request");
        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(result.plan.routes.empty());
        EXPECT_NE(result.error.ToString().find("Prefill is missing draft layers required by Decode"),
                  std::string::npos);
    }
    for (int published = 2; published <= 3; ++published) {
        src_layers.resize(8 + published, {kFullTag});
        const auto result = KVCacheTransferPlanner::planStages(src, dst, "request");
        ASSERT_TRUE(result.ok()) << result.error.ToString();
        std::set<int> transferred_draft_layers;
        for (const auto& route : result.plan.routes) {
            for (const auto& layer : route.layers) {
                if (layer.global_id >= 8) {
                    transferred_draft_layers.insert(layer.global_id);
                    EXPECT_EQ(route.dst_rank, 1);
                }
            }
        }
        EXPECT_EQ(transferred_draft_layers, (std::set<int>{8, 9}));
    }
}

TEST(PlannerPipeline, UsesExplicitWorkerIndicesInRoutesAndDigest) {
    const auto src = pipeline({8}, mlaLayout(1));
    auto       dst = pipeline({3, 5}, mlaLayout(2));
    const auto original = KVCacheTransferPlanner::planStages(src, dst, "request");
    ASSERT_TRUE(original.ok()) << original.error.ToString();

    std::swap(dst.stages[0].worker_indices[1], dst.stages[1].worker_indices[1]);
    const auto remapped = KVCacheTransferPlanner::planStages(src, dst, "request");
    ASSERT_TRUE(remapped.ok()) << remapped.error.ToString();
    ASSERT_EQ(remapped.plan.routes.size(), 4u);
    ASSERT_FALSE(remapped.plan.routes[1].layers.empty());
    ASSERT_FALSE(remapped.plan.routes[3].layers.empty());
    EXPECT_EQ(remapped.plan.routes[1].dst_rank, 3);
    EXPECT_EQ(remapped.plan.routes[1].layers.front().global_id, 0);
    EXPECT_EQ(remapped.plan.routes[3].dst_rank, 1);
    EXPECT_EQ(remapped.plan.routes[3].layers.front().global_id, 3);
    EXPECT_NE(remapped.plan.digest(), original.plan.digest());
}

// ===========================================================================
// 组 A：plan() 产出的 route 集与字段
// ===========================================================================

// 用例 A1: 对称 TP、无 CP 的基线
TEST(PlannerGroupA, A1_SymmetricTpNoCp) {
    const auto src = mhaLayout(8);
    const auto dst = mhaLayout(8);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});

    ASSERT_TRUE(result.ok()) << result.error.ToString();
    EXPECT_EQ(result.plan.routes.size(), 8u);

    std::set<std::pair<int, int>> expected;
    for (int k = 0; k < 8; ++k) {
        expected.insert({k, k});
    }
    EXPECT_EQ(rankPairs(result.plan, kFullTag), expected);

    for (const auto& r : result.plan.routes) {
        EXPECT_EQ(r.cache_tag, kFullTag);
        // 两侧都不分片 -> 键规则退化为「取全部逻辑位置」
        EXPECT_EQ(r.src_keys, (KeyShardSpec{1, 0, false, 0, 1, 0}));
    }
    expectNoHeadPartitioning(result.plan);
    expectNoSlicing(result.plan);
    expectAllRouteBytesMatch(result.plan);
    expectUniqueRouteIds(result.plan);
}

// 用例 A2: RR CP 对称（现状行为的等价基准）
TEST(PlannerGroupA, A2_SymmetricRrCp) {
    const auto src = mhaLayout(8, CPRotateMethod::ALL_GATHER, /*sharded=*/true);
    const auto dst = mhaLayout(8, CPRotateMethod::ALL_GATHER, /*sharded=*/true);

    // ALL_GATHER 使 get_attn_tp_size() 压成 1 -> head 不切分，序列按 RR 切分
    EXPECT_EQ(src.headShardCount(kFullTag), 1);
    EXPECT_EQ(src.cpSize(), 8);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    EXPECT_EQ(result.plan.routes.size(), 8u);
    std::set<std::pair<int, int>> expected;
    for (int k = 0; k < 8; ++k) {
        expected.insert({k, k});  // RR 下只有源、目的 CP rank 相同时键集才有交集
    }
    EXPECT_EQ(rankPairs(result.plan, kFullTag), expected);

    for (const auto* r : result.plan.forTag(kFullTag)) {
        EXPECT_EQ(r->src_keys.modulus, 8);
        EXPECT_EQ(r->src_keys.residue, src.cpRank(r->src_rank));
        EXPECT_FALSE(r->src_keys.include_final_key);
    }
    expectNoHeadPartitioning(result.plan);
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A3: 非对称 CP —— prefill 按 CP 分 4 片，decode 不分片
TEST(PlannerGroupA, DecodeWithPeerCpFlagReceivesEveryLogicalKey) {
    auto src = mlaLayout(2, CPRotateMethod::ALL_GATHER, true);
    src.pc.role_type = RoleType::PREFILL;
    auto dst = mlaLayout(2, CPRotateMethod::PREFILL_CP, true);
    dst.pc.role_type = RoleType::DECODE;
    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    for (int rank = 0; rank < 2; ++rank) {
        std::set<size_t> received;
        for (const auto* route : result.plan.forDecodeRank(rank)) {
            auto keys = KVCacheTransferPlanner::resolveKeys(route->src_keys, 6);
            received.insert(keys.begin(), keys.end());
        }
        EXPECT_EQ(received, (std::set<size_t>{0, 1, 2, 3, 4, 5}));
    }
}

TEST(PlannerGroupA, A3_AsymmetricCpShrinkToOne) {
    const auto src = mhaLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true);
    const auto dst = mhaLayout(1);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // 4 条而不是 1 条 —— 与现状最大的行为差异
    EXPECT_EQ(result.plan.routes.size(), 4u);
    EXPECT_EQ(rankPairs(result.plan, kFullTag), (std::set<std::pair<int, int>>{{0, 0}, {1, 0}, {2, 0}, {3, 0}}));

    EXPECT_EQ(result.plan.forDecodeRank(0).size(), 4u);
    for (int j = 0; j < 4; ++j) {
        ASSERT_EQ(result.plan.forPrefillRank(j).size(), 1u);
        const auto* r = result.plan.forPrefillRank(j).front();
        EXPECT_EQ(r->src_keys.modulus, 4);
        EXPECT_EQ(r->src_keys.residue, j);
    }
    expectNoHeadPartitioning(result.plan);
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A4: ND1P —— prefill TP 小于 decode TP
TEST(PlannerGroupA, A4_Nd1p) {
    const auto src = mhaLayout(4);
    const auto dst = mhaLayout(8);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // 每个 decode rank 恰好一条
    EXPECT_EQ(result.plan.routes.size(), 8u);
    for (int d = 0; d < 8; ++d) {
        ASSERT_EQ(result.plan.forDecodeRank(d).size(), 1u);
        const auto* r = result.plan.forDecodeRank(d).front();
        EXPECT_EQ(r->src_rank, d / 2);
        // 源端按 head 维切分后只发自己负责的那份，目的端接收整块
        EXPECT_EQ(r->src_partition, (PartitionSpec{2, d % 2}));
        EXPECT_EQ(r->dst_partition, (PartitionSpec{1, 0}));
    }
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A5: NP1D —— prefill TP 大于 decode TP（§2.3 回归）
TEST(PlannerGroupA, A5_Np1pRegressionSrcSendsWholeBlock) {
    const auto src = mhaLayout(8);
    const auto dst = mhaLayout(4);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // 每个 prefill rank 恰好一条，多个 prefill rank 汇聚到同一个 decode rank
    EXPECT_EQ(result.plan.routes.size(), 8u);
    for (int s = 0; s < 8; ++s) {
        ASSERT_EQ(result.plan.forPrefillRank(s).size(), 1u);
        const auto* r = result.plan.forPrefillRank(s).front();
        EXPECT_EQ(r->dst_rank, s / 2);

        // 回归核心：源端发送整块、不做 head 切分
        EXPECT_EQ(r->src_partition, (PartitionSpec{1, 0}))
            << "src must send the whole local block; a second split on the source side is the §2.3 defect";
        EXPECT_EQ(r->dst_partition, (PartitionSpec{2, s % 2}));
    }
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A6: MLA 的 NP1D —— 副本类内选举
TEST(PlannerGroupA, A6_MlaNp1dElectsOneSenderPerDst) {
    const auto src = mlaLayout(8);
    const auto dst = mlaLayout(4);

    // MLA 的 latent 在各 rank 间复制，head 分片数恒为 1（spec 类型的属性）
    EXPECT_EQ(src.headShardCount(kFullTag), 1);
    EXPECT_EQ(dst.headShardCount(kFullTag), 1);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // 每个 decode rank 只从一个 prefill 副本取数
    EXPECT_EQ(result.plan.routes.size(), 4u);
    EXPECT_EQ(rankPairs(result.plan, kFullTag), (std::set<std::pair<int, int>>{{0, 0}, {2, 1}, {4, 2}, {6, 3}}));

    // 落选的 prefill rank 无任务
    for (int idle : {1, 3, 5, 7}) {
        EXPECT_TRUE(result.plan.forPrefillRank(idle).empty()) << "src_rank " << idle << " should not be elected";
    }
    expectNoHeadPartitioning(result.plan);
    expectAllRouteBytesMatch(result.plan);
}

TEST(PlannerGroupA, A7a_PrefillCpModeKeepsMlaKeysReplicated) {
    const auto src = mlaLayout(8, CPRotateMethod::PREFILL_CP, /*sharded=*/true);
    const auto dst = mlaLayout(8);
    EXPECT_EQ(src.pc.get_attn_tp_size(), 8);
    EXPECT_EQ(src.headShardCount(kFullTag), 1);
    EXPECT_EQ(src.cpSize(), 1);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    EXPECT_EQ(result.plan.routes.size(), 8u);
    for (int rank = 0; rank < 8; ++rank) {
        const auto routes = result.plan.forDecodeRank(rank);
        ASSERT_EQ(routes.size(), 1u);
        EXPECT_EQ(routes.front()->src_keys.modulus, 1);
    }
    expectNoHeadPartitioning(result.plan);
    expectAllRouteBytesMatch(result.plan);
}

TEST(PlannerGroupA, A7b_PrefillCpModeKeepsMhaHeadOwnership) {
    const auto src = mhaLayout(8, CPRotateMethod::PREFILL_CP, /*sharded=*/true);
    const auto dst = mhaLayout(8);
    EXPECT_EQ(src.headShardCount(kFullTag), 8);
    EXPECT_EQ(src.cpSize(), 1);
    EXPECT_EQ(src.effectiveGlobalBlockBytes(kFullTag), dst.effectiveGlobalBlockBytes(kFullTag));

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    EXPECT_EQ(result.plan.routes.size(), 8u);
    for (const auto& route : result.plan.routes) {
        EXPECT_EQ(route.src_rank, route.dst_rank);
        EXPECT_EQ(route.src_keys.modulus, 1);
    }
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A8: hybrid 下 RR 与非 RR group 共存
TEST(PlannerGroupA, A8_HybridRrAndCompactCoexist) {
    const auto src = makeLayout(4,
                                CPRotateMethod::ALL_GATHER,
                                /*sharded=*/true,
                                {{kFullTag, makeMhaFullGroup()}, {kSwaTag, makeSwaGroup()}});
    const auto dst = makeLayout(4,
                                CPRotateMethod::ALL_GATHER,
                                /*sharded=*/true,
                                {{kFullTag, makeMhaFullGroup()}, {kSwaTag, makeSwaGroup()}});

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag, kSwaTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    EXPECT_EQ(result.plan.routes.size(), 8u);

    // FULL group：RR -> 对角线配对
    EXPECT_EQ(result.plan.forTag(kFullTag).size(), 4u);
    EXPECT_EQ(rankPairs(result.plan, kFullTag),
              (std::set<std::pair<int, int>>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}));

    // SWA group：COMPACT 是复制型子集，全部 CP rank 都是候选源，选举后统一收敛到最小者
    EXPECT_EQ(result.plan.forTag(kSwaTag).size(), 4u);
    EXPECT_EQ(rankPairs(result.plan, kSwaTag), (std::set<std::pair<int, int>>{{0, 0}, {0, 1}, {0, 2}, {0, 3}}));
    for (const auto* r : result.plan.forTag(kSwaTag)) {
        EXPECT_TRUE(r->src_keys.include_final_key) << "COMPACT group must carry the final-key exception";
    }

    expectAllRouteBytesMatch(result.plan);
}

TEST(PlannerGroupA, AsymmetricDecodeKeepsCompactGeometryButFullPagesAreUnsharded) {
    auto state = makeSwaGroup();
    state.spec_type = KVCacheSpecType::OpaqueState;
    auto src = makeLayout(4, CPRotateMethod::ALL_GATHER, true,
                          {{kFullTag, makeMlaFullGroup()}, {kFixedTag, state}});
    src.pc.role_type = RoleType::PREFILL;
    auto dst = ShardLayoutFactory::peerOf(src, 1, true, RoleType::DECODE);
    ASSERT_EQ(dst.cpSize(), 1);
    EXPECT_EQ(dst.physicalCpSize(kFullTag), 1);
    EXPECT_EQ(dst.physicalCpSize(kFixedTag), 4);
    auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag, kFixedTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    std::set<size_t> full_positions;
    for (const auto* route : result.plan.forTag(kFullTag)) {
        for (auto pos : KVCacheTransferPlanner::resolveKeys(route->src_keys, 9)) {
            full_positions.insert(pos);
        }
    }
    EXPECT_EQ(full_positions, (std::set<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
    for (const auto* route : result.plan.forTag(kFixedTag)) {
        EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(route->src_keys, 9), (std::vector<size_t>{7, 8}));
        EXPECT_EQ(CPSlotMapper::physicalBlockPosition(dst.group(kFixedTag).policy, 7, 9, 0,
                                                     dst.physicalCpSize(kFixedTag)), 1u);
        EXPECT_EQ(CPSlotMapper::physicalBlockPosition(dst.group(kFixedTag).policy, 8, 9, 0,
                                                     dst.physicalCpSize(kFixedTag)), 2u);
    }
    EXPECT_FALSE(result.plan.forTag(kFixedTag).empty());
}

// 用例 A9: CP 字节切分（prefill 预切片、decode 持整块）
TEST(PlannerGroupA, A9_CpByteSlicing) {
    auto sliced_group      = makeSwaGroup();
    sliced_group.policy.cp_slice = CpBlockSliceMode::PAYLOAD_BYTES;
    sliced_group.pre_sliced      = true;

    auto whole_group        = makeSwaGroup();
    whole_group.policy.cp_slice = CpBlockSliceMode::PAYLOAD_BYTES;
    whole_group.pre_sliced      = false;

    auto src = makeLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true, {{kFixedTag, sliced_group}});
    auto dst = makeLayout(1, CPRotateMethod::DISABLED, /*sharded=*/false, {{kFixedTag, whole_group}});

    // PAYLOAD_BYTES 模式的分母是 k_block_payload_bytes，不是 stride。这里刻意让
    // payload < stride（模拟 blockStrideBytes 的对齐上取整），以区分两种 slice 模式的分母。
    dst.groups[kFixedTag].k_block_payload_bytes = dst.group(kFixedTag).kv_block_stride_bytes - 4100;
    src.groups[kFixedTag].k_block_payload_bytes = dst.group(kFixedTag).k_block_payload_bytes / 4;
    // src 已预切片，其本地块尺寸对齐到 payload 分片
    src.groups[kFixedTag].kv_block_stride_bytes = src.group(kFixedTag).k_block_payload_bytes;

    EXPECT_EQ(src.group(kFixedTag).kv_block_stride_bytes * 4, dst.group(kFixedTag).k_block_payload_bytes);
    EXPECT_NE(src.group(kFixedTag).k_block_payload_bytes % 4, 0u);
    // dst 带对齐填充：stride > payload。PAYLOAD_BYTES 模式下守恒量是 payload 而非 stride。
    EXPECT_GT(dst.group(kFixedTag).kv_block_stride_bytes, dst.group(kFixedTag).k_block_payload_bytes);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFixedTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // 字节切片场景下各源持有互不相同的字节区间 -> 不选举、全部取用
    EXPECT_EQ(result.plan.routes.size(), 4u);
    for (int j = 0; j < 4; ++j) {
        ASSERT_EQ(result.plan.forPrefillRank(j).size(), 1u);
        const auto* r = result.plan.forPrefillRank(j).front();
        EXPECT_EQ(r->dst_rank, 0);
        // 源端不再做字节切片（其本地块本身就是切片）
        EXPECT_EQ(r->src_slice.mode, CpBlockSliceMode::NONE);
        // 目的端按源端的 CP 几何切出落点偏移
        EXPECT_EQ(r->dst_slice, (SliceSpec{CpBlockSliceMode::PAYLOAD_BYTES, 4, j}));
    }
    expectNoHeadPartitioning(result.plan);
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A10: CP 与 TP 同时不对称
TEST(PlannerGroupA, A10_CpAndTpAsymmetricSimultaneously) {
    // prefill: 序列切 4 份、head 复制
    const auto src = mhaLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true);
    // decode: head 切 4 份、序列完整
    const auto dst = mhaLayout(4);

    EXPECT_EQ(src.cpSize(), 4);
    EXPECT_EQ(src.headShardCount(kFullTag), 1);
    EXPECT_EQ(dst.cpSize(), 1);
    EXPECT_EQ(dst.headShardCount(kFullTag), 4);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // route 数是两个维度的乘积
    EXPECT_EQ(result.plan.routes.size(), 16u);
    for (int p = 0; p < 4; ++p) {
        EXPECT_EQ(result.plan.forPrefillRank(p).size(), 4u);
    }
    for (int d = 0; d < 4; ++d) {
        EXPECT_EQ(result.plan.forDecodeRank(d).size(), 4u);
    }

    for (const auto& r : result.plan.routes) {
        // 键规则由 **源** rank 的 CP 位置决定
        EXPECT_EQ(r.src_keys.modulus, 4);
        EXPECT_EQ(r.src_keys.residue, src.cpRank(r.src_rank));
        // head 切分参数由 **目的** rank 的 head 位置决定
        EXPECT_EQ(r.src_partition, (PartitionSpec{4, dst.headShard(r.dst_rank, kFullTag)}));
        EXPECT_EQ(r.dst_partition, (PartitionSpec{1, 0}));
    }
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A11: 两侧都分片且不相等 —— 不支持，必须拒绝
//
// 我们今天只支持两侧 CP 完全相等（§2.1），本设计新增的能力是 prefill CP N -> decode CP 1。
// 「两侧都分片且不相等」（如 p cp=2 -> d cp=4）需要「模 lcm 的剩余类 + CRT」整套机制，
// 且 vLLM / SGLang 同样禁止两侧 CP 并存，故明确不支持。
TEST(PlannerGroupA, A11_BothSidesShardedButUnequalIsRejected) {
    const auto src = mhaLayout(2, CPRotateMethod::ALL_GATHER, /*sharded=*/true);
    const auto dst = mhaLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true);

    EXPECT_EQ(src.cpSize(), 2);
    EXPECT_EQ(dst.cpSize(), 4);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});

    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("unsupported CP topology"), std::string::npos)
        << result.error.ToString();
    EXPECT_TRUE(result.plan.routes.empty());
}

// 用例 A11b: 两侧 CP 相等仍然支持（今天的对称形态不能被 A11 的白名单误伤）
TEST(PlannerGroupA, A11b_BothSidesShardedAndEqualIsAllowed) {
    const auto src = mhaLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true);
    const auto dst = mhaLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    EXPECT_EQ(result.plan.routes.size(), 4u);
    EXPECT_EQ(rankPairs(result.plan, kFullTag),
              (std::set<std::pair<int, int>>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}));
    for (const auto* r : result.plan.forTag(kFullTag)) {
        // 白名单挡住不等情形之后，模数恒为源端 CP 片数、剩余值恒为源端 CP 位置
        EXPECT_EQ(r->src_keys.modulus, 4);
        EXPECT_EQ(r->src_keys.residue, src.cpRank(r->src_rank));
    }
    expectAllRouteBytesMatch(result.plan);
}

TEST(PlannerGroupA, LocalFullPageRoundRobinRoutesEachKeyOnce) {
    for (const auto& [src_tp, dst_tp] : {std::pair<int, int>{8, 4}, {4, 8}, {8, 8}}) {
        auto src               = mlaLayout(src_tp, CPRotateMethod::ALL_GATHER, true);
        auto dst               = mlaLayout(dst_tp);
        src.local_full_page_rr = true;
        dst.local_full_page_rr = true;
        dst.pc.role_type       = RoleType::DECODE;

        const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
        ASSERT_TRUE(result.ok()) << result.error.ToString();
        ASSERT_EQ(result.plan.routes.size(), static_cast<size_t>(std::max(src_tp, dst_tp)));
        constexpr size_t kKeyCount = 17;
        std::vector<int> visits(kKeyCount, 0);
        for (const auto& route : result.plan.routes) {
            EXPECT_EQ(route.filter_prefill_keys, src_tp < dst_tp);
            for (const auto pos : KVCacheTransferPlanner::resolveKeys(route.src_keys, kKeyCount)) {
                EXPECT_EQ(route.src_rank, static_cast<int>(pos % src_tp));
                EXPECT_EQ(route.dst_rank, static_cast<int>(pos % dst_tp));
                ++visits[pos];
            }
        }
        EXPECT_EQ(visits, std::vector<int>(kKeyCount, 1));
        expectAllRouteBytesMatch(result.plan);
    }
}

// ===========================================================================
// 组 B：键选择规则的展开
// ===========================================================================

// 用例 B1: 剩余类规则展开
TEST(PlannerGroupB, B1_ResidueClassExpansion) {
    const KeyShardSpec spec{/*modulus=*/4, /*residue=*/1};
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(spec, 8), (std::vector<size_t>{1, 5}));
}

// 用例 B2: 全取规则展开
TEST(PlannerGroupB, B2_TakeAllExpansion) {
    const KeyShardSpec spec{/*modulus=*/1, /*residue=*/0};
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(spec, 8), (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

// 用例 B3: 尾键例外
TEST(PlannerGroupB, B3_FinalKeyException) {
    KeyShardSpec spec{/*modulus=*/4, /*residue=*/3};
    spec.include_final_key = true;

    // 末位 9 不落在剩余类 {3,7} 上 -> 必须被额外取到
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(spec, 10), (std::vector<size_t>{3, 7, 9}));

    // 末位恰好落在剩余类内时不应重复
    const auto keys = KVCacheTransferPlanner::resolveKeys(spec, 8);
    EXPECT_EQ(keys, (std::vector<size_t>{3, 7}));
    EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
    EXPECT_EQ(std::set<size_t>(keys.begin(), keys.end()).size(), keys.size());
}

// 用例 B7: active_tail_blocks 折进规则后只取末尾若干项
TEST(PlannerGroupB, B7_TailCountKeepsOnlyTheLastEntries) {
    KeyShardSpec spec{/*modulus=*/1, /*residue=*/0};
    spec.tail_count = 2;  // SWA 语义：只传末两块
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(spec, 8), (std::vector<size_t>{6, 7}));

    spec.tail_count = 1;  // LINEAR 语义：只传尾块
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(spec, 8), (std::vector<size_t>{7}));

    // 键数不足 tail_count 时全取，不报错
    spec.tail_count = 5;
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(spec, 3), (std::vector<size_t>{0, 1, 2}));

    // 与剩余类叠加：先筛剩余类，再取尾
    KeyShardSpec rr{/*modulus=*/4, /*residue=*/1};
    rr.tail_count = 2;
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(rr, 16), (std::vector<size_t>{9, 13}));
}

// 用例 B8: SWA group 的 tail_count 由编排层写入，两侧必然相同
TEST(PlannerGroupB, B8_TailCountIsFoldedByPlanner) {
    const auto src = makeLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true, {{kSwaTag, makeSwaGroup()}});
    const auto dst = makeLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true, {{kSwaTag, makeSwaGroup()}});

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kSwaTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // defaultCacheGroupPolicy(SWA) 的 active_tail_blocks == 2
    for (const auto* r : result.plan.forTag(kSwaTag)) {
        EXPECT_EQ(r->src_keys.tail_count, 2);
    }
}

TEST(PlannerGroupB, StateKeySpanDoesNotFollowDecodeCpSize) {
    const auto src    = makeLayout(4, CPRotateMethod::ALL_GATHER, true, {{kSwaTag, makeSwaGroup()}});
    const auto dst    = makeLayout(1, CPRotateMethod::PREFILL_CP, true, {{kSwaTag, makeSwaGroup()}});
    const auto result = KVCacheTransferPlanner::plan(src, dst, {kSwaTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    ASSERT_EQ(result.plan.routes.size(), 1u);
    const auto& rule = result.plan.routes.front().src_keys;
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(rule, 280), (std::vector<size_t>{275, 279}));
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(rule, 281), (std::vector<size_t>{279, 280}));
    const std::map<size_t, std::vector<size_t>> partial_tails = {
        {1, {0}}, {3, {2}}, {4, {3}}, {5, {3, 4}}, {7, {3, 6}}, {8, {3, 7}}, {9, {7, 8}}};
    for (const auto& [count, expected] : partial_tails) {
        EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(rule, count), expected) << "key_count=" << count;
    }
}

TEST(PlannerGroupB, LinearStateRejectsUnsupportedAsymmetricHeadSharding) {
    auto group      = makeMhaFullGroup();
    group.spec_type = KVCacheSpecType::LinearAttention;
    group.policy    = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    group.policy.cp_slice = CpBlockSliceMode::NONE;
    for (const auto& sizes : {std::pair<int, int>{2, 1}, {1, 2}}) {
        const auto src = makeLayout(sizes.first, CPRotateMethod::DISABLED, false, {{kFixedTag, group}});
        const auto dst = makeLayout(sizes.second, CPRotateMethod::DISABLED, false, {{kFixedTag, group}});
        ASSERT_EQ(src.headShardCount(kFixedTag), sizes.first);
        ASSERT_EQ(dst.headShardCount(kFixedTag), sizes.second);
        const auto result = KVCacheTransferPlanner::plan(src, dst, {kFixedTag});
        ASSERT_FALSE(result.ok());
        EXPECT_NE(result.error.ToString().find("LinearAttention asymmetric head sharding"), std::string::npos);
    }
    const auto symmetric = makeLayout(2, CPRotateMethod::DISABLED, false, {{kFixedTag, group}});
    const auto result = KVCacheTransferPlanner::plan(symmetric, symmetric, {kFixedTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    for (const auto& route : result.plan.routes) {
        EXPECT_EQ(route.src_partition.count, 1);
        EXPECT_EQ(route.dst_partition.count, 1);
    }
}

TEST(PlannerGroupB, LegacyLinearStateUsesHeadShardsWithLocalPageRoundRobin) {
    auto linear_group      = makeMhaFullGroup();
    linear_group.spec_type = KVCacheSpecType::LinearAttention;
    linear_group.policy    = defaultCacheGroupPolicy(CacheGroupType::LINEAR);

    auto src =
        makeLayout(2, CPRotateMethod::DISABLED, true, {{kFullTag, makeMlaFullGroup()}, {kFixedTag, linear_group}});
    src.pc.role_type       = RoleType::PREFILL;
    src.local_full_page_rr = true;
    auto dst               = src;
    dst.pc.role_type       = RoleType::DECODE;

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag, kFixedTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    EXPECT_EQ(rankPairs(result.plan, kFixedTag), (std::set<std::pair<int, int>>{{0, 0}, {1, 1}}));
    const auto linear_routes = result.plan.forTag(kFixedTag);
    ASSERT_EQ(linear_routes.size(), 2u);
    for (const auto* route : linear_routes) {
        EXPECT_FALSE(route->linear_qkv_segments);
        EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(route->src_keys, 5), (std::vector<size_t>{4}));
    }
    const auto full_routes = result.plan.forTag(kFullTag);
    ASSERT_EQ(full_routes.size(), 2u);
    for (const auto* route : full_routes) {
        EXPECT_EQ(route->src_rank, route->dst_rank);
        EXPECT_EQ(route->src_keys.modulus, 2);
        EXPECT_EQ(route->src_keys.residue, route->src_rank);
    }
    expectNoHeadPartitioning(result.plan);
    expectNoSlicing(result.plan);
    expectAllRouteBytesMatch(result.plan);
}

TEST(PlannerGroupB, LinearQkvRoutesHeadPartitionsIndependentlyOfPageRoundRobin) {
    const auto linear_layout = [](int tp, RoleType role) {
        ShardLayout layout;
        layout.pc                                      = makePc(tp, CPRotateMethod::ALL_GATHER, true);
        layout.pc.role_type                            = role;
        layout.pc.prefill_cp_config.preserve_tp_layout = true;
        layout.local_full_page_rr                      = true;
        layout.cache_key_tokens_per_block              = kSeqSizePerBlock;
        auto&        group                             = layout.groups[kFixedTag];
        const size_t factor                            = 8 / tp;
        group.policy                                   = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
        group.spec_type                                = KVCacheSpecType::LinearAttention;
        group.seq_size_per_block                       = kSeqSizePerBlock;
        group.k_block_payload_bytes                    = 64 * factor;
        group.linear_history_rows                      = 3;
        group.linear_q_row_bytes                       = 8 * factor;
        group.linear_k_row_bytes                       = 8 * factor;
        group.linear_v_row_bytes                       = 16 * factor;
        group.kv_block_stride_bytes                    = (64 + 3 * (8 + 8 + 16)) * factor;
        layout.deriveHeadShardCounts();
        return layout;
    };

    const auto src    = linear_layout(8, RoleType::PREFILL);
    auto       dst    = linear_layout(4, RoleType::DECODE);
    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFixedTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    ASSERT_EQ(result.plan.routes.size(), 8u);
    for (const auto& route : result.plan.routes) {
        EXPECT_TRUE(route.linear_qkv_segments);
        EXPECT_EQ(route.dst_rank, route.src_rank / 2);
        EXPECT_EQ(route.src_partition, (PartitionSpec{1, 0}));
        EXPECT_EQ(route.dst_partition, (PartitionSpec{2, route.src_rank % 2}));
        EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(route.src_keys, 17), (std::vector<size_t>{16}));
    }
    expectAllRouteBytesMatch(result.plan);

    // Total bytes still match, but changing the component boundaries is unsafe.
    ++dst.groups[kFixedTag].linear_q_row_bytes;
    --dst.groups[kFixedTag].linear_v_row_bytes;
    const auto mismatched = KVCacheTransferPlanner::plan(src, dst, {kFixedTag});
    EXPECT_FALSE(mismatched.ok());
    EXPECT_NE(mismatched.error.ToString().find("incompatible LINEAR_QKV layout"), std::string::npos);
}

TEST(PlannerGroupB, UnscaledStateGroupDoesNotCompactKeysByTransportCp) {
    auto group               = makeSwaGroup();
    group.seq_size_per_block = kSeqSizePerBlock;
    const auto src           = makeLayout(4, CPRotateMethod::ALL_GATHER, true, {{kSwaTag, group}});
    const auto dst           = makeLayout(1, CPRotateMethod::PREFILL_CP, true, {{kSwaTag, group}});
    const auto result        = KVCacheTransferPlanner::plan(src, dst, {kSwaTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    ASSERT_EQ(result.plan.routes.size(), 1u);
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(result.plan.routes.front().src_keys, 8), (std::vector<size_t>{6, 7}));
}

TEST(PlannerGroupB, TokenStrideMismatchIsRejectedAndStrideChangesAffectDigest) {
    auto       src      = makeLayout(4, CPRotateMethod::ALL_GATHER, true, {{kSwaTag, makeSwaGroup()}});
    auto       dst      = makeLayout(1, CPRotateMethod::PREFILL_CP, true, {{kSwaTag, makeSwaGroup()}});
    const auto original = KVCacheTransferPlanner::plan(src, dst, {kSwaTag});
    ASSERT_TRUE(original.ok()) << original.error.ToString();
    dst.groups.at(kSwaTag).seq_size_per_block = kSeqSizePerBlock;
    EXPECT_FALSE(KVCacheTransferPlanner::plan(src, dst, {kSwaTag}).ok());
    dst.groups.at(kSwaTag).seq_size_per_block = 4 * kSeqSizePerBlock;
    dst.cache_key_tokens_per_block *= 2;
    EXPECT_FALSE(KVCacheTransferPlanner::plan(src, dst, {kSwaTag}).ok());
    src.cache_key_tokens_per_block *= 2;
    src.groups.at(kSwaTag).seq_size_per_block *= 2;
    dst.groups.at(kSwaTag).seq_size_per_block *= 2;
    const auto scaled = KVCacheTransferPlanner::plan(src, dst, {kSwaTag});
    ASSERT_TRUE(scaled.ok()) << scaled.error.ToString();
    EXPECT_EQ(original.plan.routes.front().src_keys, scaled.plan.routes.front().src_keys);
    EXPECT_NE(original.plan.digest(), scaled.plan.digest());
    dst.groups.at(kSwaTag).seq_size_per_block += 1;
    EXPECT_FALSE(KVCacheTransferPlanner::plan(dst, dst, {kSwaTag}).ok());
    dst.cache_key_tokens_per_block = 0;
    EXPECT_FALSE(KVCacheTransferPlanner::plan(dst, dst, {kSwaTag}).ok());
}

// 用例 B9: 两侧 active_tail_blocks 不一致必须拒绝
TEST(PlannerGroupB, B9_TailCountMismatchIsRejected) {
    auto shifted                      = makeSwaGroup();
    shifted.policy.active_tail_blocks = 3;

    const auto src = makeLayout(4, CPRotateMethod::ALL_GATHER, true, {{kSwaTag, makeSwaGroup()}});
    const auto dst = makeLayout(4, CPRotateMethod::ALL_GATHER, true, {{kSwaTag, shifted}});

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kSwaTag});
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("active_tail_blocks mismatch"), std::string::npos)
        << result.error.ToString();
}

TEST(PlannerGroupB, NonFullGroupCannotUseRoundRobinCp) {
    auto sharded_swa              = makeSwaGroup();
    sharded_swa.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    const auto src                = makeLayout(2, CPRotateMethod::ALL_GATHER, true, {{kSwaTag, sharded_swa}});
    const auto dst                = makeLayout(1, CPRotateMethod::DISABLED, false, {{kSwaTag, sharded_swa}});
    const auto result             = KVCacheTransferPlanner::plan(src, dst, {kSwaTag});
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("round-robin CP requires a FULL P2P group"), std::string::npos)
        << result.error.ToString();
}

TEST(PlannerGroupB, MlaModelCannotPartitionAnMhaGroupAsymmetrically) {
    auto two_shards  = mhaLayout(2);
    auto four_shards = mhaLayout(4);
    ASSERT_TRUE(KVCacheTransferPlanner::plan(two_shards, four_shards, {kFullTag}).ok());
    two_shards.use_mla = four_shards.use_mla = true;

    for (const auto& result : {KVCacheTransferPlanner::plan(two_shards, four_shards, {kFullTag}),
                               KVCacheTransferPlanner::plan(four_shards, two_shards, {kFullTag})}) {
        ASSERT_FALSE(result.ok());
        EXPECT_NE(result.error.ToString().find("asymmetric MHA head sharding requires block partitioning"),
                  std::string::npos);
    }
    EXPECT_TRUE(KVCacheTransferPlanner::plan(two_shards, two_shards, {kFullTag}).ok());
}

TEST(PlannerGroupB, HybridPrefillCannotTailLimitRoundRobinFullGroup) {
    auto full_tail                      = makeMlaFullGroup();
    full_tail.policy.active_tail_blocks = 2;
    auto src = makeLayout(2, CPRotateMethod::ALL_GATHER, true, {{kFullTag, full_tail}, {kSwaTag, makeSwaGroup()}});
    src.use_group_cache_transfer_policy = true;
    auto dst = makeLayout(1, CPRotateMethod::DISABLED, false, {{kFullTag, full_tail}, {kSwaTag, makeSwaGroup()}});
    dst.use_group_cache_transfer_policy = true;
    const auto peer_src                 = ShardLayout::forPeer(dst, src.pc, true);
    const auto on_prefill               = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    const auto on_decode                = KVCacheTransferPlanner::plan(peer_src, dst, {kFullTag});
    ASSERT_FALSE(on_prefill.ok());
    ASSERT_FALSE(on_decode.ok());
    EXPECT_NE(on_prefill.error.ToString().find("cannot use active_tail_blocks in a hybrid Prefill layout"),
              std::string::npos)
        << on_prefill.error.ToString();
    EXPECT_EQ(on_decode.error.ToString(), on_prefill.error.ToString());
}

// 用例 B4: 解析为空不算错误
TEST(PlannerGroupB, B4_EmptyResolutionIsNotAnError) {
    const KeyShardSpec spec{/*modulus=*/4, /*residue=*/3};
    EXPECT_TRUE(KVCacheTransferPlanner::resolveKeys(spec, 2).empty());
}

// 用例 B5: 非对称 CP 的 route 集在真实长度下完备且互斥
TEST(PlannerGroupB, B5_AsymmetricCpRoutesAreCompleteAndDisjoint) {
    const auto src    = mhaLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true);
    const auto dst    = mhaLayout(1);
    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    constexpr size_t kCount = 8;
    std::multiset<size_t> all;
    for (const auto* r : result.plan.forDecodeRank(0)) {
        for (size_t key : KVCacheTransferPlanner::resolveKeys(r->src_keys, kCount)) {
            all.insert(key);
        }
    }
    // 两两不相交 == 没有重复项
    EXPECT_EQ(std::set<size_t>(all.begin(), all.end()).size(), all.size());
    // 并集等于 decode rank0 实际需要的全部键
    std::set<size_t> expected;
    for (size_t i = 0; i < kCount; ++i) {
        expected.insert(i);
    }
    EXPECT_EQ(std::set<size_t>(all.begin(), all.end()), expected);
}

// 用例 B6: 复制型源的键集折叠成「全取」
//
// 原 B6 覆盖的是「CP 变细时同一源拆给多个 dst」，该形态已在 A11 被明确拒绝。
// 这里改为覆盖 collapseResidues 真正还需要的场合：源端 mapping 为 NONE（复制）且 cp_size > 1 时，
// 选举出的那条 route 覆盖全部剩余类，必须被折叠成 modulus=1 的「全取」，
// 否则只会取到 1/cp_size 的键。
TEST(PlannerGroupB, B6_ReplicatedSourceCollapsesToTakeAll) {
    auto replicated_group           = makeMhaFullGroup();
    replicated_group.policy.cp_mapping = CpBlockMappingMode::NONE;  // 复制而非 RR

    const auto src = makeLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true, {{kFullTag, replicated_group}});
    const auto dst = makeLayout(1, CPRotateMethod::DISABLED, /*sharded=*/false, {{kFullTag, replicated_group}});
    EXPECT_EQ(src.cpSize(), 4);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // 选举后每个 dst rank 只有一条 route，且键规则被折叠成「全取」
    ASSERT_EQ(result.plan.forDecodeRank(0).size(), 1u);
    const auto* r = result.plan.forDecodeRank(0).front();
    EXPECT_EQ(r->src_keys.modulus, 1);
    EXPECT_EQ(r->src_keys.residue, 0);
    EXPECT_EQ(src.cpRank(r->src_rank), 0);

    constexpr size_t kCount = 8;
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(r->src_keys, kCount).size(), kCount);
}

// ===========================================================================
// 组 C：编排期前置校验
// ===========================================================================

// 用例 C1: head 分片数不整除
TEST(PlannerGroupC, C1_HeadShardCountNotDivisible) {
    auto src = mhaLayout(6);
    auto dst = mhaLayout(4);
    // 让两侧还原出的全局块大小严格一致（须同时被 6 与 4 整除），把失败原因隔离到 head 整除性上。
    // 基线常量 262144 不能被 6 整除，直接沿用会先触发字节维度校验。
    ASSERT_EQ(src.headShardCount(kFullTag), 6);
    ASSERT_EQ(dst.headShardCount(kFullTag), 4);
    src.groups[kFullTag].kv_block_stride_bytes = 131072;  // 还原全局 = 6 * 131072 = 786432
    dst.groups[kFullTag].kv_block_stride_bytes = 196608;  // 还原全局 = 4 * 196608 = 786432
    ASSERT_EQ(src.effectiveGlobalBlockBytes(kFullTag), dst.effectiveGlobalBlockBytes(kFullTag));

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("not divisible"), std::string::npos) << result.error.ToString();
    EXPECT_NE(result.error.ToString().find("6"), std::string::npos);
    EXPECT_NE(result.error.ToString().find("4"), std::string::npos);
    EXPECT_TRUE(result.plan.routes.empty());
}

// 用例 C2: 两侧全局块大小不一致
TEST(PlannerGroupC, C2_GlobalBlockBytesMismatch) {
    auto       src = mhaLayout(8);
    const auto dst = mhaLayout(4);
    src.groups[kFullTag].kv_block_stride_bytes = 16384;  // 全局还原为 131072 != 262144

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("global block bytes mismatch"), std::string::npos)
        << result.error.ToString();
    EXPECT_NE(result.error.ToString().find("131072"), std::string::npos);
    EXPECT_NE(result.error.ToString().find("262144"), std::string::npos);
    EXPECT_TRUE(result.plan.routes.empty());
}

// 用例 C3: 两侧 spec 类型不一致（MLA vs MHA）
TEST(PlannerGroupC, C3_SpecTypeMismatch) {
    const auto src = mlaLayout(8);
    const auto dst = mhaLayout(8);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("spec_type mismatch"), std::string::npos) << result.error.ToString();
}

// 用例 C4: 同名 group 的类型不一致
TEST(PlannerGroupC, C4_GroupTypeMismatch) {
    const auto src = makeLayout(4, CPRotateMethod::DISABLED, false, {{kFullTag, makeMhaFullGroup()}});
    auto       swa_as_full = makeSwaGroup();
    const auto dst         = makeLayout(4, CPRotateMethod::DISABLED, false, {{kFullTag, swa_as_full}});

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("group_type mismatch"), std::string::npos) << result.error.ToString();
    EXPECT_NE(result.error.ToString().find(kFullTag), std::string::npos);
}

// 用例 C6: tag 在一侧缺失
TEST(PlannerGroupC, C6_MissingTag) {
    const auto src = mhaLayout(4);
    const auto dst = mhaLayout(4);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {"nonexistent"});
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("missing"), std::string::npos) << result.error.ToString();
}

// ===========================================================================
// 组 D：属性测试
// ===========================================================================
namespace {

struct LayoutCandidate {
    int             tp_size;
    CPRotateMethod  method;
    bool            sharded;
    KVCacheSpecType spec_type;
};

std::vector<LayoutCandidate> propertyCandidates() {
    std::vector<LayoutCandidate> out;
    for (int tp : {1, 2, 4, 8}) {
        for (auto spec : {KVCacheSpecType::MultiHeadAttention, KVCacheSpecType::MultiHeadLatentAttention}) {
            out.push_back({tp, CPRotateMethod::DISABLED, false, spec});
            if (tp > 1) {
                out.push_back({tp, CPRotateMethod::ALL_GATHER, true, spec});
            }
        }
    }
    return out;
}

ShardLayout buildCandidate(const LayoutCandidate& c) {
    ShardLayout::GroupLayout g;
    g.policy             = defaultCacheGroupPolicy(CacheGroupType::FULL);
    g.spec_type          = c.spec_type;
    g.seq_size_per_block = kSeqSizePerBlock;
    return makeLayout(c.tp_size, c.method, c.sharded, {{kFullTag, g}});
}

/// dst rank d 在给定序列长度下实际需要的逻辑位置。
std::set<size_t> neededKeys(const ShardLayout& dst, const std::string& tag, int dst_rank, size_t count) {
    std::set<size_t>  out;
    CacheGroupPolicy policy = dst.group(tag).policy;
    policy.cp_mapping        = dst.effectiveMapping(tag);
    for (size_t pos = 0; pos < count; ++pos) {
        if (CPSlotMapper::physicalBlockPosition(policy, pos, count, dst.cpRank(dst_rank), dst.cpSize())
                .has_value()) {
            out.insert(pos);
        }
    }
    return out;
}

}  // namespace

// 用例 D1 + D2: 完备且互斥、两侧形状相等
//
// 互斥性必须按「目的端字节区间」判定，不能只按 key 判定：当 head 轴被切分（或 CP 字节
// 切分生效）时，同一个 cache_key 会合法地出现在多条 route 上，各自写入目的块的不同字节段。
// 因此判据是 (key, dst_partition.id, dst_slice.index) 三元组，且对每个 key 这些下标必须
// 恰好覆盖 dst_partition.count * dst_slice.count 个格子 —— 即不重叠且无空洞。
TEST(PlannerGroupD, D1_D2_CompletenessDisjointnessAndShape) {
    const auto candidates = propertyCandidates();
    int        checked    = 0;

    for (const auto& sc : candidates) {
        for (const auto& dc : candidates) {
            if (sc.spec_type != dc.spec_type) {
                continue;  // Step 1 会拒绝，另有 C3 覆盖
            }
            const auto src    = buildCandidate(sc);
            const auto dst    = buildCandidate(dc);
            const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
            if (!result.ok()) {
                continue;  // 未通过 Step 1 的组合不在本属性的范围内
            }
            ++checked;

            const std::string ctx = "src(tp=" + std::to_string(sc.tp_size) + ",sharded=" + std::to_string(sc.sharded)
                                    + ") dst(tp=" + std::to_string(dc.tp_size) + ",sharded="
                                    + std::to_string(dc.sharded) + ")";

            // D2：两侧字节数相等
            expectAllRouteBytesMatch(result.plan);

            for (size_t count : {size_t(1), size_t(7), size_t(8), size_t(33)}) {
                for (int d = 0; d < dst.rankCount(); ++d) {
                    // key -> 该 key 上被写入的目的字节格子集合
                    std::map<size_t, std::multiset<std::pair<int, int>>> cells;
                    std::map<size_t, std::pair<int, int>>                grid;  // key -> (partition.count, slice.count)

                    for (const auto* r : result.plan.forDecodeRankAndTag(d, kFullTag)) {
                        const int p_count = std::max(1, r->dst_partition.count);
                        const int s_count = std::max(1, r->dst_slice.count);
                        for (size_t key : KVCacheTransferPlanner::resolveKeys(r->src_keys, count)) {
                            auto it = grid.find(key);
                            if (it == grid.end()) {
                                grid[key] = {p_count, s_count};
                            } else {
                                EXPECT_EQ(it->second, std::make_pair(p_count, s_count))
                                    << "inconsistent destination grid for key " << key << ", " << ctx;
                            }
                            cells[key].insert({r->dst_partition.id, r->dst_slice.index});
                        }
                    }

                    // 完备：覆盖到的 key 集合等于该 dst rank 实际需要的 key 集合
                    std::set<size_t> covered;
                    for (const auto& [key, _] : cells) {
                        covered.insert(key);
                    }
                    EXPECT_EQ(covered, neededKeys(dst, kFullTag, d, count))
                        << "incomplete key coverage: " << ctx << " count=" << count << " dst_rank=" << d;

                    // 互斥且无空洞：每个 key 上的字节格子恰好铺满一次
                    for (const auto& [key, occupied] : cells) {
                        const std::set<std::pair<int, int>> unique(occupied.begin(), occupied.end());
                        EXPECT_EQ(unique.size(), occupied.size())
                            << "overlapping destination byte range on key " << key << ", " << ctx
                            << " count=" << count << " dst_rank=" << d;
                        const auto [p_count, s_count] = grid.at(key);
                        EXPECT_EQ(unique.size(), static_cast<size_t>(p_count * s_count))
                            << "destination byte range not fully covered on key " << key << ", " << ctx
                            << " count=" << count << " dst_rank=" << d;
                    }
                }
            }
        }
    }
    EXPECT_GT(checked, 0);
}

// 用例 D3: 幂等与稳定
TEST(PlannerGroupD, D3_IdempotentAndStable) {
    const auto src = mhaLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true);
    const auto dst = mhaLayout(2);

    const auto a = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    const auto b = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(a.ok()) << a.error.ToString();
    ASSERT_TRUE(b.ok());

    ASSERT_EQ(a.plan.routes.size(), b.plan.routes.size());
    for (size_t i = 0; i < a.plan.routes.size(); ++i) {
        const auto& x = a.plan.routes[i];
        const auto& y = b.plan.routes[i];
        EXPECT_EQ(x.route_id, y.route_id);
        EXPECT_EQ(x.src_rank, y.src_rank);
        EXPECT_EQ(x.dst_rank, y.dst_rank);
        EXPECT_EQ(x.cache_tag, y.cache_tag);
        EXPECT_EQ(x.src_keys, y.src_keys);
        EXPECT_EQ(x.src_partition, y.src_partition);
        EXPECT_EQ(x.dst_partition, y.dst_partition);
        EXPECT_EQ(x.src_slice, y.src_slice);
        EXPECT_EQ(x.dst_slice, y.dst_slice);
    }
    EXPECT_EQ(a.plan.digest(), b.plan.digest());
}

TEST(PlannerGroupD, DigestDistinguishesTailKeySelection) {
    const auto result = KVCacheTransferPlanner::plan(mhaLayout(1), mhaLayout(1), {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();
    ASSERT_EQ(result.plan.routes.size(), 1u);

    auto last_one = result.plan;
    auto last_two = result.plan;
    last_one.routes[0].src_keys.tail_count = 1;
    last_two.routes[0].src_keys.tail_count = 2;

    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(last_one.routes[0].src_keys, 4),
              (std::vector<size_t>{3}));
    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(last_two.routes[0].src_keys, 4),
              (std::vector<size_t>{2, 3}));
    EXPECT_NE(last_one.digest(), last_two.digest());
    EXPECT_NE(result.plan.digest(), last_one.digest());
    EXPECT_NE(result.plan.digest(), last_two.digest());
}

// 从真实状态缓存 spec 构造两端，避免手工填写 payload 掩盖 factory 的推导错误。
TEST(PlannerGroupD, FactoryMirrorsCpStatePayloadFromRealSpecs) {
    for (int cp_size : {2, 4}) {
        for (int decode_tp_size : {1, cp_size}) {
            for (auto mode : {CpBlockSliceMode::PAYLOAD_BYTES, CpBlockSliceMode::EQUAL_BYTES}) {
                SCOPED_TRACE(::testing::Message() << "cp_size=" << cp_size << " decode_tp=" << decode_tp_size
                                                  << " mode=" << static_cast<int>(mode));
                auto prefill_pc = makePc(cp_size, CPRotateMethod::ALL_GATHER, true);
                prefill_pc.role_type = RoleType::PREFILL;
                auto decode_pc = makePc(decode_tp_size, CPRotateMethod::PREFILL_CP, true);
                decode_pc.role_type = RoleType::DECODE;
                decode_pc.prefill_cp_config.prefill_cp_size = cp_size;

                auto local_layout = [mode](const ParallelismConfig& pc) {
                    KVCacheSpecDesc desc;
                    desc.tag                  = kFixedTag;
                    desc.cache_type           = KVCacheSpecType::OpaqueState;
                    desc.dtype                = DataType::TYPE_UINT8;
                    desc.entry_dtype          = DataType::TYPE_UINT8;
                    desc.entry_elems           = 64;
                    desc.entry_count_mode      = OpaqueBlockEntryCountMode::EXPLICIT;
                    desc.explicit_entry_count = 8;
                    desc.cp                   = CacheCpPolicyDesc{};
                    desc.cp->slice            = mode;
                    desc.cp->prefill_slice_layout = mode == CpBlockSliceMode::PAYLOAD_BYTES ?
                                                       CpPrefillSliceLayout::PAYLOAD :
                                                       CpPrefillSliceLayout::BLOCK_STRIDE;
                    SpecBuildContext ctx;
                    ctx.seq_size_per_block = kSeqSizePerBlock;
                    ctx.parallelism_config = &pc;
                    auto spec = FixedStateCacheSpec::build(desc, ctx);
                    GroupBase group;
                    group.tag                       = kFixedTag;
                    group.spec                      = spec;
                    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::SWA);
                    group.policy.cp_slice           = mode;
                    group.block_num                 = 8;
                    auto topology = CacheTopology::create({group}, {{0, {kFixedTag}}});
                    return ShardLayoutFactory::fromTopology(*topology, pc, pc.role_type, kSeqSizePerBlock);
                };
                const auto prefill = local_layout(prefill_pc);
                const auto decode = local_layout(decode_pc);
                const auto inferred_prefill =
                    ShardLayoutFactory::peerOf(decode, cp_size, true, RoleType::PREFILL);
                const auto inferred_decode =
                    ShardLayoutFactory::peerOf(prefill, decode_tp_size, true, RoleType::DECODE);
                EXPECT_EQ(inferred_prefill.group(kFixedTag).k_block_payload_bytes,
                          prefill.group(kFixedTag).k_block_payload_bytes);
                EXPECT_EQ(inferred_decode.group(kFixedTag).k_block_payload_bytes,
                          decode.group(kFixedTag).k_block_payload_bytes);
                EXPECT_EQ(inferred_prefill.group(kFixedTag).kv_block_stride_bytes,
                          prefill.group(kFixedTag).kv_block_stride_bytes);
                EXPECT_EQ(inferred_decode.group(kFixedTag).kv_block_stride_bytes,
                          decode.group(kFixedTag).kv_block_stride_bytes);

                const auto on_decode = KVCacheTransferPlanner::plan(inferred_prefill, decode, {kFixedTag});
                const auto on_prefill = KVCacheTransferPlanner::plan(prefill, inferred_decode, {kFixedTag});
                ASSERT_TRUE(on_decode.ok()) << on_decode.error.ToString();
                ASSERT_TRUE(on_prefill.ok()) << on_prefill.error.ToString();
                EXPECT_FALSE(on_decode.plan.routes.empty());
                EXPECT_EQ(on_decode.plan.digest(), on_prefill.plan.digest());
                for (const auto& route : on_decode.plan.routes) {
                    EXPECT_EQ(route.src_bytes, prefill.group(kFixedTag).kv_block_stride_bytes);
                    EXPECT_EQ(route.dst_bytes, route.src_bytes);
                }
            }
        }
    }
}

TEST(PlannerGroupD, FactoryMirrorsReplicatedMhaHeadShards) {
    AttentionConfigs attention{};
    attention.kv_head_num   = 4;
    attention.size_per_head = 128;
    auto make_mha_layout    = [&attention](int tp_size, RoleType role) {
        auto pc      = makePc(tp_size, CPRotateMethod::DISABLED, false);
        pc.role_type = role;
        KVCacheSpecDesc desc;
        desc.tag        = kFullTag;
        desc.cache_type = KVCacheSpecType::MultiHeadAttention;
        desc.dtype      = DataType::TYPE_FP16;
        SpecBuildContext ctx;
        ctx.seq_size_per_block = kSeqSizePerBlock;
        ctx.parallelism_config = &pc;
        ctx.attn_config        = &attention;
        GroupBase group;
        group.tag       = desc.tag;
        group.spec      = MHAKVCacheSpec::build(desc, ctx);
        group.policy    = defaultCacheGroupPolicy(CacheGroupType::FULL);
        group.block_num = 8;
        auto topology   = CacheTopology::create({group}, {{0, {kFullTag}}});
        return ShardLayoutFactory::fromTopology(*topology, pc, role, kSeqSizePerBlock);
    };

    auto prefill = make_mha_layout(8, RoleType::PREFILL);
    auto decode  = make_mha_layout(2, RoleType::DECODE);
    ASSERT_EQ(prefill.headShardCount(kFullTag), 4);
    EXPECT_EQ(prefill.headShard(0, kFullTag), 0);
    EXPECT_EQ(prefill.headShard(1, kFullTag), 0);
    EXPECT_EQ(prefill.headShard(2, kFullTag), 1);
    EXPECT_EQ(prefill.headShard(7, kFullTag), 3);
    ASSERT_EQ(decode.headShardCount(kFullTag), 2);
    const auto inferred_prefill = ShardLayoutFactory::peerOf(decode, 8, false, RoleType::PREFILL);
    const auto inferred_decode  = ShardLayoutFactory::peerOf(prefill, 2, false, RoleType::DECODE);
    EXPECT_EQ(inferred_prefill.group(kFullTag).kv_block_stride_bytes, prefill.group(kFullTag).kv_block_stride_bytes);
    EXPECT_EQ(inferred_decode.group(kFullTag).kv_block_stride_bytes, decode.group(kFullTag).kv_block_stride_bytes);

    const auto on_decode  = KVCacheTransferPlanner::plan(inferred_prefill, decode, {kFullTag});
    const auto on_prefill = KVCacheTransferPlanner::plan(prefill, inferred_decode, {kFullTag});
    ASSERT_TRUE(on_decode.ok()) << on_decode.error.ToString();
    ASSERT_TRUE(on_prefill.ok()) << on_prefill.error.ToString();
    EXPECT_EQ(on_decode.plan.digest(), on_prefill.plan.digest());
    ASSERT_EQ(on_decode.plan.routes.size(), 4u);
    expectAllRouteBytesMatch(on_decode.plan);
    for (const auto& route : on_decode.plan.routes) {
        EXPECT_EQ(prefill.headShard(route.src_rank, kFullTag) / 2, decode.headShard(route.dst_rank, kFullTag));
    }
}

TEST(PlannerGroupD, RejectsFullGroupsCoveringMultipleCacheKeys) {
    auto src                                   = mlaLayout(2, CPRotateMethod::ALL_GATHER, true);
    auto dst                                   = mlaLayout(1);
    src.groups.at(kFullTag).seq_size_per_block = 2 * kSeqSizePerBlock;
    dst.groups.at(kFullTag).seq_size_per_block = 2 * kSeqSizePerBlock;
    const auto result                          = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("exactly one cache key per block"), std::string::npos);
}

// 用例 D4: 两端独立求值结果一致（「跨端协议零改动」的正确性根据）
TEST(PlannerGroupD, D4_MirrorConsistencyAcrossSides) {
    const auto candidates = propertyCandidates();
    int        checked    = 0;

    for (const auto& sc : candidates) {
        for (const auto& dc : candidates) {
            if (sc.spec_type != dc.spec_type) {
                continue;
            }
            const auto prefill_self = buildCandidate(sc);
            const auto decode_self  = buildCandidate(dc);

            // decode 端：本端布局 + 从本端配置推导出的 prefill 布局
            const auto prefill_as_seen_by_decode =
                ShardLayout::forPeer(decode_self, prefill_self.pc, /*peer_is_prefill_role=*/true);
            // prefill 端：本端布局 + 推导出的 decode 布局
            const auto decode_as_seen_by_prefill =
                ShardLayout::forPeer(prefill_self, decode_self.pc, /*peer_is_prefill_role=*/false);

            const auto on_decode =
                KVCacheTransferPlanner::plan(prefill_as_seen_by_decode, decode_self, {kFullTag});
            const auto on_prefill =
                KVCacheTransferPlanner::plan(prefill_self, decode_as_seen_by_prefill, {kFullTag});

            ASSERT_EQ(on_decode.ok(), on_prefill.ok())
                << "src tp=" << sc.tp_size << " dst tp=" << dc.tp_size
                << " decode_err=" << on_decode.error.ToString()
                << " prefill_err=" << on_prefill.error.ToString();
            if (!on_decode.ok()) {
                continue;
            }
            ++checked;

            ASSERT_EQ(on_decode.plan.routes.size(), on_prefill.plan.routes.size());
            for (size_t i = 0; i < on_decode.plan.routes.size(); ++i) {
                const auto& x = on_decode.plan.routes[i];
                const auto& y = on_prefill.plan.routes[i];
                EXPECT_EQ(x.route_id, y.route_id);
                EXPECT_EQ(x.src_rank, y.src_rank);
                EXPECT_EQ(x.dst_rank, y.dst_rank);
                EXPECT_EQ(x.src_keys, y.src_keys);
                EXPECT_EQ(x.src_partition, y.src_partition);
                EXPECT_EQ(x.dst_partition, y.dst_partition);
                EXPECT_EQ(x.src_slice, y.src_slice);
                EXPECT_EQ(x.dst_slice, y.dst_slice);
            }
            EXPECT_EQ(on_decode.plan.digest(), on_prefill.plan.digest())
                << "src tp=" << sc.tp_size << " dst tp=" << dc.tp_size;
        }
    }
    EXPECT_GT(checked, 0);
}

// A non-sharded CP Prefill still has different attention head geometry from
// ordinary TP. Decode must consume the advertised layout, not infer the CP
// method from its own configuration or from cp_size == 1.
TEST(PlannerGroupD, D4b_AdvertisedLayoutConsistencyForPrefillCpToDecodeTp) {
    auto prefill_self = mhaLayout(2, CPRotateMethod::ALL_GATHER, /*sharded=*/false);
    auto decode_self  = mhaLayout(2, CPRotateMethod::PREFILL_CP, /*sharded=*/false);
    prefill_self.pc.role_type = RoleType::PREFILL;
    decode_self.pc.role_type  = RoleType::DECODE;

    P2PShardLayoutPB advertised_prefill;
    encodeP2PShardLayout(prefill_self, &advertised_prefill);
    ShardLayout prefill_as_seen_by_decode;
    ASSERT_TRUE(decodeP2PShardLayout(advertised_prefill, RoleType::PREFILL, prefill_as_seen_by_decode));
    P2PShardLayoutPB advertised_decode;
    encodeP2PShardLayout(decode_self, &advertised_decode);
    ShardLayout decode_as_seen_by_prefill;
    ASSERT_TRUE(decodeP2PShardLayout(advertised_decode, RoleType::DECODE, decode_as_seen_by_prefill));

    EXPECT_EQ(prefill_as_seen_by_decode.pc.prefill_cp_config.method, CPRotateMethod::ALL_GATHER);
    EXPECT_EQ(decode_as_seen_by_prefill.pc.prefill_cp_config.method, CPRotateMethod::PREFILL_CP);
    EXPECT_EQ(prefill_as_seen_by_decode.headShardCount(kFullTag), 1);
    EXPECT_EQ(decode_as_seen_by_prefill.headShardCount(kFullTag), 2);

    const auto on_decode =
        KVCacheTransferPlanner::plan(prefill_as_seen_by_decode, decode_self, {kFullTag});
    const auto on_prefill =
        KVCacheTransferPlanner::plan(prefill_self, decode_as_seen_by_prefill, {kFullTag});
    ASSERT_TRUE(on_decode.ok()) << on_decode.error.ToString();
    ASSERT_TRUE(on_prefill.ok()) << on_prefill.error.ToString();
    ASSERT_EQ(on_decode.plan.routes.size(), on_prefill.plan.routes.size());
    EXPECT_EQ(on_decode.plan.digest(), on_prefill.plan.digest());
    for (size_t i = 0; i < on_decode.plan.routes.size(); ++i) {
        const auto& x = on_decode.plan.routes[i];
        const auto& y = on_prefill.plan.routes[i];
        EXPECT_EQ(x.route_id, y.route_id);
        EXPECT_EQ(x.src_rank, y.src_rank);
        EXPECT_EQ(x.dst_rank, y.dst_rank);
        EXPECT_EQ(x.src_keys, y.src_keys);
        EXPECT_EQ(x.src_partition, y.src_partition);
        EXPECT_EQ(x.dst_partition, y.dst_partition);
        EXPECT_EQ(x.src_slice, y.src_slice);
        EXPECT_EQ(x.dst_slice, y.dst_slice);
    }
}

// 用例 D5: 复制型 group 的选举结构（P5 前置）
TEST(PlannerGroupD, D5_ReplicatedGroupElectionStructure) {
    // COMPACT + 不带字节切片 + CP 片数大于 1
    const auto src = makeLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true, {{kSwaTag, makeSwaGroup()}});
    const auto dst = makeLayout(4, CPRotateMethod::ALL_GATHER, /*sharded=*/true, {{kSwaTag, makeSwaGroup()}});

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kSwaTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    for (int d = 0; d < dst.rankCount(); ++d) {
        const auto routes = result.plan.forDecodeRankAndTag(d, kSwaTag);
        EXPECT_EQ(routes.size(), 1u) << "election must collapse replicas to one route per dst rank";
        // 被选中的源始终是 CP rank 最小的那个
        EXPECT_EQ(src.cpRank(routes.front()->src_rank), 0);
    }
}

}  // namespace rtp_llm

namespace rtp_llm {
TEST(PlannerReplicaBalance, SingleRankDecodeRequestsUseAllPrefillReplicas) {
    EXPECT_EQ(KVCacheTransferPlanner::sourceReplicaOffset("hello", 8), 3u);
    const auto    src = mlaLayout(8);
    const auto    dst = mlaLayout(1);
    std::set<int> selected;
    for (int i = 0; i < 8; ++i) {
        const auto key    = "decode-dp-" + std::to_string(i);
        const auto offset = KVCacheTransferPlanner::sourceReplicaOffset(key, src.rankCount());
        ASSERT_LT(offset, 8u);
        const auto decode  = KVCacheTransferPlanner::plan(src, dst, {kFullTag}, offset);
        const auto prefill = KVCacheTransferPlanner::plan(
            src, dst, {kFullTag}, KVCacheTransferPlanner::sourceReplicaOffset(key, src.rankCount()));
        ASSERT_TRUE(decode.ok());
        ASSERT_TRUE(prefill.ok());
        ASSERT_EQ(decode.plan.routes.size(), 1u);
        EXPECT_EQ(decode.plan.digest(), prefill.plan.digest());
        selected.insert(decode.plan.routes.front().src_rank);
    }
    EXPECT_EQ(selected, (std::set<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    const auto first  = KVCacheTransferPlanner::plan(src, dst, {kFullTag}, 0);
    const auto second = KVCacheTransferPlanner::plan(src, dst, {kFullTag}, 1);
    EXPECT_NE(first.plan.digest(), second.plan.digest());
}

TEST(PlannerReplicaBalance, MultipleDecodeDpRequestsBalanceFullMlaTransfers) {
    constexpr int kDecodeDpCount = 8;
    for (int prefill_ranks : {2, 4, 8}) {
        auto src                = mlaLayout(prefill_ranks);
        auto dst                = mlaLayout(1);
        src.pc.role_type        = RoleType::PREFILL;
        dst.pc.role_type        = RoleType::DECODE;
        const auto inferred_src = ShardLayoutFactory::peerOf(dst, prefill_ranks, false, RoleType::PREFILL);
        const auto inferred_dst = ShardLayoutFactory::peerOf(src, 1, false, RoleType::DECODE);

        for (size_t block_count : {1u, 7u, 16u, 33u}) {
            SCOPED_TRACE(::testing::Message() << "prefill_ranks=" << prefill_ranks << " blocks=" << block_count);
            std::vector<size_t> expected_keys(block_count);
            std::iota(expected_keys.begin(), expected_keys.end(), 0);
            std::vector<size_t> request_counts(prefill_ranks, 0);
            std::vector<size_t> transfer_bytes(prefill_ranks, 0);
            constexpr int       kRequestsPerDp = 16;
            for (int request = 0; request < kRequestsPerDp; ++request) {
                std::vector<size_t> round_counts(prefill_ranks, 0);
                for (int dp = 0; dp < kDecodeDpCount; ++dp) {
                    // Each DP has its own single-rank plan. Use the entrance key
                    // shape ip_counter_time, with equal counters/times across DPs:
                    // the address must still disperse otherwise identical requests.
                    // This fixed corpus is balanced for the power-of-two replica counts;
                    // arbitrary production keys are not guaranteed collision-free.
                    const auto key = "10.0.0." + std::to_string(dp) + "_" + std::to_string(request) + "_"
                                     + std::to_string(1700000000000000LL + request);
                    SCOPED_TRACE(key);
                    const auto on_decode = KVCacheTransferPlanner::plan(
                        inferred_src, dst, {kFullTag}, KVCacheTransferPlanner::sourceReplicaOffset(key, prefill_ranks));
                    const auto on_prefill = KVCacheTransferPlanner::plan(
                        src, inferred_dst, {kFullTag}, KVCacheTransferPlanner::sourceReplicaOffset(key, prefill_ranks));
                    ASSERT_TRUE(on_decode.ok()) << on_decode.error.ToString();
                    ASSERT_TRUE(on_prefill.ok()) << on_prefill.error.ToString();
                    ASSERT_EQ(on_decode.plan.routes.size(), 1u);
                    ASSERT_EQ(on_prefill.plan.routes.size(), 1u);
                    EXPECT_EQ(on_decode.plan.digest(), on_prefill.plan.digest());
                    const auto& route = on_decode.plan.routes.front();
                    ASSERT_GE(route.src_rank, 0);
                    ASSERT_LT(route.src_rank, prefill_ranks);
                    EXPECT_EQ(route.dst_rank, 0);
                    EXPECT_EQ(route.cache_tag, kFullTag);
                    const auto keys = KVCacheTransferPlanner::resolveKeys(route.src_keys, block_count);
                    EXPECT_EQ(keys, expected_keys);
                    EXPECT_EQ(route.src_bytes, kMlaBlockBytes);
                    EXPECT_EQ(route.dst_bytes, kMlaBlockBytes);
                    expectNoHeadPartitioning(on_decode.plan);
                    expectNoSlicing(on_decode.plan);
                    ++round_counts[route.src_rank];
                    ++request_counts[route.src_rank];
                    transfer_bytes[route.src_rank] += keys.size() * route.src_bytes;
                }
                for (int rank = 0; rank < prefill_ranks; ++rank) {
                    EXPECT_EQ(round_counts[rank], static_cast<size_t>(kDecodeDpCount / prefill_ranks))
                        << "request=" << request << " source_rank=" << rank;
                }
            }
            const size_t expected_requests = kDecodeDpCount * kRequestsPerDp / prefill_ranks;
            for (int rank = 0; rank < prefill_ranks; ++rank) {
                SCOPED_TRACE(::testing::Message() << "source_rank=" << rank);
                EXPECT_EQ(request_counts[rank], expected_requests);
                EXPECT_EQ(transfer_bytes[rank], expected_requests * block_count * kMlaBlockBytes);
            }
        }
    }
}

TEST(PlannerReplicaBalance, SameHostDecodeDpsDisperseFullMlaRequestsWithIndependentCounters) {
    constexpr int kDecodeDpCount = 8;
    const std::vector<size_t> expected_keys{0, 1, 2, 3, 4, 5, 6};
    for (int prefill_ranks : {2, 4, 8}) {
        const auto src = mlaLayout(prefill_ranks);
        for (int request = 0; request < 32; ++request) {
            std::vector<size_t> transferred_bytes(prefill_ranks, 0);
            for (int dp = 0; dp < kDecodeDpCount; ++dp) {
                auto dst = mlaLayout(1);
                dst.pc.dp_size = kDecodeDpCount;
                dst.pc.dp_rank = dp;
                // Same IP and independent counters starting at the same value.
                // Distinct arrival times supply distinct ip_counter_time handoff keys.
                const auto key = "10.0.0.1_" + std::to_string(request) + "_"
                                 + std::to_string(1700000000000000LL + request * 100 + dp);
                const auto offset = KVCacheTransferPlanner::sourceReplicaOffset(key, prefill_ranks);
                const auto on_decode = KVCacheTransferPlanner::plan(
                    ShardLayoutFactory::peerOf(dst, prefill_ranks, false, RoleType::PREFILL),
                    dst, {kFullTag}, offset);
                const auto on_prefill = KVCacheTransferPlanner::plan(
                    src, ShardLayoutFactory::peerOf(src, 1, false, RoleType::DECODE), {kFullTag}, offset);
                ASSERT_TRUE(on_decode.ok()) << on_decode.error.ToString();
                ASSERT_TRUE(on_prefill.ok()) << on_prefill.error.ToString();
                EXPECT_EQ(on_decode.plan.digest(), on_prefill.plan.digest());
                ASSERT_EQ(on_decode.plan.routes.size(), 1u);
                const auto& route = on_decode.plan.routes.front();
                ASSERT_GE(route.src_rank, 0);
                ASSERT_LT(route.src_rank, prefill_ranks);
                EXPECT_EQ(route.dst_rank, 0);
                EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(route.src_keys, expected_keys.size()), expected_keys);
                EXPECT_EQ(route.src_bytes, kMlaBlockBytes);
                EXPECT_EQ(route.dst_bytes, kMlaBlockBytes);
                expectNoHeadPartitioning(on_decode.plan);
                expectNoSlicing(on_decode.plan);
                transferred_bytes[route.src_rank] += expected_keys.size() * route.src_bytes;
            }
            // This corpus varies the final timestamp digit across 0..7, which
            // evenly spans the power-of-two replica counts. Arbitrary keys may collide.
            for (int rank = 0; rank < prefill_ranks; ++rank) {
                EXPECT_EQ(transferred_bytes[rank],
                          kDecodeDpCount / prefill_ranks * expected_keys.size() * kMlaBlockBytes)
                    << "request=" << request << " source_rank=" << rank;
            }
        }
    }
}

TEST(PlannerReplicaBalance, EveryOffsetBalancesFullMlaWithoutDroppingDecodeRanks) {
    for (int prefill_ranks : {1, 2, 3, 4, 8}) {
        for (int decode_ranks : {1, 2, 3, 4, 8}) {
            SCOPED_TRACE(::testing::Message() << "P=" << prefill_ranks << " D=" << decode_ranks);
            const auto                       src = mlaLayout(prefill_ranks);
            const auto                       dst = mlaLayout(decode_ranks);
            std::vector<std::vector<size_t>> counts(decode_ranks, std::vector<size_t>(prefill_ranks, 0));
            const std::vector<size_t>        expected_keys{0, 1, 2, 3, 4, 5, 6};
            for (int offset = 0; offset < prefill_ranks; ++offset) {
                SCOPED_TRACE(offset);
                const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag}, offset);
                ASSERT_TRUE(result.ok()) << result.error.ToString();
                ASSERT_EQ(result.plan.routes.size(), static_cast<size_t>(decode_ranks));
                expectUniqueRouteIds(result.plan);
                expectNoHeadPartitioning(result.plan);
                expectNoSlicing(result.plan);
                std::vector<size_t> offset_counts(prefill_ranks, 0);
                for (int decode_rank = 0; decode_rank < decode_ranks; ++decode_rank) {
                    const auto routes = result.plan.forDecodeRank(decode_rank);
                    ASSERT_EQ(routes.size(), 1u);
                    const auto& route = *routes.front();
                    ASSERT_GE(route.src_rank, 0);
                    ASSERT_LT(route.src_rank, prefill_ranks);
                    EXPECT_EQ(route.src_bytes, kMlaBlockBytes);
                    EXPECT_EQ(route.dst_bytes, kMlaBlockBytes);
                    EXPECT_EQ(KVCacheTransferPlanner::resolveKeys(route.src_keys, expected_keys.size()), expected_keys);
                    ++counts[decode_rank][route.src_rank];
                    ++offset_counts[route.src_rank];
                }
                const auto bounds = std::minmax_element(offset_counts.begin(), offset_counts.end());
                EXPECT_LE(*bounds.second - *bounds.first, 1u);
            }
            // Across a full offset cycle, every decode rank must use each source
            // exactly once, including unequal and non-power-of-two rank counts.
            for (int decode_rank = 0; decode_rank < decode_ranks; ++decode_rank) {
                for (int source_rank = 0; source_rank < prefill_ranks; ++source_rank) {
                    EXPECT_EQ(counts[decode_rank][source_rank], 1u)
                        << "decode_rank=" << decode_rank << " source_rank=" << source_rank;
                }
            }
        }
    }
}

TEST(PlannerReplicaBalance, OffsetPreservesCpAndHeadOwnership) {
    const auto src      = mlaLayout(8, CPRotateMethod::PREFILL_CP, true);
    const auto dst      = mlaLayout(4);
    const auto baseline = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(baseline.ok());
    for (size_t offset = 0; offset < 8; ++offset) {
        const auto rotated = KVCacheTransferPlanner::plan(src, dst, {kFullTag}, offset);
        ASSERT_TRUE(rotated.ok());
        ASSERT_EQ(rotated.plan.routes.size(), baseline.plan.routes.size());
        for (size_t i = 0; i < rotated.plan.routes.size(); ++i) {
            const auto& actual   = rotated.plan.routes[i];
            const auto& expected = baseline.plan.routes[i];
            EXPECT_EQ(src.cpRank(actual.src_rank), src.cpRank(expected.src_rank));
            EXPECT_EQ(src.headShard(actual.src_rank, kFullTag), src.headShard(expected.src_rank, kFullTag));
            EXPECT_EQ(actual.dst_rank, expected.dst_rank);
            EXPECT_EQ(actual.src_keys, expected.src_keys);
            EXPECT_EQ(actual.src_bytes, expected.src_bytes);
            EXPECT_EQ(actual.dst_bytes, expected.dst_bytes);
        }
    }
}
}  // namespace rtp_llm
