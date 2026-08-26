#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"

#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include <gtest/gtest.h>
#include <tuple>

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace rtp_llm {

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
    g.seq_size_per_block = kSeqSizePerBlock;
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

}  // namespace

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

        // 反向断言：若源端也切分，两侧字节数会差一倍
        EXPECT_NE(r->src_partition, (PartitionSpec{2, s % 2}));
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
    // 反向断言：全局只选一个发送方会让除第一个以外的 decode rank 拿不到数据
    EXPECT_NE(rankPairs(result.plan, kFullTag), (std::set<std::pair<int, int>>{{0, 0}}));

    expectNoHeadPartitioning(result.plan);
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A7a: PREFILL_CP + CP 分片，MLA group（§3.3 回归）
TEST(PlannerGroupA, A7a_PrefillCpWithMlaGroup) {
    const auto src = mlaLayout(8, CPRotateMethod::PREFILL_CP, /*sharded=*/true);
    const auto dst = mlaLayout(8);

    // PREFILL_CP 被 is_enabled() 排除，故 attention TP 大小不会被压成 1
    EXPECT_EQ(src.pc.get_attn_tp_size(), 8);
    // 但 head 分片数是 spec 类型的属性：MLA latent 复制，与 attention TP 无关
    EXPECT_EQ(src.headShardCount(kFullTag), 1);
    EXPECT_EQ(src.cpSize(), 8);

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});
    ASSERT_TRUE(result.ok()) << result.error.ToString();

    // decode 侧 8 个 rank 各需全量序列，prefill 侧 8 个 rank 各持 1/8 -> 必须 8x8
    EXPECT_EQ(result.plan.routes.size(), 64u);
    for (int d = 0; d < 8; ++d) {
        const auto routes = result.plan.forDecodeRank(d);
        ASSERT_EQ(routes.size(), 8u);
        std::set<int> residues;
        for (const auto* r : routes) {
            EXPECT_EQ(r->src_keys.modulus, 8);
            residues.insert(r->src_keys.residue);
        }
        EXPECT_EQ(residues.size(), 8u) << "dst rank " << d << " must cover every residue class";
    }
    expectNoHeadPartitioning(result.plan);
    expectAllRouteBytesMatch(result.plan);
}

// 用例 A7b: PREFILL_CP + CP 分片，MHA group —— 单侧不完备，必须拒绝
TEST(PlannerGroupA, A7b_PrefillCpWithMhaGroupIsRejected) {
    const auto src = mhaLayout(8, CPRotateMethod::PREFILL_CP, /*sharded=*/true);
    const auto dst = mhaLayout(8);

    // MHA spec 会按 attention TP 切 head，而 PREFILL_CP 下 attention TP 仍是 tp_size
    EXPECT_EQ(src.headShardCount(kFullTag), 8);
    EXPECT_EQ(src.cpSize(), 8);
    // 字节维度校验通不住这个错误：两侧还原出的全局块大小相等
    EXPECT_EQ(src.effectiveGlobalBlockBytes(kFullTag), dst.effectiveGlobalBlockBytes(kFullTag));

    const auto result = KVCacheTransferPlanner::plan(src, dst, {kFullTag});

    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error.ToString().find("double-sharded"), std::string::npos) << result.error.ToString();
    EXPECT_TRUE(result.plan.routes.empty());
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

    // 反向断言：不选举会产生 cp_size 平方条冗余 route
    EXPECT_NE(result.plan.forTag(kSwaTag).size(), 16u);
    expectAllRouteBytesMatch(result.plan);
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
    dst.groups[kFixedTag].k_block_payload_bytes = dst.group(kFixedTag).kv_block_stride_bytes - 4096;
    src.groups[kFixedTag].k_block_payload_bytes = dst.group(kFixedTag).k_block_payload_bytes / 4;
    // src 已预切片，其本地块尺寸对齐到 payload 分片
    src.groups[kFixedTag].kv_block_stride_bytes = src.group(kFixedTag).k_block_payload_bytes;

    EXPECT_EQ(src.group(kFixedTag).kv_block_stride_bytes * 4, dst.group(kFixedTag).k_block_payload_bytes);
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
