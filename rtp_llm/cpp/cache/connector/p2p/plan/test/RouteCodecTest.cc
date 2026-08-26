#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

namespace rtp_llm {

namespace {

TransferRoute makeRoute() {
    TransferRoute r;
    r.route_id      = 5;
    r.src_rank      = 3;
    r.dst_rank      = 1;
    r.cache_tag     = "full";
    r.src_keys      = KeyShardSpec{/*modulus=*/4, /*residue=*/3};
    r.src_partition = PartitionSpec{1, 0};
    r.dst_partition = PartitionSpec{2, 1};
    r.src_slice     = SliceSpec{CpBlockSliceMode::NONE, 1, 0};
    r.dst_slice     = SliceSpec{CpBlockSliceMode::PAYLOAD_BYTES, 4, 3};
    return r;
}

}  // namespace

// 编码是有方向的：prefill 只拿到 src_*，对端那一半永不上线。
TEST(RouteCodec, PrefillDirectionCarriesOnlySrcHalf) {
    const auto      route = makeRoute();
    TransferRoutePB pb;
    RouteCodec::encodeForPrefill(route, /*peer_index=*/1, &pb);

    EXPECT_EQ(pb.route_id(), 5);
    EXPECT_EQ(pb.cache_tag(), "full");
    EXPECT_EQ(pb.peer_index(), 1);

    const auto local = RouteCodec::decode(pb);
    EXPECT_EQ(local.partition, route.src_partition);
    EXPECT_EQ(local.slice, route.src_slice);
    // 目的端的 partition / slice 不应出现在 prefill 的那份里
    EXPECT_NE(local.partition, route.dst_partition);
    EXPECT_NE(local.slice, route.dst_slice);
}

TEST(RouteCodec, DecodeDirectionCarriesOnlyDstHalf) {
    const auto      route = makeRoute();
    TransferRoutePB pb;
    RouteCodec::encodeForDecode(route, &pb);

    const auto local = RouteCodec::decode(pb);
    EXPECT_EQ(local.route_id, 5);
    EXPECT_EQ(local.cache_tag, "full");
    EXPECT_EQ(local.partition, route.dst_partition);
    EXPECT_EQ(local.slice, route.dst_slice);
}

// slice_mode 走的是裸 int32，必须能无损往返，且非法值收敛到 NONE。
TEST(RouteCodec, SliceModeRoundTrip) {
    for (auto mode : {CpBlockSliceMode::NONE, CpBlockSliceMode::EQUAL_BYTES, CpBlockSliceMode::PAYLOAD_BYTES}) {
        auto route      = makeRoute();
        route.dst_slice = SliceSpec{mode, 4, 2};
        TransferRoutePB pb;
        RouteCodec::encodeForDecode(route, &pb);
        EXPECT_EQ(RouteCodec::decode(pb).slice.mode, mode);
    }
    EXPECT_EQ(RouteCodec::toSliceMode(99), CpBlockSliceMode::NONE);
    EXPECT_EQ(RouteCodec::toSliceMode(-1), CpBlockSliceMode::NONE);
}

// count 为 0（proto 默认值 / 老对端）时必须收敛到 1，不能产生除零。
TEST(RouteCodec, ZeroCountsDefaultToOne) {
    TransferRoutePB pb;
    pb.set_route_id(0);
    const auto local = RouteCodec::decode(pb);
    EXPECT_EQ(local.partition.count, 1);
    EXPECT_EQ(local.slice.count, 1);
    EXPECT_EQ(local.slice.mode, CpBlockSliceMode::NONE);
}

// 传输 key 是两侧唯一的汇合点：同 route 必须逐字节相同，不同 route / 不同层 / 不同 plan 必须不同。
TEST(P2PKeyUtilTest, RouteLayerKeyIsStableAndDiscriminating) {
    const std::string base   = "uk";
    const uint64_t    digest = 0xdeadbeefcafe1234ull;

    EXPECT_EQ(P2PKeyUtil::makeRouteLayerKey(base, 3, "full", 7, digest),
              P2PKeyUtil::makeRouteLayerKey(base, 3, "full", 7, digest));

    std::set<std::string> keys{
        P2PKeyUtil::makeRouteLayerKey(base, 3, "full", 7, digest),
        P2PKeyUtil::makeRouteLayerKey(base, 4, "full", 7, digest),  // 换层
        P2PKeyUtil::makeRouteLayerKey(base, 3, "swa", 7, digest),   // 换 tag
        P2PKeyUtil::makeRouteLayerKey(base, 3, "full", 8, digest),  // 换 route
        P2PKeyUtil::makeRouteLayerKey(base, 3, "full", 7, 0),       // 换 plan
    };
    EXPECT_EQ(keys.size(), 5u);

    // digest 分歧 ⇒ key 不匹配 ⇒ 退化为 TIMEOUT 而不是拷错字节
    EXPECT_NE(P2PKeyUtil::makeRouteLayerKey(base, 3, "full", 7, digest),
              P2PKeyUtil::makeRouteLayerKey(base, 3, "full", 7, digest + 1));

    EXPECT_EQ(P2PKeyUtil::shortDigest(0x0000000100000000ull), "00000001");
    EXPECT_EQ(P2PKeyUtil::shortDigest(0xabcdef1200000000ull), "abcdef12");
}

}  // namespace rtp_llm
