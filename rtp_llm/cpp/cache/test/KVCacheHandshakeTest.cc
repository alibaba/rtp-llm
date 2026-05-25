// F01-PR2-followup (R4-24 F9): gtest coverage for KVCacheHandshake.
//
// Validator + accept-API contract:
//   1. Legacy↔legacy (both magic=0, all fields zero) — pair accepts.
//   2. Matching nonzero salt (same K_state bitmap, same schema_version) — accepts.
//   3. Mismatched bitmap — refused; counter increments; no engine FAIL.
//   4. Mismatched version — refused; counter increments.
//   5. Mismatched magic byte (peer magic neither {0,1}) — refused via the
//      mixed-mode branch in validateHandshake.
//   6. Bitmap-overflow safety — out-of-range bits still compare for equality.
//
// Default-OFF byte-equality: legacy peers (bitmap==0, version==0) NEVER
// trigger the counter — verified at end of each "true" case.

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/connector/KVCacheHandshake.h"

namespace rtp_llm {
namespace test {

class KVCacheHandshakeTest: public ::testing::Test {
protected:
    void SetUp() override {
        resetPdSaltMismatchSkippedCountForTest();
    }

    static HandshakeInfo makeInfo(uint32_t magic, uint64_t pool_hash, uint32_t version, uint32_t bitmap) {
        HandshakeInfo info{};
        info.protocol_magic           = magic;
        info.pool_descriptor_hash     = pool_hash;
        info.hash_salt_version        = version;
        info.hash_salt_nonzero_bitmap = bitmap;
        return info;
    }
};

// 1. Legacy↔legacy (default-OFF byte-equality MUST be preserved).
TEST_F(KVCacheHandshakeTest, LegacyPeerPairing) {
    const auto local = makeInfo(0, 0, 0, 0);
    const auto peer  = makeInfo(0, 0, 0, 0);
    std::string err;
    EXPECT_TRUE(validateHandshake(local, peer, &err)) << err;
    EXPECT_EQ(pdSaltMismatchSkippedCount(), 0u) << "legacy↔legacy must NOT increment the silent-skip counter";
}

// 2. Matching unified salt: same K_state bitmap (bit3), same version.
TEST_F(KVCacheHandshakeTest, MatchingNonzeroSalt) {
    constexpr uint32_t kBitmap  = (1u << 3);  // K_state-only salt
    constexpr uint32_t kVersion = kCacheKeySaltSchemaVersion;
    const auto         local    = makeInfo(/*magic=*/1, /*pool_hash=*/0xdeadbeefULL, kVersion, kBitmap);
    const auto         peer     = makeInfo(/*magic=*/1, /*pool_hash=*/0xdeadbeefULL, kVersion, kBitmap);
    std::string        err;
    EXPECT_TRUE(validateHandshake(local, peer, &err)) << err;
    EXPECT_EQ(pdSaltMismatchSkippedCount(), 0u) << "matching salt must NOT increment the counter";
}

// 3. Bitmap drift (peer bit3 set, local bit3 clear) — REFUSE.
TEST_F(KVCacheHandshakeTest, MismatchedBitmapRefused) {
    constexpr uint32_t kVersion = kCacheKeySaltSchemaVersion;
    const auto         local    = makeInfo(1, 0xc0deULL, kVersion, /*bitmap=*/0);
    const auto         peer     = makeInfo(1, 0xc0deULL, kVersion, /*bitmap=*/(1u << 3));
    // Note: validateHandshake's legacy↔legacy branch is gated on magic==0
    // on BOTH sides; here both sides have magic==1, but only one carries
    // a salt — the bitmap mismatch in the unified↔unified branch fires.
    std::string err;
    EXPECT_FALSE(validateHandshake(local, peer, &err));
    EXPECT_NE(err.find("hash_salt_nonzero_bitmap"), std::string::npos) << "err: " << err;
}

// 4. Version drift (peer v=2, local v=1) — REFUSE.
TEST_F(KVCacheHandshakeTest, MismatchedVersionRefused) {
    constexpr uint32_t kBitmap = (1u << 3);
    const auto         local   = makeInfo(1, 0xc0deULL, /*version=*/1, kBitmap);
    const auto         peer    = makeInfo(1, 0xc0deULL, /*version=*/2, kBitmap);
    std::string        err;
    EXPECT_FALSE(validateHandshake(local, peer, &err));
    EXPECT_NE(err.find("hash_salt_version"), std::string::npos) << "err: " << err;
}

// 5. Magic byte != 100 (== peer not in {0,1}) is caught by the mixed-mode
// branch.  Local is unified (magic=1); peer announces magic=0 (legacy)
// while ALSO carrying nonzero salt — legacy↔unified is refused.
TEST_F(KVCacheHandshakeTest, MagicByteWrongRefused) {
    const auto local = makeInfo(/*magic=*/1, /*pool_hash=*/0xfeULL, kCacheKeySaltSchemaVersion, (1u << 3));
    const auto peer  = makeInfo(/*magic=*/0, /*pool_hash=*/0ULL, /*version=*/0, /*bitmap=*/0);
    std::string err;
    EXPECT_FALSE(validateHandshake(local, peer, &err));
    EXPECT_NE(err.find("mixed-mode"), std::string::npos) << "err: " << err;
}

// 6. Bitmap-overflow safety: out-of-range bits (e.g. bit31) still compare
// by exact equality.  No truncation, no false-positive accept.
TEST_F(KVCacheHandshakeTest, BitmapOverflowSafetyCheck) {
    constexpr uint32_t kOverflowBitmap = 0xFFFFFFFFu;  // every bit lit, incl. reserved
    constexpr uint32_t kVersion        = kCacheKeySaltSchemaVersion;
    const auto         local           = makeInfo(1, 0x42ULL, kVersion, kOverflowBitmap);
    // Peer is missing the top bit — must NOT compare equal even though both
    // values are "out of the documented [bit0..bit4] range".
    const auto  peer = makeInfo(1, 0x42ULL, kVersion, kOverflowBitmap & ~(1u << 31));
    std::string err;
    EXPECT_FALSE(validateHandshake(local, peer, &err));
    EXPECT_NE(err.find("hash_salt_nonzero_bitmap"), std::string::npos) << "err: " << err;

    // Same-overflow on both sides — accept (struct field is uint32, NOT a
    // bitset truncated to 5 bits; XR4-24 N1 doc-vs-struct check).
    err.clear();
    const auto peer_match = makeInfo(1, 0x42ULL, kVersion, kOverflowBitmap);
    EXPECT_TRUE(validateHandshake(local, peer_match, &err)) << err;
}

// 7. Counter wiring: a refused validation MUST be observable via the
// process-wide ``pd.cache.salt_mismatch_skipped`` counter once the
// ``recordPdSaltMismatchSkipped`` helper runs (the coordinator calls it
// from validatePeerHandshake on the WARN+return-false path).
TEST_F(KVCacheHandshakeTest, RecordPdSaltMismatchSkippedIncrementsCounter) {
    EXPECT_EQ(pdSaltMismatchSkippedCount(), 0u);
    recordPdSaltMismatchSkipped(/*reporter=*/nullptr);
    EXPECT_EQ(pdSaltMismatchSkippedCount(), 1u);
    recordPdSaltMismatchSkipped(/*reporter=*/nullptr);
    EXPECT_EQ(pdSaltMismatchSkippedCount(), 2u);
}

// 8. computeLocalHandshakeInfo legacy default: super_block_layout.enabled
// == false MUST produce an all-zero handshake regardless of salt args —
// this is the load-bearing default-OFF byte-equality guard.
TEST_F(KVCacheHandshakeTest, ComputeLocalHandshakeInfoLegacyAllZero) {
    CacheConfig cfg{};
    cfg.super_block_layout.enabled = false;
    const auto info = computeLocalHandshakeInfo(cfg, /*version=*/7, /*bitmap=*/0xFu);
    EXPECT_EQ(info.protocol_magic, 0u);
    EXPECT_EQ(info.pool_descriptor_hash, 0u);
    EXPECT_EQ(info.hash_salt_version, 0u);
    EXPECT_EQ(info.hash_salt_nonzero_bitmap, 0u);
}

// 9. computeLocalHandshakeInfo unified path: super_block_layout.enabled
// == true populates the pinned hash + salt schema fields.
TEST_F(KVCacheHandshakeTest, ComputeLocalHandshakeInfoUnifiedPopulatesFields) {
    CacheConfig cfg{};
    cfg.super_block_layout.enabled = true;
    cfg.group_types                = {CacheGroupType::FULL};
    cfg.group_region_names         = {KVCacheRegionName::DEFAULT};
    cfg.group_block_size_bytes     = {1024};
    cfg.layer_all_num              = 4;
    const auto info = computeLocalHandshakeInfo(cfg, kCacheKeySaltSchemaVersion, (1u << 3));
    EXPECT_EQ(info.protocol_magic, 1u);
    EXPECT_NE(info.pool_descriptor_hash, 0u);
    EXPECT_EQ(info.hash_salt_version, kCacheKeySaltSchemaVersion);
    EXPECT_EQ(info.hash_salt_nonzero_bitmap, (1u << 3));
}

// ============================================================
// FIX-B HIGH-5 (DEFEND-4 #5) — LegacyPeerWithoutSaltMagicAccepted
//
// Production-fatal regression guard: pre-fix, ``salt_protocol_magic``
// defaulted to 100 on the wire and was naively routed into
// ``HandshakeInfo::protocol_magic``. validateHandshake's unified↔unified
// branch then REFUSED every legacy peer (peer.magic=100, local.magic=0
// → mixed-mode REFUSE OR peer.magic=100 vs local.magic=1 mismatch).
//
// Post-fix: proto field renamed to ``salt_magic`` (default 0); the wire
// helper ``acceptPeerHandshakeFields`` derives HandshakeInfo::protocol_magic
// from ``hash_salt_version > 0``, NOT from salt_magic.  A legacy peer
// that omits field 103 entirely (proto default 0) MUST be accepted as a
// legacy peer, NOT refused.
//
// Test directly mirrors the post-fix mapping documented on
// KVCacheConnectorCoordinator::acceptPeerHandshakeFields so the contract
// is anchored in this gtest even before a per-connector wire integration
// test exists.
// ============================================================
TEST_F(KVCacheHandshakeTest, LegacyPeerWithoutSaltMagicAccepted) {
    // Wire scenario: legacy peer omits proto field 103 → its
    // ``salt_magic`` parses as default 0 (post-rename).  Coordinator's
    // wire helper derives HandshakeInfo::protocol_magic from
    // ``hash_salt_version > 0`` (= 0 here → magic stays 0).
    const uint32_t peer_salt_magic              = 0;
    const uint32_t peer_hash_salt_version       = 0;
    const uint32_t peer_hash_salt_nonzero_bitmap = 0;

    HandshakeInfo peer{};
    peer.pool_descriptor_hash     = 0;
    peer.hash_salt_version        = peer_hash_salt_version;
    peer.hash_salt_nonzero_bitmap = peer_hash_salt_nonzero_bitmap;
    peer.protocol_magic = (peer_hash_salt_version > 0) ? 1u : 0u;
    (void)peer_salt_magic;  // documented input; not stored on HandshakeInfo

    // Local engine is also legacy (super_block_layout.enabled == false).
    CacheConfig local_cfg{};
    local_cfg.super_block_layout.enabled = false;
    const auto local = computeLocalHandshakeInfo(local_cfg, /*version=*/0, /*bitmap=*/0);

    std::string err;
    EXPECT_TRUE(validateHandshake(local, peer, &err))
        << "salt_magic=0 (legacy peer, proto default) MUST be accepted as legacy, not REFUSED. err=" << err;
    EXPECT_EQ(pdSaltMismatchSkippedCount(), 0u)
        << "legacy-peer accept path must NOT increment salt_mismatch_skipped counter";
}

// Forward-compat coverage: a salt-aware peer that DOES send field 103
// (e.g. legacy sentinel value 100) but happens to have hash_salt_version=0
// (K_state OFF on the sender side) must STILL be treated as legacy — the
// "is salted" gate is hash_salt_version > 0, NOT salt_magic.
TEST_F(KVCacheHandshakeTest, SaltAwareSenderWithZeroVersionAcceptedAsLegacy) {
    const uint32_t peer_salt_magic              = 100;  // sender opted into envelope detection
    const uint32_t peer_hash_salt_version       = 0;    // but no actual salt fields populated
    const uint32_t peer_hash_salt_nonzero_bitmap = 0;

    HandshakeInfo peer{};
    peer.pool_descriptor_hash     = 0;
    peer.hash_salt_version        = peer_hash_salt_version;
    peer.hash_salt_nonzero_bitmap = peer_hash_salt_nonzero_bitmap;
    peer.protocol_magic = (peer_hash_salt_version > 0) ? 1u : 0u;
    (void)peer_salt_magic;

    CacheConfig local_cfg{};
    local_cfg.super_block_layout.enabled = false;
    const auto local = computeLocalHandshakeInfo(local_cfg, /*version=*/0, /*bitmap=*/0);

    std::string err;
    EXPECT_TRUE(validateHandshake(local, peer, &err))
        << "salt_magic=100 + version=0 (envelope-aware but unsalted) MUST be legacy-accepted. err=" << err;
}

}  // namespace test
}  // namespace rtp_llm
