#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"

namespace rtp_llm {
namespace transfer {
namespace {

TEST(TransferBackendConfigTest, ResolveBackendDefaultsToTcp) {
    TransferBackendConfig config;
    EXPECT_EQ(config.resolveBackend(), TransferBackend::kTcp);
}

TEST(TransferBackendConfigTest, ResolveBackendUsesLegacyRdmaModeWhenMooncakeDisabled) {
    TransferBackendConfig config;
    config.cache_store_rdma_mode = true;
    EXPECT_EQ(config.resolveBackend(), TransferBackend::kBarexRdma);
}

TEST(TransferBackendConfigTest, ResolveBackendPrefersMooncakeMode) {
    TransferBackendConfig config;
    config.cache_store_rdma_mode      = true;
    config.cache_store_mooncake_mode  = true;
    EXPECT_EQ(config.resolveBackend(), TransferBackend::kMooncake);
}

TEST(TransferBackendConfigTest, MooncakeConfigKeepsDedicatedDefaults) {
    TransferBackendConfig config;
    EXPECT_EQ(config.mooncake.classic.transport, "tcp");
    EXPECT_EQ(config.mooncake.location, "*");
    EXPECT_TRUE(config.mooncake.remote_accessible);
    EXPECT_TRUE(config.mooncake.update_metadata);
    EXPECT_EQ(config.mooncake.control_plane_port, 0);
}

TEST(TransferBackendConfigTest, NormalizeMooncakeTransportSupportsNvlinkAlias) {
    EXPECT_EQ(normalizeMooncakeTransport(""), "tcp");
    EXPECT_EQ(normalizeMooncakeTransport("nvlink"), "nvlink");
    EXPECT_EQ(normalizeMooncakeTransport("nvlink_intraNode"), "nvlink_intra");
}

}  // namespace
}  // namespace transfer
}  // namespace rtp_llm
