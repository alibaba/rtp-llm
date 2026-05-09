#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"

namespace rtp_llm {
namespace {

TEST(P2PConnectorConfigLiteTest, WorkerConfigCarriesMooncakeTcpTransportSettings) {
    RuntimeConfig runtime_config;
    CacheStoreConfig cache_store_config;
    cache_store_config.cache_store_mooncake_mode = true;
    cache_store_config.cache_store_mooncake_transport = "tcp";
    cache_store_config.cache_store_mooncake_location = "tp0";
    cache_store_config.cache_store_mooncake_control_plane_port = 23456;
    cache_store_config.cache_store_mooncake_rpc_port = 34567;
    cache_store_config.cache_store_mooncake_ip_or_host_name = "127.0.0.1";

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 2;
    parallelism_config.tp_rank = 1;

    PDSepConfig pd_sep_config;
    pd_sep_config.cache_store_listen_port = 12345;

    auto config = P2PConnectorConfig::create(
        runtime_config, cache_store_config, parallelism_config, pd_sep_config, /*layer_all_num=*/8);

    EXPECT_TRUE(config.worker_config.transfer_backend_config.cache_store_mooncake_mode);
    EXPECT_EQ(config.worker_config.transfer_backend_config.resolveBackend(), transfer::TransferBackend::kMooncake);
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.classic.transport, "tcp");
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.location, "tp0");
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.control_plane_port, 23456);
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.classic.rpc_port, 34567);
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.classic.ip_or_host_name, "127.0.0.1");
    EXPECT_EQ(config.worker_config.transfer_backend_config.cache_store_listen_port, 12345);
}

TEST(P2PConnectorConfigLiteTest, WorkerConfigNormalizesMooncakeNvlinkAlias) {
    RuntimeConfig runtime_config;
    CacheStoreConfig cache_store_config;
    cache_store_config.cache_store_mooncake_mode = true;
    cache_store_config.cache_store_mooncake_transport = "nvlink_intraNode";

    ParallelismConfig parallelism_config;
    PDSepConfig pd_sep_config;

    auto config = P2PConnectorConfig::create(
        runtime_config, cache_store_config, parallelism_config, pd_sep_config, /*layer_all_num=*/4);

    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.classic.transport, "nvlink_intra");
}

TEST(P2PConnectorConfigLiteTest, WorkerConfigCarriesMooncakeBarexTransportSettings) {
    RuntimeConfig runtime_config;
    CacheStoreConfig cache_store_config;
    cache_store_config.cache_store_mooncake_mode = true;
    cache_store_config.cache_store_mooncake_transport = "barex";
    cache_store_config.cache_store_mooncake_location = "*";
    cache_store_config.cache_store_mooncake_rpc_port = 34678;
    cache_store_config.cache_store_mooncake_ip_or_host_name = "127.0.0.1";

    ParallelismConfig parallelism_config;
    PDSepConfig pd_sep_config;

    auto config = P2PConnectorConfig::create(
        runtime_config, cache_store_config, parallelism_config, pd_sep_config, /*layer_all_num=*/4);

    EXPECT_TRUE(config.worker_config.transfer_backend_config.cache_store_mooncake_mode);
    EXPECT_EQ(config.worker_config.transfer_backend_config.resolveBackend(), transfer::TransferBackend::kMooncake);
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.classic.transport, "barex");
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.location, "*");
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.classic.rpc_port, 34678);
    EXPECT_EQ(config.worker_config.transfer_backend_config.mooncake.classic.ip_or_host_name, "127.0.0.1");
}

TEST(P2PConnectorConfigLiteTest, WorkerConfigPrefersMooncakeBackendOverLegacyRdmaFlag) {
    RuntimeConfig runtime_config;
    CacheStoreConfig cache_store_config;
    cache_store_config.cache_store_rdma_mode = true;
    cache_store_config.cache_store_mooncake_mode = true;
    cache_store_config.cache_store_mooncake_transport = "tcp";

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 2;
    parallelism_config.tp_rank = 0;

    PDSepConfig pd_sep_config;
    auto config = P2PConnectorConfig::create(
        runtime_config, cache_store_config, parallelism_config, pd_sep_config, /*layer_all_num=*/4);

    EXPECT_TRUE(config.worker_config.transfer_backend_config.cache_store_rdma_mode);
    EXPECT_TRUE(config.worker_config.transfer_backend_config.cache_store_mooncake_mode);
    EXPECT_EQ(config.worker_config.transfer_backend_config.resolveBackend(), transfer::TransferBackend::kMooncake);
}

}  // namespace
}  // namespace rtp_llm
