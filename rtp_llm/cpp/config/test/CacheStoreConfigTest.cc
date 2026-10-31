#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RdmaDeviceHealthConfig.h"

namespace rtp_llm {

namespace {

CacheStoreConfig makeNonDefaultConfig() {
    CacheStoreConfig config;
    config.rdma_device_health_enabled                  = true;
    config.rdma_device_health_fault_handler            = "ABORT";
    config.rdma_device_health_probe_interval_ms        = 2000;
    config.rdma_device_health_device_fault_confirm_ms  = 5000;
    config.rdma_device_health_fault_window_ms          = 900000;
    config.rdma_device_health_unavailable_threshold_ms = 300000;
    config.rdma_device_health_flap_warn_threshold      = 20;
    return config;
}

void expectNonDefaultValues(const RdmaDeviceHealthConfig& config) {
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.fault_handler, "ABORT");
    EXPECT_EQ(config.probe_interval_ms, 2000);
    EXPECT_EQ(config.device_fault_confirm_ms, 5000);
    EXPECT_EQ(config.fault_window_ms, 900000);
    EXPECT_EQ(config.unavailable_threshold_ms, 300000);
    EXPECT_EQ(config.flap_warn_threshold, 20);
}

}  // namespace

class CacheStoreConfigTest: public ::testing::Test {};

// CacheStoreConfig and RdmaDeviceHealthConfig declare separate defaults that must remain consistent.
TEST_F(CacheStoreConfigTest, testRdmaDeviceHealthDefaultsMatchConfig) {
    const CacheStoreConfig       config;
    const RdmaDeviceHealthConfig expected;
    const auto                   actual = config.toRdmaDeviceHealthConfig();

    EXPECT_FALSE(actual.enabled);
    EXPECT_EQ(actual.enabled, expected.enabled);
    EXPECT_EQ(actual.fault_handler, expected.fault_handler);
    EXPECT_EQ(actual.probe_interval_ms, expected.probe_interval_ms);
    EXPECT_EQ(actual.device_fault_confirm_ms, expected.device_fault_confirm_ms);
    EXPECT_EQ(actual.fault_window_ms, expected.fault_window_ms);
    EXPECT_EQ(actual.unavailable_threshold_ms, expected.unavailable_threshold_ms);
    EXPECT_EQ(actual.flap_warn_threshold, expected.flap_warn_threshold);
}

TEST_F(CacheStoreConfigTest, testToRdmaDeviceHealthConfigCopiesAllFields) {
    expectNonDefaultValues(makeNonDefaultConfig().toRdmaDeviceHealthConfig());
}

TEST_F(CacheStoreConfigTest, testP2PWorkerConfigCarriesRdmaDeviceHealth) {
    const auto worker_config =
        P2PConnectorWorkerConfig::create(makeNonDefaultConfig(), PDSepConfig(), ParallelismConfig(), 1);
    expectNonDefaultValues(worker_config.transfer_backend_config.rdma_device_health);
}

TEST_F(CacheStoreConfigTest, testToStringContainsRdmaDeviceHealthFields) {
    const auto str = makeNonDefaultConfig().to_string();
    EXPECT_NE(str.find("rdma_device_health_enabled: 1\n"), std::string::npos);
    EXPECT_NE(str.find("rdma_device_health_fault_handler: ABORT\n"), std::string::npos);
    EXPECT_NE(str.find("rdma_device_health_probe_interval_ms: 2000\n"), std::string::npos);
    EXPECT_NE(str.find("rdma_device_health_device_fault_confirm_ms: 5000\n"), std::string::npos);
    EXPECT_NE(str.find("rdma_device_health_fault_window_ms: 900000\n"), std::string::npos);
    EXPECT_NE(str.find("rdma_device_health_unavailable_threshold_ms: 300000\n"), std::string::npos);
    EXPECT_NE(str.find("rdma_device_health_flap_warn_threshold: 20\n"), std::string::npos);
}

}  // namespace rtp_llm
