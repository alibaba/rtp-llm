#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/RdmaDeviceHealthTypes.h"

#include <gtest/gtest.h>

namespace rtp_llm {

class CacheStoreConfigTest: public ::testing::Test {};

TEST_F(CacheStoreConfigTest, testRdmaDeviceHealthMonitorConfigDefaultsAreOff) {
    CacheStoreConfig config;
    auto             monitor_config = config.makeRdmaDeviceHealthMonitorConfig(true);
    ASSERT_FALSE(monitor_config.enabled);
    ASSERT_EQ(monitor_config.fault_handler, RdmaDeviceHealthFaultHandler::LOG);
    ASSERT_EQ(monitor_config.probe_interval_ms, kDefaultRdmaDeviceHealthProbeIntervalMs);
    ASSERT_EQ(monitor_config.fault_confirm_count, kDefaultRdmaDeviceHealthFaultConfirmCount);
}

TEST_F(CacheStoreConfigTest, testRdmaDeviceHealthMonitorConfigNeedsRdmaMode) {
    CacheStoreConfig config;
    config.rdma_device_health_check_enabled = true;

    ASSERT_TRUE(config.makeRdmaDeviceHealthMonitorConfig(true).enabled);
    // TCP 模式下即使显式开启也不得生效，否则会在没有 RDMA 设备的部署上误触发 fault handler
    ASSERT_FALSE(config.makeRdmaDeviceHealthMonitorConfig(false).enabled);
}

TEST_F(CacheStoreConfigTest, testRdmaDeviceHealthMonitorConfigPassesThroughValues) {
    CacheStoreConfig config;
    config.rdma_device_health_check_enabled       = true;
    config.rdma_device_health_fault_handler       = RdmaDeviceHealthFaultHandler::ABORT;
    config.rdma_device_health_probe_interval_ms   = 2000;
    config.rdma_device_health_fault_confirm_count = 5;

    auto monitor_config = config.makeRdmaDeviceHealthMonitorConfig(true);
    ASSERT_TRUE(monitor_config.enabled);
    ASSERT_EQ(monitor_config.fault_handler, RdmaDeviceHealthFaultHandler::ABORT);
    ASSERT_EQ(monitor_config.probe_interval_ms, 2000u);
    ASSERT_EQ(monitor_config.fault_confirm_count, 5u);

    // fault handler 与数值即使在探测未生效时也要保持原样，便于日志与后续排障
    auto disabled_config = config.makeRdmaDeviceHealthMonitorConfig(false);
    ASSERT_FALSE(disabled_config.enabled);
    ASSERT_EQ(disabled_config.fault_handler, RdmaDeviceHealthFaultHandler::ABORT);
    ASSERT_EQ(disabled_config.probe_interval_ms, 2000u);
    ASSERT_EQ(disabled_config.fault_confirm_count, 5u);
}

TEST_F(CacheStoreConfigTest, testRdmaDeviceHealthFaultHandlerName) {
    ASSERT_STREQ(rdmaDeviceHealthFaultHandlerName(RdmaDeviceHealthFaultHandler::LOG), "LOG");
    ASSERT_STREQ(rdmaDeviceHealthFaultHandlerName(RdmaDeviceHealthFaultHandler::ABORT), "ABORT");
}

}  // namespace rtp_llm
