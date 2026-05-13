package org.flexlb.config;

import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(SystemStubsExtension.class)
class ConfigServiceTest {

    private static final String FLEXLB_CONFIG = """
            {
                "enableQueueing": true,
                "deploy": "DISAGGREGATED",
                "loadBalanceStrategy": "SHORTEST_TTFT",
                "prefillBatchWaitTimeMs": 100,
                "kvCache": "LOCAL_STATIC",
                "staticCacheBlockSize": 500,
                "batchSize": 1,
                "prefillLbTimeoutMs": 300,
                "prefillGenerateTimeoutMs": 5000,
                "enableGrpcPrefillMaster": false,
                "enableGrpcCacheStatus": true,
                "enableGrpcEngineStatus": true,
                "maxPrefillQueueSize": 10,
                "prefillQueueSizeThreshold": 8,
                "decodeConcurrencyLimit": 128,
                "maxQueueSize": 20
            }
            """;

    @SystemStub
    private EnvironmentVariables environmentVariables = new EnvironmentVariables();

    @Test
    void loadBalanceConfig_shouldParseFlexlbConfigFromEnvironment() {
        environmentVariables.set("FLEXLB_CONFIG", FLEXLB_CONFIG);

        FlexlbConfig config = new ConfigService().loadBalanceConfig();

        assertTrue(config.isEnableQueueing());
        assertEquals(LoadBalanceStrategyEnum.SHORTEST_TTFT, config.getLoadBalanceStrategy());
        assertEquals(20, config.getMaxQueueSize());
        assertEquals(10, config.getMaxPrefillQueueSize());
        assertEquals(8, config.getPrefillQueueSizeThreshold());
        assertEquals("DISAGGREGATED", config.getDeploy());
        assertEquals("LOCAL_STATIC", config.getKvCache());
        assertEquals(500, config.getStaticCacheBlockSize());
        assertEquals(1, config.getBatchSize());
        assertEquals(100, config.getPrefillBatchWaitTimeMs());
        assertEquals(300, config.getPrefillLbTimeoutMs());
        assertEquals(5000, config.getPrefillGenerateTimeoutMs());
        assertFalse(config.isEnableGrpcPrefillMaster());
        assertTrue(config.isEnableGrpcCacheStatus());
        assertTrue(config.isEnableGrpcEngineStatus());
        assertEquals(128, config.getDecodeConcurrencyLimit());
    }

    @Test
    void loadBalanceConfig_shouldApplyPrimitiveEnvironmentOverrides() {
        environmentVariables.set("FLEXLB_CONFIG", FLEXLB_CONFIG);
        environmentVariables.set("ENABLE_QUEUEING", "false");
        environmentVariables.set("MAX_QUEUE_SIZE", "7");
        environmentVariables.set("PREFILL_QUEUE_SIZE_THRESHOLD", "3");
        environmentVariables.set("KV_CACHE", "LOCAL_DYNAMIC");

        FlexlbConfig config = new ConfigService().loadBalanceConfig();

        assertFalse(config.isEnableQueueing());
        assertEquals(7, config.getMaxQueueSize());
        assertEquals(3, config.getPrefillQueueSizeThreshold());
        assertEquals("LOCAL_DYNAMIC", config.getKvCache());
    }
}
