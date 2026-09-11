package org.flexlb.config;

import org.flexlb.enums.EngineType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbConfigTest {
    @Test
    void batchOptionsUseSchemaVersionTwo() {
        FlexlbConfig config = ConfigService.parse("""
                {"schemaVersion":2,"router":{"batchScheduleMaxCount":32},
                 "workerRegistry":{"engineType":"EMBEDDING"}}
                """);
        assertEquals(32, config.getRouter().getBatchScheduleMaxCount());
        assertEquals(EngineType.EMBEDDING, config.getWorkerRegistry().getEngineType());
    }

    @Test
    void defaultsPreserveMainlineScheduling() {
        FlexlbConfig config = ConfigService.parse("{\"schemaVersion\":2}");
        assertEquals(1000, config.getRouter().getBatchScheduleMaxCount());
        assertEquals(EngineType.LLM, config.getWorkerRegistry().getEngineType());
        assertTrue(config.isQueue());
        assertEquals(DispatcherConfig.Type.BATCH, config.getDispatcher().getType());
    }

    @Test
    void oldBatchEnvironmentCannotSilentlySelectTheWrongEngineProtocol() {
        for (String key : new String[]{"ENGINE_TYPE", "FLEXLB_ENGINE_TYPE",
                "BATCH_SCHEDULE_MAX_COUNT", "BATCH_LOAD_BALANCE_STRATEGY"}) {
            assertThrows(ConfigValidationException.class,
                    () -> new ConfigService(java.util.Map.of(key, "EMBEDDING")));
        }
    }

    @Test
    void invalidBatchConfigurationFailsAtStartup() {
        for (String document : new String[] {
                "{\"schemaVersion\":2,\"router\":{\"batchScheduleMaxCount\":0}}",
                "{\"schemaVersion\":2,\"workerRegistry\":{\"engineType\":\"UNKNOWN\"}}",
                "{\"schemaVersion\":2,\"batchScheduleMaxCount\":32}"}) {
            assertThrows(ConfigValidationException.class, () -> ConfigService.parse(document));
        }
    }
}
