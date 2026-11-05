package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExecutorConfigTest {
    @Test
    void defaultsPreserveExistingPoolSizes() {
        var config = ConfigTestFixtures.parse("{}");
        var workers = config.getWorkerRegistry().getExecutors();
        assertEquals(32, workers.getCacheStatusThreads());
        assertEquals(8, workers.getCacheDiffParallelism());
        assertEquals(32, config.getEngineGrpc().getExecutorThreads());
        assertEquals(10_000, config.getEngineGrpc().getExecutorQueueCapacity());
        assertEquals(8, config.getEngineGrpc().getEventLoopThreads());
    }

    @Test
    void eachPoolCanBeSizedIndependentlyAndRejectsInvalidValues() {
        String[] paths = {
            "workerRegistry.executors.cacheStatusThreads",
            "workerRegistry.executors.cacheDiffParallelism",
            "engineGrpc.executorThreads",
            "engineGrpc.executorQueueCapacity",
            "engineGrpc.eventLoopThreads"
        };
        for (String path : paths) {
            ConfigTestFixtures.parse(patch(path, "16"));
            for (String value :
                    new String[] {"0", "-1", "null", "1.5", "\"16\"", "true", "2147483648"}) {
                var error =
                        assertThrows(
                                ConfigValidationException.class,
                                () -> ConfigTestFixtures.parse(patch(path, value)),
                                path + "=" + value);
                assertTrue(
                        error.getMessage().contains(path.substring(path.lastIndexOf('.') + 1)),
                        error.getMessage());
            }
        }
        assertThrows(
                ConfigValidationException.class,
                () ->
                        ConfigTestFixtures.parse(
                                patch("workerRegistry.executors.cacheDiffParallelism", "32768")));
        ConfigTestFixtures.parse(patch("workerRegistry.executors.cacheDiffParallelism", "32767"));
    }

    private static String patch(String path, String value) {
        String[] parts = path.split("\\.");
        for (int index = parts.length - 1; index >= 0; index--) {
            value = "{\"" + parts[index] + "\":" + value + "}";
        }
        return value;
    }
}
