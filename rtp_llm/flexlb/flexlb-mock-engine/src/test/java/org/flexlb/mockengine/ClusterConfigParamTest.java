package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ClusterConfigParamTest {
    private String[] args(String... extra) {
        var values = new java.util.ArrayList<>(java.util.List.of("--endpoint-file", "endpoints.json",
                "--engine-config", "../tools/mock_bundle/engine-config.json", "--calibration", "calibration.json"));
        values.addAll(java.util.List.of(extra));
        return values.toArray(String[]::new);
    }
    @Test void configurationOwnsRoleCapacityAndConcurrency() {
        var config = JavaMockEngineCluster.Config.parse(args("--n-prefill", "1", "--n-decode", "0"));
        assertEquals(11042, config.engineConfig.blocks("prefill"));
        assertEquals(128, config.engineConfig.concurrency("decode"));
        assertEquals(1, config.nPrefill);
    }
    @Test void removedCapacityAndPerformanceArgumentsAreRejected() {
        for (String flag : new String[]{"--total-kv-tokens", "--block-size", "--prefill-block-size", "--decode-block-size",
                "--decode-max-concurrency", "--prefill-kv-pool-blocks", "--decode-kv-pool-blocks", "--performance"}) {
            assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(args(flag, "1")), flag);
        }
    }
    @Test void runtimeStillValidatesThreadAndTransportLimits() {
        assertEquals(7, JavaMockEngineCluster.Config.parse(args("--event-loop-threads", "7")).eventLoopThreads);
        for (String flag : new String[]{"--event-loop-threads", "--completion-threads", "--rpc-threads", "--rpc-max-calls", "--grpc-max-inbound-message-bytes"})
            assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(args(flag, "0")));
    }
    @Test void missingEngineConfigurationFailsBeforeStartup() {
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(new String[]{
                "--endpoint-file", "endpoints.json", "--calibration", "calibration.json",}));
    }
}
