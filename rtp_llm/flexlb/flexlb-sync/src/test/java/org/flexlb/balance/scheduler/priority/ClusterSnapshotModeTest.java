package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * O(1) snapshot tests for {@link ClusterSnapshot}: capture is fixed to the
 * summary mode (aggregate scalars per decode endpoint, no eager full
 * snapshots), and {@link ClusterSnapshot#decodes()} lazily builds full
 * snapshots that agree with a direct per-endpoint capture.
 */
class ClusterSnapshotModeTest {

    private static final String DECODE_A = "10.0.0.2:8081";
    private static final String DECODE_B = "10.0.0.3:8081";

    private FlexlbConfig config;
    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setDecodeConcurrencyLimit(4);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = new EndpointRegistry(configService, () -> null,
                mock(BatchSchedulerReporter.class));
        registerDecode(DECODE_A, "10.0.0.2");
        registerDecode(DECODE_B, "10.0.0.3");
        DecodeEndpoint a = registry.getDecode(DECODE_A);
        a.reserve(1L, 500, 600, 30, 1_000);
        a.reserve(2L, 300, 350, 40, 2_000);
        registry.getDecode(DECODE_B).reserve(3L, 200, 220, 50, 3_000);
    }

    // ==================== summary capture + lazy upgrade parity ====================

    @Test
    void capture_capturesAggregatesOnly_andLazyDecodesMatchDirectCapture() {
        ClusterSnapshot snapshot = ClusterSnapshot.capture(registry, config);

        // Summary shape: aggregates for every endpoint, no eager full snapshots.
        assertEquals(2, snapshot.decodeSummaries().size());
        DecodeEndpointSummary summaryA = snapshot.decodeSummaries().get(DECODE_A);
        DecodeEndpoint liveA = registry.getDecode(DECODE_A);
        assertEquals(liveA.getTotalLoad(), summaryA.totalLoad());
        assertEquals(liveA.inflightHardKvReserved(), summaryA.hardKvReserved());
        assertEquals(config.getDecodeConcurrencyLimit(), summaryA.concurrencyLimit());

        // Lazy upgrade builds a fresh map per call, with direct-capture parity.
        Map<String, DecodeEndpointSnapshot> lazy = snapshot.decodes();
        assertNotSame(lazy, snapshot.decodes());
        assertEquals(2, lazy.size());
        for (String key : lazy.keySet()) {
            DecodeEndpointSnapshot expected = DecodeEndpointSnapshot.capture(
                    registry.getDecode(key), config.getDecodeConcurrencyLimit());
            DecodeEndpointSnapshot actual = lazy.get(key);
            assertEquals(expected.admissionVersion(), actual.admissionVersion());
            assertEquals(expected.totalLoad(), actual.totalLoad());
            assertEquals(expected.engineLoad(), actual.engineLoad());
            assertEquals(expected.hardKvReserved(), actual.hardKvReserved());
            assertEquals(expected.reserved().size(), actual.reserved().size());
        }
    }

    // ==================== summary mode reflects mutations at upgrade time ====================

    @Test
    void lazyUpgradeSeesMutationsAfterCapture() {
        ClusterSnapshot snapshot = ClusterSnapshot.capture(registry, config);
        DecodeEndpointSummary before = snapshot.decodeSummaries().get(DECODE_B);
        assertEquals(1, before.totalLoad());

        // Mutation after capture: the frozen aggregates keep the old value,
        // the lazy upgrade (used only by eviction planning) sees live state.
        registry.getDecode(DECODE_B).release(3L);
        assertEquals(1, before.totalLoad());
        assertEquals(0, snapshot.decodes().get(DECODE_B).totalLoad());
    }

    private void registerDecode(String ipPort, String ip) {
        WorkerStatus ws = new WorkerStatus();
        ws.setIp(ip);
        ws.setPort(8081);
        ws.setGrpcPort(8082);
        ws.setAvailableKvCacheTokens(new AtomicLong(100_000L));
        ws.setTotalKvCacheTokens(new AtomicLong(200_000L));
        registry.ensureEndpoint(RoleType.DECODE, ipPort, ws);
    }
}
