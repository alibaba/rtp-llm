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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * O(1) snapshot redesign mode tests for {@link ClusterSnapshot}: with the
 * {@code autoTpmSnapshotMode} switch at its default ({@code full}, i.e. the
 * gray switch OFF), capture must keep the legacy eager per-endpoint full
 * snapshots so the decision path is unchanged; with {@code summary}, capture
 * must take only aggregate scalars and {@link ClusterSnapshot#decodes()} must
 * lazily build full snapshots that agree with a direct full capture.
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

    // ==================== switch OFF (default): legacy eager behavior ====================

    @Test
    void defaultMode_isFull_andCapturesEagerDecodeSnapshots() {
        assertFalse(config.isAutoTpmSnapshotSummaryMode(), "gray switch must default to OFF");

        ClusterSnapshot snapshot = ClusterSnapshot.capture(registry, config);

        // Legacy shape: the full snapshots are captured eagerly at capture()
        // time and every decodes() call returns that same map — no lazy
        // re-capture may sneak into the default decision path.
        assertNotNull(snapshot.eagerDecodes());
        assertSame(snapshot.eagerDecodes(), snapshot.decodes());
        assertSame(snapshot.decodes(), snapshot.decodes());

        DecodeEndpointSnapshot direct =
                DecodeEndpointSnapshot.capture(registry.getDecode(DECODE_A),
                        config.getDecodeConcurrencyLimit());
        DecodeEndpointSnapshot captured = snapshot.decodes().get(DECODE_A);
        assertEquals(direct.admissionVersion(), captured.admissionVersion());
        assertEquals(direct.totalLoad(), captured.totalLoad());
        assertEquals(direct.hardKvReserved(), captured.hardKvReserved());
        assertEquals(direct.reserved().size(), captured.reserved().size());
        assertEquals(2, captured.reserved().size());
    }

    // ==================== switch ON: summary capture + lazy upgrade parity ====================

    @Test
    void summaryMode_capturesAggregatesOnly_andLazyDecodesMatchFullCapture() {
        config.setAutoTpmSnapshotMode("summary");
        assertTrue(config.isAutoTpmSnapshotSummaryMode());

        ClusterSnapshot snapshot = ClusterSnapshot.capture(registry, config);

        // Summary shape: no eager full snapshots, aggregates for every endpoint.
        assertNull(snapshot.eagerDecodes());
        assertEquals(2, snapshot.decodeSummaries().size());
        DecodeEndpointSummary summaryA = snapshot.decodeSummaries().get(DECODE_A);
        DecodeEndpoint liveA = registry.getDecode(DECODE_A);
        assertEquals(liveA.getTotalLoad(), summaryA.totalLoad());
        assertEquals(liveA.inflightHardKvReserved(), summaryA.hardKvReserved());
        assertEquals(config.getDecodeConcurrencyLimit(), summaryA.concurrencyLimit());

        // Lazy upgrade builds a fresh map per call, with full-capture parity.
        Map<String, DecodeEndpointSnapshot> lazy = snapshot.decodes();
        assertNotSame(lazy, snapshot.decodes());
        assertEquals(2, lazy.size());
        ClusterSnapshot fullSnapshot = ClusterSnapshot.captureFull(registry, config);
        for (String key : fullSnapshot.decodes().keySet()) {
            DecodeEndpointSnapshot expected = fullSnapshot.decodes().get(key);
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
    void summaryMode_lazyUpgradeSeesMutationsAfterCapture() {
        config.setAutoTpmSnapshotMode("summary");
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
