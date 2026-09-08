package org.flexlb.cache.match.localsync;

import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class KvCacheManagerTest {

    @Mock
    private GlobalCacheIndex globalCacheIndex;

    @Mock
    private EngineLocalView engineLocalView;

    @Mock
    private WorkerStatusProvider workerStatusProvider;

    @Mock
    private CacheMetricsReporter cacheMetricsReporter;

    @InjectMocks
    private KvCacheManager kvCacheManager;

    @Test
    void findsRtpSingleEngineCacheByLogicalWorkerStatusIdentity() {
        KvCacheManager manager = kvCacheManagerWithRealIndexes();
        manager.updateEngineCache(workerStatus(0, 1), "PREFILL", Set.of(11L, 22L));
        when(workerStatusProvider.getWorkerStatuses(RoleType.PREFILL, "rtp"))
                .thenReturn(List.of(workerStatus(0, 1)));

        Map<String, Integer> matches = manager.findMatchingEngines(
                List.of(11L, 22L, 33L), RoleType.PREFILL, "rtp");

        assertEquals(Map.of("10.0.0.1:8080@0", 2), matches);
        assertFalse(matches.containsKey("10.0.0.1:8080"));
    }

    @Test
    void reportsRtpSingleEngineCacheMetricsWithPhysicalAddress() {
        WorkerStatus status = workerStatus(0, 1);
        when(engineLocalView.calculateDiff("10.0.0.1:8080@0", Set.of()))
                .thenReturn(DiffResult.empty("10.0.0.1:8080@0"));

        kvCacheManager.updateEngineCache(status, "PREFILL", Set.of());

        verify(cacheMetricsReporter).reportCacheDiffMetrics(
                "10.0.0.1:8080", "PREFILL", 0, 0);
    }

    @Test
    void removesStaleCacheWhenActiveWorkerIsReplacedAtTheSameCount() {
        when(engineLocalView.getAllEngineIpPorts())
                .thenReturn(Set.of("10.0.0.1:8080"));

        kvCacheManager.removeStaleEngineCaches(List.of("10.0.0.2:8080"));

        verify(engineLocalView).removeAllCacheBlockOfEngine("10.0.0.1:8080");
        verify(globalCacheIndex).removeAllCacheBlockOfEngine("10.0.0.1:8080");
        verify(engineLocalView, never()).removeAllCacheBlockOfEngine("10.0.0.2:8080");
    }

    @Test
    void keepsLogicalCacheWhenItsPhysicalEngineRemainsDiscoverable() {
        when(engineLocalView.getAllEngineIpPorts())
                .thenReturn(Set.of("10.0.0.1:8080@0"));
        when(engineLocalView.calculateDiff("10.0.0.1:8080@0", Set.of()))
                .thenReturn(DiffResult.empty("10.0.0.1:8080@0"));
        kvCacheManager.updateEngineCache(
                workerStatus(0, 1), "PREFILL", Set.of());

        kvCacheManager.removeStaleEngineCaches(List.of("10.0.0.1:8080"));

        verify(engineLocalView, never()).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
        verify(globalCacheIndex, never()).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
    }

    @Test
    void removesLogicalCacheWhenItsPhysicalEngineDisappears() {
        when(engineLocalView.getAllEngineIpPorts())
                .thenReturn(Set.of("10.0.0.1:8080@0"));
        when(engineLocalView.calculateDiff("10.0.0.1:8080@0", Set.of()))
                .thenReturn(DiffResult.empty("10.0.0.1:8080@0"));
        kvCacheManager.updateEngineCache(
                workerStatus(0, 1), "PREFILL", Set.of());

        kvCacheManager.removeStaleEngineCaches(List.of("10.0.0.2:8080"));

        verify(engineLocalView).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
        verify(globalCacheIndex).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
    }

    @Test
    void directRetirementRemovesTheLogicalToPhysicalMapping() {
        String logicalIpPort = "10.0.0.1:8080@0";
        when(engineLocalView.calculateDiff(logicalIpPort, Set.of()))
                .thenReturn(DiffResult.empty(logicalIpPort));
        kvCacheManager.updateEngineCache(
                workerStatus(0, 1), "PREFILL", Set.of());

        kvCacheManager.removeEngineCache(logicalIpPort);
        when(engineLocalView.getAllEngineIpPorts()).thenReturn(Set.of(logicalIpPort));
        kvCacheManager.removeStaleEngineCaches(List.of("10.0.0.1:8080"));

        verify(engineLocalView, times(2)).removeAllCacheBlockOfEngine(logicalIpPort);
        verify(globalCacheIndex, times(2)).removeAllCacheBlockOfEngine(logicalIpPort);
    }

    @Test
    void keepsAllSiblingLogicalCachesWhenTheirPhysicalEngineRemainsDiscoverable() {
        when(engineLocalView.getAllEngineIpPorts())
                .thenReturn(Set.of("10.0.0.1:8080@0", "10.0.0.1:8080@1"));
        when(engineLocalView.calculateDiff("10.0.0.1:8080@0", Set.of()))
                .thenReturn(DiffResult.empty("10.0.0.1:8080@0"));
        when(engineLocalView.calculateDiff("10.0.0.1:8080@1", Set.of()))
                .thenReturn(DiffResult.empty("10.0.0.1:8080@1"));
        kvCacheManager.updateEngineCache(
                workerStatus(0, 2), "PREFILL", Set.of());
        kvCacheManager.updateEngineCache(
                workerStatus(1, 2), "PREFILL", Set.of());

        kvCacheManager.removeStaleEngineCaches(List.of("10.0.0.1:8080"));

        verify(engineLocalView, never()).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
        verify(engineLocalView, never()).removeAllCacheBlockOfEngine("10.0.0.1:8080@1");
        verify(globalCacheIndex, never()).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
        verify(globalCacheIndex, never()).removeAllCacheBlockOfEngine("10.0.0.1:8080@1");
    }

    @Test
    void removesAllSiblingLogicalCachesWhenTheirPhysicalEngineDisappears() {
        when(engineLocalView.getAllEngineIpPorts())
                .thenReturn(Set.of("10.0.0.1:8080@0", "10.0.0.1:8080@1"));
        when(engineLocalView.calculateDiff("10.0.0.1:8080@0", Set.of()))
                .thenReturn(DiffResult.empty("10.0.0.1:8080@0"));
        when(engineLocalView.calculateDiff("10.0.0.1:8080@1", Set.of()))
                .thenReturn(DiffResult.empty("10.0.0.1:8080@1"));
        kvCacheManager.updateEngineCache(
                workerStatus(0, 2), "PREFILL", Set.of());
        kvCacheManager.updateEngineCache(
                workerStatus(1, 2), "PREFILL", Set.of());

        kvCacheManager.removeStaleEngineCaches(List.of("10.0.0.2:8080"));

        verify(engineLocalView).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
        verify(engineLocalView).removeAllCacheBlockOfEngine("10.0.0.1:8080@1");
        verify(globalCacheIndex).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
        verify(globalCacheIndex).removeAllCacheBlockOfEngine("10.0.0.1:8080@1");
    }

    @Test
    void treatsUnmappedLogicalKeyAsStaleEvenWhenItsPhysicalEngineRemainsDiscoverable() {
        when(engineLocalView.getAllEngineIpPorts())
                .thenReturn(Set.of("10.0.0.1:8080@0"));

        kvCacheManager.removeStaleEngineCaches(List.of("10.0.0.1:8080"));

        verify(engineLocalView).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
        verify(globalCacheIndex).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
    }

    @Test
    void ignoresNullAddressUpdates() {
        kvCacheManager.removeStaleEngineCaches(null);

        verifyNoInteractions(engineLocalView, globalCacheIndex);
    }

    private static WorkerStatus workerStatus(int engineIndex, int multiEngineNum) {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, "rtp", "10.0.0.1", 8080, 9090,
                "test-site", "rtp-deploy", engineIndex, multiEngineNum);
    }

    private KvCacheManager kvCacheManagerWithRealIndexes() {
        KvCacheManager manager = new KvCacheManager();
        EngineLocalView localView = new EngineLocalView();
        ReflectionTestUtils.setField(
                localView, "dynamicIntervalManager",
                (DynamicCacheIntervalService) diffSize -> { });
        ReflectionTestUtils.setField(
                manager, "globalCacheIndex", new GlobalCacheIndex());
        ReflectionTestUtils.setField(
                manager, "engineLocalView", localView);
        ReflectionTestUtils.setField(
                manager, "workerStatusProvider", workerStatusProvider);
        ReflectionTestUtils.setField(
                manager, "cacheMetricsReporter", cacheMetricsReporter);
        return manager;
    }
}
