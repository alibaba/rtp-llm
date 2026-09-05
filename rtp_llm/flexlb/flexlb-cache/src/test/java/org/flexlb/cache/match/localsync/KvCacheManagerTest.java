package org.flexlb.cache.match.localsync;

import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Set;

import static org.mockito.Mockito.never;
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
                new WorkerIdentity("10.0.0.1", 8080, 0), "PREFILL", Set.of());

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
                new WorkerIdentity("10.0.0.1", 8080, 0), "PREFILL", Set.of());

        kvCacheManager.removeStaleEngineCaches(List.of("10.0.0.2:8080"));

        verify(engineLocalView).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
        verify(globalCacheIndex).removeAllCacheBlockOfEngine("10.0.0.1:8080@0");
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
                new WorkerIdentity("10.0.0.1", 8080, 0), "PREFILL", Set.of());
        kvCacheManager.updateEngineCache(
                new WorkerIdentity("10.0.0.1", 8080, 1), "PREFILL", Set.of());

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
                new WorkerIdentity("10.0.0.1", 8080, 0), "PREFILL", Set.of());
        kvCacheManager.updateEngineCache(
                new WorkerIdentity("10.0.0.1", 8080, 1), "PREFILL", Set.of());

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
}
