package org.flexlb.cache.core;

import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class KvCacheManagerTest {

    private static final String ENGINE = "127.0.0.1:8080";

    @Mock
    private GlobalCacheIndex globalCacheIndex;
    @Mock
    private EngineLocalView engineLocalView;
    @Mock
    private WorkerStatusProvider workerStatusProvider;
    @Mock
    private CacheMetricsReporter cacheMetricsReporter;
    @InjectMocks
    private KvCacheManager manager;

    private Set<Long> oldSnapshot;
    private Set<Long> newSnapshot;

    @BeforeEach
    void setUp() {
        oldSnapshot = Set.of(1L);
        newSnapshot = Set.of(2L);
        when(engineLocalView.getEngineCacheBlocks(ENGINE)).thenReturn(oldSnapshot);
    }

    @Test
    void should_commit_local_snapshot_only_after_global_publication_succeeds() {
        when(globalCacheIndex.applyEngineCacheSnapshot(ENGINE, oldSnapshot, newSnapshot))
                .thenThrow(new IllegalStateException("injected global failure"))
                .thenReturn(new GlobalCacheIndex.CacheDiffStats(1, 1));

        assertThrows(IllegalStateException.class,
                () -> manager.updateEngineCache(ENGINE, "prefill", newSnapshot));
        verify(engineLocalView, never()).commitSnapshot(ENGINE, newSnapshot);

        manager.updateEngineCache(ENGINE, "prefill", newSnapshot);
        verify(engineLocalView).commitSnapshot(ENGINE, newSnapshot);
    }

    @Test
    void should_keep_local_clear_marker_until_global_cleanup_succeeds() {
        doThrow(new IllegalStateException("injected global failure"))
                .doNothing()
                .when(globalCacheIndex).removeEngineCacheBlocks(ENGINE, oldSnapshot);

        assertThrows(IllegalStateException.class, () -> manager.clearEngineCache(ENGINE));
        verify(engineLocalView, never()).removeAllCacheBlockOfEngine(ENGINE);

        manager.clearEngineCache(ENGINE);
        verify(engineLocalView).removeAllCacheBlockOfEngine(ENGINE);
    }

    @Test
    void should_retry_failed_clear_before_publishing_next_generation() {
        Set<Long> emptySnapshot = Set.of();
        when(engineLocalView.getEngineCacheBlocks(ENGINE))
                .thenReturn(oldSnapshot, oldSnapshot, emptySnapshot);
        doThrow(new IllegalStateException("injected global failure"))
                .doNothing()
                .when(globalCacheIndex).removeEngineCacheBlocks(ENGINE, oldSnapshot);
        when(globalCacheIndex.applyEngineCacheSnapshot(ENGINE, emptySnapshot, newSnapshot))
                .thenReturn(new GlobalCacheIndex.CacheDiffStats(1, 0));

        assertThrows(IllegalStateException.class, () -> manager.clearEngineCache(ENGINE));
        manager.updateEngineCache(ENGINE, "prefill", newSnapshot);

        verify(globalCacheIndex, times(2)).removeEngineCacheBlocks(ENGINE, oldSnapshot);
        verify(engineLocalView).removeAllCacheBlockOfEngine(ENGINE);
        verify(globalCacheIndex).applyEngineCacheSnapshot(ENGINE, emptySnapshot, newSnapshot);
        verify(engineLocalView).commitSnapshot(ENGINE, newSnapshot);
    }

}
