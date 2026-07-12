package org.flexlb.cache.core;

import org.flexlb.cache.monitor.CacheMetricsReporter;
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
    void ignoresNullAddressUpdates() {
        kvCacheManager.removeStaleEngineCaches(null);

        verifyNoInteractions(engineLocalView, globalCacheIndex);
    }
}
