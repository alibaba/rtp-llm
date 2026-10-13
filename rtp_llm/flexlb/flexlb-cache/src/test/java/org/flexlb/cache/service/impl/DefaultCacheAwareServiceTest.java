package org.flexlb.cache.service.impl;

import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DefaultCacheAwareServiceTest {

    @Test
    void cacheLookupFailureIsNotConvertedIntoACacheMiss() {
        KvCacheManager cache = mock(KvCacheManager.class);
        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        DefaultCacheAwareService service =
                new DefaultCacheAwareService(cache, metrics);
        IllegalStateException failure =
                new IllegalStateException("index unavailable");
        List<Long> keys = List.of(1L);
        List<String> candidates = List.of("127.0.0.1:8080");
        when(cache.findMatchingEngines(keys, candidates)).thenThrow(failure);

        IllegalStateException actual = assertThrows(
                IllegalStateException.class,
                () -> service.findMatchingEngines(
                        keys, RoleType.PREFILL, candidates));

        assertSame(failure, actual);
        verify(metrics).reportFindMatchingEnginesRT(
                eq(RoleType.PREFILL), anyLong(), eq("1"));
    }
}
