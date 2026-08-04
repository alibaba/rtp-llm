package org.flexlb.cache.match.localstandby;

import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.LocalStandbyHashResult;
import org.flexlb.cache.hash.LocalStandbyHashService;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class LocalStandbyCacheMatchProviderTest {

    @Test
    void waitsForStandbyHashBeforeMatching() throws Exception {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyHashService hashService = mock(LocalStandbyHashService.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                new CacheMatchConfiguration(modelMetaConfig()), cacheManager, hashService);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1", List.of(11L), 2192, null, 4096, RoleType.PREFILL, "default");
        CompletableFuture<LocalStandbyHashResult> pendingHash = new CompletableFuture<>();
        when(hashService.getHashResult("request-1", null, 4096)).thenReturn(pendingHash);
        when(cacheManager.findMatchingEngines(List.of(101L), RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.1:8080", 1));

        try {
            CompletableFuture<CacheMatchResult> match = provider.asyncLocalStandbyMatch(query);
            assertFalse(match.isDone());

            pendingHash.complete(new LocalStandbyHashResult(List.of(101L), 4096));
            CacheMatchResult result = match.get(1, TimeUnit.SECONDS);

            assertEquals(Map.of("10.0.0.1:8080", 1), result.matches());
            assertEquals(4096, result.blockSize());
            verify(cacheManager).findMatchingEngines(List.of(101L), RoleType.PREFILL, "default");
        } finally {
            provider.shutdown();
        }
    }

    @Test
    void updatesRequestDerivedCacheMetadataAsynchronously() {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyHashService hashService = mock(LocalStandbyHashService.class);
        LocalStandbyCacheMatchProvider provider =
                new LocalStandbyCacheMatchProvider(
                        new CacheMatchConfiguration(modelMetaConfig()),
                        cacheManager,
                        hashService);

        Request request = new Request();
        request.setRequestId("request-1");
        request.setLocalStandbyBlockSize(4096);
        CompletableFuture<LocalStandbyHashResult> pendingHash = new CompletableFuture<>();
        when(hashService.getHashResult("request-1", null, 4096)).thenReturn(pendingHash);

        ServerStatus selectedWorker = new ServerStatus();
        selectedWorker.setSuccess(true);
        selectedWorker.setServerIp("10.0.0.1");
        selectedWorker.setHttpPort(8080);
        selectedWorker.setRole(RoleType.PREFILL);
        selectedWorker.setGroup("default");

        try {
            provider.updateFromRoutedRequest(request, List.of(selectedWorker));
            verifyNoInteractions(cacheManager);

            pendingHash.complete(new LocalStandbyHashResult(List.of(11L, 22L), 4096));

            verify(cacheManager, timeout(1_000))
                    .addRoutedRequestBlocks("10.0.0.1:8080", List.of(11L, 22L));
        } finally {
            provider.shutdown();
        }
    }

    @Test
    void ignoresCacheMetadataForNonPrefillWorkers() {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyHashService hashService = mock(LocalStandbyHashService.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                new CacheMatchConfiguration(modelMetaConfig()), cacheManager, hashService);

        Request request = new Request();
        request.setRequestId("request-1");
        request.setLocalStandbyBlockSize(4096);
        when(hashService.getHashResult("request-1", null, 4096))
                .thenReturn(CompletableFuture.completedFuture(
                        new LocalStandbyHashResult(List.of(11L), 4096)));

        ServerStatus decodeWorker = new ServerStatus();
        decodeWorker.setSuccess(true);
        decodeWorker.setServerIp("10.0.0.1");
        decodeWorker.setHttpPort(8080);
        decodeWorker.setRole(RoleType.DECODE);

        try {
            provider.updateFromRoutedRequest(request, List.of(decodeWorker));
            verify(cacheManager, after(200).never())
                    .addRoutedRequestBlocks(anyString(), anyList());
        } finally {
            provider.shutdown();
        }
    }

    @Test
    void runsReactiveMatchingOnTheBoundedMatcherExecutor() {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyHashService hashService = mock(LocalStandbyHashService.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                new CacheMatchConfiguration(modelMetaConfig()), cacheManager, hashService);
        AtomicReference<String> matchingThread = new AtomicReference<>();
        when(cacheManager.findMatchingEngines(List.of(101L), RoleType.PREFILL, "default"))
                .thenAnswer(invocation -> {
                    matchingThread.set(Thread.currentThread().getName());
                    return Map.of("10.0.0.1:8080", 1);
                });

        try {
            Map<String, Integer> matches = provider.findMatchingEngines(
                    "request-1", List.of(101L), 4096, RoleType.PREFILL, "default").block();

            assertEquals(Map.of("10.0.0.1:8080", 1), matches);
            assertTrue(matchingThread.get().startsWith("local-standby-cache-matcher"));
        } finally {
            provider.shutdown();
        }
    }

    private ModelMetaConfig modelMetaConfig() {
        LocalStandbyConfig standby = new LocalStandbyConfig();

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setLocalStandby(standby);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return config;
    }
}
