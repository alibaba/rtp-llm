package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;

class ExpirationCleanerTest {

    @Test
    void detachesEveryExpiredWorkerBeforeAwaitingAnyRetirement()
            throws Exception {
        EndpointRegistry registry = mock(EndpointRegistry.class);
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatus first = status("127.0.0.1", 8080);
        WorkerStatus second = status("127.0.0.2", 8080);
        WorkerDirectory directory = new WorkerDirectory(registry);
        directory.currentOrDiscover(
                RoleType.PREFILL, first.getIpPort(), () -> first);
        directory.currentOrDiscover(
                RoleType.PREFILL, second.getIpPort(), () -> second);

        EndpointRegistry.DetachedGeneration firstDetached =
                mock(EndpointRegistry.DetachedGeneration.class);
        EndpointRegistry.DetachedGeneration secondDetached =
                mock(EndpointRegistry.DetachedGeneration.class);
        WorkerEndpoint firstEndpoint = mock(WorkerEndpoint.class);
        WorkerEndpoint secondEndpoint = mock(WorkerEndpoint.class);
        configureDetach(
                registry, first, firstEndpoint, firstDetached);
        configureDetach(
                registry, second, secondEndpoint, secondDetached);

        CountDownLatch firstAwaitEntered = new CountDownLatch(1);
        CountDownLatch releaseFirstAwait = new CountDownLatch(1);
        org.mockito.Mockito.doAnswer(invocation -> {
            firstAwaitEntered.countDown();
            assertTrue(releaseFirstAwait.await(5, TimeUnit.SECONDS));
            return null;
        }).when(firstDetached).retireAndAwait();

        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        config.getWorkerRegistry().getHealth().setStatusStaleAfterMs(0L);
        when(configService.loadBalanceConfig()).thenReturn(config);
        ExpirationCleaner cleaner = new ExpirationCleaner(
                configService, cache, directory);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> cleaning = executor.submit(
                    cleaner::cleanExpiredWorkers);

            assertTrue(firstAwaitEntered.await(2, TimeUnit.SECONDS));
            assertFalse(first.isActiveGeneration());
            assertFalse(second.isActiveGeneration(),
                    "the second routing gate must close before the first drain waits");
            releaseFirstAwait.countDown();
            cleaning.get(5, TimeUnit.SECONDS);
            assertTrue(directory.statusSnapshot(RoleType.PREFILL).isEmpty());
            verify(cache).removeEngineBlockCache(first.getIpPort());
            verify(cache).removeEngineBlockCache(second.getIpPort());
        } finally {
            releaseFirstAwait.countDown();
            executor.shutdownNow();
        }
    }

    private static void configureDetach(
            EndpointRegistry registry,
            WorkerStatus status,
            WorkerEndpoint endpoint,
            EndpointRegistry.DetachedGeneration detached) {
        when(registry.get(
                RoleType.PREFILL, status.getIpPort(), status))
                .thenReturn(endpoint);
        when(detached.ownsEndpoint(endpoint)).thenReturn(true);
        when(registry.detachAndBeginRetirement(
                RoleType.PREFILL, status.getIpPort(), status))
                .thenAnswer(invocation -> {
                    status.beginRetirementAfterEndpointGateClosed();
                    return detached;
                });
    }

    private static WorkerStatus status(String ip, int port) {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, ip, port, port + 1, "test-site");
    }
}
