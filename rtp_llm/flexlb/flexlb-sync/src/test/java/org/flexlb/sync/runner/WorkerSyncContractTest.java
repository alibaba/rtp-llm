package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.schedule.WorkerRetirementCleaner;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.flexlb.sync.runner.RunnerTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class WorkerSyncContractTest {
    private final ConfigService config = mock(ConfigService.class);
    private final WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
    private final CacheAwareService cache = mock(CacheAwareService.class);
    private final EngineHealthReporter reporter = mock(EngineHealthReporter.class);
    private final DynamicCacheIntervalService interval = mock(DynamicCacheIntervalService.class);
    private final WorkerStatus status = RunnerTestSupport.discovered(
            RoleType.PREFILL, "", "127.0.0.1", 8080, 8081, null);
    private final EndpointRegistry registry;

    WorkerSyncContractTest() {
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        registry = RunnerTestSupport.endpointRegistry(config);
        registry.currentOrDiscover(status.getRole(), status.getIpPort(), () -> status);
    }

    @AfterEach
    void closeRegistry() { registry.close(); }

    @Test
    void queuedApplicationDoesNotBlockHeartbeatsOrAccumulateApplications() {
        ArrayDeque<Runnable> applications = new ArrayDeque<>();
        poll(true, 1L, applications::add);
        assertEquals(-1L, status.appliedStatusCursor().statusVersion());
        assertTrue(status.pollHealth().reportedAlive());

        poll(false, 1L, applications::add);
        poll(true, 2L, applications::add);
        assertEquals(1, applications.size(), "at most one pending application per generation");
        assertTrue(status.pollHealth().reportedAlive());
        assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
        applications.remove().run();
        assertEquals(1L, status.appliedStatusCursor().statusVersion());
        poll(true, 2L, Runnable::run);
        assertEquals(2L, status.appliedStatusCursor().statusVersion());
    }

    @Test
    void rejectedApplicationReleasesBothOwnersAndKeepsHeartbeatEvidence() {
        poll(true, 1L, task -> { throw new RejectedExecutionException("full"); });
        assertTrue(status.pollHealth().reportedAlive());
        assertEquals(-1L, status.appliedStatusCursor().statusVersion());
        poll(true, 1L, Runnable::run);
        assertEquals(1L, status.appliedStatusCursor().statusVersion());
        assertNotNull(registry.get(status.getRole(), status.getIpPort()));
    }

    @Test
    void thirdNotAliveResponseFencesRoutingEvenWhileAnOlderApplicationIsQueued() {
        RunnerTestSupport.publishEndpoint(registry, status.getRole(), status.getIpPort(), status);
        ArrayDeque<Runnable> applications = new ArrayDeque<>();
        poll(true, 2L, applications::add);
        for (int i = 0; i < 3; i++) { poll(false, 2L, applications::add); }
        assertEquals(WorkerStatus.RetirementReason.CONSECUTIVE_WORKER_FAILURES, status.retirementReason());
        assertNull(registry.capture(status.getRole(), status.getIpPort()));
        applications.remove().run();
        org.flexlb.sync.runner.RunnerTestSupport.awaitCondition(() -> registry.statusSnapshot(status.getRole()).isEmpty());
        assertEquals(1L, status.appliedStatusCursor().statusVersion(), "older queued alive response cannot revive the generation");
        verify(cache).removeEngineBlockCache(status.getIpPort());
    }

    @Test
    void discoveryFailureAndBlockedGenerationLockStillAllowCachedWorkerPolling() throws Exception {
        var discoverySource = mock(org.flexlb.discovery.ServiceDiscovery.class);
        var metadata = mock(org.flexlb.config.ModelMetaConfig.class);
        var endpoint = new org.flexlb.dao.route.Endpoint();
        endpoint.setAddress("workers");
        endpoint.setProtocol("http");
        when(metadata.endpointsWithGroup("model", status.getRole()))
                .thenReturn(List.of(org.apache.commons.lang3.tuple.Pair.of("", endpoint)));
        when(discoverySource.getHosts("workers")).thenThrow(new IllegalStateException("discovery unavailable"));
        WorkerAddressService addresses = new WorkerAddressService(reporter, metadata, discoverySource, config);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response(true, 1L)));
        when(grpc.getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(new CompletableFuture<>());
        try (var rpcExecutor = Executors.newSingleThreadExecutor();
             var cacheExecutor = Executors.newSingleThreadExecutor();
             var applications = Executors.newSingleThreadExecutor();
             var discovery = Executors.newSingleThreadExecutor()) {
            EngineSyncRunner runner = new EngineSyncRunner("model", registry, addresses, rpcExecutor,
                    reporter, grpc, status.getRole(), cache, interval, 500L, 0L, 20L,
                    false, cacheExecutor, applications);
            status.lock.lock();
            try {
                discovery.submit(runner).get(2, TimeUnit.SECONDS);
                org.flexlb.sync.runner.RunnerTestSupport.awaitCondition(() -> status.pollHealth().reportedAlive());
                assertEquals(-1L, status.appliedStatusCursor().statusVersion());
                new WorkerRetirementCleaner(cache, registry).retireUnavailableWorkers();
                assertNull(status.retirementReason());
                assertSame(status, registry.statusSnapshot(status.getRole()).get(status.getIpPort()));
                awaitCondition(() -> {
                    try (var lease = status.tryBeginStatusPoll()) { return lease != null; }
                });
                doReturn(List.of(new WorkerHost("127.0.0.1", 8080))).when(discoverySource).getHosts("workers");
                discovery.submit(runner).get(2, TimeUnit.SECONDS);
                verify(grpc, timeout(2000).times(2))
                        .getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any());
            } finally { status.lock.unlock(); }
        } finally { addresses.destroy(); }
    }

    @Test
    void blockedCacheWriteDoesNotBlockHeartbeatAndCannotCrossReplacement() throws Exception {
        CountDownLatch updating = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        when(cache.updateEngineBlockCache(status)).thenAnswer(invocation -> {
            assertFalse(status.lock.isHeldByCurrentThread());
            updating.countDown();
            assertTrue(release.await(5, TimeUnit.SECONDS));
            return WorkerCacheUpdateResult.builder().success(true).build();
        });
        when(grpc.getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(cacheResponse(1L, 100L)));
        try (var worker = Executors.newSingleThreadExecutor()) {
            var writing = worker.submit(() -> cachePoll(Runnable::run));
            try {
                assertTrue(updating.await(2, TimeUnit.SECONDS));
                poll(true, 1L, Runnable::run);
                assertEquals(1L, status.appliedStatusCursor().statusVersion());
                status.markRemovedFromDiscovery();
                var retirement = registry.beginRequestedRetirement(status.getRole(), status.getIpPort(), status);
                assertNotNull(retirement);
                retirement.completeAsync(cache, org.slf4j.LoggerFactory.getLogger(getClass()));
                WorkerStatus replacement = RunnerTestSupport.discovered(
                        status.getRole(), null, "127.0.0.1", 8080, 8081, null);
                assertSame(status, registry.currentOrDiscover(status.getRole(), status.getIpPort(), () -> replacement));
                release.countDown();
                writing.get(2, TimeUnit.SECONDS);
                org.flexlb.sync.runner.RunnerTestSupport.awaitCondition(() -> registry.statusSnapshot(status.getRole()).isEmpty());
                assertSame(replacement, registry.currentOrDiscover(status.getRole(), status.getIpPort(), () -> replacement));
                var order = inOrder(cache);
                order.verify(cache).updateEngineBlockCache(status);
                order.verify(cache).removeEngineBlockCache(status.getIpPort());
                assertFalse(status.cacheIndexSnapshot().indexInitialized(), "a retired writer cannot advance its cursor");
            } finally { release.countDown(); }
        }
    }

    @Test
    void sameVersionReusesKeysButRefreshesCapacityWithoutReindexing() {
        when(cache.updateEngineBlockCache(status)).thenReturn(WorkerCacheUpdateResult.builder().success(true).build());
        when(grpc.getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(cacheResponse(1L, 100L)),
                        CompletableFuture.completedFuture(cacheResponse(1L, 200L)));
        cachePoll(Runnable::run);
        var keys = status.getCacheStatus().getCachedKeys();
        cachePoll(Runnable::run);
        assertSame(keys, status.getCacheStatus().getCachedKeys());
        assertEquals(200L, status.getCacheStatus().getAvailableKvCache());
        verify(cache, times(1)).updateEngineBlockCache(status);
    }

    private void poll(boolean alive, long version, Executor executor) {
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response(alive, version)));
        var lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        new GrpcWorkerStatusRunner("model", status, lease, registry, reporter, grpc, 500L, cache, executor).run();
    }

    private EngineRpcService.WorkerStatusPB response(boolean alive, long version) {
        return EngineRpcService.WorkerStatusPB.newBuilder().setRole(status.getRole().getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL).setAlive(alive)
                .setStatusVersion(version).setDpSize(1).setAvailableKvCache(1000L).setTotalKvCache(2000L).build();
    }

    private EngineRpcService.CacheStatusPB cacheResponse(long version, long available) {
        return EngineRpcService.CacheStatusPB.newBuilder().setVersion(version).setBlockSize(128L)
                .setAvailableKvCache(available).setTotalKvCache(2000L).putCacheKeys(42L, true).build();
    }

    private void cachePoll(Executor executor) {
        new GrpcCacheStatusCheckRunner("model", status, status.tryBeginCachePoll(), registry,
                reporter, grpc, cache, interval, 500L, false, executor).run();
    }
}
