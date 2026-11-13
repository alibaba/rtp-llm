package org.flexlb.worker.endpoint;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.rpc.WorkerStatusRpcClient;
import org.flexlb.worker.rpc.protocol.EngineRpcService;
import org.flexlb.worker.status.GrpcWorkerStatusRunner;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.LoggerFactory;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.worker.endpoint.EndpointTestSupport.decodeRoutingSnapshot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Verifies registry shutdown rejects new publications and waits for admitted generation cleanup.
 * Late worker feedback must not restore closed resource records.
 */
class EndpointRegistryCloseTest {
    @Test
    void retirementScanSkipsBusyAdmissionAndClosesAnotherWorker() throws Exception {
        var registry = registry();
        var first = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8970, 8971);
        var second = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.2", 8970, 8971);
        var endpoint =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.VIT, first.getLogicalIpPort(), first);
        EndpointTestSupport.publishEndpoint(
                registry, RoleType.VIT, second.getLogicalIpPort(), second);
        first.markRemovedFromDiscovery();
        second.markRemovedFromDiscovery();
        endpoint.stateLock().lock();
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            try {
                var skipped =
                        executor.submit(
                                () ->
                                        registry.closeUnavailableWorker(
                                                RoleType.VIT, first.getLogicalIpPort(), first));
                assertNull(skipped.get(1, TimeUnit.SECONDS));
                executor.submit(registry::closeUnavailableWorkers).get(1, TimeUnit.SECONDS);
                org.flexlb.worker.status.RunnerTestSupport.awaitCondition(
                        () -> !registry.statusSnapshot(RoleType.VIT).containsValue(second));
                assertTrue(first.isActiveGeneration());
                assertSame(endpoint, registry.getEndpoint(RoleType.VIT, first.getLogicalIpPort()));
            } finally {
                endpoint.stateLock().unlock();
            }
        }
        registry.closeUnavailableWorkers();
        org.flexlb.worker.status.RunnerTestSupport.awaitCondition(
                () -> registry.statusSnapshot(RoleType.VIT).isEmpty());
        registry.close();
    }

    @Test
    void failedDetachedEndpointCleanupIsReportedByEveryRegistryCloser() throws Exception {
        var registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8960, 8961);
        String address = status.getLogicalIpPort();
        registry.currentOrDiscover(RoleType.VIT, address, () -> status);
        var endpoint =
                org.mockito.Mockito.spy(
                        EndpointTestSupport.publishEndpoint(
                                registry, RoleType.VIT, address, status));
        endpointMaps(registry).get(RoleType.VIT).put(address, endpoint);
        var failure = new IllegalStateException("endpoint cleanup failed");
        org.mockito.Mockito.doThrow(failure).when(endpoint).closeEndpoint();
        var cleanup = registry.closeWorker(RoleType.VIT, address, status);
        var reported =
                assertThrows(
                        ExecutionException.class,
                        () -> cleanup.toCompletableFuture().get(2, TimeUnit.SECONDS));
        assertSame(failure, reported.getCause());
        assertSame(status, registry.statusSnapshot(RoleType.VIT).get(address));
        assertThrows(IllegalStateException.class, registry::close);
        assertThrows(IllegalStateException.class, registry::close);
    }

    @Test
    void cacheCleanupDependencyCanResolveTheRegistryAsWorkerStatusProvider() throws Exception {
        var cache = mock(CacheAwareService.class);
        try (var context =
                new org.springframework.context.annotation.AnnotationConfigApplicationContext()) {
            context.getBeanFactory().registerSingleton("configService", mock(ConfigService.class));
            context.registerBean(org.flexlb.balance.schedule.RequestRepository.class);
            context.registerBean(
                    DeliveryMetricsReporter.class, () -> mock(DeliveryMetricsReporter.class));
            context.registerBean(CapacityNotifier.class);
            context.registerBean(EndpointRegistry.class);
            context.registerBean(
                    CacheAwareService.class,
                    () -> {
                        assertSame(
                                context.getBean(EndpointRegistry.class),
                                context.getBean(
                                        org.flexlb.worker.status.WorkerStatusProvider.class));
                        return cache;
                    });
            context.refresh();
            var registry = context.getBean(EndpointRegistry.class);
            var status =
                    EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8950, 8951);
            var address = status.getLogicalIpPort();
            registry.currentOrDiscover(RoleType.PREFILL, address, () -> status);
            registry.closeWorker(RoleType.PREFILL, address, status)
                    .toCompletableFuture()
                    .get(2, TimeUnit.SECONDS);
            org.mockito.Mockito.verify(cache).removeEngineBlockCache(address);
            registry.close();
        }
    }

    @Test
    void retiringWorkerWaitingForAdmissionLockDoesNotBlockAnotherWorker() throws Exception {
        var registry = registry();
        var first = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8900, 8901);
        var second = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.2", 8900, 8901);
        registry.currentOrDiscover(RoleType.VIT, first.getLogicalIpPort(), () -> first);
        var endpoint =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.VIT, first.getLogicalIpPort(), first);
        var started = new CountDownLatch(1);
        endpoint.stateLock().lock();
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var closing =
                    executor.submit(
                            () -> {
                                first.lock.lock();
                                try {
                                    started.countDown();
                                    return registry.closeWorker(
                                            RoleType.VIT, first.getLogicalIpPort(), first);
                                } finally {
                                    first.lock.unlock();
                                }
                            });
            try {
                assertTrue(started.await(2, TimeUnit.SECONDS));
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
                while (!endpoint.stateLock().hasQueuedThreads() && System.nanoTime() < deadline) {
                    Thread.onSpinWait();
                }
                assertTrue(endpoint.stateLock().hasQueuedThreads());
                var other =
                        executor.submit(
                                () -> {
                                    registry.currentOrDiscover(
                                            RoleType.VIT, second.getLogicalIpPort(), () -> second);
                                    return EndpointTestSupport.publishEndpoint(
                                            registry,
                                            RoleType.VIT,
                                            second.getLogicalIpPort(),
                                            second);
                                });
                assertNotNull(other.get(1, TimeUnit.SECONDS));
            } finally {
                endpoint.stateLock().unlock();
                closing.get(2, TimeUnit.SECONDS).toCompletableFuture().get(2, TimeUnit.SECONDS);
                registry.close();
            }
        }
    }

    @Test
    void asynchronousRetirementsDoNotOccupyThreadsNeededToDrainTheirWorkers() throws Exception {
        CapacityNotifier availability = mock(CapacityNotifier.class);
        EndpointRegistry registry = registry(availability);
        CacheAwareService cache = mock(CacheAwareService.class);
        org.springframework.test.util.ReflectionTestUtils.setField(
                registry, "cacheAwareService", cache);
        var selectedWorkers = new java.util.ArrayList<WorkerEndpoint>();
        var statuses = new java.util.ArrayList<org.flexlb.worker.status.WorkerStatus>();
        CountDownLatch retirementStarted = new CountDownLatch(4);
        try {
            // Fill the shared cleanup pool while each endpoint still has an accepted handoff.
            for (int i = 0; i < 4; i++) {
                var status =
                        EndpointTestSupport.workerStatus(
                                RoleType.DECODE, "127.0.0.1", 8600 + i, 8700 + i);
                registry.currentOrDiscover(
                        RoleType.DECODE, status.getLogicalIpPort(), () -> status);
                var endpoint =
                        EndpointTestSupport.publishEndpoint(
                                registry, RoleType.DECODE, status.getLogicalIpPort(), status);
                selectedWorkers.add(endpoint);
                statuses.add(status);
            }
            doAnswer(
                            call -> {
                                retirementStarted.countDown();
                                return null;
                            })
                    .when(availability)
                    .notifyChanged(any());
            for (var status : statuses) {
                status.markRemovedFromDiscovery();
                var cleanup =
                        registry.closeUnavailableWorker(
                                RoleType.DECODE, status.getLogicalIpPort(), status);
                assertNotNull(cleanup);
            }
            assertTrue(retirementStarted.await(5, TimeUnit.SECONDS));
            selectedWorkers.forEach(org.flexlb.worker.endpoint.EndpointTestSupport::closeAndWait);
            assertTimeoutPreemptively(
                    java.time.Duration.ofSeconds(5),
                    () -> {
                        while (pendingWorkerCleanups(registry) != 0) {
                            Thread.sleep(1L);
                        }
                    });
            assertEquals(0, registry.discoveredCount());
        } finally {
            selectedWorkers.forEach(org.flexlb.worker.endpoint.EndpointTestSupport::closeAndWait);
            registry.close();
        }
    }

    @Test
    void closeWaitsForAcceptedPublicationAndRejectsNewPublications() throws Exception {
        CapacityNotifier availability = mock(CapacityNotifier.class);
        EndpointRegistry registry = registry(availability);
        CountDownLatch publicationEntered = new CountDownLatch(1);
        CountDownLatch finishPublication = new CountDownLatch(1);
        doAnswer(
                        call -> {
                            publicationEntered.countDown();
                            assertTrue(finishPublication.await(5, TimeUnit.SECONDS));
                            return null;
                        })
                .when(availability)
                .notifyChanged(any());
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8300, 8301);
        try (var executor = Executors.newFixedThreadPool(3)) {
            Future<WorkerEndpoint> publication =
                    executor.submit(
                            () ->
                                    EndpointTestSupport.publishEndpoint(
                                            registry,
                                            RoleType.DECODE,
                                            status.getLogicalIpPort(),
                                            status));
            try {
                assertTrue(publicationEntered.await(2, TimeUnit.SECONDS));
                assertFalse(decodeRoutingSnapshot(registry, null).isEmpty());
                Future<?> owner =
                        executor.submit(
                                () -> {
                                    Thread.currentThread().interrupt();
                                    try {
                                        registry.close();
                                    } finally {
                                        assertTrue(
                                                Thread.interrupted(),
                                                "publication drain must preserve interruption");
                                    }
                                });
                assertTimeoutPreemptively(
                        java.time.Duration.ofSeconds(2),
                        () -> {
                            while (!decodeRoutingSnapshot(registry, null).isEmpty()) {
                                Thread.sleep(1L);
                            }
                        });
                Future<?> waiter = executor.submit(registry::close);
                var late = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8400, 8401);
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                EndpointTestSupport.publishEndpoint(
                                        registry, RoleType.VIT, late.getLogicalIpPort(), late));
                assertThrows(TimeoutException.class, () -> owner.get(50, TimeUnit.MILLISECONDS));
                assertThrows(TimeoutException.class, () -> waiter.get(50, TimeUnit.MILLISECONDS));
                finishPublication.countDown();
                WorkerEndpoint published = publication.get(2, TimeUnit.SECONDS);
                owner.get(2, TimeUnit.SECONDS);
                waiter.get(2, TimeUnit.SECONDS);
                assertTrue(published.hasStoppedAcceptingRequests());
                assertTrue(decodeRoutingSnapshot(registry, null).isEmpty());
            } finally {
                finishPublication.countDown();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void concurrentClosersShareSettlementAndPreserveInterruption(boolean fail) throws Exception {
        EndpointRegistry registry = registry();
        CountDownLatch cleanupStarted = new CountDownLatch(1);
        CountDownLatch finishCleanup = new CountDownLatch(1);
        CountDownLatch waiterStarted = new CountDownLatch(1);
        AtomicInteger cleanups = new AtomicInteger();
        IllegalStateException failure = new IllegalStateException("endpoint cleanup failed");
        WorkerEndpoint endpoint =
                new WorkerEndpoint(
                        EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8100, 8101)) {
                    @Override
                    protected void closeEndpoint() {
                        cleanups.incrementAndGet();
                        cleanupStarted.countDown();
                        try {
                            assertTrue(finishCleanup.await(5, TimeUnit.SECONDS));
                        } catch (InterruptedException interruption) {
                            throw new AssertionError(interruption);
                        }
                        if (fail) {
                            throw failure;
                        }
                    }
                };
        // Inject the same exact generation twice to exercise shutdown's identity deduplication.
        endpointMaps(registry).get(RoleType.VIT).put("first", endpoint);
        endpointMaps(registry).get(RoleType.DECODE).put("alias", endpoint);
        try (var executor = Executors.newFixedThreadPool(2)) {
            Future<?> owner = executor.submit(registry::close);
            try {
                assertTrue(cleanupStarted.await(2, TimeUnit.SECONDS));
                Future<?> waiter =
                        executor.submit(
                                () -> {
                                    Thread.currentThread().interrupt();
                                    waiterStarted.countDown();
                                    try {
                                        registry.close();
                                    } finally {
                                        assertTrue(
                                                Thread.interrupted(),
                                                "close must preserve interruption even on failure");
                                    }
                                });
                assertTrue(waiterStarted.await(2, TimeUnit.SECONDS));
                assertThrows(TimeoutException.class, () -> waiter.get(50, TimeUnit.MILLISECONDS));
                finishCleanup.countDown();
                if (fail) {
                    assertSame(
                            failure,
                            assertThrows(
                                            ExecutionException.class,
                                            () -> owner.get(2, TimeUnit.SECONDS))
                                    .getCause());
                    assertSame(
                            failure,
                            assertThrows(
                                            ExecutionException.class,
                                            () -> waiter.get(2, TimeUnit.SECONDS))
                                    .getCause());
                    assertSame(failure, assertThrows(IllegalStateException.class, registry::close));
                } else {
                    owner.get(2, TimeUnit.SECONDS);
                    waiter.get(2, TimeUnit.SECONDS);
                    registry.close();
                }
                assertEquals(1, cleanups.get());
                assertTrue(endpoint.hasStoppedAcceptingRequests());
            } finally {
                finishCleanup.countDown();
            }
        }
    }

    @Test
    void closeWaitsForAlreadyDetachedGeneration() throws Exception {
        var registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8200, 8201);
        String address = status.getLogicalIpPort();
        registry.currentOrDiscover(RoleType.VIT, address, () -> status);
        var endpoint = EndpointTestSupport.publishEndpoint(registry, RoleType.VIT, address, status);
        status.lock.lock();
        var cleanup = registry.closeWorker(RoleType.VIT, address, status);
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var closing = executor.submit(registry::close);
            try {
                assertThrows(TimeoutException.class, () -> closing.get(50, TimeUnit.MILLISECONDS));
            } finally {
                status.lock.unlock();
            }
            cleanup.toCompletableFuture().get(2, TimeUnit.SECONDS);
            closing.get(2, TimeUnit.SECONDS);
            assertTrue(endpoint.hasStoppedAcceptingRequests());
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void closeWaitsForExactCacheFinalizerEvenWithoutPublishedEndpoint(boolean published)
            throws Exception {
        var registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8500, 8501);
        String address = status.getLogicalIpPort();
        registry.currentOrDiscover(RoleType.PREFILL, address, () -> status);
        if (published)
            EndpointTestSupport.publishEndpoint(registry, RoleType.PREFILL, address, status);
        var cache = mock(CacheAwareService.class);
        ReflectionTestUtils.setField(registry, "cacheAwareService", cache);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        doAnswer(
                        call -> {
                            entered.countDown();
                            assertTrue(release.await(5, TimeUnit.SECONDS));
                            return null;
                        })
                .when(cache)
                .removeEngineBlockCache(address);
        var cleanup = registry.closeWorker(RoleType.PREFILL, address, status);
        assertSame(cleanup, registry.closeWorker(RoleType.PREFILL, address, status));
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            try {
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                var other =
                        executor.submit(
                                () -> {
                                    var second =
                                            EndpointTestSupport.workerStatus(
                                                    RoleType.PREFILL, "127.0.0.2", 8500, 8501);
                                    String secondAddress = second.getLogicalIpPort();
                                    registry.currentOrDiscover(
                                            RoleType.PREFILL, secondAddress, () -> second);
                                    var endpoint =
                                            EndpointTestSupport.publishEndpoint(
                                                    registry,
                                                    RoleType.PREFILL,
                                                    secondAddress,
                                                    second);
                                    assertSame(
                                            endpoint,
                                            registry.getEndpoint(RoleType.PREFILL, secondAddress));
                                    return registry.closeWorker(
                                            RoleType.PREFILL, secondAddress, second);
                                });
                other.get(1, TimeUnit.SECONDS).toCompletableFuture().get(1, TimeUnit.SECONDS);
                var closing = executor.submit(registry::close);
                assertThrows(TimeoutException.class, () -> closing.get(50, TimeUnit.MILLISECONDS));
                assertSame(status, registry.statusSnapshot(RoleType.PREFILL).get(address));
                release.countDown();
                cleanup.toCompletableFuture().get(2, TimeUnit.SECONDS);
                closing.get(2, TimeUnit.SECONDS);
                assertTrue(registry.statusSnapshot(RoleType.PREFILL).isEmpty());
            } finally {
                release.countDown();
            }
        }
    }

    @Test
    void lateRpcFrameCannotRepopulateRetiredDecodeLedger() {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8600, 8601);
        registry.currentOrDiscover(RoleType.DECODE, status.getLogicalIpPort(), () -> status);
        DecodeEndpoint endpoint =
                (DecodeEndpoint)
                        EndpointTestSupport.publishEndpoint(
                                registry, RoleType.DECODE, status.getLogicalIpPort(), status);
        long committedVersion = status.appliedStatusCursor().statusVersion();
        CompletableFuture<EngineRpcService.WorkerStatusPB> response = new CompletableFuture<>();
        WorkerStatusRpcClient rpc = mock(WorkerStatusRpcClient.class);
        when(rpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(response);
        new GrpcWorkerStatusRunner(
                        "test-model",
                        status,
                        status.tryBeginStatusPoll(),
                        registry,
                        mock(ServiceMetricsReporter.class),
                        rpc,
                        5000L,
                        mock(CacheAwareService.class),
                        Runnable::run)
                .run();
        registry.close();
        response.complete(runningStatus(committedVersion + 1L));
        assertTrue(
                endpoint.resourceSnapshot().requests().isEmpty(),
                "an already-drained Decode generation must not acquire late Engine ownership");
        assertEquals(committedVersion, status.appliedStatusCursor().statusVersion());
        assertFalse(status.isActiveGeneration());
        assertFalse(registry.isCurrentStatus(RoleType.DECODE, status.getLogicalIpPort(), status));
        assertThrows(
                IllegalStateException.class,
                () ->
                        registry.currentOrDiscover(
                                RoleType.DECODE, status.getLogicalIpPort(), () -> status));
    }

    @Test
    void closeWaitsForAlreadyAdmittedGenerationTransactionBeforeDraining() throws Exception {
        EndpointRegistry registry = registry();
        var requests =
                (org.flexlb.balance.schedule.RequestRepository)
                        ReflectionTestUtils.getField(registry, "requests");
        when(requests.findActive("123")).thenReturn(null);
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8700, 8701);
        registry.currentOrDiscover(RoleType.DECODE, status.getLogicalIpPort(), () -> status);
        DecodeEndpoint endpoint =
                (DecodeEndpoint)
                        EndpointTestSupport.publishEndpoint(
                                registry, RoleType.DECODE, status.getLogicalIpPort(), status);
        assertNotNull(EndpointTestSupport.registerDecode(endpoint, 123L, 1L, 1L, 50));
        status.lock.lock();
        try (var executor = Executors.newSingleThreadExecutor()) {
            Future<?> closing = executor.submit(registry::close);
            try {
                assertTimeoutPreemptively(
                        java.time.Duration.ofSeconds(2),
                        () -> {
                            while (registry.isCurrentStatus(
                                    RoleType.DECODE, status.getLogicalIpPort(), status)) {
                                Thread.onSpinWait();
                            }
                        });
                assertThrows(TimeoutException.class, () -> closing.get(50, TimeUnit.MILLISECONDS));
                // This transaction passed its generation check before close set
                // the registry gate. Its resource commit must precede drain.
                var observation =
                        WorkerStatusRpcClient.convertToStatusObservation(
                                status,
                                runningStatus(status.appliedStatusCursor().statusVersion() + 1L));
                endpoint.applyWorkerStatus(status, status.prepareStatus(observation));
                assertTrue(
                        endpoint.resourceSnapshot().requests().containsKey(String.valueOf(123L)));
            } finally {
                status.lock.unlock();
            }
            closing.get(2, TimeUnit.SECONDS);
            assertTrue(endpoint.resourceSnapshot().requests().isEmpty());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void waitTimeoutRetainsExactGenerationUntilRealCleanupCompletes(boolean closeWhileDraining)
            throws Exception {
        EndpointRegistry registry = registry();
        ReflectionTestUtils.setField(registry, "closeTimeoutMs", 25L);
        var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 9000, 9001);
        String address = status.getLogicalIpPort();
        registry.currentOrDiscover(RoleType.PREFILL, address, () -> status);
        WorkerEndpoint endpoint =
                EndpointTestSupport.publishEndpoint(registry, RoleType.PREFILL, address, status);
        var finishCleanup = new CountDownLatch(1);
        WorkerEndpoint slowEndpoint = org.mockito.Mockito.spy(endpoint);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            assertTrue(finishCleanup.await(5, TimeUnit.SECONDS));
                            return call.callRealMethod();
                        })
                .when(slowEndpoint)
                .closeEndpoint();
        endpointMaps(registry).get(RoleType.PREFILL).put(address, slowEndpoint);
        endpoint = slowEndpoint;
        CacheAwareService cache = mock(CacheAwareService.class);
        org.springframework.test.util.ReflectionTestUtils.setField(
                registry, "cacheAwareService", cache);
        var callbackOutsideMonitor = new java.util.concurrent.atomic.AtomicBoolean();
        CountDownLatch actualRetirement = new CountDownLatch(1);
        Object lifecycle = ReflectionTestUtils.getField(endpoint, "closeCompleted");
        {
            var selectedEndpoint = registry.getEndpoint(RoleType.PREFILL, address);

            assertNotNull(selectedEndpoint);
            java.util.concurrent.CompletionStage<Void> cleanup;
            status.lock.lock();
            try {
                cleanup = registry.closeWorker(RoleType.PREFILL, address, status);
            } finally {
                status.lock.unlock();
            }
            endpoint.closeAsync();
            assertSame(status, registry.statusSnapshot(RoleType.PREFILL).get(address));
            assertFalse(status.isActiveGeneration());
            assertSame(
                    status,
                    registry.currentOrDiscover(
                            RoleType.PREFILL,
                            address,
                            () -> {
                                throw new AssertionError(
                                        "replacement must stay fenced until real cleanup");
                            }));
            org.mockito.Mockito.verify(cache, org.mockito.Mockito.never())
                    .removeEngineBlockCache(address);
            assertEquals(1, pendingWorkerCleanups(registry));
            if (closeWhileDraining) {
                assertTimeoutPreemptively(
                        java.time.Duration.ofSeconds(2),
                        () -> {
                            IllegalStateException failure =
                                    assertThrows(IllegalStateException.class, registry::close);
                            assertTrue(failure.getMessage().contains("timed out"));
                        });
                assertSame(status, registry.statusSnapshot(RoleType.PREFILL).get(address));
                assertEquals(1, pendingWorkerCleanups(registry));
            }
            endpoint.closeAsync()
                    .thenRun(
                            () -> {
                                callbackOutsideMonitor.set(!Thread.holdsLock(lifecycle));
                                actualRetirement.countDown();
                            });
        }
        finishCleanup.countDown();
        assertTrue(actualRetirement.await(2, TimeUnit.SECONDS));
        assertTrue(
                callbackOutsideMonitor.get(),
                "completion callbacks must not add L -> S lock edges");
        assertTimeoutPreemptively(
                java.time.Duration.ofSeconds(2),
                () -> {
                    while (!registry.statusSnapshot(RoleType.PREFILL).isEmpty()
                            || pendingWorkerCleanups(registry) != 0) {
                        Thread.sleep(1L);
                    }
                });
        org.mockito.Mockito.verify(cache, org.mockito.Mockito.times(1))
                .removeEngineBlockCache(address);
        if (!closeWhileDraining) {
            registry.close();
        } else {
            assertThrows(
                    IllegalStateException.class,
                    registry::close,
                    "all closers must retain the original bounded shutdown failure");
        }
    }

    @Test
    void discoveryRemovalLoggingCannotLoseRetirementCapability() {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 9100, 9101);
        registry.currentOrDiscover(RoleType.DECODE, status.getLogicalIpPort(), () -> status);
        WorkerEndpoint endpoint =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.DECODE, status.getLogicalIpPort(), status);
        var loggingFailed = new java.util.concurrent.atomic.AtomicBoolean();
        ch.qos.logback.classic.Logger syncLogger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(EndpointRegistry.class);
        var appender =
                new ch.qos.logback.core.AppenderBase<ch.qos.logback.classic.spi.ILoggingEvent>() {
                    @Override
                    protected void append(ch.qos.logback.classic.spi.ILoggingEvent event) {
                        if (event.getMessage().startsWith("Retiring unavailable worker")
                                && status.getIpPort().equals(event.getArgumentArray()[0])) {
                            loggingFailed.set(true);
                            throw new AssertionError("logging boundary failed after exact detach");
                        }
                    }
                };
        appender.start();
        syncLogger.addAppender(appender);
        try {
            status.markRemovedFromDiscovery();
            var cleaner =
                    org.flexlb.worker.status.WorkerSyncTestSupport.cleanups(
                            mock(CacheAwareService.class), registry);
            assertThrows(AssertionError.class, cleaner::closeUnavailableWorkers);
            org.flexlb.worker.status.RunnerTestSupport.awaitCondition(
                    () -> pendingWorkerCleanups(registry) == 0);
            assertTrue(loggingFailed.get());
            assertTrue(endpoint.hasStoppedAcceptingRequests());
            assertTrue(registry.statusSnapshot(RoleType.DECODE).isEmpty());
            assertEquals(0, pendingWorkerCleanups(registry));
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), registry::close);
        } finally {
            syncLogger.detachAppender(appender);
            appender.stop();
        }
    }

    @Test
    void retirementNotificationFailureCannotHideItsCapabilityOrSkipDrain() {
        CapacityNotifier availability = mock(CapacityNotifier.class);
        EndpointRegistry registry = registry(availability);
        var status = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8900, 8901);
        registry.currentOrDiscover(RoleType.VIT, status.getLogicalIpPort(), () -> status);
        WorkerEndpoint endpoint =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.VIT, status.getLogicalIpPort(), status);
        var notifiedOutsideLock = new java.util.concurrent.atomic.AtomicBoolean();
        doAnswer(
                        call -> {
                            notifiedOutsideLock.set(!status.lock.isHeldByCurrentThread());
                            throw new IllegalStateException("notification unavailable");
                        })
                .when(availability)
                .notifyChanged(any());
        java.util.concurrent.CompletionStage<Void> cleanup;
        status.lock.lock();
        try {
            cleanup = registry.closeWorker(RoleType.VIT, status.getLogicalIpPort(), status);
        } finally {
            status.lock.unlock();
        }
        assertFalse(
                notifiedOutsideLock.get(),
                "detach only returns ownership; it must not publish inline");
        EndpointTestSupport.awaitWorkerCleanup(cleanup);
        assertTrue(notifiedOutsideLock.get());
        assertTrue(endpoint.hasStoppedAcceptingRequests());
        assertTrue(registry.statusSnapshot(RoleType.VIT).isEmpty());
        assertTimeoutPreemptively(
                java.time.Duration.ofSeconds(2),
                registry::close,
                "the failed notification must not strand the detached retirement barrier");
    }

    @Test
    void discoveryBuiltOutsideCloseGateCannotPublishAfterClose() throws Exception {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8800, 8801);
        CountDownLatch factoryEntered = new CountDownLatch(1);
        CountDownLatch finishFactory = new CountDownLatch(1);
        try (var executor = Executors.newSingleThreadExecutor()) {
            Future<?> discovery =
                    executor.submit(
                            () ->
                                    registry.currentOrDiscover(
                                            RoleType.VIT,
                                            status.getLogicalIpPort(),
                                            () -> {
                                                factoryEntered.countDown();
                                                try {
                                                    assertTrue(
                                                            finishFactory.await(
                                                                    5, TimeUnit.SECONDS));
                                                } catch (InterruptedException interrupted) {
                                                    throw new AssertionError(interrupted);
                                                }
                                                return status;
                                            }));
            try {
                assertTrue(factoryEntered.await(2, TimeUnit.SECONDS));
                registry.close();
                finishFactory.countDown();
                ExecutionException rejected =
                        assertThrows(
                                ExecutionException.class, () -> discovery.get(2, TimeUnit.SECONDS));
                assertTrue(rejected.getCause() instanceof IllegalStateException);
                assertEquals(0, registry.discoveredCount());
            } finally {
                finishFactory.countDown();
            }
        }
    }

    private static EngineRpcService.WorkerStatusPB runningStatus(long version) {
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.DECODE.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setAlive(true)
                .setStatusVersion(version)
                .addRunningTaskInfo(
                        EngineRpcService.TaskInfoPB.newBuilder()
                                .setRequestId(123L)
                                .setInputLength(64)
                                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING))
                .build();
    }

    private static int pendingWorkerCleanups(EndpointRegistry registry) {
        Object gate = ReflectionTestUtils.getField(registry, "lifecycleGate");
        synchronized (gate) {
            return (int) ReflectionTestUtils.getField(registry, "pendingWorkerCleanups");
        }
    }

    private static EndpointRegistry registry() {
        return registry(new CapacityNotifier());
    }

    private static EndpointRegistry registry(CapacityNotifier availability) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.schedule.SchedulingTestConfig.newConfig());
        return new EndpointRegistry(
                config,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                        mock(AbstractRequestScheduler.class)),
                mock(DeliveryMetricsReporter.class),
                availability,
                org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class));
    }

    @SuppressWarnings("unchecked")
    private static Map<RoleType, Map<String, WorkerEndpoint>> endpointMaps(
            EndpointRegistry registry) {
        return (Map<RoleType, Map<String, WorkerEndpoint>>)
                ReflectionTestUtils.getField(registry, "endpointsByRole");
    }
}
