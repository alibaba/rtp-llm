package org.flexlb.balance.endpoint;

import static org.flexlb.balance.endpoint.EndpointTestSupport.decodeRoutingSnapshot;

import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.GrpcWorkerStatusRunner;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;

class EndpointRegistryCloseTest {
    @Test
    void closeWaitsForAcceptedPublicationAndRejectsNewPublications() throws Exception {
        PlacementAvailability availability = mock(PlacementAvailability.class);
        EndpointRegistry registry = registry(availability);
        CountDownLatch publicationEntered = new CountDownLatch(1);
        CountDownLatch finishPublication = new CountDownLatch(1);
        doAnswer(call -> {
            publicationEntered.countDown();
            assertTrue(finishPublication.await(5, TimeUnit.SECONDS));
            return null;
        }).when(availability).changed(any(), any(), any());
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8300, 8301);
        try (var executor = Executors.newFixedThreadPool(3)) {
            Future<WorkerEndpoint> publication = executor.submit(() -> EndpointTestSupport.publishEndpoint(
                    registry, RoleType.DECODE, status.getIpPort(), status));
            try {
                assertTrue(publicationEntered.await(2, TimeUnit.SECONDS));
                assertFalse(decodeRoutingSnapshot(registry, null).isEmpty());
                Future<?> owner = executor.submit(() -> {
                    Thread.currentThread().interrupt();
                    try {
                        registry.close();
                    } finally {
                        assertTrue(Thread.interrupted(), "publication drain must preserve interruption");
                    }
                });
                assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), () -> {
                    while (!decodeRoutingSnapshot(registry, null).isEmpty()) { Thread.sleep(1L); }
                });
                Future<?> waiter = executor.submit(registry::close);
                var late = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8400, 8401);
                assertThrows(IllegalStateException.class, () -> EndpointTestSupport.publishEndpoint(
                        registry, RoleType.VIT, late.getIpPort(), late));
                assertThrows(TimeoutException.class, () -> owner.get(50, TimeUnit.MILLISECONDS));
                assertThrows(TimeoutException.class, () -> waiter.get(50, TimeUnit.MILLISECONDS));
                finishPublication.countDown();
                WorkerEndpoint published = publication.get(2, TimeUnit.SECONDS);
                owner.get(2, TimeUnit.SECONDS);
                waiter.get(2, TimeUnit.SECONDS);
                assertNull(published.tryPinGeneration());
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
        WorkerEndpoint endpoint = new WorkerEndpoint(
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
                if (fail) { throw failure; }
            }
        };
        // Inject the same exact generation twice to exercise shutdown's identity deduplication.
        endpointMaps(registry).get(RoleType.VIT).put("first", endpoint);
        endpointMaps(registry).get(RoleType.DECODE).put("alias", endpoint);
        try (var executor = Executors.newFixedThreadPool(2)) {
            Future<?> owner = executor.submit(registry::close);
            try {
                assertTrue(cleanupStarted.await(2, TimeUnit.SECONDS));
                Future<?> waiter = executor.submit(() -> {
                    Thread.currentThread().interrupt();
                    waiterStarted.countDown();
                    try {
                        registry.close();
                    } finally {
                        assertTrue(Thread.interrupted(), "close must preserve interruption even on failure");
                    }
                });
                assertTrue(waiterStarted.await(2, TimeUnit.SECONDS));
                assertThrows(TimeoutException.class, () -> waiter.get(50, TimeUnit.MILLISECONDS));
                finishCleanup.countDown();
                if (fail) {
                    assertSame(failure, assertThrows(ExecutionException.class,
                            () -> owner.get(2, TimeUnit.SECONDS)).getCause());
                    assertSame(failure, assertThrows(ExecutionException.class,
                            () -> waiter.get(2, TimeUnit.SECONDS)).getCause());
                    assertSame(failure, assertThrows(IllegalStateException.class, registry::close));
                } else {
                    owner.get(2, TimeUnit.SECONDS);
                    waiter.get(2, TimeUnit.SECONDS);
                    registry.close();
                }
                assertEquals(1, cleanups.get());
                assertNull(endpoint.tryPinGeneration());
            } finally {
                finishCleanup.countDown();
            }
        }
    }

    @Test
    void closeWaitsForAlreadyDetachedGeneration() throws Exception {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8200, 8201);
        String address = status.getIpPort();
        registry.currentOrDiscover(RoleType.VIT, address, () -> status);
        EndpointTestSupport.publishEndpoint(registry, RoleType.VIT, address, status);
        EndpointRegistry.Retirement detached;
        status.lock.lock();
        try {
            detached = registry.beginRetirement(RoleType.VIT, address, status);
        } finally {
            status.lock.unlock();
        }
        CountDownLatch closing = new CountDownLatch(1);
        try (var executor = Executors.newSingleThreadExecutor()) {
            Future<?> owner = executor.submit(() -> {
                Thread.currentThread().interrupt();
                closing.countDown();
                try {
                    registry.close();
                } finally {
                    assertTrue(Thread.interrupted());
                }
            });
            try {
                assertTrue(closing.await(2, TimeUnit.SECONDS));
                assertThrows(TimeoutException.class, () -> owner.get(50, TimeUnit.MILLISECONDS));
            } finally {
                detached.complete(mock(CacheAwareService.class), LoggerFactory.getLogger(getClass()));
            }
            owner.get(2, TimeUnit.SECONDS);
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void closeWaitsForExactCacheFinalizerEvenWithoutPublishedEndpoint(boolean published) throws Exception {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8500, 8501);
        String address = status.getIpPort();
        registry.currentOrDiscover(RoleType.PREFILL, address, () -> status);
        if (published) { EndpointTestSupport.publishEndpoint(registry, RoleType.PREFILL, address, status); }
        EndpointRegistry.Retirement retirement;
        status.lock.lock();
        try { retirement = registry.beginRetirement(RoleType.PREFILL, address, status); }
        finally { status.lock.unlock(); }
        CountDownLatch finalizerEntered = new CountDownLatch(1);
        CountDownLatch finishFinalizer = new CountDownLatch(1);
        CacheAwareService cache = mock(CacheAwareService.class);
        doAnswer(call -> {
            finalizerEntered.countDown();
            assertTrue(finishFinalizer.await(5, TimeUnit.SECONDS));
            return null;
        }).when(cache).removeEngineBlockCache(address);
        var closingThread = new java.util.concurrent.atomic.AtomicReference<Thread>();
        try (var executor = Executors.newFixedThreadPool(2)) {
            Future<?> closing = executor.submit(() -> {
                closingThread.set(Thread.currentThread());
                registry.close();
            });
            Future<?> finalizer = null;
            try {
                // Reach the detached barrier before finalization grabs S. This
                // rules out close merely waiting on the finalizer's status lock.
                assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), () -> {
                    while (closingThread.get() == null || java.util.Arrays.stream(closingThread.get().getStackTrace())
                            .noneMatch(frame -> frame.getMethodName().equals("awaitDetachedRetirements"))) {
                        Thread.sleep(1L);
                    }
                });
                finalizer = executor.submit(() -> retirement.complete(cache, LoggerFactory.getLogger(getClass())));
                assertTrue(finalizerEntered.await(2, TimeUnit.SECONDS));
                assertThrows(TimeoutException.class, () -> closing.get(50, TimeUnit.MILLISECONDS),
                        "close must wait through cache cleanup and exact status removal");
                finishFinalizer.countDown();
                finalizer.get(2, TimeUnit.SECONDS);
                closing.get(2, TimeUnit.SECONDS);
                assertTrue(registry.statusSnapshot(RoleType.PREFILL).isEmpty());
            } finally {
                finishFinalizer.countDown();
                if (finalizer == null) { retirement.complete(cache, LoggerFactory.getLogger(getClass())); }
            }
        }
    }

    @Test
    void lateRpcFrameCannotRepopulateRetiredDecodeLedger() {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8600, 8601);
        registry.currentOrDiscover(RoleType.DECODE, status.getIpPort(), () -> status);
        DecodeEndpoint endpoint = (DecodeEndpoint) EndpointTestSupport.publishEndpoint(
                registry, RoleType.DECODE, status.getIpPort(), status);
        long committedVersion = status.appliedStatusCursor().statusVersion();
        CompletableFuture<EngineRpcService.WorkerStatusPB> response = new CompletableFuture<>();
        WorkerStatusRpcClient rpc = mock(WorkerStatusRpcClient.class);
        when(rpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any())).thenReturn(response);
        new GrpcWorkerStatusRunner("test-model", status, status.tryBeginStatusPoll(), registry,
                mock(EngineHealthReporter.class), rpc, 5000L, mock(CacheAwareService.class), Runnable::run).run();
        registry.close();
        response.complete(runningStatus(committedVersion + 1L));
        assertTrue(endpoint.resourceSnapshot().requests().isEmpty(),
                "an already-drained Decode generation must not acquire late Engine ownership");
        assertEquals(committedVersion, status.appliedStatusCursor().statusVersion());
        assertFalse(status.isActiveGeneration());
        assertFalse(registry.isCurrentStatus(RoleType.DECODE, status.getIpPort(), status));
        assertThrows(IllegalStateException.class, () -> registry.currentOrDiscover(
                RoleType.DECODE, status.getIpPort(), () -> status));
    }

    @Test
    void closeWaitsForAlreadyAdmittedGenerationTransactionBeforeDraining() throws Exception {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8700, 8701);
        registry.currentOrDiscover(RoleType.DECODE, status.getIpPort(), () -> status);
        DecodeEndpoint endpoint = (DecodeEndpoint) EndpointTestSupport.publishEndpoint(
                registry, RoleType.DECODE, status.getIpPort(), status);
        status.lock.lock();
        try (var executor = Executors.newSingleThreadExecutor()) {
            Future<?> closing = executor.submit(registry::close);
            try {
                assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), () -> {
                    while (registry.isCurrentStatus(RoleType.DECODE, status.getIpPort(), status)) { Thread.onSpinWait(); }
                });
                assertThrows(TimeoutException.class, () -> closing.get(50, TimeUnit.MILLISECONDS));
                // This transaction passed its generation check before close set
                // the registry gate. Its resource commit must precede drain.
                var observation = EngineStatusConverter.convertToStatusObservation(status,
                        runningStatus(status.appliedStatusCursor().statusVersion() + 1L));
                endpoint.applyPreparedStatus(status, status.prepareNewStatus(observation));
                assertTrue(endpoint.resourceSnapshot().requests().containsKey(123L));
            } finally { status.lock.unlock(); }
            closing.get(2, TimeUnit.SECONDS);
            assertTrue(endpoint.resourceSnapshot().requests().isEmpty());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void waitTimeoutRetainsExactGenerationUntilRealCleanupCompletes(boolean closeWhileDraining) throws Exception {
        EndpointRegistry registry = registry();
        ReflectionTestUtils.setField(registry, "retirementTimeoutMs", 25L);
        var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 9000, 9001);
        String address = status.getIpPort();
        registry.currentOrDiscover(RoleType.PREFILL, address, () -> status);
        WorkerEndpoint endpoint = EndpointTestSupport.publishEndpoint(registry, RoleType.PREFILL, address, status);
        CacheAwareService cache = mock(CacheAwareService.class);
        var callbackOutsideMonitor = new java.util.concurrent.atomic.AtomicBoolean();
        CountDownLatch actualRetirement = new CountDownLatch(1);
        Object lifecycle = ReflectionTestUtils.getField(endpoint, "generationLifecycle");
        try (var pin = registry.capture(RoleType.PREFILL, address)) {
            assertNotNull(pin);
            EndpointRegistry.Retirement retirement;
            status.lock.lock();
            try { retirement = registry.beginRetirement(RoleType.PREFILL, address, status); }
            finally { status.lock.unlock(); }
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(2),
                    () -> retirement.complete(cache, LoggerFactory.getLogger(getClass())));
            assertSame(status, registry.statusSnapshot(RoleType.PREFILL).get(address));
            assertFalse(status.isActiveGeneration());
            assertSame(status, registry.currentOrDiscover(RoleType.PREFILL, address,
                    () -> { throw new AssertionError("replacement must stay fenced until real cleanup"); }));
            org.mockito.Mockito.verify(cache, org.mockito.Mockito.never()).removeEngineBlockCache(address);
            assertEquals(1, pendingRetirements(registry));
            if (closeWhileDraining) {
                assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), () -> {
                    IllegalStateException failure = assertThrows(IllegalStateException.class, registry::close);
                    assertTrue(failure.getMessage().contains("timed out"));
                });
                assertSame(status, registry.statusSnapshot(RoleType.PREFILL).get(address));
                assertEquals(1, pendingRetirements(registry));
            }
            endpoint.onRetired(() -> {
                callbackOutsideMonitor.set(!Thread.holdsLock(lifecycle));
                actualRetirement.countDown();
            });
        }
        assertTrue(actualRetirement.await(2, TimeUnit.SECONDS));
        assertTrue(callbackOutsideMonitor.get(), "completion callbacks must not add L -> S lock edges");
        assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), () -> {
            while (!registry.statusSnapshot(RoleType.PREFILL).isEmpty() || pendingRetirements(registry) != 0) {
                Thread.sleep(1L);
            }
        });
        org.mockito.Mockito.verify(cache, org.mockito.Mockito.times(1)).removeEngineBlockCache(address);
        if (!closeWhileDraining) {
            registry.close();
        } else {
            assertThrows(IllegalStateException.class, registry::close,
                    "all closers must retain the original bounded shutdown failure");
        }
    }

    @Test
    void transportFailureLoggingCannotLoseRetirementCapability() {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 9100, 9101);
        registry.currentOrDiscover(RoleType.DECODE, status.getIpPort(), () -> status);
        WorkerEndpoint endpoint = EndpointTestSupport.publishEndpoint(registry, RoleType.DECODE, status.getIpPort(), status);
        WorkerStatusRpcClient rpc = mock(WorkerStatusRpcClient.class);
        when(rpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(new IllegalStateException("transport unavailable")));
        var loggingFailed = new java.util.concurrent.atomic.AtomicBoolean();
        ch.qos.logback.classic.Logger syncLogger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("syncLogger");
        var appender = new ch.qos.logback.core.AppenderBase<ch.qos.logback.classic.spi.ILoggingEvent>() {
            @Override protected void append(ch.qos.logback.classic.spi.ILoggingEvent event) {
                if (event.getMessage().startsWith("worker {} marked dead")
                        && status.getIpPort().equals(event.getArgumentArray()[0])) {
                    loggingFailed.set(true);
                    throw new AssertionError("logging boundary failed after exact detach");
                }
            }
        };
        appender.start();
        syncLogger.addAppender(appender);
        try {
            for (int failure = 0; failure < 3; failure++) {
                var lease = status.tryBeginStatusPoll();
                assertNotNull(lease);
                new GrpcWorkerStatusRunner("test-model", status, lease, registry,
                        mock(EngineHealthReporter.class), rpc, 5000L, mock(CacheAwareService.class), Runnable::run).run();
            }
            assertTrue(loggingFailed.get());
            assertTrue(endpoint.isGenerationRetiringOrRetired());
            assertTrue(registry.statusSnapshot(RoleType.DECODE).isEmpty());
            assertEquals(0, pendingRetirements(registry));
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), registry::close);
        } finally {
            syncLogger.detachAppender(appender);
            appender.stop();
        }
    }

    @Test
    void retirementNotificationFailureCannotHideItsCapabilityOrSkipDrain() {
        PlacementAvailability availability = mock(PlacementAvailability.class);
        EndpointRegistry registry = registry(availability);
        var status = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8900, 8901);
        registry.currentOrDiscover(RoleType.VIT, status.getIpPort(), () -> status);
        WorkerEndpoint endpoint = EndpointTestSupport.publishEndpoint(registry, RoleType.VIT, status.getIpPort(), status);
        var notifiedOutsideLock = new java.util.concurrent.atomic.AtomicBoolean();
        doAnswer(call -> {
            notifiedOutsideLock.set(!status.lock.isHeldByCurrentThread());
            throw new IllegalStateException("notification unavailable");
        }).when(availability).changed(any(), any(), any());
        EndpointRegistry.Retirement retirement;
        status.lock.lock();
        try { retirement = registry.beginRetirement(RoleType.VIT, status.getIpPort(), status); }
        finally { status.lock.unlock(); }
        assertFalse(notifiedOutsideLock.get(), "detach only returns ownership; it must not publish inline");
        retirement.complete(mock(CacheAwareService.class), LoggerFactory.getLogger(getClass()));
        assertTrue(notifiedOutsideLock.get());
        assertTrue(endpoint.isGenerationRetiringOrRetired());
        assertTrue(registry.statusSnapshot(RoleType.VIT).isEmpty());
        assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), registry::close,
                "the failed notification must not strand the detached retirement barrier");
    }

    @Test
    void discoveryBuiltOutsideCloseGateCannotPublishAfterClose() throws Exception {
        EndpointRegistry registry = registry();
        var status = EndpointTestSupport.workerStatus(RoleType.VIT, "127.0.0.1", 8800, 8801);
        CountDownLatch factoryEntered = new CountDownLatch(1);
        CountDownLatch finishFactory = new CountDownLatch(1);
        try (var executor = Executors.newSingleThreadExecutor()) {
            Future<?> discovery = executor.submit(() -> registry.currentOrDiscover(RoleType.VIT, status.getIpPort(), () -> {
                factoryEntered.countDown();
                try { assertTrue(finishFactory.await(5, TimeUnit.SECONDS)); }
                catch (InterruptedException interrupted) { throw new AssertionError(interrupted); }
                return status;
            }));
            try {
                assertTrue(factoryEntered.await(2, TimeUnit.SECONDS));
                registry.close();
                finishFactory.countDown();
                ExecutionException rejected = assertThrows(ExecutionException.class,
                        () -> discovery.get(2, TimeUnit.SECONDS));
                assertTrue(rejected.getCause() instanceof IllegalStateException);
                assertEquals(0, registry.discoveredCount());
            } finally { finishFactory.countDown(); }
        }
    }

    private static EngineRpcService.WorkerStatusPB runningStatus(long version) {
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.DECODE.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setAlive(true).setStatusVersion(version)
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(123L).setInputLength(64)
                        .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING))
                .build();
    }

    private static int pendingRetirements(EndpointRegistry registry) {
        Object gate = ReflectionTestUtils.getField(registry, "lifecycleGate");
        synchronized (gate) {
            return (int) ReflectionTestUtils.getField(registry, "inflightDetachedRetirements");
        }
    }

    private static EndpointRegistry registry() {
        return registry(new PlacementAvailability());
    }

    private static EndpointRegistry registry(PlacementAvailability availability) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        return new EndpointRegistry(config, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(mock(AbstractRequestScheduler.class)), mock(DeliveryMetricsReporter.class), org.flexlb.balance.scheduler.SchedulerTestSupport.dispatcher(), availability);
    }

    @SuppressWarnings("unchecked")
    private static Map<RoleType, Map<String, WorkerEndpoint>> endpointMaps(EndpointRegistry registry) {
        return (Map<RoleType, Map<String, WorkerEndpoint>>) ReflectionTestUtils.getField(registry, "endpointsByRole");
    }
}
