package org.flexlb.worker.endpoint;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.balance.schedule.SchedulingTestConfig;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.worker.endpoint.EndpointTestSupport.runWithStateLock;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks failed endpoint cleanup retains the exact generation for retry and monitoring remains readable while resource locks are held.
 */
class ReviewLifecycleRegressionTest {
    @Test
    void failedCleanupCannotRunASuccessfulGenerationFinalizer() {
        var failure = new IllegalStateException("cleanup failed");
        var lifecycle =
                new EndpointCloseTestSupport(
                        () -> {
                            throw failure;
                        });
        AtomicBoolean finalized = new AtomicBoolean();
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(
                                        lifecycle)));
        lifecycle.closeAsync().thenRun(() -> finalized.set(true));
        assertFalse(finalized.get());
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () -> EndpointTestSupport.awaitClosed(lifecycle, 30_000L)));
    }

    @Test
    void failedCacheCleanupKeepsTheExactGenerationUntilRetrySucceeds() throws Exception {
        EndpointRegistry registry = registry();
        WorkerStatus status =
                EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8081);
        String address = status.getLogicalIpPort();
        registry.currentOrDiscover(RoleType.PREFILL, address, () -> status);
        EndpointTestSupport.publishEndpoint(registry, RoleType.PREFILL, address, status);
        CacheAwareService cache = mock(CacheAwareService.class);
        org.springframework.test.util.ReflectionTestUtils.setField(
                registry, "cacheAwareService", cache);
        AtomicInteger attempts = new AtomicInteger();
        var firstFailure = new java.util.concurrent.CountDownLatch(1);
        doAnswer(
                        invocation -> {
                            if (attempts.incrementAndGet() == 1) {
                                firstFailure.countDown();
                                throw new IllegalStateException("cache unavailable");
                            }
                            return null;
                        })
                .when(cache)
                .removeEngineBlockCache(address);
        java.util.concurrent.CompletionStage<Void> cleanup;
        status.lock.lock();
        try {
            cleanup = registry.closeWorker(RoleType.PREFILL, address, status);
        } finally {
            status.lock.unlock();
        }
        try {
            assertTrue(firstFailure.await(2, TimeUnit.SECONDS));
            assertSame(status, registry.statusSnapshot(RoleType.PREFILL).get(address));
            assertSame(
                    status,
                    registry.currentOrDiscover(
                            RoleType.PREFILL,
                            address,
                            () ->
                                    EndpointTestSupport.workerStatus(
                                            RoleType.PREFILL, "127.0.0.1", 8080, 8081)));
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3L);
            while (registry.statusSnapshot(RoleType.PREFILL).containsKey(address)
                    && System.nanoTime() < deadline) {
                Thread.sleep(5L);
            }
            assertTrue(registry.statusSnapshot(RoleType.PREFILL).isEmpty());
            assertTrue(attempts.get() >= 2);
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "DECODE"})
    void monitoringDoesNotWaitForAnEndpointOwnershipLock(RoleType role) throws Exception {
        EndpointRegistry registry = registry();
        WorkerStatus status = EndpointTestSupport.workerStatus(role, "127.0.0.1", 8080, 8081);
        WorkerEndpoint endpoint =
                EndpointTestSupport.publishEndpoint(
                        registry, role, status.getLogicalIpPort(), status);
        ReentrantLock lock =
                role == RoleType.PREFILL
                        ? ((PrefillEndpoint) endpoint).stateLock()
                        : endpoint.stateLock();
        CountDownLatch held = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try (var owner = Executors.newSingleThreadExecutor()) {
            var holding =
                    owner.submit(
                            () -> {
                                runWithStateLock(
                                        lock,
                                        () -> {
                                            try {
                                                held.countDown();
                                                release.await();
                                            } catch (InterruptedException failure) {
                                                Thread.currentThread().interrupt();
                                            }
                                        });
                            });
            try {
                assertTrue(held.await(2L, TimeUnit.SECONDS));
                var sample =
                        CompletableFuture.supplyAsync(endpoint::tryGetLoadMetric)
                                .get(500L, TimeUnit.MILLISECONDS);
                assertTrue(sample.isEmpty());
            } finally {
                release.countDown();
                holding.get(2L, TimeUnit.SECONDS);
            }
        } finally {
            registry.close();
        }
    }

    private EndpointRegistry registry() {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        return new EndpointRegistry(
                config,
                new RequestRepository(),
                mock(DeliveryMetricsReporter.class),
                new CapacityNotifier(),
                org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class));
    }
}
