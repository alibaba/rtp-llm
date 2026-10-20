package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.slf4j.LoggerFactory;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ReviewLifecycleRegressionTest {
    private EndpointRegistry registry() {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        return new EndpointRegistry(config, new RequestRepository(), mock(DeliveryMetricsReporter.class),
                EndpointTestSupport.routeStrategy(EndpointTestSupport.requestRuntime()), new PlacementAvailability());
    }

    @Test
    void failedCleanupCannotRunASuccessfulGenerationFinalizer() {
        var lifecycle = new EndpointGenerationLifecycle(() -> { });
        AtomicBoolean finalized = new AtomicBoolean();
        lifecycle.onRetired(() -> finalized.set(true));
        lifecycle.beginRetirement();
        assertTrue(lifecycle.tryStartCleanup());
        lifecycle.beginCleanup();
        var failure = new IllegalStateException("cleanup failed");
        lifecycle.completeRetirement(failure);
        assertFalse(finalized.get());
        assertSame(failure, assertThrows(IllegalStateException.class, lifecycle::awaitRetirement));
    }

    @Test
    void failedCacheCleanupKeepsTheExactGenerationUntilRetrySucceeds() throws Exception {
        EndpointRegistry registry = registry();
        WorkerStatus status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8081);
        String address = status.getIpPort();
        registry.currentOrDiscover(RoleType.PREFILL, address, () -> status);
        EndpointTestSupport.publishEndpoint(registry, RoleType.PREFILL, address, status);
        CacheAwareService cache = mock(CacheAwareService.class);
        AtomicInteger attempts = new AtomicInteger();
        doAnswer(invocation -> {
            if (attempts.incrementAndGet() == 1) { throw new IllegalStateException("cache unavailable"); }
            return null;
        }).when(cache).removeEngineBlockCache(address);
        EndpointRegistry.Retirement retirement;
        status.lock.lock();
        try { retirement = registry.beginRetirement(RoleType.PREFILL, address, status); }
        finally { status.lock.unlock(); }
        try {
            retirement.complete(cache, LoggerFactory.getLogger(getClass()));
            assertSame(status, registry.statusSnapshot(RoleType.PREFILL).get(address));
            assertSame(status, registry.currentOrDiscover(RoleType.PREFILL, address,
                    () -> EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8081)));
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3L);
            while (registry.statusSnapshot(RoleType.PREFILL).containsKey(address) && System.nanoTime() < deadline) {
                Thread.sleep(5L);
            }
            assertTrue(registry.statusSnapshot(RoleType.PREFILL).isEmpty());
            assertTrue(attempts.get() >= 2);
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void monitoringDoesNotWaitForAnEndpointOwnershipLock(RoleType role) throws Exception {
        EndpointRegistry registry = registry();
        WorkerStatus status = EndpointTestSupport.workerStatus(role, "127.0.0.1", 8080, 8081);
        WorkerEndpoint endpoint = EndpointTestSupport.publishEndpoint(registry, role, status.getIpPort(), status);
        ReentrantLock lock = role == RoleType.PREFILL ? ((PrefillEndpoint) endpoint).ownership.lock
                : (ReentrantLock) ReflectionTestUtils.getField(ReflectionTestUtils.getField(endpoint, "state"), "admissionLock");
        CountDownLatch held = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try (var owner = Executors.newSingleThreadExecutor()) {
            var holding = owner.submit(() -> {
                lock.lock();
                try { held.countDown(); release.await(); }
                catch (InterruptedException failure) { Thread.currentThread().interrupt(); }
                finally { lock.unlock(); }
            });
            try {
                assertTrue(held.await(2L, TimeUnit.SECONDS));
                var sample = CompletableFuture.supplyAsync(endpoint::tryGetLoadMetric).get(500L, TimeUnit.MILLISECONDS);
                assertTrue(sample.isEmpty());
            } finally {
                release.countDown();
                holding.get(2L, TimeUnit.SECONDS);
            }
        } finally {
            registry.close();
        }
    }
}
