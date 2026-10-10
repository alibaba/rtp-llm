package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DeliveryStrategy;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.LoggerFactory;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;

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
                assertFalse(registry.decodeRoutingSnapshot(null).isEmpty());
                Future<?> owner = executor.submit(() -> {
                    Thread.currentThread().interrupt();
                    try {
                        registry.close();
                    } finally {
                        assertTrue(Thread.interrupted(), "publication drain must preserve interruption");
                    }
                });
                assertTimeoutPreemptively(java.time.Duration.ofSeconds(2), () -> {
                    while (!registry.decodeRoutingSnapshot(null).isEmpty()) { Thread.sleep(1L); }
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
                assertTrue(registry.decodeRoutingSnapshot(null).isEmpty());
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

    private static EndpointRegistry registry() {
        return registry(new PlacementAvailability());
    }

    private static EndpointRegistry registry(PlacementAvailability availability) {
        return new EndpointRegistry(mock(ConfigService.class), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(mock(AbstractRequestScheduler.class)), mock(DeliveryMetricsReporter.class), mock(DeliveryStrategy.class), availability);
    }

    @SuppressWarnings("unchecked")
    private static Map<RoleType, Map<String, WorkerEndpoint>> endpointMaps(EndpointRegistry registry) {
        return (Map<RoleType, Map<String, WorkerEndpoint>>) ReflectionTestUtils.getField(registry, "endpointsByRole");
    }
}
