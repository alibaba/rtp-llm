package org.flexlb.worker.endpoint;

import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.worker.endpoint.EndpointTestSupport.runWithStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies endpoint retirement completes resource cleanup despite notification failures.
 */
class WorkerEndpointCloseTest {
    @Test
    void decodeNotificationFailuresDoNotBlockOtherRequestsOrFinalization() throws Exception {
        var repository = mock(RequestRepository.class);
        var notifier = mock(CapacityNotifier.class);
        var endpoint =
                new DecodeEndpoint(
                        EndpointTestSupport.workerStatus(RoleType.DECODE, "10.0.0.1", 8080, 8081),
                        repository,
                        notifier);
        var status = new WorkerStatusResponse();
        status.setTotalKvCacheTokens(1000L);
        status.setAvailableKvCacheTokens(1000L);
        EndpointTestSupport.applyStatus(endpoint, status).run();
        for (var id : java.util.List.of("first", "second")) {
            assertNotNull(
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(),
                                    id,
                                    10,
                                    10,
                                    1,
                                    new DecodeResources.AdmissionCapacity(2, 100))
                            .value());
            when(repository.findActive(id))
                    .thenThrow(new IllegalStateException("request notification failed"));
        }
        clearInvocations(notifier);
        doThrow(new IllegalStateException("capacity notification failed"))
                .when(notifier)
                .notifyChanged(endpoint);
        var finalized = new AtomicInteger();
        endpoint.closeAsync()
                .thenRun(finalized::incrementAndGet)
                .toCompletableFuture()
                .get(2, TimeUnit.SECONDS);
        verify(repository).findActive("first");
        verify(repository).findActive("second");
        verify(notifier).notifyChanged(endpoint);
        assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(1, finalized.get());
    }

    @Test
    void stopAdmissionAlsoStopsWorkerUpdatesWithoutStartingCleanup() {
        var cleanups = new AtomicInteger();
        var endpoint = new EndpointCloseTestSupport(cleanups::incrementAndGet);
        endpoint.stopAcceptingRequests();
        assertTrue(endpoint.hasStoppedAcceptingRequests());
        org.junit.jupiter.api.Assertions.assertFalse(endpoint.getStatus().isActiveGeneration());
        assertEquals(0, cleanups.get());
    }

    @Test
    void closeProtocolHasPackageVisibilityAndNoSynchronousEntry() throws Exception {
        for (var name : java.util.List.of("stopAcceptingRequests", "closeAsync")) {
            int modifiers = WorkerEndpoint.class.getDeclaredMethod(name).getModifiers();
            assertEquals(
                    0,
                    modifiers
                            & (java.lang.reflect.Modifier.PUBLIC
                                    | java.lang.reflect.Modifier.PROTECTED
                                    | java.lang.reflect.Modifier.PRIVATE));
        }
        assertThrows(
                NoSuchMethodException.class, () -> WorkerEndpoint.class.getDeclaredMethod("close"));
    }

    @Test
    void asyncCloseReportsCompletionAndProtectsTheSharedResult() throws Exception {
        var started = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var cleanups = new AtomicInteger();
        var endpoint =
                new EndpointCloseTestSupport(
                        () -> {
                            cleanups.incrementAndGet();
                            started.countDown();
                            com.google.common.util.concurrent.Uninterruptibles.awaitUninterruptibly(
                                    release);
                        });
        var closing = endpoint.closeAsync();
        try {
            assertTrue(started.await(2, TimeUnit.SECONDS));
            var repeated = endpoint.closeAsync();
            closing.toCompletableFuture().complete(null);
            org.junit.jupiter.api.Assertions.assertFalse(repeated.toCompletableFuture().isDone());
            assertEquals(1, cleanups.get());
        } finally {
            release.countDown();
        }
        closing.toCompletableFuture().get(2, TimeUnit.SECONDS);
    }

    @Test
    void concurrentCloseCleansOnceAndCompletesAfterNotifications() throws Exception {
        var started = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var cleanups = new AtomicInteger();
        var completions = new AtomicInteger();
        var endpoint =
                new EndpointCloseTestSupport(
                        () -> {
                            cleanups.incrementAndGet();
                            started.countDown();
                            com.google.common.util.concurrent.Uninterruptibles.awaitUninterruptibly(
                                    release);
                        });
        var notified = endpoint.closeAsync().thenRun(completions::incrementAndGet);
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var closing = executor.submit(endpoint::closeAsync);
            try {
                assertTrue(started.await(2, TimeUnit.SECONDS));
                assertTrue(endpoint.hasStoppedAcceptingRequests());
                for (int i = 0; i < 20; i++)
                    executor.submit(endpoint::closeAsync).get(2, TimeUnit.SECONDS);
                assertEquals(0, completions.get());
                assertThrows(
                        IllegalStateException.class,
                        () -> EndpointTestSupport.awaitClosed(endpoint, 25));
            } finally {
                release.countDown();
            }
            closing.get(2, TimeUnit.SECONDS);
            EndpointTestSupport.awaitClosed(endpoint, 2000);
            notified.toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(1, cleanups.get());
            assertEquals(1, completions.get());
        }
    }

    @Test
    void failedCleanupIsReportedToEveryWaiterAndDoesNotFinalize() {
        var failure = new IllegalStateException("cleanup failed");
        var completions = new AtomicInteger();
        var endpoint =
                new EndpointCloseTestSupport(
                        () -> {
                            throw failure;
                        });
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(
                                        endpoint)));
        endpoint.closeAsync().thenRun(completions::incrementAndGet);
        for (int i = 0; i < 2; i++) {
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () -> EndpointTestSupport.awaitClosed(endpoint, 2000)));
        }
        assertEquals(0, completions.get());
        assertTrue(endpoint.hasStoppedAcceptingRequests());
    }

    @Test
    void closingFromAnEventRunsCleanupOutsideThatEvent() throws Exception {
        var cleanupOutsideEvent = new CompletableFuture<Boolean>();
        var ref = new java.util.concurrent.atomic.AtomicReference<WorkerEndpoint>();
        var endpoint =
                new EndpointCloseTestSupport(
                        () ->
                                cleanupOutsideEvent.complete(
                                        !ref.get().stateLock().isHeldByCurrentThread()));
        ref.set(endpoint);
        runWithStateLock(
                endpoint.stateLock(),
                () -> org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint));
        assertTrue(cleanupOutsideEvent.get(2, TimeUnit.SECONDS));
        EndpointTestSupport.awaitClosed(endpoint, 2000);
    }
}
