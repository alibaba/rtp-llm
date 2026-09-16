package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointCleanupTestSupport;
import org.flexlb.balance.endpoint.EndpointCleanupTestSupport.PrefillLedger;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.lang.management.ManagementFactory;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;
import java.util.function.Supplier;
import java.util.function.ToIntFunction;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/** Cleanup uses current directory membership, including terminal records, rather than live Slot state. */
class EndpointCleanupOwnershipTest {
    private FlexlbConfig config;
    private RequestRegistry registry;

    @BeforeEach void setup() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
    }

    @AfterEach void close() {
        if (registry.closeAdmissionAndAwaitMutations()) {
            registry.closeOutstandingAndTerminalize();
            registry.closeExpiration();
            registry.closePublisher();
        }
    }

    @Test void currentDirectoryProtectsReservationButEarlierSnapshotCanEvictIt() {
        Set<Long> beforeRegistration = registry.snapshotSlots().stream()
                .map(RequestSlot::requestId).collect(java.util.stream.Collectors.toSet());
        long id = 7001L;
        registry.register(RequestLifecycleTestSupport.context(config, id));
        var endpoint = new DecodeEndpoint(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null), mock(EndpointEventProjector.class));
        try (var pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            assertNotNull(endpoint.reservePinned(pin, id, 1L, 1L, 50));
        }
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertEquals(1, endpoint.getInflightCount());
        // Models moving the ownership scan outside the endpoint lock without revalidation.
        assertEquals(1, endpoint.evictExpiredRequests(-1L, beforeRegistration::contains));
        assertEquals(0, endpoint.getInflightCount());
        assertTrue(registry.retainForSchedulerCleanup(id), "request is still registered despite premature eviction");
    }

    @Test void terminalRecordMembershipIsConservativeButNotEquivalentToLiveOwnership() throws Exception {
        long id = 7002L;
        var future = registry.register(RequestLifecycleTestSupport.context(config, id));
        RequestSlot original = registry.requestSlot(id);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        future.get(5, java.util.concurrent.TimeUnit.SECONDS);
        assertEquals(0, registry.liveRequestCount());
        assertTrue(registry.retainForSchedulerCleanup(id));
        assertSame(original, registry.requestSlot(id), "terminal record remains indexed");
        assertTrue(registry.removeExactTerminalRecord(original, Long.MAX_VALUE));
        assertFalse(registry.retainForSchedulerCleanup(id));
        registry.register(RequestLifecycleTestSupport.context(config, id));
        assertNotSame(original, registry.requestSlot(id));
        assertTrue(registry.retainForSchedulerCleanup(id));
        assertFalse(registry.removeExactTerminalRecord(original, Long.MAX_VALUE));
        assertTrue(registry.retainForSchedulerCleanup(id), "old generation cleanup must preserve replacement");
    }

    @ParameterizedTest(name = "Decode confirmed={0}: absent lookup cannot release the replacement")
    @ValueSource(booleans = {false, true})
    void registrationAfterAbsentCheckAcquiresFreshReservation(boolean confirmed) throws Exception {
        long id = 7003L;
        var endpoint = new DecodeEndpoint(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null), mock(EndpointEventProjector.class));
        var oldReservation = reserve(endpoint, id, 100L, 150L);
        if (confirmed) {
            EndpointCleanupTestSupport.confirmDecode(endpoint, id);
        }
        Runnable assertOldLedger = () -> {
            assertDecodeLedger(endpoint, confirmed ? 0 : 1, confirmed ? 1 : 0,
                    confirmed ? 0L : 100L, confirmed ? 0L : 150L);
            assertEquals(confirmed, endpoint.isReservationAccepted(oldReservation));
        };
        assertOldLedger.run();

        // Confirmed entries are queried once in the shadow loop, then again in
        // the confirmed-record loop. Force registration at the latter lookup.
        var result = raceRegistrationAfterAbsentCheck(id, confirmed ? 1 : 0,
                retain -> endpoint.evictExpiredRequests(-1L, retain),
                () -> reserve(endpoint, id, 200L, 350L), assertOldLedger);
        assertEquals(confirmed ? 0 : 1, result.evicted(),
                "the return value counts shadow evictions, not confirmed-record purges");
        var replacement = result.owner();
        assertNotEquals(oldReservation.reservationToken(), replacement.reservationToken());
        assertDecodeLedger(endpoint, 1, 0, 200L, 350L);
        endpoint.releaseReservationExact(oldReservation);
        assertDecodeLedger(endpoint, 1, 0, 200L, 350L);
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertDecodeLedger(endpoint, 1, 0, 200L, 350L);

        RequestSlot slot = registry.requestSlot(id);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertDecodeLedger(endpoint, 1, 0, 200L, 350L);
        assertTrue(registry.removeExactTerminalRecord(slot, Long.MAX_VALUE));
        assertEquals(1, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertDecodeLedger(endpoint, 0, 0, 0L, 0L);
        endpoint.releaseReservationExact(replacement);
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertDecodeLedger(endpoint, 0, 0, 0L, 0L);
    }

    @ParameterizedTest(name = "Prefill batch={0}: absent lookup cannot release the replacement")
    @ValueSource(booleans = {false, true})
    void prefillRegistrationAfterAbsentCheckPreservesFreshWork(boolean batch) throws Exception {
        long id = 7004L;
        var ledger = new PrefillLedger(batch);
        var old = ledger.commit(id, 1L, 20L);
        ledger.advanceBeyondTtl();
        var result = raceRegistrationAfterAbsentCheck(id, 0, ledger::sweep,
                () -> ledger.commit(id, 2L, 70L), () -> ledger.assertOwned(old, 20L));
        assertEquals(1, result.evicted());
        var replacement = result.owner();
        assertNotSame(old.item(), replacement.item());
        assertNotSame(old.reservation(), replacement.reservation());
        ledger.assertOwned(replacement, 70L);
        ledger.assertStaleReleaseIsIgnored(old);
        ledger.assertOwned(replacement, 70L);

        ledger.advanceBeyondTtl();
        assertEquals(0, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertOwned(replacement, 70L);
        RequestSlot slot = registry.requestSlot(id);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertOwned(replacement, 70L);
        assertTrue(registry.removeExactTerminalRecord(slot, Long.MAX_VALUE));
        assertEquals(1, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertEmpty();
        ledger.assertStaleReleaseIsIgnored(replacement);
        assertEquals(0, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertEmpty();

        // Reacquisition with a limit of one checks that cleanup returned the
        // actual admission capacity, including the batch lease, exactly once.
        registry.register(RequestLifecycleTestSupport.context(config, id));
        var third = ledger.commit(id, 3L, 90L);
        ledger.assertOwned(third, 90L);
        ledger.assertStaleReleaseIsIgnored(replacement);
        ledger.assertOwned(third, 90L);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        assertTrue(registry.removeExactTerminalRecord(registry.requestSlot(id), Long.MAX_VALUE));
        ledger.advanceBeyondTtl();
        assertEquals(1, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertEmpty();
    }

    @ParameterizedTest(name = "Prefill batch={0}: replacement directory entry retains old work")
    @ValueSource(booleans = {false, true})
    void prefillReplacementDirectoryEntryConservativelyRetainsOldWork(boolean batch) {
        long id = 7006L;
        registry.register(RequestLifecycleTestSupport.context(config, id));
        var ledger = new PrefillLedger(batch);
        var old = ledger.commit(id, 1L, 20L);
        ledger.advanceBeyondTtl();
        RequestSlot original = registry.requestSlot(id);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertOwned(old, 20L);
        assertTrue(registry.removeExactTerminalRecord(original, Long.MAX_VALUE));

        registry.register(RequestLifecycleTestSupport.context(config, id));
        assertFalse(registry.removeExactTerminalRecord(original, Long.MAX_VALUE));
        assertEquals(0, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertOwned(old, 20L);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        assertTrue(registry.removeExactTerminalRecord(registry.requestSlot(id), Long.MAX_VALUE));
        assertEquals(1, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertStaleReleaseIsIgnored(old);
        assertEquals(0, ledger.sweep(registry::retainForSchedulerCleanup));
        ledger.assertEmpty();
    }

    @Test void confirmedRecordIsRetainedUntilExactTerminalRecordRemoval() {
        long id = 7005L;
        registry.register(RequestLifecycleTestSupport.context(config, id));
        var endpoint = new DecodeEndpoint(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null), mock(EndpointEventProjector.class));
        var reservation = reserve(endpoint, id, 100L, 150L);
        EndpointCleanupTestSupport.confirmDecode(endpoint, id);
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertDecodeLedger(endpoint, 0, 1, 0L, 0L);
        RequestSlot original = registry.requestSlot(id);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertDecodeLedger(endpoint, 0, 1, 0L, 0L);
        assertTrue(registry.removeExactTerminalRecord(original, Long.MAX_VALUE));

        registry.register(RequestLifecycleTestSupport.context(config, id));
        assertFalse(registry.removeExactTerminalRecord(original, Long.MAX_VALUE));
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertTrue(endpoint.isReservationAccepted(reservation));
        assertDecodeLedger(endpoint, 0, 1, 0L, 0L);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        assertTrue(registry.removeExactTerminalRecord(registry.requestSlot(id), Long.MAX_VALUE));
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertDecodeLedger(endpoint, 0, 0, 0L, 0L);
        assertFalse(endpoint.isReservationAccepted(reservation));
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertDecodeLedger(endpoint, 0, 0, 0L, 0L);
    }

    private static DecodeEndpoint.ReservationHandle reserve(
            DecodeEndpoint endpoint, long id, long hardKv, long expectedKv) {
        try (var pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            var reservation = endpoint.reservePinned(pin, id, hardKv, expectedKv, 50);
            assertNotNull(reservation);
            return reservation;
        }
    }

    private static void assertDecodeLedger(DecodeEndpoint endpoint, int reserved, int confirmed,
                                          long hardKv, long expectedKv) {
        var view = endpoint.layeredAdmissionView();
        assertEquals(reserved, endpoint.getInflightCount());
        assertEquals(reserved, view.reserved().size());
        assertEquals(confirmed, view.confirmed().size());
        assertEquals(reserved + confirmed, view.routing().totalLoad());
        // These are immediate reservations, so both unqueued shadows and
        // confirmed Engine owners occupy dispatch capacity.
        assertEquals(reserved + confirmed, view.engineCapacityUsed());
        assertEquals(hardKv, view.routing().inflightHardKv());
        assertEquals(expectedKv, view.routing().inflightExpectedKv());
    }

    private record CleanupRace<T>(int evicted, T owner) { }

    private <T> CleanupRace<T> raceRegistrationAfterAbsentCheck(
            long id, int earlierQueries, ToIntFunction<LongPredicate> sweep,
            Supplier<T> acquire, Runnable assertOldLedger) throws Exception {
        CountDownLatch checkedAbsent = new CountDownLatch(1);
        CountDownLatch registered = new CountDownLatch(1);
        AtomicReference<Thread> acquirer = new AtomicReference<>();
        AtomicInteger queries = new AtomicInteger();
        var executor = Executors.newFixedThreadPool(2, task -> {
            Thread thread = new Thread(task, "cleanup-registration-race");
            thread.setDaemon(true);
            return thread;
        });
        try {
            var cleanup = executor.submit(() -> sweep.applyAsInt(requestId -> {
                assertEquals(id, requestId);
                boolean retain = registry.retainForSchedulerCleanup(requestId);
                assertFalse(retain);
                if (queries.getAndIncrement() < earlierQueries) {
                    return retain;
                }
                assertOldLedger.run();
                checkedAbsent.countDown();
                await(registered);
                // Do not unblock cleanup until the new generation has actually
                // attempted resource acquisition and is waiting on this lock.
                awaitEndpointLockWait(acquirer.get(), Thread.currentThread());
                assertOldLedger.run();
                return retain;
            }));
            var replacement = executor.submit(() -> {
                await(checkedAbsent);
                registry.register(RequestLifecycleTestSupport.context(config, id));
                acquirer.set(Thread.currentThread());
                registered.countDown();
                return acquire.get();
            });
            int evicted = cleanup.get(10, TimeUnit.SECONDS);
            T owner = replacement.get(10, TimeUnit.SECONDS);
            assertEquals(earlierQueries + 1, queries.get());
            return new CleanupRace<>(evicted, owner);
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS), "race threads did not finish");
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS), "race barrier timed out");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(interrupted);
        }
    }

    private static void awaitEndpointLockWait(Thread acquirer, Thread cleaner) {
        var threads = ManagementFactory.getThreadMXBean();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            var info = threads.getThreadInfo(acquirer.threadId());
            if (info != null && info.getLockOwnerId() == cleaner.threadId()) {
                return;
            }
            Thread.yield();
        }
        fail("new resource acquisition did not wait for the cleanup thread's endpoint lock");
    }
}
