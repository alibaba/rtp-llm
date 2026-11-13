package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.EndpointCleanupTestSupport;
import org.flexlb.worker.endpoint.EndpointTestSupport;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.function.ToIntFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Cleanup uses current directory membership, including terminal records, rather than live request state.
 */
class EndpointCleanupOwnershipTest {
    private FlexlbConfig config;
    private AbstractRequestScheduler registry;

    @BeforeEach
    void setup() {
        config = SchedulingTestConfig.newConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
    }

    @AfterEach
    void close() {
        if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(registry)) {
            registry.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(
                            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(registry))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(registry));
        }
    }

    @Test
    void currentDirectoryProtectsReservationButEarlierSnapshotCanEvictIt() {
        Set<String> beforeRegistration =
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                        .snapshotActive()
                        .stream()
                        .map(context -> context.getRequestId())
                        .collect(java.util.stream.Collectors.toSet());
        long id = 7001L;
        registry.submit(RequestProtocolTestSupport.context(config, id), () -> {});
        var endpoint =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null),
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                                mock(AbstractRequestScheduler.class)));
        {
            var selectedEndpoint = endpoint;

            assertNotNull(selectedEndpoint);
            assertNotNull(EndpointTestSupport.registerDecode(endpoint, id, 1L, 1L, 50));
        }
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());
        // Models moving the ownership scan outside the endpoint lock without revalidation.
        assertEquals(1, endpoint.evictExpiredRequests(-1L, beforeRegistration::contains));
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());
        assertTrue(
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                        .retainsIdentity(String.valueOf(id)),
                "request is still registered despite premature eviction");
    }

    @ParameterizedTest(name = "Decode confirmed={0}: absent lookup cannot release the replacement")
    @ValueSource(booleans = {false, true})
    void registrationAfterAbsentCheckAcquiresFreshReservation(boolean confirmed) throws Exception {
        long id = 7003L;
        var endpoint =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null),
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                                mock(AbstractRequestScheduler.class)));
        var oldReservation = reserve(endpoint, id, 100L, 150L);
        if (confirmed) {
            EndpointCleanupTestSupport.confirmDecode(endpoint, id);
        }
        Runnable assertOldLedger =
                () -> {
                    assertDecodeLedger(
                            endpoint,
                            confirmed ? 0 : 1,
                            confirmed ? 1 : 0,
                            confirmed ? 0L : 100L,
                            confirmed ? 0L : 150L);
                    assertEquals(
                            confirmed
                                    ? org.flexlb.worker.endpoint.DecodeResources.RegistrationState
                                            .CONFIRMED
                                    : org.flexlb.worker.endpoint.DecodeResources.RegistrationState
                                            .UNCONFIRMED,
                            endpoint.registrationState(oldReservation));
                };
        assertOldLedger.run();

        // Every owned entry now has one retention check in the canonical sweep.
        var result =
                raceRegistrationAfterAbsentCheck(
                        id,
                        retain -> endpoint.evictExpiredRequests(-1L, retain),
                        () -> {
                            if (confirmed) {
                                var acquired =
                                        endpoint.tryRegisterSelection(
                                                endpoint.selectionSnapshot(),
                                                String.valueOf(id),
                                                200L,
                                                350L,
                                                50,
                                                new DecodeResources.AdmissionCapacity(100, 100));
                                assertEquals(
                                        WorkerEndpoint.RegistrationStatus.ALREADY_REGISTERED,
                                        acquired.status());
                                EndpointCleanupTestSupport.finishDecode(endpoint, id);
                            }
                            return reserve(endpoint, id, 200L, 350L);
                        },
                        assertOldLedger);
        assertEquals(
                confirmed ? 0 : 1,
                result.evicted(),
                "the return value counts shadow evictions, not confirmed-record purges");
        var replacement = result.owner();
        org.junit.jupiter.api.Assertions.assertNotSame(oldReservation, replacement);
        assertDecodeLedger(endpoint, 1, 0, 200L, 350L);
        endpoint.releaseRequest(oldReservation);
        assertDecodeLedger(endpoint, 1, 0, 200L, 350L);
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertDecodeLedger(endpoint, 1, 0, 200L, 350L);

        RequestContext requestContext = registry.repository.findActive(String.valueOf(id));
        registry.cancel(String.valueOf(id), CancelReason.CLIENT_CANCELLED);
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertDecodeLedger(endpoint, 1, 0, 200L, 350L);
        RequestResult retiredRecord1 =
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                        .getRequestState(requestContext.getRequestId());
        assertTrue(
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                        .removeExactTerminal(
                                org.flexlb.balance.schedule.SchedulerTestSupport
                                        .requestRecordForArchive(registry, retiredRecord1),
                                Long.MAX_VALUE));
        assertEquals(
                1,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertDecodeLedger(endpoint, 0, 0, 0L, 0L);
        endpoint.releaseRequest(replacement);
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertDecodeLedger(endpoint, 0, 0, 0L, 0L);
    }

    @Test
    void confirmedRecordIsRetainedUntilExactTerminalRecordRemoval() {
        long id = 7005L;
        registry.submit(RequestProtocolTestSupport.context(config, id), () -> {});
        var endpoint =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null),
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                                mock(AbstractRequestScheduler.class)));
        var reservation = reserve(endpoint, id, 100L, 150L);
        EndpointCleanupTestSupport.confirmDecode(endpoint, id);
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertDecodeLedger(endpoint, 0, 1, 0L, 0L);
        RequestContext original = registry.repository.findActive(String.valueOf(id));
        registry.cancel(String.valueOf(id), CancelReason.CLIENT_CANCELLED);
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertDecodeLedger(endpoint, 0, 1, 0L, 0L);
        RequestResult retiredRecord3 =
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                        .getRequestState(original.getRequestId());
        assertTrue(
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                        .removeExactTerminal(
                                org.flexlb.balance.schedule.SchedulerTestSupport
                                        .requestRecordForArchive(registry, retiredRecord3),
                                Long.MAX_VALUE));

        registry.submit(RequestProtocolTestSupport.context(config, id), () -> {});
        assertFalse(
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                        .removeExactTerminal(
                                org.flexlb.balance.schedule.SchedulerTestSupport
                                        .requestRecordForArchive(registry, retiredRecord3),
                                Long.MAX_VALUE));
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertEquals(
                org.flexlb.worker.endpoint.DecodeResources.RegistrationState.CONFIRMED,
                endpoint.registrationState(reservation));
        assertDecodeLedger(endpoint, 0, 1, 0L, 0L);
        registry.cancel(String.valueOf(id), CancelReason.CLIENT_CANCELLED);
        assertTrue(
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                        .removeExactTerminal(
                                org.flexlb.balance.schedule.SchedulerTestSupport
                                        .requestRecordForArchive(
                                                registry,
                                                org.flexlb.balance.schedule.SchedulerTestSupport
                                                        .repository(registry)
                                                        .getRequestState(String.valueOf(id))),
                                Long.MAX_VALUE));
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertTrue(endpoint.resourceSnapshot().requests().isEmpty());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(0L, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
        assertEquals(
                org.flexlb.worker.endpoint.DecodeResources.RegistrationState.NOT_REGISTERED,
                endpoint.registrationState(reservation));
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                                ::retainsIdentity));
        assertTrue(endpoint.resourceSnapshot().requests().isEmpty());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(0L, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
    }

    private static DecodeEndpoint.RequestRegistration reserve(
            DecodeEndpoint endpoint, long id, long hardKv, long expectedKv) {
        {
            var selectedEndpoint = endpoint;

            assertNotNull(selectedEndpoint);
            var reservation =
                    EndpointTestSupport.registerDecode(endpoint, id, hardKv, expectedKv, 50);
            assertNotNull(reservation);
            return reservation;
        }
    }

    private static void assertDecodeLedger(
            DecodeEndpoint endpoint, int reserved, int confirmed, long hardKv, long expectedKv) {
        var view = endpoint.resourceSnapshot();
        assertEquals(
                reserved,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());
        assertEquals(
                reserved,
                view.requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());
        assertEquals(
                confirmed,
                view.requests().values().stream()
                        .filter(snapshotRequest -> snapshotRequest.phase().isEngineConfirmed())
                        .count());
        assertEquals(reserved + confirmed, view.selection().usage().occupiedRequests());
        // Queued selections own KV, while confirmed Engine owners consume dispatch capacity.
        assertEquals(reserved + confirmed, view.selection().usage().occupiedRequests());
        assertEquals(
                reserved,
                view.requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());
        assertEquals(hardKv, view.selection().usage().hardReservedKvTokens());
        assertEquals(expectedKv, EndpointTestSupport.expectedReservedKv(view));
    }

    private <T> CleanupRace<T> raceRegistrationAfterAbsentCheck(
            long id,
            ToIntFunction<Predicate<String>> sweep,
            Supplier<T> acquire,
            Runnable assertOldLedger)
            throws Exception {
        CountDownLatch checkedAbsent = new CountDownLatch(1);
        CountDownLatch registered = new CountDownLatch(1);
        CountDownLatch swept = new CountDownLatch(1);
        AtomicInteger queries = new AtomicInteger();
        var executor =
                Executors.newFixedThreadPool(
                        2,
                        task -> {
                            Thread thread = new Thread(task, "cleanup-registration-race");
                            thread.setDaemon(true);
                            return thread;
                        });
        try {
            var cleanup =
                    executor.submit(
                            () -> {
                                try {
                                    return sweep.applyAsInt(
                                            requestId -> {
                                                assertEquals(String.valueOf(id), requestId);
                                                boolean retain =
                                                        org.flexlb.balance.schedule
                                                                .SchedulerTestSupport.repository(
                                                                        registry)
                                                                .retainsIdentity(
                                                                        String.valueOf(requestId));
                                                assertFalse(retain);
                                                queries.incrementAndGet();
                                                assertOldLedger.run();
                                                checkedAbsent.countDown();
                                                await(registered);
                                                // The retention check runs outside State's lock.
                                                // Exact identities and current
                                                // resource facts fence replacement acquisition when
                                                // cleanup resumes.
                                                return retain;
                                            });
                                } finally {
                                    swept.countDown();
                                }
                            });
            var replacement =
                    executor.submit(
                            () -> {
                                await(checkedAbsent);
                                registry.submit(
                                        RequestProtocolTestSupport.context(config, id), () -> {});
                                registered.countDown();
                                await(swept);
                                return acquire.get();
                            });
            int evicted = cleanup.get(10, TimeUnit.SECONDS);
            T owner = replacement.get(10, TimeUnit.SECONDS);
            assertEquals(1, queries.get());
            return new CleanupRace<>(evicted, owner);
        } finally {
            executor.shutdownNow();
            assertTrue(
                    executor.awaitTermination(5, TimeUnit.SECONDS), "race threads did not finish");
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

    private record CleanupRace<T>(int evicted, T owner) {}
}
