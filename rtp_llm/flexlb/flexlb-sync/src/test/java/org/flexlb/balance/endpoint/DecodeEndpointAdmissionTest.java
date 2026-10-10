package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;

import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.LongStream;

import static org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
import static org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
import static org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * Phase 4 tests for the decode admission state of {@link DecodeEndpoint}:
 * priority-carrying reservations, the admission version, the atomic
 * release-victims-and-reserve-incoming commit (all-or-nothing, design doc
 * 11.5/17.2), and the reserved-only view after calibrate (10.1).
 */
class DecodeEndpointAdmissionTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final Map<Long, DecodeResources.ReservationHandle> reservations =
            new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.1", 8080, 8081);
        endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
    }

    // ==================== realKvAvailable = reported - hard reservations ====================

    @Test
    void realKvAvailable_subtractsHardNotExpectedReservations() {
        updateStatus(null, null, 10_000);
        reserve(1L, 500, 600, 70);

        // Hard (500), not expected (600), is subtracted from the report.
        assertEquals(9_500, endpoint.routingView().realKvAvailable());
        assertEquals(500, endpoint.routingView().inflightHardKv());
        assertEquals(600, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));

        DecodeResources.DecodeRequestView entry = reserved().get(1L);
        assertEquals(70, entry.priority());
        assertEquals(DecodeTaskPhase.LOCAL_RESERVED, entry.phase());
    }

    // ==================== reserve / release bump the admission version ====================

    @Test
    void reserveAndRelease_bumpVersion_andReverseShadowAccounting() {
        long v0 = endpoint.routingView().admissionVersion();

        reserve(1L, 500, 600, 30);
        assertEquals(v0 + 1, endpoint.routingView().admissionVersion());
        assertEquals(1, endpoint.routingView().totalLoad());

        release(1L);
        assertEquals(v0 + 2, endpoint.routingView().admissionVersion());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(0, endpoint.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
    }

    @Test
    void everyExactReservationReleaseSignalsPlacementCapacity() {
        PlacementAvailability availability =
                mock(PlacementAvailability.class);
        DecodeEndpoint exactEndpoint = new DecodeEndpoint(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(mock(AbstractRequestScheduler.class)), availability);

        DecodeResources.ReservationHandle speculative;
        try (WorkerEndpoint.GenerationPin pin =
                     exactEndpoint.tryPinGeneration()) {
            assertNotNull(pin);
            speculative = exactEndpoint.tryReserveQueuedRequest(pin, 11L, 100L, 110L, 10, null);
        }
        exactEndpoint.release(speculative, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(availability).changed(
                RoleType.DECODE, null, "10.0.0.1:8080");

        DecodeResources.ReservationHandle published;
        try (WorkerEndpoint.GenerationPin pin =
                     exactEndpoint.tryPinGeneration()) {
            assertNotNull(pin);
            published = exactEndpoint.tryReserveQueuedRequest(pin, 12L, 100L, 110L, 10, null);
        }
        exactEndpoint.release(published, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(availability, times(2)).changed(
                RoleType.DECODE, null, "10.0.0.1:8080");
    }

    @Test
    void conditionalOrphanReleasePreservesReplacementReservation() {
        long requestId = 2L;
        DecodeResources.ReservationHandle stale =
                reserve(requestId, 100, 110, 30);
        DecodeResources.DecodeRequestView staleSnapshot = reserved().get(requestId);
        release(requestId);

        DecodeResources.ReservationHandle current =
                reserve(requestId, 200, 220, 70);
        DecodeResources.DecodeRequestView replacement = reserved().get(requestId);
        assertNotEquals(staleSnapshot.reservationToken(), replacement.reservationToken());

        assertEquals(DecodeResources.ReservationReleaseResult.STALE,
                endpoint.release(stale, DecodeResources.ReleaseReason.COUNTERPART_FINISHED));
        assertReservationIdentity(replacement, reserved().get(requestId));
        assertEquals(200, endpoint.routingView().inflightHardKv());
        assertEquals(220, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));

        assertEquals(DecodeResources.ReservationReleaseResult.RELEASED,
                endpoint.release(current, DecodeResources.ReleaseReason.COUNTERPART_FINISHED));
        assertFalse(reserved().containsKey(requestId));
        assertEquals(0, endpoint.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
    }

    // ==================== atomic release+reserve: success ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_success_appliesAtomically() {
        updateStatus(Map.of(), Map.of(), 708L);
        reserve(1L, 100, 110, 30);
        reserve(2L, 200, 220, 40);
        markQueued(1L);
        markQueued(2L);
        long version = endpoint.routingView().admissionVersion();

        var incoming = endpoint.replaceQueuedRequests(handles(1L, 2L), 9L, 700, 708, 70,
                new DecodeResources.AdmissionCapacity(2, 100));
        assertNotNull(incoming);
        assertEquals(EndpointTestSupport.decodeReservation(endpoint, 9L), incoming);
        assertFalse(reserved().containsKey(1L));
        assertFalse(reserved().containsKey(2L));
        assertEquals(70, reserved().get(9L).priority());
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(700, endpoint.routingView().inflightHardKv());
        assertTrue(endpoint.routingView().admissionVersion() > version);
    }

    // ==================== atomic release+reserve: validation failures apply nothing ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_identityMismatch_appliesNothing() {
        DecodeResources.ReservationHandle exact =
                reserve(1L, 100, 110, 30);
        markQueued(1L);
        DecodeResources.ReservationHandle stale =
                new DecodeResources.ReservationHandle(
                        exact.endpointGenerationId(),
                        exact.requestId(),
                        exact.reservationToken() + 1L);
        long version = endpoint.routingView().admissionVersion();

        assertNull(endpoint.replaceQueuedRequests(List.of(stale), 9L, 700, 708, 70, new DecodeResources.AdmissionCapacity(1, 100)));
        assertTrue(reserved().containsKey(1L));
        assertFalse(reserved().containsKey(9L));
        assertEquals(100, endpoint.routingView().inflightHardKv());
        assertEquals(version, endpoint.routingView().admissionVersion());
    }

    @Test
    void tryReleaseVictimsAndReserveIncoming_victimGone_appliesNothing() {
        DecodeResources.ReservationHandle exact =
                reserve(1L, 100, 110, 30);
        markQueued(1L);
        long version = endpoint.routingView().admissionVersion();
        DecodeResources.ReservationHandle absent =
                new DecodeResources.ReservationHandle(
                        exact.endpointGenerationId(), 42L,
                        exact.reservationToken() + 1L);

        assertNull(endpoint.replaceQueuedRequests(List.of(exact, absent), 9L, 700, 708, 70, new DecodeResources.AdmissionCapacity(1, 100)));
        assertTrue(reserved().containsKey(1L));
        assertFalse(reserved().containsKey(9L));
        assertEquals(100, endpoint.routingView().inflightHardKv());
        assertEquals(version, endpoint.routingView().admissionVersion());
    }

    @Test
    void engineDispatchPermitAndLocalEvictionHaveOneAdmissionLockWinner() {
        // Eviction wins: it removes the reservation, so the batch item must be
        // skipped before acquiring an engine-dispatch permit / gRPC publication.
        reserve(1L, 100, 110, 30);
        markQueued(1L);
        assertTrue(releaseLocalShadow(1L));
        DecodeEndpoint.EngineDispatchPermitAcquisition released =
                endpoint.acquireDispatchPermit(reservations.get(1L), new DecodeResources.AdmissionCapacity(5, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                released.status());
        assertNull(released.permit());

        // Dispatch wins: commit atomically clears queued ownership while retaining
        // the reservation, so a later local-eviction attempt cannot touch it.
        reserve(2L, 100, 110, 30);
        markQueued(2L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(2L, 5);
        assertEquals(TRANSFERRED, permit.dispatch());
        assertFalse(releaseLocalShadow(2L));
        assertTrue(reserved().containsKey(2L));

        // Engine-facing reservations are not eligible for a pre-delivery permit.
        reserve(3L, 100, 110, 30);
        DecodeEndpoint.EngineDispatchPermitAcquisition engineFacing =
                endpoint.acquireDispatchPermit(reservations.get(3L), new DecodeResources.AdmissionCapacity(5, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.NOT_QUEUED,
                engineFacing.status());
        assertNull(engineFacing.permit());
    }

    @Test
    void engineDispatchPermit_stopsAtConfiguredEngineFacingLimit() {
        // Four non-queued reservations already face the Engine.
        for (long requestId = 100; requestId < 104; requestId++) {
            reserve(requestId, 100, 110, 30);
        }
        // A single Prefill batch may contain many reservations which are all
        // deliberately invisible to getEngineLoad while still queued.
        for (long requestId = 1; requestId <= 20; requestId++) {
            reserve(requestId, 100, 110, 50);
            markQueued(requestId);
        }

        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(1L, 5);
        assertEquals(TRANSFERRED, first.dispatch());
        List<DecodeResources.EngineDispatchPermitAcquireStatus> results =
                LongStream.rangeClosed(2, 20)
                        .mapToObj(requestId -> endpoint
                                .acquireDispatchPermit(reservations.get(requestId), new DecodeResources.AdmissionCapacity(5, 100L)).status())
                        .toList();

        assertTrue(results.stream().allMatch(result ->
                result == DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL));
        assertEquals(5, endpoint.routingView().engineLoad());
        assertEquals(19, endpoint.resourceSnapshot().queuedCount(),
                "capacity-blocked reservations must remain queued");
    }

    @Test
    void engineDispatchPermit_preservesUnlimitedAndRejectsNonQueuedReservations() {
        reserve(1L, 100, 110, 30);
        reserve(2L, 100, 110, 30);
        markQueued(1L);
        markQueued(2L);

        assertEquals(TRANSFERRED,
                acquirePermit(1L, 0).dispatch());
        assertEquals(TRANSFERRED,
                acquirePermit(2L, 0).dispatch());

        // A non-queued reservation is already engine-facing and has no
        // pre-delivery ownership transition to reserve.
        reserve(3L, 100, 110, 30);
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireDispatchPermit(reservations.get(3L), new DecodeResources.AdmissionCapacity(1, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.NOT_QUEUED,
                acquisition.status());
        assertNull(acquisition.permit());
    }

    @Test
    void queuedKvIsSoftUntilDispatchPermit_thenKvLimitIsAuthoritative() {
        updateStatus(null, null, 1_000);

        reserveQueued(1L, 400, 900, 50);

        assertEquals(900, endpoint.routingView().realKvUsed(),
                "placement scoring must retain queued expected KV");
        assertEquals(0, endpoint.routingView().dispatchUsage().expectedKvUsed(),
                "queued expected KV must not poison the dispatch gate");
        assertEquals(600, endpoint.routingView().realKvAvailable());
        assertEquals(1_000, endpoint.routingView().dispatchUsage().hardKvAvailable());

        DecodeEndpoint.EngineDispatchPermitAcquisition first =
                endpoint.acquireDispatchPermit(reservations.get(1L), new DecodeResources.AdmissionCapacity(256, 90));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                first.status());
        assertNotNull(first.permit());
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 1L),
                "capacity is occupied before queued ownership is transferred");
        assertEquals(0, endpoint.routingView().engineLoad(),
                "a pre-delivery permit is not engine-facing load");

        reserveQueued(2L, 100, 100, 50);
        DecodeEndpoint.EngineDispatchPermitAcquisition second =
                endpoint.acquireDispatchPermit(reservations.get(2L), new DecodeResources.AdmissionCapacity(256, 90));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                second.status(),
                "a permit already occupying the 90% KV fence must prevent oversubscription");
        assertNull(second.permit());
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 2L));
        assertEquals(900, endpoint.routingView().dispatchUsage().expectedKvUsed(),
                "the failed candidate must not add KV beyond the first acquired permit");

        assertEquals(TRANSFERRED, first.permit().dispatch());
        assertEquals(900, endpoint.routingView().dispatchUsage().expectedKvUsed());
        assertEquals(600, endpoint.routingView().dispatchUsage().hardKvAvailable());
    }

    @Test
    void engineDispatchPermit_reportsNotOwnedAfterReleaseOrPreemptionClaim() {
        reserve(1L, 100, 110, 30);
        markQueued(1L);
        release(1L);
        DecodeEndpoint.EngineDispatchPermitAcquisition released =
                endpoint.acquireDispatchPermit(reservations.get(1L), new DecodeResources.AdmissionCapacity(5, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                released.status());
        assertNull(released.permit());

        reserve(2L, 100, 110, 30);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(
                        101L, List.of(2L), 9L, 100, 110, 70));
        DecodeEndpoint.EngineDispatchPermitAcquisition preempted =
                endpoint.acquireDispatchPermit(reservations.get(2L), new DecodeResources.AdmissionCapacity(5, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                preempted.status());
        assertNull(preempted.permit());
    }

    @Test
    void engineDispatchPermit_capacityFullLeavesReservationQueued() {
        reserve(100L, 100, 110, 30);
        reserve(1L, 100, 110, 50);
        markQueued(1L);
        long versionBeforeAcquire = endpoint.routingView().admissionVersion();

        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireDispatchPermit(reservations.get(1L), new DecodeResources.AdmissionCapacity(1, 100L));

        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                acquisition.status());
        assertNull(acquisition.permit());
        assertEquals(versionBeforeAcquire, endpoint.routingView().admissionVersion());
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 1L));
        assertEquals(1, endpoint.routingView().engineLoad());
    }

    @Test
    void engineDispatchPermit_occupiesHardGateWithoutChangingEngineLoad() {
        reserve(1L, 100, 110, 50);
        reserve(2L, 100, 110, 50);
        markQueued(1L);
        markQueued(2L);

        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(1L, 1);
        DecodeEndpoint.EngineDispatchPermitAcquisition duplicate =
                endpoint.acquireDispatchPermit(reservations.get(1L), new DecodeResources.AdmissionCapacity(1, 100L));
        DecodeEndpoint.EngineDispatchPermitAcquisition second =
                endpoint.acquireDispatchPermit(reservations.get(2L), new DecodeResources.AdmissionCapacity(1, 100L));

        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED,
                duplicate.status());
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                second.status());
        assertEquals(0, endpoint.routingView().engineLoad(),
                "a pre-delivery permit is not engine-facing load");
        assertEquals(2, endpoint.resourceSnapshot().queuedCount());
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 1L));
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 2L));
        assertTrue(first.release());
    }

    @Test
    void engineDispatchPermit_releaseRestoresHardGateCapacityAndIsIdempotent() {
        reserve(1L, 100, 110, 50);
        reserve(2L, 100, 110, 50);
        markQueued(1L);
        markQueued(2L);
        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(1L, 1);

        assertTrue(first.release());
        assertFalse(first.release());
        assertEquals(OWNERSHIP_LOST, first.dispatch());

        DecodeEndpoint.EngineDispatchPermit second = acquirePermit(2L, 1);
        assertEquals(2, endpoint.resourceSnapshot().queuedCount());
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 1L));
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 2L));
        assertTrue(second.release());
    }

    @Test
    void closedEndpointRejectsPermitAcquisitionAsRetired() {
        reserveQueued(1L, 100L, 110L, 50);
        endpoint.close();

        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireDispatchPermit(reservations.get(1L), new DecodeResources.AdmissionCapacity(1, 100L));

        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ENDPOINT_RETIRED,
                acquisition.status());
        assertNull(acquisition.permit());
    }

    @Test
    void closeBeforeReservationRejectsEveryNewOwnershipEntryPoint() {
        DecodeResources.ReservationHandle stale =
                reserve(10L, 10, 10, 1);
        release(10L);
        endpoint.close();

        assertNull(endpoint.tryPinGeneration());
        assertNull(endpoint.replaceQueuedRequests(List.of(stale), 3L, 100, 110, 50, new DecodeResources.AdmissionCapacity(1, 100)));
        assertEquals(DecodeResources.PreemptionBeginResult.ENDPOINT_RETIRED,
                endpoint.beginPreemption(101L, List.of(stale), 5L, 100, 110, 50, new DecodeResources.AdmissionCapacity(1, 100)));

        assertTrue(reserved().isEmpty());
        assertEquals(0L, endpoint.routingView().inflightHardKv());
        assertEquals(0L, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
    }

    @Test
    void reserveQueuedBeforeCloseRetiresUnpublishedOwnership() {
        DecodeResources.ReservationHandle reservation = reserveQueued(
                1L, 100, 110, 50);
        assertEquals(reservation, EndpointTestSupport.decodeReservation(endpoint, 1L));

        endpoint.close();

        assertNull(EndpointTestSupport.decodeReservation(endpoint, 1L));
        assertTrue(endpoint.resourceSnapshot().queuedCount() == 0);
        assertEquals(0L, endpoint.routingView().inflightHardKv());
        assertEquals(0L, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
    }

    @Test
    void closeBetweenPermitAcquisitionAndTransferRetiresExactPermitAndReleasesHardGate()
            throws InterruptedException {
        reserve(1L, 100, 110, 50);
        reserve(2L, 100, 110, 50);
        markQueued(1L);
        markQueued(2L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(1L, 1);
        assertFalse(endpoint.shouldRetryDispatch(2L, new DecodeResources.AdmissionCapacity(1L, 100L)));

        CountDownLatch capacityWakeup = new CountDownLatch(1);
        AtomicInteger capacityNotifications = new AtomicInteger();
        endpoint.addEngineDispatchCapacityListener(() -> {
            capacityNotifications.incrementAndGet();
            capacityWakeup.countDown();
        });

        endpoint.close();
        endpoint.close();

        assertTrue(capacityWakeup.await(1, TimeUnit.SECONDS));
        assertEquals(1, capacityNotifications.get(),
                "retiring one endpoint generation publishes one capacity transition");
        assertEquals(ENDPOINT_RETIRED, permit.dispatch());
        assertEquals(ENDPOINT_RETIRED, permit.dispatch(),
                "the exact retired permit keeps its typed terminal result");
        assertFalse(permit.release(), "dispatch already consumed the retired result");
        assertEquals(ENDPOINT_RETIRED, permit.dispatch(),
                "a failed release cannot replace the cached dispatch result");
        assertTrue(endpoint.shouldRetryDispatch(2L, new DecodeResources.AdmissionCapacity(1L, 100L)),
                "close must remove the outstanding permit from hard-gate usage");
        assertEquals(0, endpoint.routingView().engineLoad());
        assertTrue(endpoint.resourceSnapshot().queuedCount() == 0,
                "retirement must release queued ownership not yet transferred to the Engine");
    }

    @Test
    void releaseAcknowledgesTheExactPermitInvalidatedByRetirement() {
        reserve(1L, 100, 110, 50);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermit retired = acquirePermit(1L, 1);

        endpoint.close();

        assertTrue(retired.release());
        assertFalse(retired.release(),
                "retirement acknowledgement remains one-shot");
        assertEquals(OWNERSHIP_LOST, retired.dispatch(),
                "a returned permit cannot regain sending ownership, even after retirement");
    }

    @Test
    void closePreservesTransferredPermitOutcomeWhileRetiringEndpointOwnership() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, permit.dispatch());
        assertEquals(1, endpoint.routingView().engineLoad());

        endpoint.close();

        assertEquals(TRANSFERRED, permit.dispatch());
        assertNull(EndpointTestSupport.decodeReservation(endpoint, requestId));
        assertFalse(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), requestId));
        assertEquals(0, endpoint.routingView().engineLoad(),
                "retirement clears canonical generation ownership without reversing the permit result");
        assertFalse(releaseLocalShadow(requestId));
    }

    @Test
    void engineDispatchPermit_commitDoesNotRecheckCapacity() {
        reserve(1L, 100, 110, 50);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(1L, 1);

        // Independent engine-facing work fills the original limit after this
        // permit has already reserved its slot.
        reserve(100L, 100, 110, 30);
        assertEquals(1, endpoint.routingView().engineLoad());

        assertEquals(TRANSFERRED, permit.dispatch());
        assertEquals(TRANSFERRED, permit.dispatch(),
                "ownership transfer must be idempotent after handoff");
        assertFalse(permit.release());
        assertFalse(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 1L));
        assertEquals(2, endpoint.routingView().engineLoad());
    }

    @Test
    void committedPermitCannotReturnEngineOwnershipToQueuedState() {
        reserve(1L, 100, 110, 50);
        reserve(2L, 100, 110, 50);
        markQueued(1L);
        markQueued(2L);
        DecodeResources.DecodeRequestView firstReservation = reserved().get(1L);
        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(1L, 1);

        assertEquals(TRANSFERRED, first.dispatch());
        assertReservationIdentity(firstReservation, reserved().get(1L));
        assertFalse(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 1L));
        assertEquals(1, endpoint.routingView().engineLoad());
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                endpoint.acquireDispatchPermit(reservations.get(2L), new DecodeResources.AdmissionCapacity(1, 100L)).status());

        assertFalse(first.release(),
                "committed Decode ownership is irreversible through the permit");
        assertReservationIdentity(firstReservation, reserved().get(1L));
        assertFalse(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), 1L));
        assertEquals(1, endpoint.routingView().engineLoad());

        settleFromWorkerStatus(1L);
        DecodeEndpoint.EngineDispatchPermit second = acquirePermit(2L, 1);
        assertTrue(second.release());
    }

    @Test
    void committedPermitCannotAffectLaterDispatchRoundOnSameReservation() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        DecodeResources.DecodeRequestView reservation = reserved().get(requestId);
        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, first.dispatch());
        try (var pin = endpoint.tryPinGeneration()) {
            assertFalse(endpoint.markQueued(pin, reservations.get(requestId)),
                    "handoff cannot be undone by republishing queue membership");
        }
        assertFalse(first.release());
        assertReservationIdentity(reservation, reserved().get(requestId));
        assertFalse(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), requestId));
        assertEquals(1, endpoint.routingView().engineLoad());
    }

    @Test
    void markQueuedPhaseBumpsAdmissionVersionOnlyWhenOwnershipActuallyChanges() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        long versionBeforeFirstMark = endpoint.routingView().admissionVersion();

        markQueued(requestId);
        assertEquals(versionBeforeFirstMark + 1, endpoint.routingView().admissionVersion());
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), requestId));

        markQueued(requestId);
        assertEquals(versionBeforeFirstMark + 1, endpoint.routingView().admissionVersion(),
                "repeating an existing queued mark must be a no-op");

        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, permit.dispatch());
        long versionBeforeSecondRound = endpoint.routingView().admissionVersion();

        try (var pin = endpoint.tryPinGeneration()) {
            assertFalse(endpoint.markQueued(pin, reservations.get(requestId)));
        }
        assertEquals(versionBeforeSecondRound, endpoint.routingView().admissionVersion());
    }

    @Test
    void staleCommittedPermitCannotAffectReplacementGeneration() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        DecodeResources.DecodeRequestView original = reserved().get(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, stale.dispatch());

        settleFromWorkerStatus(requestId);
        reserve(requestId, 200, 220, 70);
        markQueued(requestId);
        DecodeResources.DecodeRequestView replacement = reserved().get(requestId);
        assertNotEquals(original.reservationToken(), replacement.reservationToken());
        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);

        assertFalse(stale.release(),
                "a committed old generation must not change its replacement");
        assertFalse(stale.release(), "a stale release must stay idempotent");
        assertReservationIdentity(replacement, reserved().get(requestId));
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), requestId));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED,
                endpoint.acquireDispatchPermit(reservations.get(requestId), new DecodeResources.AdmissionCapacity(1, 100L)).status());

        assertEquals(TRANSFERRED, current.dispatch(),
                "the stale token must not release or invalidate the new permit");
        assertReservationIdentity(replacement, reserved().get(requestId));
        assertFalse(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), requestId));
    }

    @Test
    void staleEngineDispatchPermitCannotAffectReplacementGeneration() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        DecodeResources.DecodeRequestView original = reserved().get(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);

        release(requestId);
        reserve(requestId, 200, 220, 70);
        markQueued(requestId);
        DecodeResources.DecodeRequestView replacement = reserved().get(requestId);
        assertNotEquals(original.reservationToken(), replacement.reservationToken());

        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);
        assertFalse(stale.release(), "the old token must not remove the new permit");
        assertEquals(OWNERSHIP_LOST, stale.dispatch());
        assertReservationIdentity(replacement, reserved().get(requestId));
        assertTrue(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), requestId));
        assertEquals(TRANSFERRED, current.dispatch(),
                "the replacement generation keeps its permit");
        assertReservationIdentity(replacement, reserved().get(requestId));
        assertFalse(EndpointTestSupport.isQueued(endpoint.resourceSnapshot(), requestId));
    }

    @Test
    void staleEngineDispatchPermitCannotUseAConfirmedReplacement() {
        long requestId = 2L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);

        release(requestId);
        reserve(requestId, 200, 220, 70);
        markQueued(requestId);
        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);
        TaskInfo running = new TaskInfo();
        running.setRequestId(requestId);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of(Long.toString(requestId), running), null, 10_000L);

        assertEquals(OWNERSHIP_LOST, stale.dispatch());
        assertTrue(endpoint.isAcceptedByEngine(reservations.get(requestId)));
        assertEquals(TRANSFERRED, current.dispatch());
    }

    @Test
    void engineAcceptanceCompletesPermitHandoffWithoutLeakingCapacity() {
        reserve(1L, 100, 110, 50);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(1L, 2);

        TaskInfo running = new TaskInfo();
        running.setRequestId(1L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("1", running), null, 10_000);

        reserve(2L, 100, 110, 50);
        markQueued(2L);
        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(2L, 2);
        assertEquals(TRANSFERRED, stale.dispatch());
        assertEquals(TRANSFERRED, stale.dispatch());
        assertFalse(stale.release());
        assertTrue(current.release());
    }

    @ParameterizedTest
    @ValueSource(longs = {-1L, Long.MIN_VALUE})
    void ttlEvictionInvalidatesPermitWithoutLeakingHardGateCapacity(long ttlMs) {
        reserve(1L, 100, 110, 50);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(1L, 1);

        assertEquals(1, endpoint.evictExpiredRequests(
                ttlMs, requestId -> false));
        reserve(2L, 100, 110, 50);
        markQueued(2L);

        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(2L, 1);
        assertFalse(stale.release());
        assertTrue(current.release());
    }

    // ==================== 10.1: confirmed requests leave the reserved view ====================

    @Test
    void calibrate_movesConfirmedOutOfReservedView() {
        reserve(1L, 500, 508, 30);

        TaskInfo running = new TaskInfo();
        running.setRequestId(1L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("1", running), null, 10_000);

        // Confirmed by the engine: no longer a reserved (evictable) entry,
        // but still counted in the total load via confirmed Engine ownership.
        assertTrue(reserved().isEmpty());
        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(0, endpoint.routingView().inflightHardKv());
    }

    @Test
    void versionedReceivedTaskEmitsActivityWithoutAdvancingAcceptance() {
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(events));
        DecodeResources.ReservationHandle reservation =
                reserve(1L, 500, 508, 30);
        TaskInfo received = new TaskInfo();
        received.setRequestId(1L);
        received.setPhase(TaskPhase.RECEIVED);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of("1", received));
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);

        EndpointTestSupport.applyStatus(endpoint, response).run();

        @SuppressWarnings("unchecked")
        ArgumentCaptor<DecodeResources.DecodeRequestStatus> requestStatuses =
                ArgumentCaptor.forClass(DecodeResources.DecodeRequestStatus.class);
        verify(events).onDecodeStatus(
                org.mockito.Mockito.any(), org.mockito.Mockito.eq(endpoint), requestStatuses.capture());
        DecodeResources.DecodeRequestStatus activeStatus = requestStatuses.getValue();
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.ACTIVE, activeStatus.kind());
        assertEquals(reservation, activeStatus.reservation());
        assertEquals(reservation, EndpointTestSupport.decodeReservation(endpoint, 1L));
    }

    @Test
    void statusFieldApplicationAndCalibrationExcludeConcurrentReserve()
            throws Exception {
        WorkerStatus blockingStatus = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.2", 8080, 8081);
        DecodeEndpoint blockingEndpoint = EndpointTestSupport.decode(blockingStatus, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setAlive(true);
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(20_000L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());

        ExecutorService executor = Executors.newFixedThreadPool(2);
        ReentrantLock admissionLock = decodeAdmissionLock(blockingEndpoint);
        admissionLock.lock();
        try {
            Future<?> statusUpdate = executor.submit(() ->
                    EndpointTestSupport.applyStatus(
                            blockingEndpoint, response));

            Future<?> reserve = executor.submit(() -> {
                try (WorkerEndpoint.GenerationPin pin =
                             blockingEndpoint.tryPinGeneration()) {
                    assertNotNull(pin);
                    EndpointTestSupport.reserveUnqueuedDecode(blockingEndpoint, pin, 71L, 500L, 600L, 80);
                }
            });
            assertThrows(TimeoutException.class,
                    () -> statusUpdate.get(100, TimeUnit.MILLISECONDS),
                    "status reduction must wait for canonical admission ownership");
            assertThrows(TimeoutException.class,
                    () -> reserve.get(100, TimeUnit.MILLISECONDS),
                    "reserve must not cross the status/calibration admission lock");

            admissionLock.unlock();
            statusUpdate.get(5, TimeUnit.SECONDS);
            reserve.get(5, TimeUnit.SECONDS);

            assertNotNull(EndpointTestSupport.decodeReservation(blockingEndpoint, 71L));
            assertEquals(500L, blockingEndpoint.routingView().inflightHardKv());
            assertEquals(600L,
                    EndpointTestSupport.expectedReservedKv(blockingEndpoint.resourceSnapshot()));
            assertEquals(9_500L, blockingEndpoint.routingView().realKvAvailable(),
                    "the post-calibration reservation must be retained");
        } finally {
            if (admissionLock.isHeldByCurrentThread()) {
                admissionLock.unlock();
            }
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
            blockingEndpoint.close();
        }
    }

    @Test
    void dispatchPermitsCheckKvAndRequestSlotsAtomically() throws Exception {
        updateStatus(Map.of(), Map.of(), 10_000L);
        ExecutorService executor = Executors.newFixedThreadPool(8);
        CountDownLatch start = new CountDownLatch(1);
        try {
            List<Future<Boolean>> attempts = new java.util.ArrayList<>();
            for (long id = 1L; id <= 32L; id++) {
                long requestId = id;
                attempts.add(executor.submit(() -> {
                    start.await();
                    DecodeResources.ReservationHandle reservation;
                    try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
                        reservation = endpoint.tryReserveQueuedRequest(pin, requestId, 100L, 3000L, 50, null);
                    }
                    assertNotNull(reservation);
                    var acquired = endpoint.acquireDispatchPermit(reservation, new DecodeResources.AdmissionCapacity(8L, 90L));
                    if (acquired.status() == DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL) {
                        endpoint.release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
                        return false;
                    }
                    assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
                    assertEquals(TRANSFERRED, acquired.permit().dispatch());
                    return true;
                }));
            }
            start.countDown();
            int admitted = 0;
            for (Future<Boolean> attempt : attempts) {
                if (attempt.get(5, TimeUnit.SECONDS)) {
                    admitted++;
                }
            }
            assertEquals(3, admitted);
            assertEquals(9000L, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
            assertEquals(3, endpoint.routingView().engineCapacityUsed());
            assertEquals(0, endpoint.resourceSnapshot().queuedCount());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void permitAcquisitionRecognizesExactAcceptanceBeforeDelivery() {
        updateStatus(Map.of(), Map.of(), 10_000L);
        var handle = reserveQueued(72L, 100L, 1000L, 50);
        assertFalse(endpoint.isAcceptedByEngine(handle));
        TaskInfo accepted = new TaskInfo();
        accepted.setRequestId(72L);
        accepted.setPhase(TaskPhase.KV_ALLOCATED);
        accepted.setInputLength(100L);
        updateStatus(Map.of("72", accepted), Map.of(), 9900L);

        var capacity = new DecodeResources.AdmissionCapacity(1L, 1L);
        var acquisition = endpoint.acquireDispatchPermit(handle, capacity);
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED, acquisition.status());
        assertNotNull(acquisition.permit(), "the handoff must retain the exact accepted reservation");
        assertEquals(TRANSFERRED, acquisition.permit().dispatch());
        assertFalse(acquisition.permit().release());
        var unused = endpoint.acquireDispatchPermit(handle, capacity);
        assertFalse(unused.permit().release(), "rollback cannot release confirmed Engine ownership");
        assertTrue(endpoint.isAcceptedByEngine(handle));
        var staleToken = new DecodeResources.ReservationHandle(
                handle.endpointGenerationId(), handle.requestId(), handle.reservationToken() + 1L);
        var staleGeneration = new DecodeResources.ReservationHandle(
                handle.endpointGenerationId() + 1L, handle.requestId(), handle.reservationToken());
        assertFalse(endpoint.isAcceptedByEngine(staleToken));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                endpoint.acquireDispatchPermit(staleToken, capacity).status());
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                endpoint.acquireDispatchPermit(staleGeneration, capacity).status());
        assertEquals(1, endpoint.routingView().engineLoad());
        assertEquals(1, endpoint.routingView().engineCapacityUsed());
        assertEquals(0L, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
    }

    @Test
    void alreadyAcceptedPermitsCannotOutliveTerminalOrRequestReplacement() {
        var handle = reserveQueued(72L, 100L, 1000L, 50);
        TaskInfo accepted = new TaskInfo();
        accepted.setRequestId(72L);
        accepted.setPhase(TaskPhase.KV_ALLOCATED);
        accepted.setInputLength(100L);
        updateStatus(Map.of("72", accepted), Map.of(), 9900L);
        var capacity = new DecodeResources.AdmissionCapacity(2L, 90L);
        var terminalCheck = endpoint.acquireDispatchPermit(handle, capacity);
        var replacementCheck = endpoint.acquireDispatchPermit(handle, capacity);
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED, terminalCheck.status());
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED, replacementCheck.status());
        assertEquals(1, endpoint.routingView().engineCapacityUsed(),
                "identity-only permits must not duplicate Engine capacity");

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(72L);
        updateStatus(Map.of(), Map.of("72", finished), 10_000L);
        assertEquals(OWNERSHIP_LOST, terminalCheck.permit().dispatch());
        assertFalse(terminalCheck.permit().release());
        assertEquals(0, endpoint.routingView().engineCapacityUsed());

        endpoint.evictExpiredRequests(-1L, ignored -> false);
        var replacement = reserveQueued(72L, 200L, 2000L, 70);
        assertNotEquals(handle.reservationToken(), replacement.reservationToken());
        updateStatus(Map.of("72", accepted), Map.of(), 9800L);
        assertTrue(endpoint.isAcceptedByEngine(replacement));
        assertEquals(OWNERSHIP_LOST, replacementCheck.permit().dispatch());
        assertFalse(replacementCheck.permit().release());
        assertEquals(1, endpoint.routingView().engineCapacityUsed());
    }

    @Test
    void permitAcquisitionCannotChargeAReplacementRequest() {
        var stale = reserveQueued(71L, 100L, 1000L, 50);
        release(71L);
        var current = reserveQueued(71L, 200L, 2000L, 50);
        var capacity = new DecodeResources.AdmissionCapacity(2L, 90L);
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                endpoint.acquireDispatchPermit(stale, capacity).status());
        assertEquals(0, endpoint.routingView().engineCapacityUsed());
        var acquired = endpoint.acquireDispatchPermit(current, capacity);
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
        assertTrue(acquired.permit().release());
    }

    @Test
    void permitHandoffPreventsLocalRollbackUntilAuthoritativeTerminal() {
        updateStatus(Map.of(), Map.of(), 10_000L);
        var handle = reserveQueued(71L, 100L, 1000L, 50);
        var acquired = endpoint.acquireDispatchPermit(handle, new DecodeResources.AdmissionCapacity(2L, 90L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
        assertEquals(TRANSFERRED, acquired.permit().dispatch());
        assertEquals(TRANSFERRED, acquired.permit().dispatch());
        assertFalse(acquired.permit().release());
        assertThrows(IllegalStateException.class,
                () -> endpoint.release(handle, DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
        assertEquals(1000L, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(71L);
        updateStatus(Map.of(), Map.of("71", finished), 10_000L);
        assertEquals(0L, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
    }

    @Test
    void invalidKvPercentageCannotReachAdmissionChecks() {
        for (long percent : new long[]{-1L, 0L, 101L}) {
            assertThrows(IllegalArgumentException.class,
                    () -> new DecodeResources.AdmissionCapacity(10L, percent));
        }
    }

    // ==================== helpers ====================

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(availableKvCacheTokens);
        EndpointTestSupport.applyStatus(endpoint, response);
    }

    private static ReentrantLock decodeAdmissionLock(
            DecodeEndpoint target) throws ReflectiveOperationException {
        java.lang.reflect.Field owner = DecodeEndpoint.class.getDeclaredField("state");
        owner.setAccessible(true);
        java.lang.reflect.Field field = DecodeState.class.getDeclaredField("admissionLock");
        field.setAccessible(true);
        return (ReentrantLock) field.get(owner.get(target));
    }

    private DecodeResources.ReservationHandle reserve(
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                throw new IllegalStateException(
                        "Decode endpoint generation is retired");
            }
            DecodeResources.ReservationHandle reservation =
                    EndpointTestSupport.reserveUnqueuedDecode(endpoint, pin, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private DecodeResources.ReservationHandle reserveQueued(
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                throw new IllegalStateException(
                        "Decode endpoint generation is retired");
            }
            DecodeResources.ReservationHandle reservation =
                    endpoint.tryReserveQueuedRequest(pin, requestId, hardKv, expectedKv, priority, null);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private void markQueued(long requestId) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            assertTrue(endpoint.markQueued(pin, reservations.get(requestId)));
        }
    }

    private void release(long requestId) {
        DecodeResources.ReservationHandle reservation =
                reservations.get(requestId);
        if (reservation != null) {
            endpoint.release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    private boolean releaseLocalShadow(long requestId) {
        DecodeResources.ReservationHandle reservation =
                reservations.get(requestId);
        return reservation != null
                && (endpoint.release(reservation, DecodeResources.ReleaseReason.COUNTERPART_FINISHED) == ReservationReleaseResult.RELEASED);
    }

    private Map<Long, DecodeResources.DecodeRequestView> reserved() {
        return endpoint.resourceSnapshot().requests().entrySet().stream()
                .filter(entry -> !entry.getValue().phase().isEngineConfirmed())
                .collect(java.util.stream.Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    private static void assertReservationIdentity(
            DecodeResources.DecodeRequestView expected,
            DecodeResources.DecodeRequestView actual) {
        assertReservationIdentity(expected, actual, "reservation identity changed");
    }

    private static void assertReservationIdentity(
            DecodeResources.DecodeRequestView expected,
            DecodeResources.DecodeRequestView actual,
            String message) {
        assertNotNull(actual, message);
        assertEquals(expected.requestId(), actual.requestId(), message);
        assertEquals(expected.reservationToken(), actual.reservationToken(), message);
    }

    private List<DecodeResources.ReservationHandle> handles(long... ids) {
        return LongStream.of(ids).mapToObj(reservations::get).toList();
    }

    private DecodeResources.PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<Long> victimIds,
            long incomingRequestId,
            long hardKv,
            long expectedKv,
            int priority) {
        return endpoint.beginPreemption(attemptToken, victimIds.stream().map(reservations::get).toList(), incomingRequestId, hardKv, expectedKv, priority, new DecodeResources.AdmissionCapacity(
                        Math.max(1, endpoint.routingView().totalLoad()), 100));
    }

    private void settleFromWorkerStatus(long requestId) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(requestId);
        finished.setErrorCode(0);
        updateStatus(Map.of(), Map.of(Long.toString(requestId), finished),
                Math.max(10_000L,
                        status.getAvailableKvCacheTokens()));
    }

    private DecodeEndpoint.EngineDispatchPermit acquirePermit(
            long requestId, long concurrencyLimit) {
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireDispatchPermit(reservations.get(requestId), new DecodeResources.AdmissionCapacity(concurrencyLimit, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                acquisition.status());
        assertNotNull(acquisition.permit());
        return acquisition.permit();
    }

    private static void await(CountDownLatch latch, String timeoutMessage) {
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                throw new AssertionError(timeoutMessage);
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError("latch wait interrupted", interrupted);
        }
    }
    @Test
    void physicalAndExpectedDeficitsRequireDifferentVictimReleases() {
        var capacity = new DecodeResources.AdmissionCapacity(0L, 90L);
        var usage = new DecodeResources.CapacityUsage(2L, 1000L, 150L, 200L, 850L);
        assertEquals(new DecodeResources.CapacityDeficit(0L, 150L, 150L),
                capacity.evaluate(usage, 100L, 200L, CapacityRelease.NONE));
        assertEquals(new DecodeResources.CapacityDeficit(0L, 100L, 0L),
                capacity.evaluate(usage, 100L, 200L, new DecodeResources.CapacityRelease(1L, 50L, 200L)));
        assertTrue(capacity.evaluate(usage, 100L, 200L,
                new DecodeResources.CapacityRelease(2L, 150L, 250L)).fits());
    }

    @Test
    void removingEveryVictimCannotMakeAnOversizedOutputAllowanceFit() {
        var capacity = new DecodeResources.AdmissionCapacity(0L, 90L);
        var usage = new DecodeResources.CapacityUsage(1L, 1000L, 500L, 0L, 500L);
        assertFalse(capacity.evaluate(usage, 100L, 901L,
                new DecodeResources.CapacityRelease(1L, 500L, 500L)).fits());
    }

    @Test
    void overflowingUsageCannotDisappearIntoASaturatedMaximumBudget() {
        var capacity = new DecodeResources.AdmissionCapacity(Long.MAX_VALUE, 100L);
        var usage = new DecodeResources.CapacityUsage(Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE,
                0L, Long.MAX_VALUE);
        assertEquals(new DecodeResources.CapacityDeficit(1L, 0L, 1L), capacity.evaluate(usage, 0L, 1L, CapacityRelease.NONE));
        assertEquals(Long.MAX_VALUE / 100L * 90L + Long.MAX_VALUE % 100L * 90L / 100L,
                new DecodeResources.AdmissionCapacity(0L, 90L).kvBudget(Long.MAX_VALUE));
    }

    @Test
    void validKvPercentagePreservesUnknownCapacityBehavior() {
        var usage = new DecodeResources.CapacityUsage(0L, 0L, 0L, 0L, 0L);
        for (long percent : new long[]{1L, 90L, 100L}) {
            var capacity = new DecodeResources.AdmissionCapacity(0L, percent);
            assertTrue(capacity.evaluate(usage, 100L, 200L, CapacityRelease.NONE).fits());
            assertEquals(percent, capacity.kvBudget(100L));
        }
    }

}
