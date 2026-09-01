package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.LongStream;

import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
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
    private final Map<Long, DecodeEndpoint.ReservationHandle> reservations =
            new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.1", 8080, 8081);
        endpoint = new DecodeEndpoint(
                status, EndpointTestSupport.noopEventSink());
    }

    // ==================== realKvAvailable = reported - hard reservations ====================

    @Test
    void realKvAvailable_subtractsHardNotExpectedReservations() {
        updateStatus(null, null, 10_000);
        reserve(1L, 500, 600, 70);

        // Hard (500), not expected (600), is subtracted from the report.
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(500, endpoint.inflightHardKvReserved());
        assertEquals(600, endpoint.routingView().inflightExpectedKv());

        RequestInflight entry = reserved().get(1L);
        assertEquals(70, entry.priority());
        assertEquals(DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN, entry.phase());
    }

    // ==================== reserve / release bump the admission version ====================

    @Test
    void reserveAndRelease_bumpVersion_andReverseShadowAccounting() {
        long v0 = endpoint.admissionVersion();

        reserve(1L, 500, 600, 30);
        assertEquals(v0 + 1, endpoint.admissionVersion());
        assertEquals(1, endpoint.getTotalLoad());

        release(1L);
        assertEquals(v0 + 2, endpoint.admissionVersion());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.routingView().inflightExpectedKv());
    }

    @Test
    void everyExactReservationReleaseSignalsPlacementCapacity() {
        PlacementAvailability availability =
                mock(PlacementAvailability.class);
        DecodeEndpoint exactEndpoint = new DecodeEndpoint(
                status, mock(EndpointEventSink.class), availability);

        DecodeEndpoint.ReservationHandle speculative;
        try (WorkerEndpoint.GenerationPin pin =
                     exactEndpoint.tryPinGeneration()) {
            assertNotNull(pin);
            speculative = exactEndpoint.reserveQueuedPinned(
                    pin, 11L, 100L, 110L, 10);
        }
        exactEndpoint.rollbackExact(speculative);
        verify(availability).capacityChanged(
                RoleType.DECODE, null);

        DecodeEndpoint.ReservationHandle published;
        try (WorkerEndpoint.GenerationPin pin =
                     exactEndpoint.tryPinGeneration()) {
            assertNotNull(pin);
            published = exactEndpoint.reserveQueuedPinned(
                    pin, 12L, 100L, 110L, 10);
        }
        exactEndpoint.releasePlacementExact(published);
        verify(availability, times(2)).capacityChanged(
                RoleType.DECODE, null);
    }

    @Test
    void conditionalOrphanReleasePreservesReplacementReservation() {
        long requestId = 2L;
        DecodeEndpoint.ReservationHandle stale =
                reserve(requestId, 100, 110, 30);
        RequestInflight staleSnapshot = reserved().get(requestId);
        release(requestId);

        DecodeEndpoint.ReservationHandle current =
                reserve(requestId, 200, 220, 70);
        RequestInflight replacement = reserved().get(requestId);
        assertNotSame(staleSnapshot, replacement);

        assertFalse(endpoint.releaseLocalShadowIfExact(stale));
        assertSame(replacement, reserved().get(requestId));
        assertEquals(200, endpoint.inflightHardKvReserved());
        assertEquals(220, endpoint.routingView().inflightExpectedKv());

        assertTrue(endpoint.releaseLocalShadowIfExact(current));
        assertFalse(reserved().containsKey(requestId));
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.routingView().inflightExpectedKv());
    }

    // ==================== atomic release+reserve: success ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_success_appliesAtomically() {
        updateStatus(Map.of(), Map.of(), 700L);
        reserve(1L, 100, 110, 30);
        reserve(2L, 200, 220, 40);
        markQueued(1L);
        markQueued(2L);
        long version = endpoint.admissionVersion();

        DecodeEndpoint.LocalEvictionResult result =
                endpoint.tryEvictLocalReservationsAndReserveIncoming(
                        handles(1L, 2L), 9L, 700, 708, 70,
                        new DecodeEndpoint.AdmissionCapacity(2, 0));

        assertEquals(DecodeEndpoint.LocalEvictionResult.COMMITTED, result);
        assertFalse(reserved().containsKey(1L));
        assertFalse(reserved().containsKey(2L));
        assertEquals(70, reserved().get(9L).priority());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(700, endpoint.inflightHardKvReserved());
        assertTrue(endpoint.admissionVersion() > version);
    }

    // ==================== atomic release+reserve: validation failures apply nothing ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_identityMismatch_appliesNothing() {
        DecodeEndpoint.ReservationHandle exact =
                reserve(1L, 100, 110, 30);
        markQueued(1L);
        DecodeEndpoint.ReservationHandle stale =
                new DecodeEndpoint.ReservationHandle(
                        exact.endpointGenerationId(),
                        exact.requestId(),
                        exact.reservationToken() + 1L);
        long version = endpoint.admissionVersion();

        DecodeEndpoint.LocalEvictionResult result =
                endpoint.tryEvictLocalReservationsAndReserveIncoming(
                        List.of(stale), 9L, 700, 708, 70,
                        new DecodeEndpoint.AdmissionCapacity(1, 0));

        assertEquals(DecodeEndpoint.LocalEvictionResult.CONFLICT, result);
        assertTrue(reserved().containsKey(1L));
        assertFalse(reserved().containsKey(9L));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(version, endpoint.admissionVersion());
    }

    @Test
    void tryReleaseVictimsAndReserveIncoming_victimGone_appliesNothing() {
        DecodeEndpoint.ReservationHandle exact =
                reserve(1L, 100, 110, 30);
        markQueued(1L);
        long version = endpoint.admissionVersion();
        DecodeEndpoint.ReservationHandle absent =
                new DecodeEndpoint.ReservationHandle(
                        exact.endpointGenerationId(), 42L,
                        exact.reservationToken() + 1L);

        DecodeEndpoint.LocalEvictionResult result =
                endpoint.tryEvictLocalReservationsAndReserveIncoming(
                        List.of(exact, absent), 9L, 700, 708, 70,
                        new DecodeEndpoint.AdmissionCapacity(1, 0));

        assertEquals(DecodeEndpoint.LocalEvictionResult.CONFLICT, result);
        assertTrue(reserved().containsKey(1L));
        assertFalse(reserved().containsKey(9L));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(version, endpoint.admissionVersion());
    }

    @Test
    void engineDispatchPermitAndLocalEvictionHaveOneAdmissionLockWinner() {
        // Eviction wins: it removes the reservation, so the batch item must be
        // skipped before acquiring an engine-dispatch permit / gRPC publication.
        reserve(1L, 100, 110, 30);
        markQueued(1L);
        assertTrue(releaseLocalShadow(1L));
        DecodeEndpoint.EngineDispatchPermitAcquisition released =
                endpoint.acquireEngineDispatchPermit(1L, 5);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                released.status());
        assertNull(released.permit());

        // Dispatch wins: commit atomically clears queued ownership while retaining
        // the reservation, so a later local-eviction attempt cannot touch it.
        reserve(2L, 100, 110, 30);
        markQueued(2L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(2L, 5);
        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        assertFalse(releaseLocalShadow(2L));
        assertTrue(reserved().containsKey(2L));

        // Engine-facing reservations are not eligible for a pre-delivery permit.
        reserve(3L, 100, 110, 30);
        DecodeEndpoint.EngineDispatchPermitAcquisition engineFacing =
                endpoint.acquireEngineDispatchPermit(3L, 5);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_QUEUED,
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
        assertEquals(TRANSFERRED, first.transferToEngineLifecycle());
        List<DecodeEndpoint.EngineDispatchPermitAcquireStatus> results =
                LongStream.rangeClosed(2, 20)
                        .mapToObj(requestId -> endpoint
                                .acquireEngineDispatchPermit(requestId, 5).status())
                        .toList();

        assertTrue(results.stream().allMatch(result ->
                result == DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL));
        assertEquals(5, endpoint.getEngineLoad());
        assertEquals(19, endpoint.layeredAdmissionView().queued().size(),
                "capacity-blocked reservations must remain queued");
    }

    @Test
    void engineDispatchPermit_preservesUnlimitedAndRejectsNonQueuedReservations() {
        reserve(1L, 100, 110, 30);
        reserve(2L, 100, 110, 30);
        markQueued(1L);
        markQueued(2L);

        assertEquals(TRANSFERRED,
                acquirePermit(1L, 0).transferToEngineLifecycle());
        assertEquals(TRANSFERRED,
                acquirePermit(2L, 0).transferToEngineLifecycle());

        // A non-queued reservation is already engine-facing and has no
        // pre-delivery ownership transition to reserve.
        reserve(3L, 100, 110, 30);
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireEngineDispatchPermit(3L, 1);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_QUEUED,
                acquisition.status());
        assertNull(acquisition.permit());
    }

    @Test
    void queuedKvIsSoftUntilDispatchPermit_thenKvLimitIsAuthoritative() {
        updateStatus(null, null, 1_000);

        reserveQueued(1L, 400, 900, 50);

        assertEquals(900, endpoint.realKvUsed(),
                "placement scoring must retain queued expected KV");
        assertEquals(0, endpoint.engineFacingKvUsed(),
                "queued expected KV must not poison the dispatch gate");
        assertEquals(600, endpoint.realKvAvailable());
        assertEquals(1_000, endpoint.engineFacingKvAvailable());

        DecodeEndpoint.EngineDispatchPermitAcquisition first =
                endpoint.acquireEngineDispatchPermit(1L, 256, 90);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
                first.status());
        assertNotNull(first.permit());
        assertTrue(endpoint.layeredAdmissionView().queued().contains(1L),
                "capacity is occupied before queued ownership is transferred");
        assertEquals(0, endpoint.getEngineLoad(),
                "a pre-delivery permit is not engine-facing load");

        reserveQueued(2L, 100, 100, 50);
        DecodeEndpoint.EngineDispatchPermitAcquisition second =
                endpoint.acquireEngineDispatchPermit(2L, 256, 90);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                second.status(),
                "a permit already occupying the 90% KV fence must prevent oversubscription");
        assertNull(second.permit());
        assertTrue(endpoint.layeredAdmissionView().queued().contains(2L));
        assertEquals(900, endpoint.engineFacingKvUsed(),
                "the failed candidate must not add KV beyond the first acquired permit");

        assertEquals(TRANSFERRED, first.permit().transferToEngineLifecycle());
        assertEquals(900, endpoint.engineFacingKvUsed());
        assertEquals(600, endpoint.engineFacingKvAvailable());
    }

    @Test
    void engineDispatchPermit_reportsNotOwnedAfterReleaseOrPreemptionClaim() {
        reserve(1L, 100, 110, 30);
        markQueued(1L);
        release(1L);
        DecodeEndpoint.EngineDispatchPermitAcquisition released =
                endpoint.acquireEngineDispatchPermit(1L, 5);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                released.status());
        assertNull(released.permit());

        reserve(2L, 100, 110, 30);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(
                        101L, List.of(2L), 9L, 100, 110, 70));
        DecodeEndpoint.EngineDispatchPermitAcquisition preempted =
                endpoint.acquireEngineDispatchPermit(2L, 5);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                preempted.status());
        assertNull(preempted.permit());
    }

    @Test
    void engineDispatchPermit_capacityFullLeavesReservationQueued() {
        reserve(100L, 100, 110, 30);
        reserve(1L, 100, 110, 50);
        markQueued(1L);
        long versionBeforeAcquire = endpoint.admissionVersion();

        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireEngineDispatchPermit(1L, 1);

        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                acquisition.status());
        assertNull(acquisition.permit());
        assertEquals(versionBeforeAcquire, endpoint.admissionVersion());
        assertTrue(endpoint.layeredAdmissionView().queued().contains(1L));
        assertEquals(1, endpoint.getEngineLoad());
    }

    @Test
    void engineDispatchPermit_occupiesHardGateWithoutChangingEngineLoad() {
        reserve(1L, 100, 110, 50);
        reserve(2L, 100, 110, 50);
        markQueued(1L);
        markQueued(2L);

        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(1L, 1);
        DecodeEndpoint.EngineDispatchPermitAcquisition duplicate =
                endpoint.acquireEngineDispatchPermit(1L, 1);
        DecodeEndpoint.EngineDispatchPermitAcquisition second =
                endpoint.acquireEngineDispatchPermit(2L, 1);

        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED,
                duplicate.status());
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                second.status());
        assertEquals(0, endpoint.getEngineLoad(),
                "a pre-delivery permit is not engine-facing load");
        assertEquals(Set.of(1L, 2L), endpoint.layeredAdmissionView().queued());
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
        assertEquals(OWNERSHIP_LOST, first.transferToEngineLifecycle());

        DecodeEndpoint.EngineDispatchPermit second = acquirePermit(2L, 1);
        assertEquals(Set.of(1L, 2L), endpoint.layeredAdmissionView().queued());
        assertTrue(second.release());
    }

    @Test
    void queuedDispatchTransactionPublishesReservationAndPermitTogether() {
        updateStatus(Map.of(), Map.of(), 10_000L);

        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition;
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            acquisition = endpoint.tryAcquireQueuedEngineDispatchPermitPinned(
                    pin, 71L, 100L, 110L, 50, 1L, 100L);
        }

        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
                acquisition.status());
        assertEquals(acquisition.reservation(),
                endpoint.reservationHandle(71L));
        assertTrue(endpoint.layeredAdmissionView().queued().contains(71L));
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED,
                endpoint.acquireEngineDispatchPermit(71L, 1L).status());

        assertTrue(acquisition.permit().release());
        endpoint.rollbackExact(acquisition.reservation());
        assertNull(endpoint.reservationHandle(71L));
        assertTrue(endpoint.layeredAdmissionView().queued().isEmpty());
        assertEquals(0L, endpoint.inflightHardKvReserved());
    }

    @Test
    void rejectedQueuedDispatchTransactionPublishesNothing() {
        reserve(90L, 100L, 110L, 50);
        assertEquals(1, endpoint.getEngineLoad());

        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition;
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            acquisition = endpoint.tryAcquireQueuedEngineDispatchPermitPinned(
                    pin, 91L, 100L, 110L, 50, 1L, 100L);
        }

        assertEquals(
                DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                acquisition.status());
        assertNull(acquisition.reservation());
        assertNull(acquisition.permit());
        assertNull(endpoint.reservationHandle(91L));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(91L));
    }

    @Test
    void closedEndpointRejectsPermitAcquisitionAsRetired() {
        endpoint.close();

        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireEngineDispatchPermit(1L, 1);

        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ENDPOINT_RETIRED,
                acquisition.status());
        assertNull(acquisition.permit());
    }

    @Test
    void closeBeforeReservationRejectsEveryNewOwnershipEntryPoint() {
        DecodeEndpoint.ReservationHandle stale =
                reserve(10L, 10, 10, 1);
        release(10L);
        endpoint.close();

        assertNull(endpoint.tryPinGeneration());
        assertEquals(DecodeEndpoint.LocalEvictionResult.ENDPOINT_RETIRED,
                endpoint.tryEvictLocalReservationsAndReserveIncoming(
                        List.of(stale), 3L, 100, 110, 50,
                        new DecodeEndpoint.AdmissionCapacity(1, 0)));
        assertEquals(DecodeEndpoint.PreemptionBeginResult.ENDPOINT_RETIRED,
                endpoint.beginPriorityPreemption(
                        101L, List.of(stale), 5L, 100, 110, 50,
                        new DecodeEndpoint.AdmissionCapacity(1, 0)));

        assertTrue(reserved().isEmpty());
        assertEquals(0L, endpoint.inflightHardKvReserved());
        assertEquals(0L, endpoint.routingView().inflightExpectedKv());
    }

    @Test
    void reserveQueuedBeforeCloseRetiresUnpublishedOwnership() {
        DecodeEndpoint.ReservationHandle reservation = reserveQueued(
                1L, 100, 110, 50);
        assertEquals(reservation, endpoint.reservationHandle(1L));

        endpoint.close();

        assertNull(endpoint.reservationHandle(1L));
        assertTrue(endpoint.layeredAdmissionView().queued().isEmpty());
        assertEquals(0L, endpoint.inflightHardKvReserved());
        assertEquals(0L, endpoint.routingView().inflightExpectedKv());
    }

    @Test
    void closeBetweenPermitAcquisitionAndTransferRetiresExactPermitAndReleasesHardGate()
            throws InterruptedException {
        reserve(1L, 100, 110, 50);
        reserve(2L, 100, 110, 50);
        markQueued(1L);
        markQueued(2L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(1L, 1);
        assertFalse(endpoint.isEngineDispatchPermitAvailable(2L, 1, -1L));

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
        assertEquals(ENDPOINT_RETIRED, permit.transferToEngineLifecycle());
        assertEquals(ENDPOINT_RETIRED, permit.transferToEngineLifecycle(),
                "the exact retired permit keeps its typed terminal result");
        assertTrue(endpoint.isEngineDispatchPermitAvailable(2L, 1, -1L),
                "close must remove the outstanding permit from hard-gate usage");
        assertEquals(0, endpoint.getEngineLoad());
        assertTrue(endpoint.layeredAdmissionView().queued().isEmpty(),
                "retirement must release queued ownership not yet transferred to the Engine");
    }

    @Test
    void closePreservesTransferredPermitOutcomeWhileRetiringEndpointOwnership() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        assertEquals(1, endpoint.getEngineLoad());

        endpoint.close();

        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        assertNull(endpoint.reservationHandle(requestId));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(0, endpoint.getEngineLoad(),
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
        assertEquals(1, endpoint.getEngineLoad());

        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle(),
                "ownership transfer must be idempotent after handoff");
        assertFalse(permit.release());
        assertFalse(endpoint.layeredAdmissionView().queued().contains(1L));
        assertEquals(2, endpoint.getEngineLoad());
    }

    @Test
    void committedPermitCannotReturnEngineOwnershipToQueuedState() {
        reserve(1L, 100, 110, 50);
        reserve(2L, 100, 110, 50);
        markQueued(1L);
        markQueued(2L);
        RequestInflight firstReservation = reserved().get(1L);
        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(1L, 1);

        assertEquals(TRANSFERRED, first.transferToEngineLifecycle());
        assertSame(firstReservation, reserved().get(1L));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(1L));
        assertEquals(1, endpoint.getEngineLoad());
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                endpoint.acquireEngineDispatchPermit(2L, 1).status());

        assertFalse(first.release(),
                "committed Decode ownership is irreversible through the permit");
        assertSame(firstReservation, reserved().get(1L));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(1L));
        assertEquals(1, endpoint.getEngineLoad());

        settleFromWorkerStatus(1L);
        DecodeEndpoint.EngineDispatchPermit second = acquirePermit(2L, 1);
        assertTrue(second.release());
    }

    @Test
    void committedPermitCannotAffectLaterDispatchRoundOnSameReservation() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        RequestInflight reservation = reserved().get(requestId);

        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, first.transferToEngineLifecycle());
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(1, endpoint.getEngineLoad());

        markQueued(requestId);
        assertSame(reservation, reserved().get(requestId),
                "the second dispatch round must reuse the same reservation");
        DecodeEndpoint.EngineDispatchPermit second = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, second.transferToEngineLifecycle());
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(1, endpoint.getEngineLoad());
        long versionBeforeStaleRelease = endpoint.admissionVersion();

        assertFalse(first.release(),
                "an older committed round must not change the current dispatch round");
        assertSame(reservation, reserved().get(requestId));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(1, endpoint.getEngineLoad(),
                "the stale token must not release current engine ownership");
        assertEquals(versionBeforeStaleRelease, endpoint.admissionVersion());

        assertFalse(second.release(),
                "the current committed round is also irreversible through its permit");
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(1, endpoint.getEngineLoad());
    }

    @Test
    void markQueuedPhaseBumpsAdmissionVersionOnlyWhenOwnershipActuallyChanges() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        long versionBeforeFirstMark = endpoint.admissionVersion();

        markQueued(requestId);
        assertEquals(versionBeforeFirstMark + 1, endpoint.admissionVersion());
        assertTrue(endpoint.layeredAdmissionView().queued().contains(requestId));

        markQueued(requestId);
        assertEquals(versionBeforeFirstMark + 1, endpoint.admissionVersion(),
                "repeating an existing queued mark must be a no-op");

        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        long versionBeforeSecondRound = endpoint.admissionVersion();

        markQueued(requestId);
        assertEquals(versionBeforeSecondRound + 1, endpoint.admissionVersion());
        markQueued(requestId);
        assertEquals(versionBeforeSecondRound + 1, endpoint.admissionVersion(),
                "repeating the second-round queued mark must also be a no-op");
    }

    @Test
    void staleCommittedPermitCannotAffectReplacementGeneration() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        RequestInflight original = reserved().get(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, stale.transferToEngineLifecycle());

        settleFromWorkerStatus(requestId);
        reserve(requestId, 200, 220, 70);
        markQueued(requestId);
        RequestInflight replacement = reserved().get(requestId);
        assertNotSame(original, replacement);
        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);

        assertFalse(stale.release(),
                "a committed old generation must not change its replacement");
        assertFalse(stale.release(), "a stale release must stay idempotent");
        assertSame(replacement, reserved().get(requestId));
        assertTrue(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED,
                endpoint.acquireEngineDispatchPermit(requestId, 1).status());

        assertEquals(TRANSFERRED, current.transferToEngineLifecycle(),
                "the stale token must not release or invalidate the new permit");
        assertSame(replacement, reserved().get(requestId));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
    }

    @Test
    void staleEngineDispatchPermitCannotAffectReplacementGeneration() {
        long requestId = 1L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        RequestInflight original = reserved().get(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);

        release(requestId);
        reserve(requestId, 200, 220, 70);
        markQueued(requestId);
        RequestInflight replacement = reserved().get(requestId);
        assertNotSame(original, replacement);

        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);
        assertFalse(stale.release(), "the old token must not remove the new permit");
        assertEquals(OWNERSHIP_LOST, stale.transferToEngineLifecycle());
        assertSame(replacement, reserved().get(requestId));
        assertTrue(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(TRANSFERRED, current.transferToEngineLifecycle(),
                "the replacement generation keeps its permit");
        assertSame(replacement, reserved().get(requestId));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
    }

    @Test
    void staleEngineDispatchPermitCommitFailsAfterDirectReplacement() {
        long requestId = 2L;
        reserve(requestId, 100, 110, 50);
        markQueued(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);

        release(requestId);
        reserve(requestId, 200, 220, 70);
        markQueued(requestId);
        RequestInflight replacement = reserved().get(requestId);
        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);

        assertEquals(OWNERSHIP_LOST, stale.transferToEngineLifecycle());
        assertSame(replacement, reserved().get(requestId));
        assertTrue(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(TRANSFERRED, current.transferToEngineLifecycle());
    }

    @Test
    void calibrateInvalidatesPermitWithoutLeakingHardGateCapacity() {
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
        assertEquals(OWNERSHIP_LOST, stale.transferToEngineLifecycle());
        assertFalse(stale.release());
        assertTrue(current.release());
    }

    @Test
    void ttlEvictionInvalidatesPermitWithoutLeakingHardGateCapacity() {
        reserve(1L, 100, 110, 50);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(1L, 1);

        assertEquals(1, endpoint.evictExpiredRequests(
                -1, requestId -> false));
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
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
    }

    @Test
    void versionedReceivedTaskEmitsActivityWithoutAdvancingAcceptance() {
        DecodeEndpoint.ReservationHandle reservation =
                reserve(1L, 500, 508, 30);
        TaskInfo received = new TaskInfo();
        received.setRequestId(1L);
        received.setPhase(TaskPhase.RECEIVED);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of("1", received));
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);

        DecodeEndpoint.StatusReduction reduction =
                (DecodeEndpoint.StatusReduction)
                        EndpointTestSupport.applyStatus(endpoint, response);

        assertEquals(1, reduction.facts().size());
        DecodeEndpoint.WorkerStatusFact active = reduction.facts().getFirst();
        assertEquals(DecodeEndpoint.WorkerStatusFact.Kind.ACTIVE, active.kind());
        assertEquals(reservation, active.reservation());
        assertEquals(reservation, endpoint.reservationHandle(1L));
    }

    @Test
    void statusFieldApplicationAndCalibrationExcludeConcurrentReserve()
            throws Exception {
        WorkerStatus blockingStatus = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.2", 8080, 8081);
        DecodeEndpoint blockingEndpoint = new DecodeEndpoint(
                blockingStatus, EndpointTestSupport.noopEventSink());
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
                    blockingEndpoint.reservePinned(
                            pin, 71L, 500L, 600L, 80);
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

            assertNotNull(blockingEndpoint.reservationHandle(71L));
            assertEquals(500L, blockingEndpoint.inflightHardKvReserved());
            assertEquals(600L,
                    blockingEndpoint.routingView().inflightExpectedKv());
            assertEquals(9_500L, blockingEndpoint.realKvAvailable(),
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
        java.lang.reflect.Field field =
                DecodeEndpoint.class.getDeclaredField("admissionLock");
        field.setAccessible(true);
        return (ReentrantLock) field.get(target);
    }

    private DecodeEndpoint.ReservationHandle reserve(
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                throw new EndpointGenerationRetiredException(
                        "Decode endpoint generation is retired");
            }
            DecodeEndpoint.ReservationHandle reservation =
                    endpoint.reservePinned(
                            pin, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private DecodeEndpoint.ReservationHandle reserveQueued(
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                throw new EndpointGenerationRetiredException(
                        "Decode endpoint generation is retired");
            }
            DecodeEndpoint.ReservationHandle reservation =
                    endpoint.reserveQueuedPinned(
                            pin, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private void markQueued(long requestId) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            assertTrue(endpoint.markQueuedExact(
                    pin, reservations.get(requestId))
                    != DecodeEndpoint.MarkQueuedResult.NOT_OWNED);
        }
    }

    private void release(long requestId) {
        DecodeEndpoint.ReservationHandle reservation =
                reservations.get(requestId);
        if (reservation != null) {
            endpoint.rollbackExact(reservation);
        }
    }

    private boolean releaseLocalShadow(long requestId) {
        DecodeEndpoint.ReservationHandle reservation =
                reservations.get(requestId);
        return reservation != null
                && endpoint.releaseLocalShadowIfExact(reservation);
    }

    private Map<Long, RequestInflight> reserved() {
        return endpoint.layeredAdmissionView().reserved();
    }

    private List<DecodeEndpoint.ReservationHandle> handles(long... ids) {
        return LongStream.of(ids).mapToObj(reservations::get).toList();
    }

    private DecodeEndpoint.PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<Long> victimIds,
            long incomingRequestId,
            long hardKv,
            long expectedKv,
            int priority) {
        return endpoint.beginPriorityPreemption(
                attemptToken,
                victimIds.stream().map(reservations::get).toList(),
                incomingRequestId,
                hardKv,
                expectedKv,
                priority,
                new DecodeEndpoint.AdmissionCapacity(
                        Math.max(1, endpoint.getTotalLoad()), 0));
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
                endpoint.acquireEngineDispatchPermit(requestId, concurrencyLimit);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
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
}
