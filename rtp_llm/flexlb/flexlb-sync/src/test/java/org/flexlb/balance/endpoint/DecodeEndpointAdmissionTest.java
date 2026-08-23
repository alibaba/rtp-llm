package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.LongStream;

import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Phase 4 tests for the decode admission state of {@link DecodeEndpoint}:
 * priority-carrying reservations, the admission version, the atomic
 * release-victims-and-reserve-incoming commit (all-or-nothing, design doc
 * 11.5/17.2), and the reserved-only view after calibrate (10.1).
 */
class DecodeEndpointAdmissionTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        endpoint = new DecodeEndpoint(status);
    }

    // ==================== realKvAvailable = reported - hard reservations ====================

    @Test
    void realKvAvailable_subtractsHardNotExpectedReservations() {
        updateStatus(null, null, 10_000);
        endpoint.reserve(1L, 500, 600, 70);

        // Hard (500), not expected (600), is subtracted from the report.
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(500, endpoint.inflightHardKvReserved());
        assertEquals(600, endpoint.inflightExpectedKvReserved());

        RequestInflight entry = endpoint.reservedView().get(1L);
        assertEquals(70, entry.priority());
        assertEquals(DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN, entry.phase());
    }

    // ==================== reserve / release bump the admission version ====================

    @Test
    void reserveAndRelease_bumpVersion_andReverseShadowAccounting() {
        long v0 = endpoint.admissionVersion();

        endpoint.reserve(1L, 500, 600, 30);
        assertEquals(v0 + 1, endpoint.admissionVersion());
        assertEquals(1, endpoint.getTotalLoad());

        endpoint.release(1L);
        assertEquals(v0 + 2, endpoint.admissionVersion());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightExpectedKvReserved());
    }

    @Test
    void conditionalOrphanReleasePreservesReplacementReservation() {
        long requestId = 2L;
        endpoint.reserve(requestId, 100, 110, 30);
        RequestInflight staleSnapshot = endpoint.reservedView().get(requestId);

        endpoint.reserve(requestId, 200, 220, 70);
        RequestInflight replacement = endpoint.reservedView().get(requestId);
        assertNotSame(staleSnapshot, replacement);

        assertFalse(endpoint.releaseReservationIfCurrent(
                requestId, staleSnapshot));
        assertSame(replacement, endpoint.reservedView().get(requestId));
        assertEquals(200, endpoint.inflightHardKvReserved());
        assertEquals(220, endpoint.inflightExpectedKvReserved());

        assertTrue(endpoint.releaseReservationIfCurrent(
                requestId, replacement));
        assertFalse(endpoint.reservedView().containsKey(requestId));
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightExpectedKvReserved());
    }

    // ==================== atomic release+reserve: success ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_success_appliesAtomically() {
        endpoint.reserve(1L, 100, 110, 30);
        endpoint.reserve(2L, 200, 220, 40);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);
        long version = endpoint.admissionVersion();

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of(1L, 2L), 9L, 700, 708, 70, version);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.SUCCESS, result);
        assertFalse(endpoint.reservedView().containsKey(1L));
        assertFalse(endpoint.reservedView().containsKey(2L));
        assertEquals(70, endpoint.reservedView().get(9L).priority());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(700, endpoint.inflightHardKvReserved());
        assertTrue(endpoint.admissionVersion() > version);
    }

    // ==================== atomic release+reserve: validation failures apply nothing ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_versionMismatch_appliesNothing() {
        endpoint.reserve(1L, 100, 110, 30);
        long staleVersion = endpoint.admissionVersion() - 1;

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of(1L), 9L, 700, 708, 70, staleVersion);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.VERSION_MISMATCH, result);
        assertTrue(endpoint.reservedView().containsKey(1L));
        assertFalse(endpoint.reservedView().containsKey(9L));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(staleVersion + 1, endpoint.admissionVersion());
    }

    @Test
    void tryReleaseVictimsAndReserveIncoming_victimGone_appliesNothing() {
        endpoint.reserve(1L, 100, 110, 30);
        endpoint.markQueuedPhase(1L);
        long version = endpoint.admissionVersion();

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of(1L, 42L), 9L, 700, 708, 70, version);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE, result);
        assertTrue(endpoint.reservedView().containsKey(1L));
        assertFalse(endpoint.reservedView().containsKey(9L));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(version, endpoint.admissionVersion());
    }

    @Test
    void engineDispatchPermitAndLocalEvictionHaveOneAdmissionLockWinner() {
        // Eviction wins: it removes the reservation, so the batch item must be
        // skipped before acquiring an engine-dispatch permit / gRPC publication.
        endpoint.reserve(1L, 100, 110, 30);
        endpoint.markQueuedPhase(1L);
        assertTrue(endpoint.releaseIfHeld(1L));
        DecodeEndpoint.EngineDispatchPermitAcquisition released =
                endpoint.acquireEngineDispatchPermit(1L, 5);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                released.status());
        assertNull(released.permit());

        // Dispatch wins: commit atomically clears queued ownership while retaining
        // the reservation, so a later local-eviction attempt cannot touch it.
        endpoint.reserve(2L, 100, 110, 30);
        endpoint.markQueuedPhase(2L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(2L, 5);
        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        assertFalse(endpoint.releaseIfHeld(2L));
        assertTrue(endpoint.reservedView().containsKey(2L));

        // Engine-facing reservations are not eligible for a pre-delivery permit.
        endpoint.reserve(3L, 100, 110, 30);
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
            endpoint.reserve(requestId, 100, 110, 30);
        }
        // A single Prefill batch may contain many reservations which are all
        // deliberately invisible to getEngineLoad while still queued.
        for (long requestId = 1; requestId <= 20; requestId++) {
            endpoint.reserve(requestId, 100, 110, 50);
            endpoint.markQueuedPhase(requestId);
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
        endpoint.reserve(1L, 100, 110, 30);
        endpoint.reserve(2L, 100, 110, 30);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);

        assertEquals(TRANSFERRED,
                acquirePermit(1L, 0).transferToEngineLifecycle());
        assertEquals(TRANSFERRED,
                acquirePermit(2L, 0).transferToEngineLifecycle());

        // A non-queued reservation is already engine-facing and has no
        // pre-delivery ownership transition to reserve.
        endpoint.reserve(3L, 100, 110, 30);
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireEngineDispatchPermit(3L, 1);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_QUEUED,
                acquisition.status());
        assertNull(acquisition.permit());
    }

    @Test
    void queuedKvIsSoftUntilDispatchPermit_thenKvLimitIsAuthoritative() {
        status.getTotalKvCacheTokens().set(1_000);
        updateStatus(null, null, 1_000);

        endpoint.reserveQueued(1L, 400, 900, 50);

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

        endpoint.reserveQueued(2L, 100, 100, 50);
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
        endpoint.reserve(1L, 100, 110, 30);
        endpoint.markQueuedPhase(1L);
        endpoint.release(1L);
        DecodeEndpoint.EngineDispatchPermitAcquisition released =
                endpoint.acquireEngineDispatchPermit(1L, 5);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                released.status());
        assertNull(released.permit());

        endpoint.reserve(2L, 100, 110, 30);
        long version = endpoint.admissionVersion();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(
                        101L, List.of(2L), 9L, 100, 110,
                        70, version, true));
        DecodeEndpoint.EngineDispatchPermitAcquisition preempted =
                endpoint.acquireEngineDispatchPermit(2L, 5);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.NOT_OWNED,
                preempted.status());
        assertNull(preempted.permit());
    }

    @Test
    void engineDispatchPermit_capacityFullLeavesReservationQueued() {
        endpoint.reserve(100L, 100, 110, 30);
        endpoint.reserve(1L, 100, 110, 50);
        endpoint.markQueuedPhase(1L);
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
        endpoint.reserve(1L, 100, 110, 50);
        endpoint.reserve(2L, 100, 110, 50);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);

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
        endpoint.reserve(1L, 100, 110, 50);
        endpoint.reserve(2L, 100, 110, 50);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);
        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(1L, 1);

        assertTrue(first.release());
        assertFalse(first.release());
        assertEquals(OWNERSHIP_LOST, first.transferToEngineLifecycle());

        DecodeEndpoint.EngineDispatchPermit second = acquirePermit(2L, 1);
        assertEquals(Set.of(1L, 2L), endpoint.layeredAdmissionView().queued());
        assertTrue(second.release());
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
    void closeBetweenPermitAcquisitionAndTransferRetiresExactPermitAndReleasesHardGate()
            throws InterruptedException {
        endpoint.reserve(1L, 100, 110, 50);
        endpoint.reserve(2L, 100, 110, 50);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(1L, 1);
        assertFalse(endpoint.hasEngineDispatchCapacity(1));

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
        assertTrue(endpoint.hasEngineDispatchCapacity(1),
                "close must remove the outstanding permit from hard-gate usage");
        assertEquals(0, endpoint.getEngineLoad());
        assertEquals(Set.of(1L, 2L), endpoint.layeredAdmissionView().queued());
    }

    @Test
    void closeDoesNotRollBackTransferredEngineLifecycleOwnership() {
        long requestId = 1L;
        endpoint.reserve(requestId, 100, 110, 50);
        endpoint.markQueuedPhase(requestId);
        RequestInflight reservation = endpoint.reservationFor(requestId);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        assertEquals(1, endpoint.getEngineLoad());

        endpoint.close();

        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        assertSame(reservation, endpoint.reservationFor(requestId));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(1, endpoint.getEngineLoad(),
                "retirement cannot return engine-owned work to Prefill ownership");
        assertFalse(endpoint.releaseIfHeld(requestId));
    }

    @Test
    void engineDispatchPermit_commitDoesNotRecheckCapacity() {
        endpoint.reserve(1L, 100, 110, 50);
        endpoint.markQueuedPhase(1L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(1L, 1);

        // Independent engine-facing work fills the original limit after this
        // permit has already reserved its slot.
        endpoint.reserve(100L, 100, 110, 30);
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
        endpoint.reserve(1L, 100, 110, 50);
        endpoint.reserve(2L, 100, 110, 50);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);
        RequestInflight firstReservation = endpoint.reservationFor(1L);
        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(1L, 1);

        assertEquals(TRANSFERRED, first.transferToEngineLifecycle());
        assertSame(firstReservation, endpoint.reservationFor(1L));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(1L));
        assertEquals(1, endpoint.getEngineLoad());
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                endpoint.acquireEngineDispatchPermit(2L, 1).status());

        assertFalse(first.release(),
                "committed Decode ownership is irreversible through the permit");
        assertSame(firstReservation, endpoint.reservationFor(1L));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(1L));
        assertEquals(1, endpoint.getEngineLoad());

        endpoint.release(1L);
        DecodeEndpoint.EngineDispatchPermit second = acquirePermit(2L, 1);
        assertTrue(second.release());
    }

    @Test
    void committedPermitCannotAffectLaterDispatchRoundOnSameReservation() {
        long requestId = 1L;
        endpoint.reserve(requestId, 100, 110, 50);
        endpoint.markQueuedPhase(requestId);
        RequestInflight reservation = endpoint.reservationFor(requestId);

        DecodeEndpoint.EngineDispatchPermit first = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, first.transferToEngineLifecycle());
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(1, endpoint.getEngineLoad());

        endpoint.markQueuedPhase(requestId);
        assertSame(reservation, endpoint.reservationFor(requestId),
                "the second dispatch round must reuse the same reservation");
        DecodeEndpoint.EngineDispatchPermit second = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, second.transferToEngineLifecycle());
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(1, endpoint.getEngineLoad());
        long versionBeforeStaleRelease = endpoint.admissionVersion();

        assertFalse(first.release(),
                "an older committed round must not change the current dispatch round");
        assertSame(reservation, endpoint.reservationFor(requestId));
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
        endpoint.reserve(requestId, 100, 110, 50);
        long versionBeforeFirstMark = endpoint.admissionVersion();

        endpoint.markQueuedPhase(requestId);
        assertEquals(versionBeforeFirstMark + 1, endpoint.admissionVersion());
        assertTrue(endpoint.layeredAdmissionView().queued().contains(requestId));

        endpoint.markQueuedPhase(requestId);
        assertEquals(versionBeforeFirstMark + 1, endpoint.admissionVersion(),
                "repeating an existing queued mark must be a no-op");

        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, permit.transferToEngineLifecycle());
        long versionBeforeSecondRound = endpoint.admissionVersion();

        endpoint.markQueuedPhase(requestId);
        assertEquals(versionBeforeSecondRound + 1, endpoint.admissionVersion());
        endpoint.markQueuedPhase(requestId);
        assertEquals(versionBeforeSecondRound + 1, endpoint.admissionVersion(),
                "repeating the second-round queued mark must also be a no-op");
    }

    @Test
    void staleCommittedPermitCannotAffectReplacementGeneration() {
        long requestId = 1L;
        endpoint.reserve(requestId, 100, 110, 50);
        endpoint.markQueuedPhase(requestId);
        RequestInflight original = endpoint.reservationFor(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);
        assertEquals(TRANSFERRED, stale.transferToEngineLifecycle());

        endpoint.reserve(requestId, 200, 220, 70);
        endpoint.markQueuedPhase(requestId);
        RequestInflight replacement = endpoint.reservationFor(requestId);
        assertNotSame(original, replacement);
        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);

        assertFalse(stale.release(),
                "a committed old generation must not change its replacement");
        assertFalse(stale.release(), "a stale release must stay idempotent");
        assertSame(replacement, endpoint.reservationFor(requestId));
        assertTrue(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED,
                endpoint.acquireEngineDispatchPermit(requestId, 1).status());

        assertEquals(TRANSFERRED, current.transferToEngineLifecycle(),
                "the stale token must not release or invalidate the new permit");
        assertSame(replacement, endpoint.reservationFor(requestId));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
    }

    @Test
    void staleEngineDispatchPermitCannotAffectReplacementGeneration() {
        long requestId = 1L;
        endpoint.reserve(requestId, 100, 110, 50);
        endpoint.markQueuedPhase(requestId);
        RequestInflight original = endpoint.reservationFor(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);

        endpoint.release(requestId);
        endpoint.reserve(requestId, 200, 220, 70);
        endpoint.markQueuedPhase(requestId);
        RequestInflight replacement = endpoint.reservationFor(requestId);
        assertNotSame(original, replacement);

        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);
        assertFalse(stale.release(), "the old token must not remove the new permit");
        assertEquals(OWNERSHIP_LOST, stale.transferToEngineLifecycle());
        assertSame(replacement, endpoint.reservationFor(requestId));
        assertTrue(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(TRANSFERRED, current.transferToEngineLifecycle(),
                "the replacement generation keeps its permit");
        assertSame(replacement, endpoint.reservationFor(requestId));
        assertFalse(endpoint.layeredAdmissionView().queued().contains(requestId));
    }

    @Test
    void staleEngineDispatchPermitCommitFailsAfterDirectReplacement() {
        long requestId = 2L;
        endpoint.reserve(requestId, 100, 110, 50);
        endpoint.markQueuedPhase(requestId);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(requestId, 1);

        endpoint.reserve(requestId, 200, 220, 70);
        endpoint.markQueuedPhase(requestId);
        RequestInflight replacement = endpoint.reservationFor(requestId);
        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(requestId, 1);

        assertEquals(OWNERSHIP_LOST, stale.transferToEngineLifecycle());
        assertSame(replacement, endpoint.reservationFor(requestId));
        assertTrue(endpoint.layeredAdmissionView().queued().contains(requestId));
        assertEquals(TRANSFERRED, current.transferToEngineLifecycle());
    }

    @Test
    void calibrateInvalidatesPermitWithoutLeakingHardGateCapacity() {
        endpoint.reserve(1L, 100, 110, 50);
        endpoint.markQueuedPhase(1L);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(1L, 2);

        TaskInfo running = new TaskInfo();
        running.setRequestId(1L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("1", running), null, 10_000);

        endpoint.reserve(2L, 100, 110, 50);
        endpoint.markQueuedPhase(2L);
        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(2L, 2);
        assertEquals(OWNERSHIP_LOST, stale.transferToEngineLifecycle());
        assertFalse(stale.release());
        assertTrue(current.release());
    }

    @Test
    void ttlEvictionInvalidatesPermitWithoutLeakingHardGateCapacity() {
        endpoint.reserve(1L, 100, 110, 50);
        endpoint.markQueuedPhase(1L);
        DecodeEndpoint.EngineDispatchPermit stale = acquirePermit(1L, 1);

        assertEquals(1, endpoint.evictExpiredRequests(-1));
        endpoint.reserve(2L, 100, 110, 50);
        endpoint.markQueuedPhase(2L);

        DecodeEndpoint.EngineDispatchPermit current = acquirePermit(2L, 1);
        assertFalse(stale.release());
        assertTrue(current.release());
    }

    // ==================== 10.1: confirmed requests leave the reserved view ====================

    @Test
    void calibrate_movesConfirmedOutOfReservedView() {
        endpoint.reserve(1L, 500, 508, 30);

        TaskInfo running = new TaskInfo();
        running.setRequestId(1L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("1", running), null, 10_000);

        // Confirmed by the engine: no longer a reserved (evictable) entry,
        // but still counted in the total load via confirmedRunningCount.
        assertTrue(endpoint.reservedView().isEmpty());
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
    }

    // ==================== helpers ====================

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        endpoint.onWorkerStatusUpdate(status, response);
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
}
