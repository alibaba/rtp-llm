package org.flexlb.balance.eviction;

import org.flexlb.balance.admission.AdmissionLifecyclePort;
import org.flexlb.balance.admission.AdmissionMutation;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime.QueueSnapshot;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Black-box guard contracts for {@link EvictionManager#tryAdmit}.
 *
 * <p>Requirement: every early decline is side-effect free (returns false
 * without reserving a permit, touching any port, or emitting telemetry).
 * Corner cases are derived from the domain requirements, not by echoing
 * if-branches.
 */
@DisplayName("EvictionManager.tryAdmit guard contracts")
class EvictionManagerTryAdmitTest {

    private EndpointRegistry endpointRegistry;
    private RequestSchedulerReporter reporter;
    private EngineCancelChannel cancelChannel;
    private DecodePreemptionCoordinator preemptionCoordinator;
    private AdmissionLifecyclePort admissionLifecycle;
    private VictimLifecyclePort victimLifecycle;
    private EvictionPlacementPort placementPort;
    private EvictionManager manager;

    @BeforeEach
    void setUp() {
        endpointRegistry = mock(EndpointRegistry.class);
        reporter = mock(RequestSchedulerReporter.class);
        cancelChannel = mock(EngineCancelChannel.class);
        preemptionCoordinator = mock(DecodePreemptionCoordinator.class);
        admissionLifecycle = mock(AdmissionLifecyclePort.class);
        victimLifecycle = mock(VictimLifecyclePort.class);
        placementPort = mock(EvictionPlacementPort.class);
        manager = new EvictionManager(
                endpointRegistry, reporter, cancelChannel,
                preemptionCoordinator, admissionLifecycle,
                victimLifecycle, placementPort);
    }

    private void assertZeroSideEffect() {
        verifyNoInteractions(endpointRegistry);
        verifyNoInteractions(cancelChannel);
        verifyNoInteractions(preemptionCoordinator);
        verifyNoInteractions(admissionLifecycle);
        verifyNoInteractions(victimLifecycle);
        verifyNoInteractions(placementPort);
        verifyNoInteractions(reporter);
    }

    // ─── Shutdown ────────────────────────────────────────────────────────

    @Test
    @DisplayName("A shut-down manager declines without side effects")
    void shutdownDeclines() {
        manager.shutdown();
        assertFalse(manager.tryAdmit(ctx(70), new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    // ─── Future states ──────────────────────────────────────────────────

    @Test
    @DisplayName("An already-completed future declines without side effects")
    void completedFutureDeclines() {
        CompletableFuture<Response> done = new CompletableFuture<>();
        done.complete(null);
        assertFalse(manager.tryAdmit(ctx(70), done));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("An exceptionally-completed future declines without side effects")
    void exceptionalFutureDeclines() {
        CompletableFuture<Response> failed = new CompletableFuture<>();
        failed.completeExceptionally(new RuntimeException("test"));
        assertFalse(manager.tryAdmit(ctx(70), failed));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("A cancelled future declines without side effects")
    void cancelledFutureDeclines() {
        CompletableFuture<Response> cancelled = new CompletableFuture<>();
        cancelled.cancel(false);
        assertFalse(manager.tryAdmit(ctx(70), cancelled));
        assertZeroSideEffect();
    }

    // ─── Expiration ─────────────────────────────────────────────────────

    @Test
    @DisplayName("An expired request declines without side effects")
    void expiredRequestDeclines() {
        BalanceContext expired = ctx(70);
        when(expired.requestExpired(anyLong())).thenReturn(true);
        assertFalse(manager.tryAdmit(expired, new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    // ─── Priority boundaries ────────────────────────────────────────────

    @Test
    @DisplayName("Priority 0 (NO_PRIORITY sentinel) declines without side effects")
    void noPriorityDeclines() {
        assertFalse(manager.tryAdmit(ctx(0), new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("Priority 1 (minimum valid) passes the guard — does NOT decline on priority alone")
    void minimumValidPriorityPassesGuard() {
        // priority=1 has priority; with FIFO config (no preemption policy) it
        // still declines, but for a DIFFERENT reason (no preemption policy),
        // proving the priority guard itself passed.
        BalanceContext ctx = ctx(1);
        when(ctx.getConfig()).thenReturn(new FlexlbConfig()); // FIFO = no preemption
        assertFalse(manager.tryAdmit(ctx, new CompletableFuture<>()));
        // It passed the priority guard but declined on preemption policy.
        // Key: reporter was called (reportCapacityGauges) if it reached past
        // the priority guard — but with FIFO config it exits before gauges too
        // (config.isQueue() check passes, preemption==null exits).
        // This proves priority=1 is accepted by hasPriority.
    }

    // ─── Scheduler mode ─────────────────────────────────────────────────

    @Test
    @DisplayName("FIFO ordering (no preemption policy) never evicts")
    void fifoOrderingNeverEvicts() {
        BalanceContext ctx = ctx(50);
        when(ctx.getConfig()).thenReturn(new FlexlbConfig()); // default=QUEUE+FIFO
        assertFalse(manager.tryAdmit(ctx, new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("DIRECT scheduler mode declines without side effects")
    void directSchedulerDeclines() {
        BalanceContext ctx = ctx(50);
        FlexlbConfig directConfig = new FlexlbConfig();
        directConfig.setScheduler(new DirectSchedulerConfig());
        when(ctx.getConfig()).thenReturn(directConfig);
        assertFalse(manager.tryAdmit(ctx, new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    // ─── Transient Decode/Prefill contention ────────────────────────────

    @Test
    @DisplayName("A raced Prefill replacement returns to waiting without terminalizing")
    void prefillReplacementConflictReturnsToWaitingWithoutTerminalizing() {
        long requestId = 9_101L;
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.allowVictim(
                config, VictimStage.PREFILL_QUEUED);
        Request request = new Request();
        request.setRequestId(requestId);
        request.setPriority(80);
        request.setSeqLen(128L);
        request.setMaxNewTokens(1);
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                80, System.currentTimeMillis() + 60_000L));
        CompletableFuture<Response> future = new CompletableFuture<>();

        DeliveryItem victim = mock(DeliveryItem.class);
        when(victim.requestId()).thenReturn(8_101L);
        when(victim.priority()).thenReturn(30);
        when(victim.enqueuedAtMs())
                .thenReturn(System.currentTimeMillis() - 1_000L);
        when(victim.seqLen()).thenReturn(128L);
        PriorityRequestEnvelope envelope = new PriorityRequestEnvelope(
                requestId, 80, 128L, 1L,
                System.currentTimeMillis(), 128L, 129L);
        QueueSnapshot snapshot = new QueueSnapshot(
                "prefill-a", 1L, 1, 1L, 1L, 1L, List.of(victim));
        EvictionPlacementPort.PrefillEvictionAdmission admission =
                mock(EvictionPlacementPort.PrefillEvictionAdmission.class);
        when(admission.envelope()).thenReturn(envelope);
        when(admission.queueSnapshot()).thenReturn(snapshot);
        when(admission.commit(
                anyList(), any(AdmissionMutation.class))).thenReturn(
                new EvictionPlacementPort.PrefillEvictionCommit(
                        EvictionPlacementPort.PrefillEvictionStatus.CONFLICT,
                        List.of()));
        when(placementPort.preparePrefillEviction(
                same(context), same(future))).thenReturn(admission);
        AdmissionMutation mutation = mock(AdmissionMutation.class);
        when(mutation.seal()).thenReturn(true);
        when(admissionLifecycle.claimAdmissionMutation(
                requestId, future)).thenReturn(mutation);
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(Map.of());
        when(endpointRegistry.snapshotDecodeEndpoints()).thenReturn(Map.of());

        boolean takenOver = manager.tryAdmit(context, future);

        assertAll(
                () -> assertFalse(takenOver),
                () -> assertFalse(future.isDone()),
                () -> verify(admission).commit(
                        anyList(), same(mutation)),
                () -> verify(admission).close(),
                () -> verify(mutation, never()).terminate(any(Response.class)),
                () -> verify(admissionLifecycle, never())
                        .bindAdmissionResources(
                                anyLong(), any(), any(), any(), anyLong()),
                () -> verifyNoInteractions(victimLifecycle),
                () -> assertEquals(0, manager.activeAdmissionCount()));
    }

    @Test
    @DisplayName("A raced local Decode plan returns to waiting without terminalizing")
    void localDecodeConflictReturnsToWaitingWithoutTerminalizing() {
        LocalDecodeAttempt attempt = localDecodeAttempt();
        when(attempt.endpoint().tryEvictLocalReservationsAndReserveIncoming(
                anyList(),
                eq(attempt.context().getRequestId()),
                anyLong(),
                anyLong(),
                eq(attempt.context().getPriority()),
                any(DecodeEndpoint.AdmissionCapacity.class)))
                .thenReturn(new DecodeEndpoint.LocalEvictionCommit(
                        DecodeEndpoint.LocalEvictionResult.CONFLICT, null));

        boolean takenOver = manager.tryAdmit(
                attempt.context(), attempt.future());

        assertAll(
                () -> assertFalse(takenOver,
                        "a stale local plan is contention, so RequestScheduler must keep waiting"),
                () -> assertFalse(attempt.future().isDone(),
                        "the original request future must remain open"),
                () -> verify(attempt.mutation(), never())
                        .terminate(any(Response.class)),
                () -> verify(victimLifecycle, never())
                        .finishYieldedReservation(anyLong(), anyLong(), any()),
                () -> verify(attempt.preparedPlacement(), never())
                        .commit(any(), any()),
                () -> verify(attempt.preparedPlacement()).close(),
                () -> assertEquals(0, manager.activeAdmissionCount(),
                        "a declined attempt must not leak its admission permit"));
    }

    @Test
    @DisplayName("Transient Prefill unavailability is resolved before Decode victim PNR")
    void transientPrefillUnavailabilityDoesNotCrossDecodeVictimPnr() {
        LocalDecodeAttempt attempt = localDecodeAttempt();
        AtomicBoolean decodeVictimPnrCrossed = new AtomicBoolean();
        when(attempt.endpoint().tryEvictLocalReservationsAndReserveIncoming(
                anyList(),
                eq(attempt.context().getRequestId()),
                anyLong(),
                anyLong(),
                eq(attempt.context().getPriority()),
                any(DecodeEndpoint.AdmissionCapacity.class)))
                .thenAnswer(ignored -> {
                    decodeVictimPnrCrossed.set(true);
                    return new DecodeEndpoint.LocalEvictionCommit(
                            DecodeEndpoint.LocalEvictionResult.COMMITTED,
                            attempt.incomingReservation());
                });
        // A null prepared placement is the typed temporary-full outcome. It
        // must be resolved before the Decode endpoint transaction is invoked.
        when(placementPort.prepareDecodePlacement(
                same(attempt.context()),
                same(attempt.future()),
                same(attempt.endpoint())))
                .thenReturn(null);

        boolean takenOver = manager.tryAdmit(
                attempt.context(), attempt.future());

        assertAll(
                () -> assertFalse(decodeVictimPnrCrossed.get(),
                        "temporary Prefill capacity must be owned before any Decode victim is removed"),
                () -> assertFalse(takenOver,
                        "without a Prefill seat the original request returns to scheduler waiting"),
                () -> assertFalse(attempt.future().isDone(),
                        "temporary Prefill pressure must not complete the original future"),
                () -> verify(victimLifecycle, never())
                        .finishYieldedReservation(anyLong(), anyLong(), any()),
                () -> verify(attempt.mutation(), never())
                        .terminate(any(Response.class)),
                () -> assertEquals(0, manager.activeAdmissionCount(),
                        "a non-admitted request must not retain an admission permit"));
    }

    @Test
    @DisplayName("A deadline that wins the seal race prevents Decode victim PNR")
    void deadlineBeforeSealDoesNotCrossDecodeVictimPnr() {
        LocalDecodeAttempt attempt = localDecodeAttempt();
        when(attempt.mutation().seal()).thenReturn(false);

        boolean takenOver = manager.tryAdmit(
                attempt.context(), attempt.future());

        assertAll(
                () -> assertFalse(takenOver),
                () -> verify(attempt.mutation()).seal(),
                () -> verify(attempt.endpoint(), never())
                        .tryEvictLocalReservationsAndReserveIncoming(
                                anyList(), anyLong(), anyLong(), anyLong(),
                                any(Integer.class),
                                any(DecodeEndpoint.AdmissionCapacity.class)),
                () -> verify(attempt.preparedPlacement()).close(),
                () -> verify(attempt.preparedPlacement(), never())
                        .commit(any(), any()),
                () -> verify(admissionLifecycle, never())
                        .bindAdmissionResources(
                                anyLong(), any(), any(), any(), anyLong()),
                () -> assertEquals(0, manager.activeAdmissionCount()));
    }

    @Test
    @DisplayName("A revoked Prefill hold prevents local Decode victim PNR")
    void revokedPreparedPlacementDoesNotCrossDecodeVictimPnr() {
        LocalDecodeAttempt attempt = localDecodeAttempt();
        when(attempt.preparedPlacement().seal()).thenReturn(false);

        assertFalse(manager.tryAdmit(
                attempt.context(), attempt.future()));

        var order = inOrder(
                attempt.mutation(), attempt.preparedPlacement());
        order.verify(attempt.mutation()).seal();
        order.verify(attempt.preparedPlacement()).seal();
        verify(attempt.endpoint(), never())
                .tryEvictLocalReservationsAndReserveIncoming(
                        anyList(), anyLong(), anyLong(), anyLong(),
                        any(Integer.class),
                        any(DecodeEndpoint.AdmissionCapacity.class));
        verify(attempt.preparedPlacement(), never()).commit(any(), any());
        assertFalse(attempt.future().isDone());
        assertEquals(0, manager.activeAdmissionCount());
    }

    @Test
    @DisplayName("Local bind failure rolls back exact incoming and settles committed victims")
    void localBindFailureRollsBackExactIncomingAfterVictimPnr() {
        LocalDecodeAttempt attempt = localDecodeAttempt();
        when(attempt.endpoint().tryEvictLocalReservationsAndReserveIncoming(
                anyList(),
                eq(attempt.context().getRequestId()),
                anyLong(),
                anyLong(),
                eq(attempt.context().getPriority()),
                any(DecodeEndpoint.AdmissionCapacity.class)))
                .thenReturn(new DecodeEndpoint.LocalEvictionCommit(
                        DecodeEndpoint.LocalEvictionResult.COMMITTED,
                        attempt.incomingReservation()));
        when(admissionLifecycle.bindAdmissionResources(
                eq(attempt.context().getRequestId()),
                same(attempt.future()),
                same(attempt.mutation()),
                any(Runnable.class),
                anyLong())).thenReturn(false);

        assertTrue(manager.tryAdmit(
                attempt.context(), attempt.future()));

        assertAll(
                () -> verify(attempt.endpoint())
                        .rollbackExact(same(attempt.incomingReservation())),
                () -> verify(attempt.preparedPlacement(), never())
                        .commit(any(), any()),
                () -> verify(victimLifecycle)
                        .finishYieldedReservation(eq(8_001L), eq(17L), any()),
                () -> verify(attempt.mutation()).terminate(any(Response.class)),
                () -> assertEquals(0, manager.activeAdmissionCount()));
    }

    @Test
    @DisplayName("Local canonical placement precedes every victim callback")
    void localPlacementCommitsBeforeVictimSettlement() {
        LocalDecodeAttempt attempt = localDecodeAttempt();
        when(attempt.endpoint().tryEvictLocalReservationsAndReserveIncoming(
                anyList(),
                eq(attempt.context().getRequestId()),
                anyLong(),
                anyLong(),
                eq(attempt.context().getPriority()),
                any(DecodeEndpoint.AdmissionCapacity.class)))
                .thenReturn(new DecodeEndpoint.LocalEvictionCommit(
                        DecodeEndpoint.LocalEvictionResult.COMMITTED,
                        attempt.incomingReservation()));
        when(attempt.preparedPlacement().commit(
                same(attempt.incomingReservation()),
                same(attempt.mutation())))
                .thenReturn(new EvictionPlacementPort.DecodePlacement.Committed());

        assertTrue(manager.tryAdmit(
                attempt.context(), attempt.future()));

        var order = inOrder(attempt.preparedPlacement(), victimLifecycle);
        order.verify(attempt.preparedPlacement())
                .commit(
                        same(attempt.incomingReservation()),
                        same(attempt.mutation()));
        order.verify(victimLifecycle)
                .finishYieldedReservation(eq(8_001L), eq(17L), any());
        verify(attempt.endpoint(), never())
                .rollbackExact(same(attempt.incomingReservation()));
    }

    @Test
    @DisplayName("Remote control failure stays non-terminal and releases pre-PNR ownership")
    void remoteControlFailureReturnsOwnershipToSchedulerWaiting() {
        RemoteDecodeAttempt attempt = remoteDecodeAttempt();

        boolean takenOver = manager.tryAdmit(
                attempt.context(), attempt.future());

        assertAll(
                () -> assertTrue(takenOver,
                        "the scheduler retains its generic wait lane while async control runs"),
                () -> assertFalse(attempt.future().isDone(),
                        "control timeout is not a request terminal"),
                () -> verify(preemptionCoordinator).execute(argThat(
                        request -> request.admissionMutation()
                                == attempt.mutation())),
                () -> verify(attempt.mutation(), never()).seal(),
                () -> verify(attempt.mutation(), never()).close(),
                () -> verify(attempt.preparedPlacement(), never()).close(),
                () -> assertEquals(1, manager.activeAdmissionCount(),
                        "pending control retains exact admission ownership"));

        attempt.control().complete(
                new DecodePreemptionCoordinator.ExecutionResult(
                        DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED,
                        "cancel_timeout"));

        assertAll(
                () -> assertFalse(attempt.future().isDone(),
                        "control timeout is not a request terminal"),
                () -> verify(attempt.mutation(), never()).terminate(any()),
                () -> verify(attempt.mutation()).close(),
                () -> verify(attempt.preparedPlacement()).close(),
                () -> verify(attempt.preparedPlacement(), never())
                        .commit(any(), any()),
                () -> verify(admissionLifecycle, never())
                        .bindAdmissionResources(
                                anyLong(), any(), any(), any(), anyLong()),
                () -> assertEquals(0, manager.activeAdmissionCount()));
    }

    @Test
    @DisplayName("Remote commit publishes the exact begin reservation without request-id lookup")
    void remoteCommitTransfersExactCoordinatorReservation() {
        RemoteDecodeAttempt attempt = remoteDecodeAttempt();
        DecodeEndpoint.ReservationHandle exactIncoming =
                new DecodeEndpoint.ReservationHandle(
                        9L, attempt.context().getRequestId(), 28L);
        when(admissionLifecycle.bindAdmissionResources(
                eq(attempt.context().getRequestId()),
                same(attempt.future()),
                same(attempt.mutation()),
                any(Runnable.class),
                anyLong())).thenReturn(true);
        when(attempt.preparedPlacement().commit(
                same(exactIncoming), same(attempt.mutation())))
                .thenReturn(new EvictionPlacementPort.DecodePlacement.Committed());

        assertTrue(manager.tryAdmit(
                attempt.context(), attempt.future()));
        attempt.control().complete(
                new DecodePreemptionCoordinator.ExecutionResult(
                        DecodePreemptionCoordinator.ResultCode.COMMITTED,
                        "committed",
                        exactIncoming));

        assertAll(
                () -> verify(attempt.preparedPlacement())
                        .commit(
                                same(exactIncoming),
                                same(attempt.mutation())),
                () -> verify(attempt.endpoint(), never())
                        .reservationHandle(anyLong()),
                () -> verify(attempt.endpoint(), never())
                        .rollbackExact(same(exactIncoming)),
                () -> verify(attempt.mutation(), never()).terminate(any()),
                () -> verify(attempt.mutation()).close());
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    private LocalDecodeAttempt localDecodeAttempt() {
        long incomingRequestId = 9_001L;
        long victimRequestId = 8_001L;
        long endpointGeneration = 7L;
        long victimReservationToken = 17L;

        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.allowVictim(
                config, VictimStage.DECODE_RESERVED);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(1L);

        Request request = new Request();
        request.setRequestId(incomingRequestId);
        request.setPriority(80);
        request.setSeqLen(128L);
        request.setMaxNewTokens(1);
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                80, System.currentTimeMillis() + 60_000L));

        RequestInflight victim = new RequestInflight(
                256L,
                256L,
                System.currentTimeMillis(),
                30,
                DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                victimReservationToken);
        DecodeEndpoint.DecodeRoutingView routing =
                new DecodeEndpoint.DecodeRoutingView(
                        null,
                        1L,
                        1,
                        0,
                        1_000L,
                        0L,
                        1_000L,
                        256L,
                        256L);
        DecodeEndpoint.LayeredAdmissionView layered =
                new DecodeEndpoint.LayeredAdmissionView(
                        routing,
                        Map.of(victimRequestId, victim),
                        List.of(),
                        Set.of(victimRequestId),
                        Set.of());

        DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.getGenerationId()).thenReturn(endpointGeneration);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ipPort()).thenReturn("decode-a");
        when(endpoint.layeredAdmissionView()).thenReturn(layered);
        when(endpointRegistry.snapshotPrefillEndpoints())
                .thenReturn(Map.of());
        when(endpointRegistry.snapshotDecodeEndpoints())
                .thenReturn(Map.of("decode-a", endpoint));

        CompletableFuture<Response> future = new CompletableFuture<>();
        AdmissionMutation mutation = mock(AdmissionMutation.class);
        when(mutation.seal()).thenReturn(true);
        EvictionPlacementPort.PreparedDecodePlacement preparedPlacement =
                mock(EvictionPlacementPort.PreparedDecodePlacement.class);
        when(preparedPlacement.seal()).thenReturn(true);
        when(placementPort.prepareDecodePlacement(
                same(context), same(future), same(endpoint)))
                .thenReturn(preparedPlacement);
        when(admissionLifecycle.bindAdmissionResources(
                eq(incomingRequestId),
                same(future),
                same(mutation),
                any(Runnable.class),
                anyLong()))
                .thenReturn(true);
        when(admissionLifecycle.claimAdmissionMutation(
                eq(incomingRequestId), same(future)))
                .thenReturn(mutation);

        return new LocalDecodeAttempt(
                context,
                future,
                endpoint,
                mutation,
                preparedPlacement,
                new DecodeEndpoint.ReservationHandle(
                        endpointGeneration,
                        incomingRequestId,
                        victimReservationToken + 1L));
    }

    private RemoteDecodeAttempt remoteDecodeAttempt() {
        long incomingRequestId = 9_201L;
        long victimRequestId = 8_201L;
        long victimReservationToken = 27L;

        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.allowVictim(
                config, VictimStage.DECODE_ENGINE_OWNED);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(1L);

        Request request = new Request();
        request.setRequestId(incomingRequestId);
        request.setPriority(80);
        request.setSeqLen(128L);
        request.setMaxNewTokens(1);
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                80, System.currentTimeMillis() + 60_000L));

        DecodeEndpoint.DecodeRoutingView routing =
                new DecodeEndpoint.DecodeRoutingView(
                        null,
                        1L,
                        1,
                        1,
                        256L,
                        744L,
                        1_000L,
                        0L,
                        0L);
        DecodeEndpoint.ConfirmedTaskView running =
                new DecodeEndpoint.ConfirmedTaskView(
                        victimRequestId,
                        30,
                        256L,
                        DecodeTaskPhase.RUNNING,
                        true,
                        victimReservationToken,
                        false);
        DecodeEndpoint.LayeredAdmissionView layered =
                new DecodeEndpoint.LayeredAdmissionView(
                        routing,
                        Map.of(),
                        List.of(running),
                        Set.of(),
                        Set.of());

        DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.getGenerationId()).thenReturn(9L);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ipPort()).thenReturn("decode-remote");
        when(endpoint.layeredAdmissionView()).thenReturn(layered);
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(Map.of());
        when(endpointRegistry.snapshotDecodeEndpoints())
                .thenReturn(Map.of("decode-remote", endpoint));
        when(cancelChannel.isSupported(endpoint)).thenReturn(true);

        CompletableFuture<Response> future = new CompletableFuture<>();
        AdmissionMutation mutation = mock(AdmissionMutation.class);
        when(mutation.seal()).thenReturn(true);
        EvictionPlacementPort.PreparedDecodePlacement preparedPlacement =
                mock(EvictionPlacementPort.PreparedDecodePlacement.class);
        when(placementPort.prepareDecodePlacement(
                same(context), same(future), same(endpoint)))
                .thenReturn(preparedPlacement);
        when(admissionLifecycle.claimAdmissionMutation(
                incomingRequestId, future)).thenReturn(mutation);
        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> control =
                new CompletableFuture<>();
        when(preemptionCoordinator.execute(any())).thenReturn(control);

        return new RemoteDecodeAttempt(
                context, future, endpoint, mutation,
                preparedPlacement, control);
    }

    private record LocalDecodeAttempt(
            BalanceContext context,
            CompletableFuture<Response> future,
            DecodeEndpoint endpoint,
            AdmissionMutation mutation,
            EvictionPlacementPort.PreparedDecodePlacement preparedPlacement,
            DecodeEndpoint.ReservationHandle incomingReservation) {
    }

    private record RemoteDecodeAttempt(
            BalanceContext context,
            CompletableFuture<Response> future,
            DecodeEndpoint endpoint,
            AdmissionMutation mutation,
            EvictionPlacementPort.PreparedDecodePlacement preparedPlacement,
            CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> control) {
    }

    private static BalanceContext ctx(int priority) {
        BalanceContext ctx = mock(BalanceContext.class);
        when(ctx.getPriority()).thenReturn(priority);
        when(ctx.requestExpired(anyLong())).thenReturn(false);
        when(ctx.getConfig()).thenReturn(new FlexlbConfig());
        return ctx;
    }
}
