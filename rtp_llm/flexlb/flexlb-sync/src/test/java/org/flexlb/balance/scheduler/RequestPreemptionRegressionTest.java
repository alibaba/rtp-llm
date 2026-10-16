package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/** Cross-component regressions: frontend progress and remote resource ownership are independent. */
@Timeout(15)
class RequestPreemptionRegressionTest {
    private static final long REQUEST_ID = 9781L;
    private static final long INCOMING_ID = 9782L;
    private static final long ATTEMPT = 41L;
    private static final long BATCH_ID = 17L;
    private static final long KV_TOKENS = 100L;
    private static final DecodeResources.AdmissionCapacity CAPACITY =
            new DecodeResources.AdmissionCapacity(1L, 100L);

    private AbstractRequestScheduler scheduler;
    private RequestContext context;
    private DecodeEndpoint decode;
    private PrefillEndpoint prefill;
    private RequestRoute route;
    private DeliveryClaim delivery;
    private CompletableFuture<Response> response;
    private long statusVersion;

    @BeforeEach
    void setUp() {
        var config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        scheduler = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
        SchedulerTestSupport.timer(scheduler.runtime).close();
        ReflectionTestUtils.setField(scheduler.runtime, "timer", mock(ExpirationTimer.class));
        SchedulerTestSupport.responseCompletions(scheduler.runtime).close();
        ReflectionTestUtils.setField(scheduler.runtime, "responseCompletions", new ResponseCompletionExecutor(1));
        context = RequestProtocolTestSupport.context(config, REQUEST_ID);
        context.data().setRequest(Request.builder().requestId(REQUEST_ID).seqLen(KV_TOKENS).maxNewTokens(0).build());
        response = RequestProtocolTestSupport.register(scheduler, context);
        decode = EndpointTestSupport.decode(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8081, 8091, null), scheduler.requests);
        reportDecode(null, false);
        DecodeResources.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = decode.tryReserveQueuedRequest(pin, REQUEST_ID, KV_TOKENS, KV_TOKENS, 50, CAPACITY).handle();
        }
        assertNotNull(reservation);
        prefill = mock(PrefillEndpoint.class);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        var server = new ServerStatus();
        server.setServerIp("127.0.0.1");
        server.setGrpcPort(8090);
        route = SchedulingTestConfig.createRoute(context, new Response(), server, null,
                prefill, decode, reservation, System.currentTimeMillis());
        try (var admission = RequestProtocolTestSupport.beginAdmission(scheduler, context);
             var finished = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                    scheduler.commitRoute(route, RequestProtocolTestSupport.publication(() -> true)));
        }
        var permit = decode.acquireDispatchPermit(reservation, CAPACITY).handle();
        assertNotNull(permit);
        delivery = scheduler.claimDelivery(route, DeliveryClaimKind.BATCH_ENQUEUE, BATCH_ID, permit);
        assertNotNull(delivery);
        assertTrue(scheduler.tryStartSend(delivery));
    }

    @AfterEach
    void tearDown() {
        if (scheduler == null) { return; }
        try {
            // Supply actual sender/Worker terminal evidence only after the regression assertions.
            if (delivery != null) { RequestProtocolTestSupport.transport(scheduler, delivery, DeliveryResult.delivered()); }
            if (route != null) {
                reportDecode(null, true);
                RequestProtocolTestSupport.applyPrefillStatus(scheduler, context, prefill, RoleType.PREFILL,
                        new PrefillState.PrefillRequestStatus(route, PrefillState.PrefillRequestStatus.Kind.COMPLETED, 0L));
            }
        } finally {
            SchedulerTestSupport.timer(scheduler.runtime).close();
            SchedulerTestSupport.closeRequestExecutors(scheduler.runtime);
            scheduler.closePlacement();
        }
    }

    @Test
    void clientCancellationDuringUnknownPreemptionStartsCleanupAndCompletesWithoutReleaseProof() throws Exception {
        var preemption = beginPreemption();
        assertTrue(RequestProtocolTestSupport.preemptionPhase(scheduler, preemption, PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertFalse(response.isDone(), "the RPC has started but its ACK has not arrived");

        scheduler.cancel(REQUEST_ID, BATCH_ID, CancelReason.CLIENT_CANCELLED);
        // Repeated nonterminal heartbeats must not gate the frontend cancellation on inactivity expiry.
        for (int heartbeat = 0; heartbeat < 3; heartbeat++) { reportDecode(TaskPhase.RECEIVED, false); }
        drainResponses();

        assertAll("frontend cancellation progresses while remote ownership remains unresolved",
                () -> verify(scheduler.runtime, times(1)).startDeliveryCleanup(delivery),
                () -> {
                    var result = response.getNow(null);
                    assertNotNull(result, "cancellation response must be published before Worker release proof");
                    assertFalse(result.isSuccess());
                    assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), result.getCode());
                },
                () -> assertTrue(scheduler.requests.isCurrent(context), "unproven remote ownership stays registered"),
                () -> assertTrue(victimClaimed(), "ordinary cancellation cannot fabricate victim release proof"),
                () -> assertNotEquals(Phase.CLOSED, context.snapshot().phase()),
                () -> assertFalse(scheduler.runtime.signals().delivery(delivery).toCompletableFuture().isDone()));
    }

    @Test
    void historicalAllocationBeforeNotFoundDoesNotReleaseVictimOrItsKvHoldOnReceived() throws Exception {
        RequestProtocolTestSupport.transport(scheduler, delivery, DeliveryResult.delivered());
        drainResponses();
        assertTrue(response.getNow(null).isSuccess(), "ACK precedes this preemption attempt");
        reportDecode(TaskPhase.KV_ALLOCATED, false);
        assertTrue(context.snapshot().engine().decodeAccepted());
        var preemption = beginPreemption();

        assertTrue(RequestProtocolTestSupport.preemptionPhase(scheduler, preemption, PreemptionCancelPhase.NOT_FOUND_STALE));
        boolean claimedAfterNotFound = victimClaimed();
        reportDecode(TaskPhase.RECEIVED, false);
        var snapshot = decode.resourceSnapshot();
        // Exclude the incoming request's separate provisional charge from the victim's conservative hold.
        long incomingKv = snapshot.requests().get(INCOMING_ID).expectedKvTokens();
        long victimKv = snapshot.routing().realKvUsed() - incomingKv;

        assertAll("historical acceptance is not fresh restoration proof for this attempt",
                () -> assertTrue(claimedAfterNotFound, "NOT_FOUND alone must preserve the exact victim claim"),
                () -> assertTrue(victimClaimed(), "RECEIVED without new allocation proof must preserve the victim claim"),
                () -> assertEquals(KV_TOKENS, victimKv, "conservative victim KV must remain charged"),
                () -> assertNotNull(context.snapshot().preemption(), "the Context must retain the unresolved attempt"),
                () -> assertFalse(scheduler.runtime.signals().preemption(preemption).toCompletableFuture().isDone()));
    }

    @Test
    void freshAllocationAfterNotFoundCanResolveTheVictimClaim() {
        var preemption = beginPreemption();
        assertFalse(context.snapshot().engine().decodeAccepted());
        assertTrue(RequestProtocolTestSupport.preemptionPhase(scheduler, preemption, PreemptionCancelPhase.NOT_FOUND_STALE));
        assertTrue(victimClaimed(), "without earlier acceptance, NOT_FOUND must wait for new evidence");

        reportDecode(TaskPhase.KV_ALLOCATED, false);

        assertFalse(victimClaimed(), "new allocation restores real Worker accounting and resolves the claim");
        assertNull(context.snapshot().preemption());
        assertEquals(2L * KV_TOKENS, decode.routingView().realKvUsed(), "Worker allocation and incoming reservation are both charged");
    }

    private PreemptionRegistration beginPreemption() {
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                decode.beginPreemption(ATTEMPT, List.of(route.decodeReservation()),
                        INCOMING_ID, KV_TOKENS, KV_TOKENS, 70, CAPACITY));
        var registration = scheduler.tryClaim(route.decodeReservation(), ATTEMPT, "higher priority incoming request").orElseThrow();
        assertTrue(RequestProtocolTestSupport.preemptionPhase(scheduler, registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(decode.updatePreemption(ATTEMPT, DecodeResources.PreemptionUpdate.handedOff(route.decodeReservation())).status().applied());
        assertTrue(victimClaimed());
        return registration;
    }

    private boolean victimClaimed() {
        var victim = decode.resourceSnapshot().requests().get(REQUEST_ID);
        return victim != null && victim.claimedForPreemption();
    }

    private void reportDecode(TaskPhase phase, boolean terminal) {
        var report = new WorkerStatusResponse();
        report.setRole(RoleType.DECODE);
        report.setAlive(true);
        report.setStatusVersion(++statusVersion);
        report.setTotalKvCacheTokens(1000L);
        report.setAvailableKvCacheTokens(phase == TaskPhase.KV_ALLOCATED ? 900L : 1000L);
        var task = new TaskInfo();
        task.setRequestId(REQUEST_ID);
        task.setInputLength(KV_TOKENS);
        task.setPhase(phase);
        report.setRunningTaskInfo(phase == null ? Map.of() : Map.of(Long.toString(REQUEST_ID), task));
        report.setFinishedTaskInfo(terminal ? Map.of(Long.toString(REQUEST_ID), task) : Map.of());
        var status = decode.getStatus();
        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange change;
        status.lock.lock();
        try {
            var prepared = status.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(status, report));
            change = decode.applyPreparedStatus(status, prepared);
        } finally { status.lock.unlock(); }
        decode.publishStatus(change);
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);
    }

    private void drainResponses() throws Exception {
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);
        var drained = new CompletableFuture<Void>();
        SchedulerTestSupport.responseCompletions(scheduler.runtime).tryRegister().submit(() -> {
            drained.complete(null);
            return true;
        });
        drained.get(5L, TimeUnit.SECONDS);
    }
}
