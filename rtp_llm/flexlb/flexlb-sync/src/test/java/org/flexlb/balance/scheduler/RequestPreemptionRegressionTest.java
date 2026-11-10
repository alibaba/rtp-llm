package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
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
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Cross-component regressions: frontend progress and remote resource ownership are independent.
 */
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
    private RouteResult route;
    private Delivery.Member delivery;
    private CompletableFuture<Response> response;
    private long statusVersion;

    @BeforeEach
    void setUp() {
        var config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        scheduler =
                SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        SchedulerTestSupport.timer(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler))
                .close();
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler),
                "timer",
                mock(ExpirationTimer.class));
        SchedulerTestSupport.responseCompletions(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler))
                .close();
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler),
                "responseCompletions",
                new ResponseCompletionExecutor(1));
        context = RequestProtocolTestSupport.context(config, REQUEST_ID);
        context.data()
                .setRequest(
                        Request.builder()
                                .requestId(String.valueOf(REQUEST_ID))
                                .seqLen(KV_TOKENS)
                                .maxNewTokens(0)
                                .build());
        response = RequestProtocolTestSupport.submitForManualPlacement(scheduler, context);
        decode =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, null, "127.0.0.1", 8081, 8091, null),
                        scheduler.repository);
        reportDecode(null, false);
        DecodeResources.RequestReservation reservation;
        {
            var selectedEndpoint = decode;

            reservation =
                    decode.tryRegisterSelection(
                                    decode.routingView(),
                                    String.valueOf(REQUEST_ID),
                                    KV_TOKENS,
                                    KV_TOKENS,
                                    50,
                                    CAPACITY)
                            .value();
        }
        assertNotNull(reservation);
        prefill = mock(PrefillEndpoint.class);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        var server = new ServerStatus();
        server.setServerIp("127.0.0.1");
        server.setGrpcPort(8090);
        route =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        server,
                        null,
                        prefill,
                        decode,
                        reservation,
                        System.currentTimeMillis());
        var admission = RequestProtocolTestSupport.beginAdmission(scheduler, context);
        try (var finished = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    org.flexlb.balance.PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            scheduler, route, RequestProtocolTestSupport.publication(() -> true)));
        }
        var permit = decode.acquireDispatchPermit(reservation, CAPACITY).value();
        assertNotNull(permit);
        delivery =
                scheduler.commitDelivery(
                        new org.flexlb.balance.scheduler.Delivery.Member(route, permit),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        BATCH_ID);
        assertNotNull(delivery);
        assertEquals(
                ContextResult.SendState.SENDING, delivery.owner().snapshot().delivery().send());
    }

    @AfterEach
    void tearDown() {
        if (scheduler == null) {
            return;
        }
        try {
            // Supply actual sender/Worker terminal evidence only after the regression assertions.
            if (delivery != null) {
                RequestProtocolTestSupport.transport(
                        scheduler, delivery, Delivery.Result.delivered());
            }
            if (route != null) {
                reportDecode(null, true);
                RequestProtocolTestSupport.applyPrefillStatus(
                        scheduler,
                        context,
                        prefill,
                        RoleType.PREFILL,
                        new PrefillEndpoint.PrefillRequestStatus(
                                route, PrefillEndpoint.PrefillRequestStatus.Kind.COMPLETED, 0L));
            }
        } finally {
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
            org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(scheduler).close();
        }
    }

    @Test
    void clientCancellationDuringUnknownPreemptionStartsCleanupAndCompletesWithoutReleaseProof()
            throws Exception {
        var preemption = beginPreemption();
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        scheduler, preemption, PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertFalse(response.isDone(), "the RPC has started but its ACK has not arrived");

        scheduler.cancel(String.valueOf(REQUEST_ID), BATCH_ID, CancelReason.CLIENT_CANCELLED);
        // Repeated nonterminal heartbeats must not gate the frontend cancellation on inactivity
        // expiry.
        for (int heartbeat = 0; heartbeat < 3; heartbeat++) {
            reportDecode(TaskPhase.RECEIVED, false);
        }
        drainResponses();

        assertAll(
                "frontend cancellation progresses while remote ownership remains unresolved",
                () ->
                        verify(
                                        SchedulerTestSupport.dispatcher(
                                                org.flexlb.balance.scheduler.SchedulerTestSupport
                                                        .runtime(scheduler)),
                                        times(1))
                                .stop(delivery),
                () -> {
                    var result = response.getNow(null);
                    assertNotNull(
                            result,
                            "cancellation response must be published before Worker release proof");
                    assertFalse(result.isSuccess());
                    assertEquals(
                            StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), result.getCode());
                },
                () ->
                        assertTrue(
                                scheduler.repository.isCurrent(context),
                                "unproven remote ownership stays registered"),
                () ->
                        assertTrue(
                                victimClaimed(),
                                "ordinary cancellation cannot fabricate victim release proof"),
                () -> assertNotEquals(Phase.CLOSED, context.snapshot().phase()),
                () -> assertFalse(delivery.completion().toCompletableFuture().isDone()));
    }

    @Test
    void historicalAllocationBeforeNotFoundDoesNotReleaseVictimOrItsKvHoldOnReceived()
            throws Exception {
        RequestProtocolTestSupport.transport(scheduler, delivery, Delivery.Result.delivered());
        drainResponses();
        assertTrue(response.getNow(null).isSuccess(), "ACK precedes this preemption attempt");
        reportDecode(TaskPhase.KV_ALLOCATED, false);
        assertTrue(context.snapshot().engine().decodeAccepted());
        var preemption = beginPreemption();

        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        scheduler, preemption, PreemptionCancelPhase.NOT_FOUND_STALE));
        boolean claimedAfterNotFound = victimClaimed();
        reportDecode(TaskPhase.RECEIVED, false);
        var snapshot = decode.resourceSnapshot();
        // Exclude the incoming request's separate provisional charge from the victim's conservative
        // hold.
        long incomingKv = snapshot.requests().get(String.valueOf(INCOMING_ID)).expectedKvTokens();
        long victimKv = snapshot.routing().realKvUsed() - incomingKv;

        assertAll(
                "historical acceptance is not fresh restoration proof for this attempt",
                () ->
                        assertTrue(
                                claimedAfterNotFound,
                                "NOT_FOUND alone must preserve the exact victim claim"),
                () ->
                        assertTrue(
                                victimClaimed(),
                                "RECEIVED without new allocation proof must preserve the victim claim"),
                () ->
                        assertEquals(
                                KV_TOKENS, victimKv, "conservative victim KV must remain charged"),
                () ->
                        assertNotNull(
                                context.snapshot().preemption(),
                                "the Context must retain the unresolved attempt"),
                () -> assertFalse(preemption.completion().toCompletableFuture().isDone()));
    }

    @Test
    void freshAllocationAfterNotFoundCanResolveTheVictimClaim() {
        var preemption = beginPreemption();
        assertFalse(context.snapshot().engine().decodeAccepted());
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        scheduler, preemption, PreemptionCancelPhase.NOT_FOUND_STALE));
        assertTrue(
                victimClaimed(),
                "without earlier acceptance, NOT_FOUND must wait for new evidence");

        reportDecode(TaskPhase.KV_ALLOCATED, false);

        assertFalse(
                victimClaimed(),
                "new allocation restores real Worker accounting and resolves the claim");
        assertNull(context.snapshot().preemption());
        assertEquals(
                2L * KV_TOKENS,
                decode.routingView().realKvUsed(),
                "Worker allocation and incoming reservation are both charged");
    }

    private PreemptionRegistration beginPreemption() {
        assertEquals(
                DecodeResources.PreemptionBeginResult.SUCCESS,
                decode.beginPreemption(
                        ATTEMPT,
                        List.of(route.decodeReservation()),
                        String.valueOf(INCOMING_ID),
                        KV_TOKENS,
                        KV_TOKENS,
                        70,
                        CAPACITY));
        var registration = scheduler.tryClaim(route.decodeReservation(), ATTEMPT).orElseThrow();
        assertTrue(
                RequestProtocolTestSupport.preemptionPhase(
                        scheduler, registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(
                decode.updatePreemption(
                                ATTEMPT,
                                DecodeResources.PreemptionUpdate.handedOff(
                                        route.decodeReservation()))
                        .status()
                        .applied());
        assertTrue(victimClaimed());
        return registration;
    }

    private boolean victimClaimed() {
        var victim = decode.resourceSnapshot().requests().get(String.valueOf(REQUEST_ID));
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
        task.setRequestId(String.valueOf(REQUEST_ID));
        task.setInputLength(KV_TOKENS);
        task.setPhase(phase);
        report.setRunningTaskInfo(
                phase == null ? Map.of() : Map.of(Long.toString(REQUEST_ID), task));
        report.setFinishedTaskInfo(terminal ? Map.of(Long.toString(REQUEST_ID), task) : Map.of());
        var status = decode.getStatus();
        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange change;
        status.lock.lock();
        try {
            var prepared =
                    status.prepareNewStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, report));
            change = decode.applyPreparedStatus(status, prepared);
        } finally {
            status.lock.unlock();
        }
        decode.publishStatus(change);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
    }

    private void drainResponses() throws Exception {
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
        var drained = new CompletableFuture<Void>();
        SchedulerTestSupport.responsePublication(
                        SchedulerTestSupport.responseCompletions(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(
                                        scheduler)))
                .submit(
                        () -> {
                            drained.complete(null);
                            return true;
                        });
        drained.get(5L, TimeUnit.SECONDS);
    }
}
