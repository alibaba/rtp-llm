package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.RequestLifecycleState;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatchers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Weak-ACK and typed original-Prefill CANCELED transaction tests. */
class AcceptedCancelSchedulerTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";
    private static final long PRIORITY_PREEMPTED = 8429L;

    private ConfigService configService;
    private Router router;
    private PriorityScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private DecodeEndpoint decodeEndpoint;
    private WorkerStatus decodeStatus;
    private FakeCancelChannel cancelChannel;
    private DecodePreemptionCoordinator coordinator;
    private FlexlbConfig config;
    private PrioritySchedulerReporter priorityReporter;
    private PriorityAdmissionScheduler priorityScheduler;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        BatchDispatcher dispatcher = mock(BatchDispatcher.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        priorityReporter = mock(PrioritySchedulerReporter.class);

        config = new FlexlbConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(100);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(10_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(16);
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_RESERVED);
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_ENGINE_OWNED);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests((long) (1));
        when(configService.loadBalanceConfig()).thenReturn(config);

        cancelChannel = new FakeCancelChannel();
        coordinator = new DecodePreemptionCoordinator(cancelChannel);
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                priorityReporter, reporter, cancelChannel, coordinator) {
            @Override
            protected ServerStatus selectPrefillForDecodeEviction(BalanceContext ctx,
                                                                  FlexlbConfig config,
                                                                  String group) {
                return server(RoleType.PREFILL, "10.0.0.1", 8080, 8081,
                        ctx.getRequestId());
            }
        };
        scheduler = new PriorityScheduler(configService, router, endpointRegistry,
                dispatcher, reporter, priorityScheduler, null, cancelChannel);

        WorkerStatus prefill = new WorkerStatus();
        prefill.setIp("10.0.0.1");
        prefill.setPort(8080);
        prefill.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefill);

        decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.2");
        decodeStatus.setPort(8081);
        decodeStatus.setGrpcPort(8082);
        decodeStatus.setAvailableKvCacheTokens(new AtomicLong(10_000));
        decodeStatus.setTotalKvCacheTokens(new AtomicLong(20_000));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeStatus);
        decodeEndpoint = endpointRegistry.getDecode(DECODE_IP_PORT);
        updateDecode(Map.of(), null);
    }

    @AfterEach
    void tearDown() {
        priorityScheduler.shutdown();
        scheduler.shutdown();
    }

    @Test
    void acceptedAckRetainsVictimUntilTypedPrefillCanceled() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.KV_ALLOCATED);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted());

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                execute("2", 70, 500, 500, () -> true);

        await(() -> decodeEndpoint.reservedView().containsKey("2"));
        assertTrue(decodeEndpoint.isConfirmedTracked("1"));
        assertFalse(result.isDone());
        assertEquals(128, decodeEndpoint.inflightHardKvReserved(),
                "the complete incoming demand is held provisionally");

        scheduler.onWorkerStatusUpdate(prefillCanceled("1"));

        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED,
                result.get(1, TimeUnit.SECONDS).code());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));
        assertTrue(decodeEndpoint.reservedView().containsKey("2"));
    }

    @Test
    void lateEnqueueSuccessDuringCancelRequestedWaitsForTypedCanceled() throws Exception {
        BatchItem victim = registerDispatchedShadowVictim("11", 30);
        RequestLifecycleSnapshot dispatched = scheduler.getRequestState("11", 0);
        long batchId = dispatched.batchId();
        assertEquals(RequestLifecycleState.DISPATCHING, dispatched.state());
        assertEquals(1, endpointRegistry.getPrefill(PREFILL_IP_PORT).getInflightBatchCount());

        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted());
        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                execute("12", 70, 500, 500, () -> true);
        await(() -> scheduler.getRequestState("11", batchId).state()
                == RequestLifecycleState.CANCEL_REQUESTED);

        assertDoesNotThrow(() -> scheduler.onDelivered(victim));
        assertFalse(victim.future().isDone(),
                "late EnqueueBatch ACK must not publish success after Cancel ACCEPTED");
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState("11", batchId).state());
        assertTrue(decodeEndpoint.reservedView().containsKey("11"),
                "weak ACK retains victim accounting");

        scheduler.onWorkerStatusUpdate(prefillCanceled("11"));

        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED,
                result.get(1, TimeUnit.SECONDS).code());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.reservedView().containsKey("11"));
        assertEquals(0, endpointRegistry.getPrefill(PREFILL_IP_PORT).getInflightBatchCount(),
                "typed cancel must retire the real Prefill batch even when WorkerStatus omits batch_id");
    }

    @Test
    void acceptedCancelDefersAdmissionDeadlineUntilTypedCanceled() throws Exception {
        BatchItem victim = registerDispatchedShadowVictim("13", 30);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted());

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                execute("14", 70, 500, 500, () -> true);
        await(() -> scheduler.getRequestState("13", 0).state()
                == RequestLifecycleState.CANCEL_REQUESTED);

        // Deterministically deliver the same reducer event used by the
        // scheduled admission deadline. The Cancel claim already owns the
        // entry, so this must be retained instead of completing the future.
        scheduler.onTimeout(victim,
                new TimeoutException("admission deadline exceeded"));

        assertFalse(victim.future().isDone(),
                "deadline must not overwrite an accepted priority Cancel");
        assertTrue(decodeEndpoint.reservedView().containsKey("13"),
                "deadline must not release claimed Decode accounting");

        scheduler.onWorkerStatusUpdate(prefillCanceled("13"));

        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED,
                result.get(1, TimeUnit.SECONDS).code());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.reservedView().containsKey("13"));
    }

    @Test
    void runningVictimUsesTheSameOriginalPrefillCancelPath() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) -> {
            scheduler.onWorkerStatusUpdate(prefillCanceled(requestId));
            return CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted());
        };

        DecodePreemptionCoordinator.ExecutionResult result =
                execute("2", 70, 500, 500, () -> true).get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED, result.code());
        assertEquals("10.0.0.1", cancelChannel.lastTarget.prefillIp());
        assertEquals(8081, cancelChannel.lastTarget.prefillGrpcPort());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
    }

    @Test
    void tombstonedVictimSettlesImmediatelyAndCommitsIncoming() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.tombstoned());

        DecodePreemptionCoordinator.ExecutionResult result =
                execute("2", 70, 500, 500, () -> true)
                        .get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED, result.code());
        assertEquals(PreemptionAttempt.State.COMMITTED, result.attempt().state());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));
        assertTrue(decodeEndpoint.reservedView().containsKey("2"),
                "TOMBSTONED capacity belongs to the committed incoming request");
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState("1", 0).state());
    }

    @Test
    void notFoundReplaysGenericTerminalCapturedDuringCancelInFlight() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) -> {
            scheduler.onWorkerStatusUpdate(decodeFinished(String.valueOf(requestId), 9001));
            return CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.notFound());
        };

        DecodePreemptionCoordinator.ExecutionResult result =
                execute("2", 70, 500, 500, () -> true).get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.REPLAN_NOT_FOUND, result.code());
        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));
        assertFalse(decodeEndpoint.reservedView().containsKey("2"));
    }

    @Test
    void notFoundReplaysDispatchTimeoutWithoutEarlyAccountingRelease() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        AtomicBoolean retainedDuringRpc = new AtomicBoolean();
        cancelChannel.handler = (ignored, requestId) -> {
            scheduler.onTimeout(victim, new TimeoutException("late stage2"));
            retainedDuringRpc.set(decodeEndpoint.isConfirmedTracked(String.valueOf(requestId))
                    && scheduler.getRequestState(String.valueOf(requestId), 0) != null
                    && !victim.future().isDone());
            return CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.notFound());
        };

        DecodePreemptionCoordinator.ExecutionResult result =
                execute("2", 70, 500, 500, () -> true).get(1, TimeUnit.SECONDS);

        assertTrue(retainedDuringRpc.get(),
                "dispatch timeout must defer while the preemption claim owns cleanup");
        assertEquals(DecodePreemptionCoordinator.ResultCode.REPLAN_NOT_FOUND, result.code());
        Response response = victim.future().get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));
    }

    @Test
    void transportUnknownRetainsLocalFailureUntilAuthoritativeWorkerTerminal() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) -> {
            scheduler.onFailure(victim, new IllegalStateException("natural enqueue failure"));
            assertTrue(decodeEndpoint.isConfirmedTracked(String.valueOf(requestId)));
            assertFalse(victim.future().isDone());
            return CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.failed());
        };

        DecodePreemptionCoordinator.ExecutionResult result =
                execute("2", 70, 500, 100, () -> true).get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED, result.code());
        assertFalse(victim.future().isDone(),
                "transport UNKNOWN must not treat a local failure as release proof");
        assertTrue(decodeEndpoint.isConfirmedTracked("1"));

        scheduler.onWorkerStatusUpdate(decodeFinished("1", 9001));

        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));
    }

    @Test
    void transportUnknownRetainsLocalTimeoutUntilLateTypedCanceled() throws Exception {
        BatchItem victim = registerConfirmedVictim("3", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) -> {
            scheduler.onTimeout(victim,
                    new TimeoutException("local admission timeout"));
            return CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.failed());
        };

        DecodePreemptionCoordinator.ExecutionResult result =
                execute("4", 70, 500, 100, () -> true).get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED, result.code());
        assertFalse(victim.future().isDone(),
                "transport UNKNOWN must retain a local timeout for typed Cancel reconciliation");
        assertTrue(decodeEndpoint.isConfirmedTracked("3"));

        scheduler.onWorkerStatusUpdate(prefillCanceled("3"));

        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.isConfirmedTracked("3"));
        scheduler.onWorkerStatusUpdate(prefillCanceled("3"));
        assertEquals(0, decodeEndpoint.getTotalLoad(),
                "late duplicate typed status must not release resources twice");
    }

    @Test
    void acceptedCancelIgnoresYieldedTerminalUntilTypedCanceled() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) -> {
            scheduler.finishYielded(victim, "ordinary yielded race");
            assertTrue(decodeEndpoint.isConfirmedTracked(String.valueOf(requestId)));
            assertFalse(victim.future().isDone());
            return CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted());
        };

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                execute("2", 70, 500, 500, () -> true);
        await(() -> cancelChannel.cancelCount.get() == 1);
        assertFalse(victim.future().isDone());

        scheduler.onWorkerStatusUpdate(prefillCanceled("1"));

        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED,
                result.get(1, TimeUnit.SECONDS).code());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
    }

    @Test
    void mixedCanceledAndNotFoundCannotReplanAfterCancellationSideEffect() throws Exception {
        List<BatchItem> victims = registerTwoConfirmedVictims();
        cancelChannel.handler = (ignored, requestId) -> CompletableFuture.completedFuture(
                requestId.equals("1")
                        ? EngineCancelChannel.CancelOutcome.accepted()
                        : EngineCancelChannel.CancelOutcome.notFound());

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                executeAllVictims("3", 70, 500, 500, () -> true);
        await(() -> cancelChannel.cancelCount.get() == 2);
        scheduler.onWorkerStatusUpdate(prefillCanceled("1"));

        DecodePreemptionCoordinator.ExecutionResult outcome =
                result.get(1, TimeUnit.SECONDS);
        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED, outcome.code());
        assertEquals("cancel_partial_not_found", outcome.detail());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victims.get(0).future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(victims.get(1).future().isDone(),
                "NOT_FOUND sibling remains live and must not be rescheduled as a new request");
        assertFalse(decodeEndpoint.reservedView().containsKey("3"),
                "partial cancellation must release the provisional incoming reservation");
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));
        assertTrue(decodeEndpoint.isConfirmedTracked("2"));
    }

    @Test
    void tenAcceptedVictimsCommitAndSettleAccountingExactlyOnce() throws Exception {
        List<BatchItem> victims = registerConfirmedVictims(10);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted());

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                executeAllVictims("100", 70, 500, 500, () -> true);
        await(() -> cancelChannel.cancelCount.get() == victims.size());
        for (BatchItem victim : victims) {
            await(() -> scheduler.getRequestState(victim.requestId(), 0).state()
                    == RequestLifecycleState.CANCEL_REQUESTED);
        }

        scheduler.onWorkerStatusUpdate(prefillCanceled(
                victims.stream().map(BatchItem::requestId).toList()));

        DecodePreemptionCoordinator.ExecutionResult outcome =
                result.get(1, TimeUnit.SECONDS);
        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED, outcome.code());
        assertEquals(10, outcome.attempt().victims().size(),
                "the execution transaction must preserve every planned victim");
        assertEquals(1, decodeEndpoint.getTotalLoad(),
                "only the committed incoming reservation remains charged");
        assertEquals(128, decodeEndpoint.inflightHardKvReserved());
        for (BatchItem victim : victims) {
            assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                    victim.future().get(1, TimeUnit.SECONDS).getCode());
            assertFalse(decodeEndpoint.isConfirmedTracked(victim.requestId()));
        }

        // Repeated authoritative WorkerStatus must be fenced by the same
        // per-victim settlement token and cannot decrement a second time.
        scheduler.onWorkerStatusUpdate(prefillCanceled(
                victims.stream().map(BatchItem::requestId).toList()));
        assertEquals(1, decodeEndpoint.getTotalLoad());
        assertEquals(128, decodeEndpoint.inflightHardKvReserved());
    }

    @Test
    void acceptedOnlyConfigPlansTenRunningVictimsAndAdmitsP70AfterTypedCanceled()
            throws Exception {
        // Mirrors the production knobs from the incident: accepted/running
        // Engine Cancel is enabled independently, while Master-local reserved
        // eviction stays disabled. Fourteen confirmed requests against a
        // concurrency limit of five require ten victims before P70 can fit.
        SchedulingTestConfig.disallowVictim(config, VictimStage.DECODE_RESERVED);
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_ENGINE_OWNED);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests((long) (5));
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(3_600_000);
        SchedulingTestConfig.engineCancellation(config).setAckTimeoutMs(500);
        SchedulingTestConfig.engineCancellation(config).setCompletionTimeoutMs(1_000);

        List<BatchItem> victims = registerConfirmedVictims(14);
        List<String> cancelRoutes = Collections.synchronizedList(
                new ArrayList<>());
        cancelChannel.handler = (target, requestId) -> {
            assertEquals("10.0.0.1", target.prefillIp());
            assertEquals(8081, target.prefillGrpcPort());
            cancelRoutes.add(requestId);
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelOutcome.accepted());
        };
        when(router.route(ArgumentMatchers.any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_DECODE_WORKER));

        BalanceContext incoming = context("100", 70);
        long now = System.currentTimeMillis();
        incoming.setSchedulingMetadata(SchedulingMetadata.explicit(70, now + 5_000));
        CompletableFuture<Response> incomingResponse = scheduler.submit(incoming);

        await(() -> cancelChannel.cancelCount.get() == 10 && cancelRoutes.size() == 10);
        assertEquals(10, cancelRoutes.size());
        assertEquals(10, cancelRoutes.stream().distinct().count());
        assertTrue(cancelRoutes.stream().allMatch(requestId -> Integer.parseInt(requestId) >= 1 && Integer.parseInt(requestId) <= 14));
        DecodeEndpointSnapshot canceling = DecodeEndpointSnapshot.capture(decodeEndpoint, 5);
        assertEquals(1, canceling.reserved().stream()
                .filter(victim -> victim.requestId().equals(String.valueOf(100L))).count(),
                "incoming reservation is provisional and cannot be dispatched yet");
        assertEquals(4, canceling.running().size(),
                "ten RUNNING victims must be hidden from the plannable layer while Cancel waits");
        assertTrue(canceling.running().stream()
                .allMatch(victim -> victim.phase() == DecodeTaskPhase.RUNNING));
        assertTrue(endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher()
                .queueManager().snapshot().items().stream()
                .noneMatch(item -> item.requestId().equals(String.valueOf(100L))),
                "weak Cancel ACK must not dispatch or queue the incoming request");
        assertFalse(incomingResponse.isDone());
        assertEquals(15, decodeEndpoint.getTotalLoad(),
                "ACK retains all fourteen victims plus the provisional incoming reservation");

        scheduler.onWorkerStatusUpdate(prefillCanceled(cancelRoutes));

        await(() -> endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher()
                .queueManager().snapshot().items().stream()
                .anyMatch(item -> item.requestId().equals(String.valueOf(100L))));
        assertFalse(incomingResponse.isDone(),
                "successful admission remains live in the Prefill queue");
        assertEquals("decode_evict", incoming.getPlanType());
        assertEquals(10, incoming.getVictimCount());
        assertEquals(5, decodeEndpoint.getTotalLoad(),
                "four unselected RUNNING requests plus P70 consume exactly five slots");
        assertEquals(128, decodeEndpoint.inflightHardKvReserved());
        for (String requestId : cancelRoutes) {
            BatchItem victim = victims.get(Integer.parseInt(requestId) - 1);
            assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                    victim.future().get(1, TimeUnit.SECONDS).getCode());
            assertFalse(decodeEndpoint.isConfirmedTracked(String.valueOf(requestId)));
        }
        assertEquals(4, victims.stream()
                .filter(victim -> !victim.future().isDone()).count());
        verify(priorityReporter, times(10)).reportVictim(
                eq(30), eq(70), eq("decode_running"), eq("decode_slot_full"));
        verify(priorityReporter, times(10)).reportCancelRequest(eq(DECODE_IP_PORT), eq(30));
        verify(priorityReporter, times(10)).reportCancelConfirm(eq(DECODE_IP_PORT), eq(30));

        // Duplicate authoritative observations are exactly-once no-ops.
        scheduler.onWorkerStatusUpdate(prefillCanceled(cancelRoutes));
        assertEquals(5, decodeEndpoint.getTotalLoad());
        assertEquals(128, decodeEndpoint.inflightHardKvReserved());
    }

    @Test
    void tenVictimsWithOneNotFoundRemainAPartialFailure() throws Exception {
        List<BatchItem> victims = registerConfirmedVictims(10);
        String notFoundId = victims.getLast().requestId();
        cancelChannel.handler = (ignored, requestId) -> CompletableFuture.completedFuture(
                requestId.equals(notFoundId)
                        ? EngineCancelChannel.CancelOutcome.notFound()
                        : EngineCancelChannel.CancelOutcome.accepted());

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                executeAllVictims("100", 70, 500, 500, () -> true);
        await(() -> cancelChannel.cancelCount.get() == victims.size());
        scheduler.onWorkerStatusUpdate(prefillCanceled(victims.stream()
                .map(BatchItem::requestId)
                .filter(requestId -> requestId != notFoundId)
                .toList()));

        DecodePreemptionCoordinator.ExecutionResult outcome =
                result.get(1, TimeUnit.SECONDS);
        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED, outcome.code());
        assertEquals("cancel_partial_not_found", outcome.detail());
        assertFalse(decodeEndpoint.reservedView().containsKey("100"),
                "partial cancellation must release the provisional incoming reservation");
        assertEquals(1, decodeEndpoint.getTotalLoad(),
                "the NOT_FOUND victim remains charged while nine canceled siblings settle");
        for (BatchItem victim : victims.subList(0, victims.size() - 1)) {
            assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                    victim.future().get(1, TimeUnit.SECONDS).getCode());
            assertFalse(decodeEndpoint.isConfirmedTracked(victim.requestId()));
        }
        assertFalse(victims.getLast().future().isDone());
        assertTrue(decodeEndpoint.isConfirmedTracked(notFoundId));

        scheduler.onWorkerStatusUpdate(prefillCanceled(victims.stream()
                .map(BatchItem::requestId)
                .filter(requestId -> requestId != notFoundId)
                .toList()));
        assertEquals(1, decodeEndpoint.getTotalLoad(),
                "duplicate CANCELED siblings cannot release the NOT_FOUND victim");
    }

    @Test
    void multipleNotFoundVictimsAreNotAReplanSafeSingleVictimCase() throws Exception {
        registerTwoConfirmedVictims();
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.notFound());

        DecodePreemptionCoordinator.ExecutionResult outcome =
                executeAllVictims("3", 70, 500, 500, () -> true)
                        .get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED, outcome.code());
        assertFalse(decodeEndpoint.reservedView().containsKey("3"));
        assertTrue(decodeEndpoint.isConfirmedTracked("1"));
        assertTrue(decodeEndpoint.isConfirmedTracked("2"));
    }

    @Test
    void failedAckDoesNotHoldARequestThatNaturallyFinishes() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.failed());

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                execute("2", 70, 500, 80, () -> true);
        await(() -> cancelChannel.cancelCount.get() == 1);
        scheduler.onWorkerStatusUpdate(decodeFinished("1", 9001));

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED,
                result.get(1, TimeUnit.SECONDS).code());
        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));
        assertFalse(decodeEndpoint.reservedView().containsKey("2"));
    }

    @Test
    void failedAckLateTypedCanceledStillSettlesAfterCompletionTimeout() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.failed());

        DecodePreemptionCoordinator.ExecutionResult result =
                execute("2", 70, 500, 20, () -> true).get(1, TimeUnit.SECONDS);
        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED, result.code());
        assertTrue(decodeEndpoint.isConfirmedTracked("1"));
        assertFalse(decodeEndpoint.reservedView().containsKey("2"));

        scheduler.onWorkerStatusUpdate(prefillCanceled("1"));
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));

        // Duplicate typed status is fenced and cannot decrement again.
        scheduler.onWorkerStatusUpdate(prefillCanceled("1"));
        assertEquals(0, decodeEndpoint.getTotalLoad());
    }

    @Test
    void acceptedAckKeepsPriorityCauseAcrossGenericTerminalAndCompletionTimeout() throws Exception {
        BatchItem victim = registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted());

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                execute("2", 70, 500, 20, () -> true);
        await(() -> cancelChannel.cancelCount.get() == 1);
        scheduler.onWorkerStatusUpdate(decodeFinished("1", 9001));

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED,
                result.get(1, TimeUnit.SECONDS).code());
        assertFalse(victim.future().isDone(),
                "ordinary Decode terminal is deferred after an accepted Cancel");
        assertTrue(decodeEndpoint.isConfirmedTracked("1"));
        assertFalse(decodeEndpoint.reservedView().containsKey("2"));

        scheduler.onWorkerStatusUpdate(prefillCanceled("1"));
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEndpoint.isConfirmedTracked("1"));

        scheduler.onWorkerStatusUpdate(prefillCanceled("1"));
        assertEquals(0, decodeEndpoint.getTotalLoad(),
                "late duplicate typed status must not settle resources twice");
    }

    @Test
    void closedAdmissionGateAbortsIncomingAfterVictimSettlement() throws Exception {
        registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        AtomicBoolean admissionOpen = new AtomicBoolean(true);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted());

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                execute("2", 70, 500, 500, admissionOpen::get);
        await(() -> decodeEndpoint.reservedView().containsKey("2"));
        admissionOpen.set(false);
        scheduler.onWorkerStatusUpdate(prefillCanceled("1"));

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED,
                result.get(1, TimeUnit.SECONDS).code());
        assertFalse(decodeEndpoint.reservedView().containsKey("2"),
                "an admission timeout must not leave an orphan incoming reservation");
    }

    @Test
    void admissionDeadlineClosesAsyncCancelBeforeInflightRegistration() throws Exception {
        BatchItem victim = registerConfirmedVictim("51", 30, TaskPhase.RUNNING);
        CompletableFuture<EngineCancelChannel.CancelOutcome> ack = new CompletableFuture<>();
        cancelChannel.handler = (ignored, requestId) -> ack;
        when(router.route(ArgumentMatchers.any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_DECODE_WORKER));

        BalanceContext incoming = context("52", 70);
        long now = System.currentTimeMillis();
        incoming.setSchedulingMetadata(SchedulingMetadata.explicit(70, now + 100));
        CompletableFuture<Response> responseFuture = scheduler.submit(incoming);

        await(() -> cancelChannel.cancelCount.get() == 1
                && decodeEndpoint.reservedView().containsKey("52"));
        await(() -> {
            RequestLifecycleSnapshot state = scheduler.getRequestState("52", 0);
            return state != null
                    && state.state() == RequestLifecycleState.CANCEL_REQUESTED;
        });
        assertFalse(responseFuture.isDone(),
                "deadline settlement waits for admission-mutation cleanup");

        ack.complete(EngineCancelChannel.CancelOutcome.accepted());
        await(() -> scheduler.getRequestState("51", 0).state()
                == RequestLifecycleState.CANCEL_REQUESTED);
        scheduler.onWorkerStatusUpdate(prefillCanceled("51"));

        Response response = responseFuture.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState("52", 0).state());

        await(() -> !decodeEndpoint.reservedView().containsKey("52"));
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
        assertTrue(endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher()
                .queueManager().snapshot().items().stream()
                .noneMatch(item -> item.requestId().equals(String.valueOf(52L))));
    }

    @Test
    void asyncCancelHandoffBeforeAdmissionDeadlineUsesInflightReducer() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(3_600_000);
        BatchItem victim = registerConfirmedVictim("61", 30, TaskPhase.RUNNING);
        CompletableFuture<EngineCancelChannel.CancelOutcome> ack = new CompletableFuture<>();
        cancelChannel.handler = (ignored, requestId) -> ack;
        when(router.route(ArgumentMatchers.any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_DECODE_WORKER));

        BalanceContext incoming = context("62", 70);
        long now = System.currentTimeMillis();
        incoming.setSchedulingMetadata(SchedulingMetadata.explicit(70, now + 500));
        CompletableFuture<Response> responseFuture = scheduler.submit(incoming);

        await(() -> cancelChannel.cancelCount.get() == 1
                && decodeEndpoint.reservedView().containsKey("62"));
        ack.complete(EngineCancelChannel.CancelOutcome.accepted());
        await(() -> scheduler.getRequestState("61", 0).state()
                == RequestLifecycleState.CANCEL_REQUESTED);
        scheduler.onWorkerStatusUpdate(prefillCanceled("61"));

        await(() -> endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher()
                .queueManager().snapshot().items().stream()
                .anyMatch(item -> item.requestId().equals(String.valueOf(62L))));
        assertFalse(responseFuture.isDone(),
                "a committed delivery remains live until the admission deadline reducer runs");

        Response response = responseFuture.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState("62", 0).state());
        assertFalse(decodeEndpoint.reservedView().containsKey("62"));
        assertTrue(endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher()
                .queueManager().snapshot().items().stream()
                .noneMatch(item -> item.requestId().equals(String.valueOf(62L))));
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                victim.future().get(1, TimeUnit.SECONDS).getCode());
    }

    @Test
    void committedEnginePreemptionIgnoresTelemetryFailure() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(3_600_000);
        registerConfirmedVictim("63", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) ->
                CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.accepted());
        when(router.route(ArgumentMatchers.any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_DECODE_WORKER));
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(priorityReporter)
                .reportEvictionCommit(eq(70), anyString(), eq("success"));
        doThrow(new IllegalStateException("cancel metrics unavailable"))
                .when(priorityReporter)
                .reportCancelRequest(eq(DECODE_IP_PORT), eq(30));

        CompletableFuture<Response> response = scheduler.submit(context("64", 70));
        await(() -> cancelChannel.cancelCount.get() == 1
                && decodeEndpoint.reservedView().containsKey("64"));
        scheduler.onWorkerStatusUpdate(prefillCanceled("63"));

        await(() -> endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher()
                .queueManager().snapshot().items().stream()
                .anyMatch(item -> item.requestId().equals(String.valueOf(64L))));
        assertFalse(response.isDone(),
                "telemetry failure must not reverse a committed admission");
        assertTrue(decodeEndpoint.reservedView().containsKey("64"));

        assertTrue(response.cancel(false));
        await(() -> !decodeEndpoint.reservedView().containsKey("64"));
        assertFalse(scheduler.ownsRequestGeneration("64"));
    }

    @Test
    void completionDeadlineStartsAfterSlowAckPhase() throws Exception {
        registerConfirmedVictim("1", 30, TaskPhase.RUNNING);
        cancelChannel.handler = (ignored, requestId) -> CompletableFuture.supplyAsync(() -> {
            sleep(60);
            CompletableFuture.runAsync(() -> {
                sleep(30);
                scheduler.onWorkerStatusUpdate(prefillCanceled(requestId));
            });
            return EngineCancelChannel.CancelOutcome.accepted();
        });

        DecodePreemptionCoordinator.ExecutionResult result =
                execute("2", 70, 100, 80, () -> true).get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.COMMITTED, result.code());
    }

    @Test
    void admissionLeaseCleanupWinningFirstPreventsCoordinatorMutation() throws Exception {
        BatchItem victim = dummyItem("21", 30);
        assertTrue(scheduler.registerInflight(victim));
        decodeEndpoint.reserve("21", 128, 136, 30);
        DecodeRequestSnapshot staleVictim = DecodeEndpointSnapshot
                .capture(decodeEndpoint, 1).reserved().getFirst();

        AdmissionLease lease = new AdmissionLease(
                victim, decodeEndpoint, null, scheduler, 0, null, null);
        lease.close();

        DecodePreemptionCoordinator.ExecutionResult result = coordinator.execute(
                request(decodeEndpoint, "22", staleVictim, () -> true), scheduler)
                .get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED, result.code());
        assertEquals(0, cancelChannel.cancelCount.get());
        assertFalse(decodeEndpoint.reservedView().containsKey("21"));
        assertFalse(decodeEndpoint.reservedView().containsKey("22"),
                "failed cleanup-vs-claim race must not provision the incoming request");
    }

    @Test
    void admissionCleanupAndPreemptionClaimHaveExactlyOneOwner() throws Exception {
        for (int i = 0; i < 50; i++) {
            long requestId = 1_000L + i;
            long token = 10_000L + i;
            BatchItem victim = dummyItem(String.valueOf(requestId), 30);
            assertTrue(scheduler.registerInflight(victim));
            decodeEndpoint.reserve(String.valueOf(requestId), 32, 40, 30);
            AdmissionLease lease = new AdmissionLease(
                    victim, decodeEndpoint, null, scheduler, 0, null, null);

            CountDownLatch start = new CountDownLatch(1);
            AtomicBoolean claimWon = new AtomicBoolean();
            CompletableFuture<Void> claim = CompletableFuture.runAsync(() -> {
                awaitLatch(start);
                claimWon.set(scheduler.claimForPreemption(
                        String.valueOf(requestId), token, "deterministic ownership race"));
            });
            CompletableFuture<Void> cleanup = CompletableFuture.runAsync(() -> {
                awaitLatch(start);
                lease.close();
            });
            start.countDown();
            CompletableFuture.allOf(claim, cleanup).get(1, TimeUnit.SECONDS);

            if (claimWon.get()) {
                assertTrue(decodeEndpoint.reservedView().containsKey(String.valueOf(requestId)),
                        "claim winner must retain Decode accounting");
                assertTrue(scheduler.releasePreemptionClaim(String.valueOf(requestId), token));
            }
            assertFalse(decodeEndpoint.reservedView().containsKey(String.valueOf(requestId)));
            assertTrue(scheduler.getRequestState(String.valueOf(requestId), 0) == null);
        }
    }

    @Test
    void softTimeoutTransfersNotFoundPreemptionToEngineFenceUntilTombstoned()
            throws Exception {
        BatchItem victim = registerDispatchedShadowVictim("31", 30);
        AdmissionLease lease = new AdmissionLease(
                victim, decodeEndpoint, null, scheduler, 0, null, null);
        lease.bindTo(victim.future());
        Response accepted = new Response();
        accepted.setSuccess(true);
        victim.future().complete(accepted);

        CompletableFuture<EngineCancelChannel.CancelOutcome> ack = new CompletableFuture<>();
        cancelChannel.handler = (ignored, requestId) -> cancelChannel.cancelCount.get() == 1
                ? ack
                : CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.tombstoned());
        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> result =
                execute("32", 70, 500, 500, () -> true);
        await(() -> cancelChannel.cancelCount.get() == 1);

        lease.reconcileAfterDeliveryTimeout();
        assertTrue(decodeEndpoint.reservedView().containsKey("31"));
        assertTrue(scheduler.getRequestState("31", 0) != null,
                "claim owns both Decode accounting and inflight registration");

        ack.complete(EngineCancelChannel.CancelOutcome.notFound());

        assertEquals(DecodePreemptionCoordinator.ResultCode.REPLAN_NOT_FOUND,
                result.get(1, TimeUnit.SECONDS).code());
        await(() -> cancelChannel.cancelCount.get() >= 2);
        await(() -> !decodeEndpoint.reservedView().containsKey("31"));
        await(() -> scheduler.getInflightSize() == 0);
        assertFalse(decodeEndpoint.reservedView().containsKey("31"));
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState("31", 0).state(),
                "TOMBSTONED releases inflight ownership but keeps a bounded terminal tombstone");
    }

    @Test
    void preRpcAbortReplaysLeaseCleanupObservedAfterClaim() throws Exception {
        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.9");
        status.setPort(8090);
        status.setAvailableKvCacheTokens(new AtomicLong(10_000));
        status.setTotalKvCacheTokens(new AtomicLong(20_000));
        AbortOnBeginDecodeEndpoint endpoint = new AbortOnBeginDecodeEndpoint(status);

        BatchItem victim = dummyItem("41", 30, endpoint);
        assertTrue(scheduler.registerInflight(victim));
        endpoint.reserve("41", 128, 136, 30);
        DecodeRequestSnapshot victimSnapshot = DecodeEndpointSnapshot
                .capture(endpoint, 1).reserved().getFirst();
        AdmissionLease lease = new AdmissionLease(victim, endpoint, null, scheduler,
                0, null, null);
        lease.bindTo(victim.future());
        endpoint.beforeBegin = () -> victim.future().completeExceptionally(
                new TimeoutException("public admission timeout"));

        DecodePreemptionCoordinator.ExecutionResult result = coordinator.execute(
                request(endpoint, "42", victimSnapshot, () -> true), scheduler)
                .get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONFLICT, result.code());
        assertEquals(0, cancelChannel.cancelCount.get(),
                "pre-RPC abort must not invoke the Cancel channel");
        assertFalse(endpoint.reservedView().containsKey("41"));
        assertFalse(endpoint.reservedView().containsKey("42"));
        assertTrue(scheduler.getRequestState("41", 0) == null,
                "releasePreemptionClaim must replay, not drop, deferred lease cleanup");
    }

    private CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> execute(
            String incomingRequestId,
            int incomingPriority,
            long ackTimeoutMs,
            long completionTimeoutMs,
            BooleanSupplier admissionOpen) {
        DecodeEndpointSnapshot snapshot = DecodeEndpointSnapshot.capture(decodeEndpoint, 1);
        List<DecodeRequestSnapshot> victims = victims(snapshot);
        DecodePreemptionCoordinator.Request request =
                new DecodePreemptionCoordinator.Request(
                        decodeEndpoint, snapshot.admissionVersion(), true,
                        incomingRequestId, 128, 136, incomingPriority,
                        List.of(victims.get(0)), ackTimeoutMs, completionTimeoutMs,
                        admissionOpen, "preempted by higher-priority request " + incomingRequestId);
        return coordinator.execute(request, scheduler);
    }

    private DecodePreemptionCoordinator.Request request(
            DecodeEndpoint endpoint,
            String incomingRequestId,
            DecodeRequestSnapshot victim,
            BooleanSupplier admissionOpen) {
        return new DecodePreemptionCoordinator.Request(
                endpoint, endpoint.admissionVersion(), false,
                incomingRequestId, 128, 136, 70,
                List.of(victim), 500, 500, admissionOpen,
                "preempted by higher-priority request " + incomingRequestId);
    }

    private CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> executeAllVictims(
            String incomingRequestId,
            int incomingPriority,
            long ackTimeoutMs,
            long completionTimeoutMs,
            BooleanSupplier admissionOpen) {
        DecodeEndpointSnapshot snapshot = DecodeEndpointSnapshot.capture(decodeEndpoint, 1);
        List<DecodeRequestSnapshot> victims = victims(snapshot);
        DecodePreemptionCoordinator.Request request =
                new DecodePreemptionCoordinator.Request(
                        decodeEndpoint, snapshot.admissionVersion(), true,
                        incomingRequestId, 128, 136, incomingPriority,
                        victims, ackTimeoutMs, completionTimeoutMs,
                        admissionOpen,
                        "preempted by higher-priority request " + incomingRequestId);
        return coordinator.execute(request, scheduler);
    }

    private BatchItem registerConfirmedVictim(String requestId, int priority, TaskPhase phase) {
        BatchItem item = dummyItem(requestId, priority);
        assertTrue(scheduler.registerInflight(item));
        decodeEndpoint.reserve(requestId, 128, 136, priority);
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(phase);
        task.setInputLength(128);
        updateDecode(Map.of(String.valueOf(requestId), task), null);
        return item;
    }

    private BatchItem registerDispatchedShadowVictim(String requestId, int priority) {
        BatchItem item = dummyItem(requestId, priority);
        assertTrue(scheduler.registerInflight(item));
        decodeEndpoint.reserve(requestId, 128, 136, priority);
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("stage2_test", 0));
        return item;
    }

    private static List<DecodeRequestSnapshot> victims(DecodeEndpointSnapshot snapshot) {
        if (!snapshot.running().isEmpty()) {
            return snapshot.running();
        }
        if (!snapshot.accepted().isEmpty()) {
            return snapshot.accepted();
        }
        return snapshot.reserved();
    }

    private List<BatchItem> registerTwoConfirmedVictims() {
        BatchItem first = dummyItem("1", 30);
        BatchItem second = dummyItem("2", 30);
        assertTrue(scheduler.registerInflight(first));
        assertTrue(scheduler.registerInflight(second));
        decodeEndpoint.reserve("1", 128, 136, 30);
        decodeEndpoint.reserve("2", 128, 136, 30);

        TaskInfo firstTask = new TaskInfo();
        firstTask.setRequestId("1");
        firstTask.setPhase(TaskPhase.RUNNING);
        firstTask.setInputLength(128);
        TaskInfo secondTask = new TaskInfo();
        secondTask.setRequestId("2");
        secondTask.setPhase(TaskPhase.RUNNING);
        secondTask.setInputLength(128);
        updateDecode(Map.of("1", firstTask, "2", secondTask), null);
        return List.of(first, second);
    }

    private List<BatchItem> registerConfirmedVictims(int count) {
        List<BatchItem> victims = new ArrayList<>(count);
        Map<String, TaskInfo> tasks = new LinkedHashMap<>();
        for (int i = 1; i <= count; i++) {
            long requestId = i;
            BatchItem victim = dummyItem(String.valueOf(requestId), 30);
            assertTrue(scheduler.registerInflight(victim));
            decodeEndpoint.reserve(String.valueOf(requestId), 128, 136, 30);
            TaskInfo task = new TaskInfo();
            task.setRequestId(String.valueOf(requestId));
            task.setPhase(TaskPhase.RUNNING);
            task.setInputLength(128);
            tasks.put(String.valueOf(requestId), task);
            victims.add(victim);
        }
        updateDecode(tasks, null);
        return List.copyOf(victims);
    }

    private void updateDecode(Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        decodeEndpoint.onWorkerStatusUpdate(decodeStatus, response);
    }

    private static WorkerStatusResponse prefillCanceled(String requestId) {
        return prefillCanceled(List.of(requestId));
    }

    private static WorkerStatusResponse prefillCanceled(List<String> requestIds) {
        Map<String, TaskInfo> finished = new LinkedHashMap<>();
        for (String requestId : requestIds) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(String.valueOf(requestId));
            task.setErrorCode(PRIORITY_PREEMPTED);
            task.setPriorityPreemptionProgress(PriorityPreemptionProgress.CANCELED);
            finished.put(String.valueOf(requestId), task);
        }
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(finished);
        return response;
    }

    private static WorkerStatusResponse decodeFinished(String requestId, long errorCode) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setErrorCode(errorCode);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setFinishedTaskInfo(Map.of(String.valueOf(requestId), task));
        return response;
    }

    private BatchItem dummyItem(String requestId, int priority) {
        return dummyItem(requestId, priority, decodeEndpoint);
    }

    private BatchItem dummyItem(String requestId, int priority, DecodeEndpoint endpoint) {
        Response route = successRoute(requestId);
        return new BatchItem(context(requestId, priority), new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL),
                PriorityScheduler.findServer(route, RoleType.DECODE),
                endpointRegistry.getPrefill(PREFILL_IP_PORT), endpoint,
                System.currentTimeMillis());
    }

    private static BalanceContext context(String requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(new FlexlbConfig());
        return context;
    }

    private static Response successRoute(String requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort,
                                       int grpcPort, String requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setGroup("g1");
        status.setRequestId(requestId);
        return status;
    }

    private static void await(BooleanSupplier condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() >= deadline) {
                throw new AssertionError("condition not met within 2 seconds");
            }
            TimeUnit.MILLISECONDS.sleep(5);
        }
    }

    private static void sleep(long millis) {
        try {
            TimeUnit.MILLISECONDS.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    private static void awaitLatch(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    private static final class FakeCancelChannel implements EngineCancelChannel {
        private final AtomicInteger cancelCount = new AtomicInteger();
        private volatile CancelTarget lastTarget;
        private volatile BiFunction<CancelTarget, String, CompletableFuture<CancelOutcome>> handler =
                (ignored, requestId) -> CompletableFuture.completedFuture(CancelOutcome.accepted());

        @Override
        public boolean isSupported(DecodeEndpoint endpoint) {
            return true;
        }

        @Override
        public CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                                       String requestId,
                                                       long timeoutMs) {
            cancelCount.incrementAndGet();
            lastTarget = target;
            return handler.apply(target, requestId);
        }
    }

    private static final class AbortOnBeginDecodeEndpoint extends DecodeEndpoint {
        private Runnable beforeBegin;

        private AbortOnBeginDecodeEndpoint(WorkerStatus status) {
            super(status);
        }

        @Override
        public PreemptionBeginResult beginPriorityPreemption(
                long attemptToken,
                List<String> victimIds,
                String incomingRequestId,
                long incomingKvTokens,
                long incomingExpectedKvTokens,
                int incomingPriority,
                long expectedAdmissionVersion,
                boolean requireVersionMatch) {
            beforeBegin.run();
            return PreemptionBeginResult.VERSION_MISMATCH;
        }
    }
}
