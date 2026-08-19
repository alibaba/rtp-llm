package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AutoTpmTransactionOwnershipTest {

    @Test
    void planConstructionFailureReleasesRouterReservation() throws Exception {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);

        Router router = mock(Router.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        PlanCommitter planCommitter = mock(PlanCommitter.class);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        org.flexlb.balance.strategy.PrefillTimePredictor predictor =
                mock(org.flexlb.balance.strategy.PrefillTimePredictor.class);
        when(prefill.getPredictor()).thenReturn(predictor);
        when(predictor.estimateMs(anyLong(), anyLong()))
                .thenThrow(new IllegalStateException("prediction failed"));

        long requestId = 7_001L;
        WorkerStatus decodeStatus = new WorkerStatus();
        decodeStatus.setRole(RoleType.DECODE);
        DecodeEndpoint decode = new DecodeEndpoint(decodeStatus);
        when(endpointRegistry.getPrefill("10.0.0.1:8080")).thenReturn(prefill);
        when(endpointRegistry.getDecode("10.0.0.2:8081")).thenReturn(decode);

        Response route = new Response();
        route.setSuccess(true);
        ServerStatus prefillStatus = server(RoleType.PREFILL, "10.0.0.1", 8080);
        ServerStatus decodeServerStatus = server(RoleType.DECODE, "10.0.0.2", 8081);
        route.setServerStatus(List.of(prefillStatus, decodeServerStatus));
        when(router.route(any(BalanceContext.class))).thenAnswer(ignored -> {
            decode.reserve(requestId, 128, 136, 50, 0);
            return route;
        });

        PriorityAdmissionScheduler scheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, planCommitter,
                mock(PrioritySloPolicy.class), mock(PrioritySchedulerReporter.class),
                mock(BatchSchedulerReporter.class), mock(EngineCancelChannel.class),
                mock(DecodePreemptionCoordinator.class));
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        ctx.setRequest(request);

        Method build = PriorityAdmissionScheduler.class.getDeclaredMethod(
                "tryNormalPlacement", BalanceContext.class,
                CompletableFuture.class, ClusterSnapshot.class);
        build.setAccessible(true);

        InvocationTargetException failure = assertThrows(InvocationTargetException.class,
                () -> build.invoke(scheduler, ctx, new CompletableFuture<Response>(),
                        new ClusterSnapshot(Map.of(), Map.of())));
        assertTrue(failure.getCause() instanceof IllegalStateException);
        assertFalse(decode.reservedView().containsKey(requestId),
                "a route reservation must have an owner before plan construction");
        verify(planCommitter, never()).commit(any(), any(), anyBoolean());
    }

    @Test
    void offerExceptionRollsBackInflightRegistration() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        WorkerBatcher batcher = mock(WorkerBatcher.class);
        BatchItem item = new BatchItem(
                null, new CompletableFuture<>(), null,
                null, null, prefill, null, 0);
        NormalPlacementPlan plan = new NormalPlacementPlan(
                null, item, null, 0, 0);

        when(prefill.getBatcher()).thenReturn(batcher);
        when(registrar.registerInflight(item)).thenReturn(true);
        when(batcher.tryOffer(item)).thenThrow(new IllegalStateException("offer failed"));

        assertThrows(IllegalStateException.class,
                () -> new PlanCommitter().commit(plan, registrar, true));
        verify(registrar).unregisterInflight(item);
    }

    @Test
    void successfulOfferTransfersRegistrationToQueue() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        WorkerBatcher batcher = mock(WorkerBatcher.class);
        BatchItem item = new BatchItem(
                null, new CompletableFuture<>(), null,
                null, null, prefill, null, 0);
        NormalPlacementPlan plan = new NormalPlacementPlan(
                null, item, null, 0, 0);

        when(prefill.getBatcher()).thenReturn(batcher);
        when(registrar.registerInflight(item)).thenReturn(true);
        when(batcher.tryOffer(item)).thenReturn(true);

        assertEquals(PlanCommitter.CommitResult.SUCCESS,
                new PlanCommitter().commit(plan, registrar, true));
        verify(registrar, never()).unregisterInflight(item);
    }

    @Test
    void abandonedLeaseDoesNotRollbackQueueOwnedAdmission() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        AtomicInteger activeLeases = new AtomicInteger(1);
        BatchItem item = new BatchItem(
                null, new CompletableFuture<>(), null,
                null, null, null, null, 0);
        AdmissionLease lease = new AdmissionLease(
                item, null, null, registrar, 0, activeLeases::decrementAndGet);

        lease.abandonWithoutCleanup();
        lease.close();

        assertEquals(0, activeLeases.get());
        verify(registrar, never()).unregisterInflight(item);
    }

    @Test
    void synchronousCancelThrowReleasesIncomingProvisionalReservation() throws Exception {
        assertCancelContractViolationCleansIncoming(true);
    }

    @Test
    void nullCancelFutureReleasesIncomingProvisionalReservation() throws Exception {
        assertCancelContractViolationCleansIncoming(false);
    }

    @Test
    void prefillSelectionFailureReleasesFreshEvictionReservation() throws Exception {
        ConfigService configService = mock(ConfigService.class);
        Router router = mock(Router.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        PrioritySloPolicy sloPolicy = mock(PrioritySloPolicy.class);
        PrioritySchedulerReporter priorityReporter = mock(PrioritySchedulerReporter.class);
        BatchSchedulerReporter batchReporter = mock(BatchSchedulerReporter.class);
        EngineCancelChannel cancelChannel = mock(EngineCancelChannel.class);
        DecodePreemptionCoordinator coordinator = mock(DecodePreemptionCoordinator.class);
        PriorityAdmissionScheduler scheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(), sloPolicy,
                priorityReporter, batchReporter, cancelChannel, coordinator) {
            @Override
            protected org.flexlb.dao.loadbalance.ServerStatus selectPrefillForDecodeEviction(
                    BalanceContext ctx, FlexlbConfig config, String group) {
                return null;
            }
        };

        long requestId = 99;
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.DECODE);
        DecodeEndpoint endpoint = new DecodeEndpoint(status);
        endpoint.reserve(requestId, 128, 136, 70, 0);
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        ctx.setRequest(request);
        CompletableFuture<Response> admission = new CompletableFuture<>();
        InflightRegistrar registrar = mock(InflightRegistrar.class);

        Method place = PriorityAdmissionScheduler.class.getDeclaredMethod(
                "placeAfterDecodeEviction",
                BalanceContext.class, CompletableFuture.class, FlexlbConfig.class,
                InflightRegistrar.class, DecodeEndpoint.class,
                DecodeReservationOwnership.class);
        place.setAccessible(true);
        try (DecodeReservationOwnership reservation =
                     DecodeReservationOwnership.own(endpoint, requestId)) {
            place.invoke(scheduler, ctx, admission, new FlexlbConfig(), registrar,
                    endpoint, reservation);
        }

        assertFalse(endpoint.reservedView().containsKey(requestId),
                "prefill selection failure must release the fresh reservation");
        assertTrue(admission.isDone());
    }

    @Test
    void deadlineFencePreventsRecursiveReplanAfterAdmissionCloses() throws Exception {
        ConfigService configService = mock(ConfigService.class);
        Router router = mock(Router.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        PrioritySloPolicy sloPolicy = mock(PrioritySloPolicy.class);
        PrioritySchedulerReporter priorityReporter = mock(PrioritySchedulerReporter.class);
        BatchSchedulerReporter batchReporter = mock(BatchSchedulerReporter.class);
        EngineCancelChannel cancelChannel = mock(EngineCancelChannel.class);
        DecodePreemptionCoordinator coordinator = mock(DecodePreemptionCoordinator.class);
        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> coordinatorResult =
                new CompletableFuture<>();
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        when(registrar.retainPendingAdmission(anyLong(), any())).thenReturn(true);
        when(coordinator.execute(any(DecodePreemptionCoordinator.Request.class), eq(registrar)))
                .thenReturn(coordinatorResult);

        AtomicInteger replans = new AtomicInteger();
        PriorityAdmissionScheduler scheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(), sloPolicy,
                priorityReporter, batchReporter, cancelChannel, coordinator) {
            @Override
            public void schedule(BalanceContext ctx,
                                 CompletableFuture<Response> future,
                                 InflightRegistrar ignored) {
                replans.incrementAndGet();
            }
        };

        long requestId = 123;
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        ctx.setRequest(request);
        CompletableFuture<Response> admission = new CompletableFuture<>();
        FlexlbConfig config = new FlexlbConfig();
        DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
        DecodeRequestSnapshot victim = new DecodeRequestSnapshot(
                41, 30, DecodeTaskPhase.RUNNING, 128, 136, 0);
        DecodeEvictionProposal proposal = new DecodeEvictionProposal(
                "decode:1", 1, List.of(victim),
                DecodeEvictionProposal.CASE_SLOT, 0, 128, null);

        Method start = PriorityAdmissionScheduler.class.getDeclaredMethod(
                "startEngineCancelPreemption",
                BalanceContext.class, CompletableFuture.class, FlexlbConfig.class,
                InflightRegistrar.class, DecodeEvictionProposal.class,
                DecodeEndpoint.class, long.class, long.class);
        start.setAccessible(true);
        start.invoke(scheduler, ctx, admission, config, registrar, proposal,
                endpoint, 128L, 136L);

        CountDownLatch completing = new CountDownLatch(1);
        CountDownLatch completed = new CountDownLatch(1);
        Thread callback = new Thread(() -> {
            completing.countDown();
            coordinatorResult.complete(DecodePreemptionCoordinator.ExecutionResult.of(
                    DecodePreemptionCoordinator.ResultCode.REPLAN_NOT_FOUND,
                    null, "not_found"));
            completed.countDown();
        });

        synchronized (admission) {
            callback.start();
            assertTrue(completing.await(1, TimeUnit.SECONDS));
            assertFalse(completed.await(100, TimeUnit.MILLISECONDS),
                    "replan callback must wait for the admission/deadline fence");
            admission.completeExceptionally(new RuntimeException("deadline"));
        }
        assertTrue(completed.await(1, TimeUnit.SECONDS));
        assertEquals(0, replans.get());
        verify(registrar).releasePendingAdmission(requestId, admission);
    }

    private static void assertCancelContractViolationCleansIncoming(boolean throwSynchronously)
            throws Exception {
        long victimId = 41;
        long incomingId = 99;
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.DECODE);
        DecodeEndpoint endpoint = new DecodeEndpoint(status);
        endpoint.reserve(victimId, 128, 136, 30, 0);
        TaskInfo running = new TaskInfo();
        running.setRequestId(victimId);
        running.setPhase(TaskPhase.RUNNING);
        running.setInputLength(128);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(Map.of(String.valueOf(victimId), running));
        endpoint.applyWorkerStatusResponse(status, response);

        InflightRegistrar registrar = mock(InflightRegistrar.class);
        PrefillEndpoint prefillGeneration = mock(PrefillEndpoint.class);
        when(prefillGeneration.getIp()).thenReturn("127.0.0.1");
        when(prefillGeneration.getGrpcPort()).thenReturn(1234);
        when(prefillGeneration.initiateGenerationDispatch(any(), any()))
                .thenAnswer(invocation -> {
                    Supplier<?> dispatch = invocation.getArgument(1);
                    return dispatch.get();
                });
        EngineCancelChannel.CancelTarget cancelTarget =
                new EngineCancelChannel.CancelTarget(prefillGeneration);
        when(registrar.resolveCancelTarget(victimId))
                .thenReturn(cancelTarget);
        when(registrar.claimForPreemption(anyLong(), anyLong(), anyString()))
                .thenReturn(true);
        when(registrar.markPreemptionCancelInFlight(anyLong(), anyLong()))
                .thenReturn(true);
        when(registrar.priorityCanceledSignal(anyLong(), anyLong()))
                .thenReturn(new CompletableFuture<>());

        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        if (throwSynchronously) {
            doThrow(new IllegalStateException("sync cancel failure"))
                    .when(channel).cancel(
                            org.mockito.ArgumentMatchers.any(), anyLong(), anyLong());
        } else {
            when(channel.cancel(
                    org.mockito.ArgumentMatchers.any(), anyLong(), anyLong()))
                    .thenReturn(null);
        }

        DecodeRequestSnapshot victim = new DecodeRequestSnapshot(
                victimId, 30, DecodeTaskPhase.RUNNING, 128, 136, 0);
        DecodePreemptionCoordinator.Request request =
                new DecodePreemptionCoordinator.Request(
                        endpoint, endpoint.admissionVersion(), false,
                        incomingId, 128, 136, 70, 0,
                        List.of(victim), 1, 1, () -> true, "test");

        DecodePreemptionCoordinator.ExecutionResult result =
                new DecodePreemptionCoordinator(channel)
                        .execute(request, registrar)
                        .get(1, TimeUnit.SECONDS);

        assertEquals(DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED,
                result.code());
        assertFalse(endpoint.releaseIfHeld(incomingId),
                "coordinator must abort the provisional incoming reservation");
        verify(registrar).markPreemptionUnknown(anyLong(), anyLong());
    }

    private static ServerStatus server(RoleType role, String ip, int port) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(port);
        return status;
    }
}
