package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeAdmission;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Exercises Encoder placement through the same submission path used by Schedule.
 */
class EncoderSchedulingIntegrationTest {

    private ConfigService configService;
    private EndpointRegistry directory;
    private RequestRepository requests;
    private AbstractRequestScheduler scheduler;
    private PlacementAvailability availability = new PlacementAvailability();
    private final java.util.List<SchedulerRuntime> runtimes = new java.util.ArrayList<>();

    @BeforeEach
    void setUp() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        config.setScheduler(org.flexlb.config.ConfigTestSupport.direct());
        configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        directory = mock(EndpointRegistry.class);
        requests = new RequestRepository();
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.ENCODER));
        RequestWorkerSelector router =
                new RequestWorkerSelector(
                        mock(CostBasedPrefillStrategy.class),
                        mock(DecodeSelector.class),
                        mock(VitWorkerSelector.class),
                        model);
        scheduler = createScheduler(router);
    }

    @AfterEach
    void tearDown() {
        runtimes.forEach(SchedulerRuntime::shutdown);
    }

    @Test
    void queueLimitsEachEncoderAndResumesAfterWorkerFinished() throws Exception {
        useEncoderQueue(1);
        EncoderEndpoint first = worker("first", 8001, true, "group", 0, Map.of(), 0, 0);
        worker("second", 8002, true, "group", 0, Map.of(), 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("first", "second"));

        assertEquals(
                "first",
                submit(context(1, 100, 0L))
                        .get(2, TimeUnit.SECONDS)
                        .getServerStatus()
                        .getFirst()
                        .getServerIp());
        assertEquals(
                "second",
                submit(context(2, 100, 0L))
                        .get(2, TimeUnit.SECONDS)
                        .getServerStatus()
                        .getFirst()
                        .getServerIp());
        CompletableFuture<Response> waiting = submit(context(3, 100, 0L));
        assertFalse(waiting.isDone());

        WorkerStatus.TaskObservation finished = task("1", 100);
        when(finished.requestId()).thenReturn("1");
        WorkerStatus status = first.getStatus();
        WorkerStatus.PreparedStatus prepared = mock(WorkerStatus.PreparedStatus.class);
        WorkerStatus.StatusObservation observation = mock(WorkerStatus.StatusObservation.class);
        when(prepared.observation()).thenReturn(observation);
        when(observation.alive()).thenReturn(true);
        when(observation.activeTasks()).thenReturn(Map.of());
        when(observation.finishedTasks()).thenReturn(Map.of("1", finished));
        first.publishStatus(first.applyPreparedStatus(status, prepared));

        assertEquals(
                "first",
                waiting.get(2, TimeUnit.SECONDS).getServerStatus().getFirst().getServerIp());
    }

    @Test
    void queueUsesRunningAndWaitingCountsAndCanCancelAWaitingRequest() throws Exception {
        useEncoderQueue(2);
        worker("busy", 8001, true, "group", 0, Map.of(), 1, 1);
        worker("open", 8002, true, "group", 0, Map.of(), 0, 1);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("busy", "open"));

        assertEquals(
                "open",
                submit(context(4, 100, 0L))
                        .get(2, TimeUnit.SECONDS)
                        .getServerStatus()
                        .getFirst()
                        .getServerIp());
        CompletableFuture<Response> waiting = submit(context(5, 100, 0L));
        assertFalse(waiting.isDone());
        scheduler.cancel("5", 0L, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                waiting.get(2, TimeUnit.SECONDS).getCode());
    }

    @Test
    void queueResumesWhenWorkerStatusReportsFreeCapacityWithoutFinishedTask() throws Exception {
        useEncoderQueue(1);
        EncoderEndpoint encoder = worker("encoder", 8001, true, "group", 0, Map.of(), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("encoder"));

        CompletableFuture<Response> waiting = submit(context(10, 100, 0L));
        assertFalse(waiting.isDone());
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (scheduler.placementCounts().blocked() == 0 && System.nanoTime() < deadline) {
            Thread.sleep(10L);
        }
        assertEquals(1, scheduler.placementCounts().blocked());

        WorkerStatus status = encoder.getStatus();
        WorkerStatus.EngineObservation engine = status.committedEngineObservation();
        when(engine.runningQueryLen()).thenReturn(0L);
        WorkerStatus.StatusObservation observation = mock(WorkerStatus.StatusObservation.class);
        when(observation.owner()).thenReturn(status);
        when(observation.activeTasks()).thenReturn(Map.of());
        encoder.publishStatus(encoder.applyStatusHeartbeat(status, observation));

        assertEquals(
                "encoder",
                waiting.get(2, TimeUnit.SECONDS).getServerStatus().getFirst().getServerIp());
    }

    @Test
    void queuedEncoderRequestExpiresWhenNoWorkerHasCapacity() throws Exception {
        useEncoderQueue(1);
        worker("busy", 8001, true, "group", 0, Map.of(), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("busy"));
        RequestData context = context(7, 100, 0L);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(50, System.currentTimeMillis() + 300L));

        Response response = submit(context).get(2, TimeUnit.SECONDS);

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
    }

    @Test
    void expiredEncoderEntryIsRemovedEvenWhenItsRequestSlotIsMissing() {
        useEncoderQueue(1);
        EncoderQueueCoordinator queue =
                (EncoderQueueCoordinator) ReflectionTestUtils.getField(scheduler, "encoderQueue");
        RequestData context = mock(RequestData.class);
        when(context.getRequestId()).thenReturn("expired-without-slot");
        when(context.getPriority()).thenReturn(50);
        when(context.getFuture()).thenReturn(new CompletableFuture<>());
        when(context.requestExpired(anyLong())).thenReturn(true);
        CompletableFuture<Response> pending = new CompletableFuture<>();

        assertTrue(queue.offer(new RequestContext(context)));

        assertTimeoutPreemptively(
                Duration.ofSeconds(2),
                () -> {
                    while (queue.size() != 0) {
                        Thread.sleep(5L);
                    }
                });
        assertFalse(pending.isDone());
    }

    @Test
    void rejectedEncoderEntryIsRemovedWhenItsRequestSlotIsMissing() {
        useEncoderQueue(1);
        var queue =
                (EncoderQueueCoordinator) ReflectionTestUtils.getField(scheduler, "encoderQueue");
        var data = context(99, 100, 0L);
        data.setFuture(new CompletableFuture<>());
        assertTrue(queue.offer(new RequestContext(data)));
        assertTimeoutPreemptively(
                Duration.ofSeconds(2),
                () -> {
                    while (queue.size() != 0) {
                        Thread.sleep(5L);
                    }
                });
        assertFalse(data.getFuture().isDone());
    }

    @Test
    void closingEncoderQueueFailsWaitingRequests() throws Exception {
        useEncoderQueue(1);
        worker("busy", 8001, true, "group", 0, Map.of(), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("busy"));
        CompletableFuture<Response> waiting = submit(context(8, 100, 0L));
        assertFalse(waiting.isDone());

        scheduler.closeAllPlacement();

        assertEquals(
                StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                waiting.get(2, TimeUnit.SECONDS).getCode());
    }

    @Test
    void queueDoesNotStartWithoutConfiguredEncoderRole() {
        scheduler.closeAllPlacement();
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        when(configService.loadBalanceConfig()).thenReturn(config);
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestWorkerSelector router =
                new RequestWorkerSelector(
                        mock(CostBasedPrefillStrategy.class),
                        mock(DecodeSelector.class),
                        mock(VitWorkerSelector.class),
                        model);
        scheduler = createScheduler(router);

        assertNull(ReflectionTestUtils.getField(scheduler, "encoderQueue"));
        RequestData encoderRequest = context(6, 100, 0L);
        encoderRequest.setRequestPhase(RequestPhase.ENCODER);
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                submit(encoderRequest).join().getCode());
        assertNull(requests.findActive("6", RequestPhase.ENCODER));
    }

    private void useEncoderQueue(int maxInflight) {
        scheduler.closeAllPlacement();
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getDispatcher().setMaxInflightPerEncoderWorker(maxInflight);
        when(configService.loadBalanceConfig()).thenReturn(config);
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.ENCODER));
        RequestWorkerSelector router =
                new RequestWorkerSelector(
                        mock(CostBasedPrefillStrategy.class),
                        mock(DecodeSelector.class),
                        mock(VitWorkerSelector.class),
                        model);
        scheduler = createScheduler(router);
    }

    @Test
    void weightedRequestChoosesLessObservedUncachedWorkDespiteMoreConcurrentTasks() {
        worker("long", 8001, true, "group", 0, Map.of("old", task("old", 1000)), 1, 0);
        worker(
                "short",
                8002,
                true,
                "group",
                -1,
                Map.of(
                        "a",
                        task("a", 10),
                        "b",
                        task("b", 10),
                        "c",
                        task("c", 10),
                        "d",
                        task("d", 10)),
                4,
                0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("long", "short"));

        Response response = submit(context(1, 800, 200L)).join();

        assertTrue(response.isSuccess());
        assertEquals("short", response.getServerStatus().getFirst().getServerIp());
        assertEquals(
                RequestPhase.ENCODER,
                requests.findActive("1", RequestPhase.ENCODER).data().getRequestPhase());
    }

    @Test
    void legacyRequestUsesConcurrencyAndIncludesLocallyPendingSelection() {
        EncoderEndpoint idle = worker("idle", 8001, true, "group", 0, Map.of(), 0, 0);
        worker("busy", 8002, true, "group", 0, Map.of("old", task("old", 1)), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("idle", "busy"));

        assertEquals(
                "idle",
                submit(context(2, 800, null)).join().getServerStatus().getFirst().getServerIp());
        assertEquals(1, idle.pendingEncoderRequestCount());
        assertEquals(
                "idle",
                submit(context(3, 800, null)).join().getServerStatus().getFirst().getServerIp());
        assertEquals(2, idle.pendingEncoderRequestCount());
        assertEquals(
                "busy",
                submit(context(6, 800, null)).join().getServerStatus().getFirst().getServerIp());
    }

    @Test
    void weightedTieUsesConcurrencyAndFullHitStillRoutesToEncoder() {
        worker("busy", 8001, true, "group", 0, Map.of("old", task("old", 0)), 1, 0);
        worker("idle", 8002, true, "group", 0, Map.of(), 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("busy", "idle"));

        Response response = submit(context(4, 800, 800L)).join();

        assertTrue(response.isSuccess());
        assertEquals("idle", response.getServerStatus().getFirst().getServerIp());
    }

    @Test
    void workerStatusActualInputLengthChangesTheNextPlacement() {
        EncoderEndpoint encoder = worker("encoder", 8001, true, "group", 0, Map.of(), 0, 0);
        worker("other", 8002, true, "group", 0, Map.of("old", task("old", 500)), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("encoder", "other"));

        assertEquals(
                "encoder",
                submit(context(7, 1000, 200L)).join().getServerStatus().getFirst().getServerIp());
        assertEquals(800L, encoder.inflightUncachedTokenEstimate());
        assertEquals(
                "other",
                submit(context(8, 1000, 200L)).join().getServerStatus().getFirst().getServerIp());

        WorkerStatus status = encoder.getStatus();
        WorkerStatus.EngineObservation engine = status.committedEngineObservation();
        WorkerStatus.TaskObservation actual = task("7", 100);
        when(actual.requestId()).thenReturn("7");
        when(engine.runningTaskList()).thenReturn(Map.of("7", actual));
        when(engine.runningQueryLen()).thenReturn(1L);
        WorkerStatus.StatusObservation running = mock(WorkerStatus.StatusObservation.class);
        when(running.owner()).thenReturn(status);
        when(running.activeTasks()).thenReturn(Map.of("7", actual));
        encoder.publishStatus(encoder.applyStatusHeartbeat(status, running));

        assertEquals(0, encoder.pendingEncoderRequestCount());
        assertEquals(100L, encoder.inflightUncachedTokenEstimate());
        assertEquals(
                "encoder",
                submit(context(9, 1000, 200L)).join().getServerStatus().getFirst().getServerIp());
    }

    @Test
    void deadOrWrongGroupWorkersCannotWinAndNoCandidateReturnsEncoderError() {
        worker("dead", 8001, false, "group", 0, Map.of(), 0, 0);
        worker("foreign", 8002, true, "other", 0, Map.of(), 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("dead", "foreign"));

        RequestData context = context(5, 800, 0L);
        TrafficPolicyConfig groupSelector = mock(TrafficPolicyConfig.class);
        context.getConfig().getRouter().setGroupSelector(groupSelector);
        when(groupSelector.resolveTargetGroup(context.getRequest()))
                .thenReturn(Optional.of("group"));
        Response response = submit(context).join();

        assertEquals(StrategyErrorType.NO_ENCODER_WORKER.getErrorCode(), response.getCode());
        assertEquals(RequestPhase.ENCODER, context.getRequestPhase());
        assertEquals(
                RequestState.Phase.FAILED,
                requests.getRequestState("5", 0L, RequestPhase.ENCODER).state());
        assertNull(requests.findActive("5", RequestPhase.ENCODER));
    }

    private AbstractRequestScheduler createScheduler(RequestWorkerSelector router) {
        ReflectionTestUtils.setField(router, "encoderSelector", new EncoderStrategy(directory));
        var reporter = mock(DeliveryMetricsReporter.class);
        var runtime =
                new SchedulerRuntime(
                        requests,
                        directory,
                        reporter,
                        mock(RequestSchedulerReporter.class),
                        SchedulerTestSupport.dispatcher(),
                        configService,
                        mock(EngineGrpcClient.class));
        runtimes.add(runtime);
        var owner =
                PlacementConfiguration.create(
                        runtime,
                        configService.loadBalanceConfig(),
                        router,
                        reporter,
                        mock(DecodeAdmission.class),
                        availability,
                        new PlacementConfiguration().decodeAdmissionCapacity(configService),
                        mock(org.flexlb.service.TheoryCacheHitReporter.class));
        runtime.registerScheduler(owner);
        return owner;
    }

    private CompletableFuture<Response> submit(RequestData data) {
        return scheduler.submit(new RequestContext(data));
    }

    private EncoderEndpoint worker(
            String ip,
            int port,
            boolean alive,
            String group,
            long availableKv,
            Map<String, WorkerStatus.TaskObservation> tasks,
            long running,
            long waiting) {
        WorkerStatus status = mock(WorkerStatus.class);
        WorkerStatus.EngineObservation engine = mock(WorkerStatus.EngineObservation.class);
        when(status.getRole()).thenReturn(RoleType.ENCODER);
        when(status.isActiveGeneration()).thenReturn(true);
        when(status.isAlive()).thenReturn(alive);
        when(status.isAvailableForScheduling()).thenReturn(alive);
        when(status.getGenerationId()).thenReturn(1L);
        when(status.getIp()).thenReturn(ip);
        when(status.getPort()).thenReturn(port);
        when(status.getLogicalIpPort()).thenReturn(ip + ":" + port);
        when(status.topologySnapshot())
                .thenReturn(new WorkerStatus.TopologySnapshot(group, ip, port, port + 1, "site"));
        when(status.committedEngineObservation()).thenReturn(engine);
        when(engine.availableKvCacheTokens()).thenReturn(availableKv);
        when(engine.runningTaskList()).thenReturn(tasks);
        when(engine.runningQueryLen()).thenReturn(running);
        when(engine.waitingQueryLen()).thenReturn(waiting);
        EncoderEndpoint endpoint = new EncoderEndpoint(status, requests, availability);
        when(directory.capture(eq(RoleType.ENCODER), eq(ip)))
                .thenAnswer(invocation -> endpoint.tryPinGeneration());
        return endpoint;
    }

    private static WorkerStatus.TaskObservation task(String requestId, long inputLength) {
        WorkerStatus.TaskObservation task = mock(WorkerStatus.TaskObservation.class);
        when(task.inputLength()).thenReturn(inputLength);
        return task;
    }

    private RequestData context(long requestId, long seqLen, Long hitLen) {
        Request request = Request.builder().build();
        request = request.toBuilder().requestId(Long.toString(requestId)).build();
        request = request.toBuilder().seqLen(seqLen).build();
        request = request.toBuilder().encoderCacheHitLen(hitLen).build();
        request = request.toBuilder().maxNewTokens(16).build();
        RequestData context = new RequestData(configService.loadBalanceConfig());
        context.setRequest(request);
        context.setRequestedRoles(Set.of(RoleType.ENCODER));
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(
                        50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }
}
