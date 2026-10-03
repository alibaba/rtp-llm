package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Exercises Encoder placement through the same submission path used by Schedule.
 */
class EncoderSchedulingIntegrationTest {

    private ConfigService configService;
    private WorkerDirectory directory;
    private RequestRegistry requests;
    private RequestScheduler scheduler;

    @BeforeEach
    void setUp() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        config.setScheduler(org.flexlb.config.SchedulerConfig.direct());
        configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        directory = mock(WorkerDirectory.class);
        requests = new RequestRegistry(configService, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.ENCODER));
        DefaultRouter router = new DefaultRouter(mock(CostBasedPrefillStrategy.class),
                mock(DecodeSelector.class), mock(RandomStrategy.class), new EncoderStrategy(directory),
                configService, model);
        scheduler = new RequestScheduler(configService, router, mock(EndpointRegistry.class),
                mock(BatchSchedulerReporter.class), mock(EvictionManager.class), requests,
                new PlacementAvailability());
    }

    @AfterEach
    void tearDown() {
        scheduler.closePlacement();
        if (requests.closeAdmissionAndAwaitMutations()) {
            requests.closeOutstandingAndTerminalize();
            requests.closeExpiration();
            requests.closePublisher();
        }
    }

    @Test
    void queueLimitsEachEncoderAndResumesAfterWorkerFinished() throws Exception {
        useEncoderQueue(1);
        EncoderEndpoint first = worker("first", 8001, true, "group", 0, Map.of(), 0, 0);
        worker("second", 8002, true, "group", 0, Map.of(), 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("first", "second"));

        assertEquals("first", scheduler.submit(context(1, 100, 0L)).get(2, TimeUnit.SECONDS)
                .getServerStatus().getFirst().getServerIp());
        assertEquals("second", scheduler.submit(context(2, 100, 0L)).get(2, TimeUnit.SECONDS)
                .getServerStatus().getFirst().getServerIp());
        CompletableFuture<Response> waiting = scheduler.submit(context(3, 100, 0L));
        assertFalse(waiting.isDone());

        WorkerStatus.TaskObservation finished = task("1", 100);
        when(finished.requestId()).thenReturn("1");
        WorkerStatus status = first.getStatus();
        WorkerStatus.PreparedStatus prepared = mock(WorkerStatus.PreparedStatus.class);
        WorkerStatus.StatusObservation observation = mock(WorkerStatus.StatusObservation.class);
        when(prepared.observation()).thenReturn(observation);
        when(observation.alive()).thenReturn(true);
        when(observation.runningTasks()).thenReturn(Map.of());
        when(observation.finishedTasks()).thenReturn(Map.of("1", finished));
        first.applyPreparedStatus(status, prepared).run();

        assertEquals("first", waiting.get(2, TimeUnit.SECONDS)
                .getServerStatus().getFirst().getServerIp());
    }

    @Test
    void queueUsesRunningAndWaitingCountsAndCanCancelAWaitingRequest() throws Exception {
        useEncoderQueue(2);
        worker("busy", 8001, true, "group", 0, Map.of(), 1, 1);
        worker("open", 8002, true, "group", 0, Map.of(), 0, 1);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("busy", "open"));

        assertEquals("open", scheduler.submit(context(4, 100, 0L)).get(2, TimeUnit.SECONDS)
                .getServerStatus().getFirst().getServerIp());
        CompletableFuture<Response> waiting = scheduler.submit(context(5, 100, 0L));
        assertFalse(waiting.isDone());
        scheduler.cancelRequest("5", 0L, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                waiting.get(2, TimeUnit.SECONDS).getCode());
    }

    @Test
    void queueResumesWhenWorkerStatusReportsFreeCapacityWithoutFinishedTask() throws Exception {
        useEncoderQueue(1);
        EncoderEndpoint encoder = worker("encoder", 8001, true, "group", 0, Map.of(), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("encoder"));

        CompletableFuture<Response> waiting = scheduler.submit(context(10, 100, 0L));
        assertFalse(waiting.isDone());
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (scheduler.getBlockedRequestCount() == 0 && System.nanoTime() < deadline) {
            Thread.sleep(10L);
        }
        assertEquals(1, scheduler.getBlockedRequestCount());

        WorkerStatus status = encoder.getStatus();
        WorkerStatus.EngineObservation engine = status.committedEngineObservation();
        when(engine.runningQueryLen()).thenReturn(0L);
        WorkerStatus.StatusObservation observation = mock(WorkerStatus.StatusObservation.class);
        when(observation.owner()).thenReturn(status);
        when(observation.runningTasks()).thenReturn(Map.of());
        encoder.observeStatusHeartbeat(status, observation).run();

        assertEquals("encoder", waiting.get(2, TimeUnit.SECONDS)
                .getServerStatus().getFirst().getServerIp());
    }

    @Test
    void queuedEncoderRequestExpiresWhenNoWorkerHasCapacity() throws Exception {
        useEncoderQueue(1);
        worker("busy", 8001, true, "group", 0, Map.of(), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("busy"));
        BalanceContext context = context(7, 100, 0L);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 300L));

        Response response = scheduler.submit(context).get(2, TimeUnit.SECONDS);

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
    }

    @Test
    void closingEncoderQueueFailsWaitingRequests() throws Exception {
        useEncoderQueue(1);
        worker("busy", 8001, true, "group", 0, Map.of(), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("busy"));
        CompletableFuture<Response> waiting = scheduler.submit(context(8, 100, 0L));
        assertFalse(waiting.isDone());

        scheduler.closePlacement();

        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                waiting.get(2, TimeUnit.SECONDS).getCode());
    }

    @Test
    void queueDoesNotStartWithoutConfiguredEncoderRole() {
        scheduler.closePlacement();
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        when(configService.loadBalanceConfig()).thenReturn(config);
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        DefaultRouter router = new DefaultRouter(mock(CostBasedPrefillStrategy.class),
                mock(DecodeSelector.class), mock(RandomStrategy.class), new EncoderStrategy(directory),
                configService, model);
        scheduler = new RequestScheduler(configService, router, mock(EndpointRegistry.class),
                mock(BatchSchedulerReporter.class), mock(EvictionManager.class), requests,
                new PlacementAvailability());

        assertNull(ReflectionTestUtils.getField(scheduler, "encoderQueue"));
        BalanceContext encoderRequest = context(6, 100, 0L);
        encoderRequest.setRequestPhase(RequestPhase.ENCODER);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                scheduler.submit(encoderRequest).join().getCode());
        assertNull(requests.requestSlot("6", RequestPhase.ENCODER));
    }

    private void useEncoderQueue(int maxInflight) {
        scheduler.closePlacement();
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getDispatcher().setMaxInflightPerEncoderWorker(maxInflight);
        when(configService.loadBalanceConfig()).thenReturn(config);
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.ENCODER));
        DefaultRouter router = new DefaultRouter(mock(CostBasedPrefillStrategy.class),
                mock(DecodeSelector.class), mock(RandomStrategy.class), new EncoderStrategy(directory),
                configService, model);
        scheduler = new RequestScheduler(configService, router, mock(EndpointRegistry.class),
                mock(BatchSchedulerReporter.class), mock(EvictionManager.class), requests,
                new PlacementAvailability());
    }

    @Test
    void weightedRequestChoosesLessObservedUncachedWorkDespiteMoreConcurrentTasks() {
        worker("long", 8001, true, "group", 0, Map.of("old", task("old", 1000)), 1, 0);
        worker("short", 8002, true, "group", -1,
                Map.of("a", task("a", 10), "b", task("b", 10),
                        "c", task("c", 10), "d", task("d", 10)), 4, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("long", "short"));

        Response response = scheduler.submit(context(1, 800, 200L)).join();

        assertTrue(response.isSuccess());
        assertEquals("short", response.getServerStatus().getFirst().getServerIp());
        assertEquals(RequestPhase.ENCODER, requests.requestSlot("1", RequestPhase.ENCODER).requestPhase());
    }

    @Test
    void legacyRequestUsesConcurrencyAndIncludesLocallyPendingSelection() {
        EncoderEndpoint idle = worker("idle", 8001, true, "group", 0, Map.of(), 0, 0);
        worker("busy", 8002, true, "group", 0, Map.of("old", task("old", 1)), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("idle", "busy"));

        assertEquals("idle", scheduler.submit(context(2, 800, null)).join()
                .getServerStatus().getFirst().getServerIp());
        assertEquals(1, idle.pendingEncoderRequestCount());
        assertEquals("idle", scheduler.submit(context(3, 800, null)).join()
                .getServerStatus().getFirst().getServerIp());
        assertEquals(2, idle.pendingEncoderRequestCount());
        assertEquals("busy", scheduler.submit(context(6, 800, null)).join()
                .getServerStatus().getFirst().getServerIp());
    }

    @Test
    void weightedTieUsesConcurrencyAndFullHitStillRoutesToEncoder() {
        worker("busy", 8001, true, "group", 0, Map.of("old", task("old", 0)), 1, 0);
        worker("idle", 8002, true, "group", 0, Map.of(), 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("busy", "idle"));

        Response response = scheduler.submit(context(4, 800, 800L)).join();

        assertTrue(response.isSuccess());
        assertEquals("idle", response.getServerStatus().getFirst().getServerIp());
    }

    @Test
    void workerStatusActualInputLengthChangesTheNextPlacement() {
        EncoderEndpoint encoder = worker("encoder", 8001, true, "group", 0, Map.of(), 0, 0);
        worker("other", 8002, true, "group", 0, Map.of("old", task("old", 500)), 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("encoder", "other"));

        assertEquals("encoder", scheduler.submit(context(7, 1000, 200L)).join()
                .getServerStatus().getFirst().getServerIp());
        assertEquals(800L, encoder.inflightUncachedTokenEstimate());
        assertEquals("other", scheduler.submit(context(8, 1000, 200L)).join()
                .getServerStatus().getFirst().getServerIp());

        WorkerStatus status = encoder.getStatus();
        WorkerStatus.EngineObservation engine = status.committedEngineObservation();
        WorkerStatus.TaskObservation actual = task("7", 100);
        when(actual.requestId()).thenReturn("7");
        when(engine.runningTaskList()).thenReturn(Map.of("7", actual));
        when(engine.runningQueryLen()).thenReturn(1L);
        WorkerStatus.StatusObservation running = mock(WorkerStatus.StatusObservation.class);
        when(running.owner()).thenReturn(status);
        when(running.runningTasks()).thenReturn(Map.of("7", actual));
        encoder.observeStatusHeartbeat(status, running).run();

        assertEquals(0, encoder.pendingEncoderRequestCount());
        assertEquals(100L, encoder.inflightUncachedTokenEstimate());
        assertEquals("encoder", scheduler.submit(context(9, 1000, 200L)).join()
                .getServerStatus().getFirst().getServerIp());
    }

    @Test
    void deadOrWrongGroupWorkersCannotWinAndNoCandidateReturnsEncoderError() {
        worker("dead", 8001, false, "group", 0, Map.of(), 0, 0);
        worker("foreign", 8002, true, "other", 0, Map.of(), 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("dead", "foreign"));

        BalanceContext context = context(5, 800, 0L);
        TrafficPolicyConfig groupSelector = mock(TrafficPolicyConfig.class);
        context.getConfig().getRouter().setGroupSelector(groupSelector);
        when(groupSelector.resolveTargetGroup(context.getRequest())).thenReturn(Optional.of("group"));
        Response response = scheduler.submit(context).join();

        assertEquals(StrategyErrorType.NO_ENCODER_WORKER.getErrorCode(), response.getCode());
        assertEquals(RequestPhase.ENCODER, requests.requestSlot("5", RequestPhase.ENCODER).requestPhase());
    }

    private EncoderEndpoint worker(String ip, int port, boolean alive, String group, long availableKv,
                                   Map<String, WorkerStatus.TaskObservation> tasks, long running, long waiting) {
        WorkerStatus status = mock(WorkerStatus.class);
        WorkerStatus.EngineObservation engine = mock(WorkerStatus.EngineObservation.class);
        when(status.isAlive()).thenReturn(alive);
        when(status.getGenerationId()).thenReturn(1L);
        when(status.getIp()).thenReturn(ip);
        when(status.getPort()).thenReturn(port);
        when(status.getLogicalIpPort()).thenReturn(ip + ":" + port);
        when(status.topologySnapshot()).thenReturn(
                new WorkerStatus.TopologySnapshot(group, ip, port, port + 1, "site"));
        when(status.committedEngineObservation()).thenReturn(engine);
        when(engine.availableKvCacheTokens()).thenReturn(availableKv);
        when(engine.runningTaskList()).thenReturn(tasks);
        when(engine.runningQueryLen()).thenReturn(running);
        when(engine.waitingQueryLen()).thenReturn(waiting);
        EncoderEndpoint endpoint = new EncoderEndpoint(status, new EndpointEventProjector(requests));
        when(directory.captureEndpoint(eq(RoleType.ENCODER), eq(ip)))
                .thenAnswer(invocation -> endpoint.tryPinGeneration());
        return endpoint;
    }

    private static WorkerStatus.TaskObservation task(String requestId, long inputLength) {
        WorkerStatus.TaskObservation task = mock(WorkerStatus.TaskObservation.class);
        when(task.inputLength()).thenReturn(inputLength);
        return task;
    }

    private BalanceContext context(long requestId, long seqLen, Long hitLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setEncoderCacheHitLen(hitLen);
        request.setMaxNewTokens(16);
        BalanceContext context = new BalanceContext(configService.loadBalanceConfig());
        context.setRequest(request);
        context.setRequestedRoles(Set.of(RoleType.ENCODER));
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }
}
