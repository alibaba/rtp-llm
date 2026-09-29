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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
        if (requests.closeAdmissionAndAwaitMutations()) {
            requests.closeOutstandingAndTerminalize();
            requests.closeExpiration();
            requests.closePublisher();
        }
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
