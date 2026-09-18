package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemptionCoordinator;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class ReturnedPreemptionSchedulingTest {
    private FlexlbConfig config;
    private RequestScheduler scheduler;
    private SchedulerRuntime runtime;
    private EvictionManager eviction;
    private EndpointRegistry endpoints;
    private PrefillEndpoint prefill;
    private DecodeEndpoint decode;
    private EngineCancelChannel cancel;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_ENGINE_OWNED);
        SchedulingTestConfig.engineCancellation(config).setMode(EngineCancellationConfig.Mode.RETURN);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        SchedulingTestConfig.useSingleDecision(config);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        FlexMonitor metrics = mock(FlexMonitor.class);
        BatchSchedulerReporter reporter = new BatchSchedulerReporter(metrics);
        RequestSchedulerReporter requestReporter = new RequestSchedulerReporter(metrics);
        cancel = mock(EngineCancelChannel.class);
        RequestRegistry requests = new RequestRegistry(service, reporter, requestReporter, cancel);
        PlacementAvailability availability = new PlacementAvailability();
        endpoints = new EndpointRegistry(service, new EndpointEventProjector(requests), reporter,
                new RouteDeliveryStrategy(requests, new DeliveryMetrics(reporter)), availability);
        prefill = (PrefillEndpoint) publish(RoleType.PREFILL, "10.0.0.1", 0, 1);
        decode = (DecodeEndpoint) publish(RoleType.DECODE, "10.0.0.2", 1, 2);
        DefaultRouter router = mock(DefaultRouter.class);
        when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);
        when(router.routeForQueue(any(), isNull())).thenAnswer(invocation -> {
            BalanceContext ctx = invocation.getArgument(0);
            ServerStatus p = server(prefill, ctx.getRequestId());
            ServerStatus d = server(decode, ctx.getRequestId());
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(List.of(p, d));
            return PlacementResult.success(QueueRouteAdmission.prepare(ctx, List.of(
                    SelectedRole.prefill(prefill.tryPinGeneration(), p, 1),
                    SelectedRole.decode(decode.tryPinGeneration(), d, 10_000)), response));
        });
        eviction = new EvictionManager(requestReporter, cancel,
                new DecodePreemptionCoordinator(cancel, requests), requests, reporter);
        scheduler = new RequestScheduler(service, router, endpoints, reporter, eviction, requests, availability);
        runtime = new SchedulerRuntime(requests, endpoints, reporter, requestReporter, scheduler);
    }

    @AfterEach
    void close() {
        runtime.shutdown();
        eviction.shutdown();
    }

    @Test
    void schedulerReturnsStringVictimsOnSelectedLogicalDecodeWithoutRpc() throws Exception {
        assertTrue(submit("victim-0001", 30).get(2, TimeUnit.SECONDS).isSuccess());
        updateDecode(Map.of("victim-0001", task("victim-0001")), Map.of(), 9_872);

        Response result = submit("new-request", 70).get(2, TimeUnit.SECONDS);

        assertTrue(result.isSuccess(), result.getErrorMessage());
        assertFalse(result.isEnqueuedByMaster());
        ServerStatus target = result.getServerStatus().get(1);
        assertEquals(List.of("victim-0001"), target.getPreemptRequestIds());
        assertEquals("10.0.0.2", target.getServerIp());
        assertEquals(1, target.getEngineIndex());
        assertTrue(result.getServerStatus().getFirst().getPreemptRequestIds().isEmpty());
        assertEquals(2, decode.routingView().totalLoad());
        verifyNoInteractions(cancel);

        updateDecode(Map.of(), Map.of("victim-0001", task("victim-0001")), 10_000);
        assertEquals(1, decode.routingView().totalLoad());
        verifyNoInteractions(cancel);
    }

    @Test
    void everyVictimIsReturnedWithItsOriginalStringId() throws Exception {
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
        assertTrue(submit("00042", 30).get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(submit("req-uuid", 30).get(2, TimeUnit.SECONDS).isSuccess());
        updateDecode(Map.of("00042", task("00042"), "req-uuid", task("req-uuid")), Map.of(), 9_744);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);

        Response result = submit("next", 80).get(2, TimeUnit.SECONDS);

        assertTrue(result.isSuccess(), result.getErrorMessage());
        assertEquals(List.of("00042", "req-uuid"),
                result.getServerStatus().get(1).getPreemptRequestIds().stream().sorted().toList());
        assertEquals(3, decode.routingView().totalLoad());
        verifyNoInteractions(cancel);
    }

    @Test
    void routeSnapshotPreservesEngineAndCopiesInstructionList() {
        ServerStatus original = server(decode, "incoming");
        List<String> ids = new ArrayList<>(List.of("00042"));
        original.setPreemptRequestIds(ids);
        ServerStatus copy = RequestRegistry.copyOf(original);
        ids.add("req-other");
        assertEquals(List.of("00042"), copy.getPreemptRequestIds());
        assertEquals(1, copy.getEngineIndex());
        assertEquals("10.0.0.2:8080@1", copy.getLogicalIpPort());
    }

    private CompletableFuture<Response> submit(String id, int priority) {
        Request request = new Request();
        request.setRequestId(id);
        request.setPriority(priority);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(config);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority,
                System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return scheduler.submit(ctx);
    }

    private WorkerEndpoint publish(RoleType role, String ip, int index, int count) {
        WorkerStatus worker = WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, null, null, index, count);
        WorkerStatusResponse response = status(role, 1, Map.of(), Map.of(), 10_000);
        worker.lock.lock();
        try {
            WorkerEndpoint endpoint = endpoints.publishPreparedEndpoint(worker.getLogicalIpPort(), worker,
                    worker.prepareNewStatus(worker.freezeStatusResponse(response))).endpoint();
            worker.recordSuccessfulPoll(true);
            return endpoint;
        } finally {
            worker.lock.unlock();
        }
    }

    private void updateDecode(Map<String, TaskInfo> running, Map<String, TaskInfo> finished, long available) {
        WorkerStatus worker = decode.getStatus();
        WorkerStatusResponse response = status(RoleType.DECODE,
                worker.appliedStatusCursor().statusVersion() + 1, running, finished, available);
        Runnable projection;
        worker.lock.lock();
        try {
            projection = decode.applyPreparedStatus(worker,
                    worker.prepareNewStatus(worker.freezeStatusResponse(response)));
        } finally {
            worker.lock.unlock();
        }
        projection.run();
    }

    private static WorkerStatusResponse status(RoleType role,
                                               long version,
                                               Map<String, TaskInfo> running,
                                               Map<String, TaskInfo> finished,
                                               long available) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(version);
        response.setLatestFinishedVersion(version);
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(available);
        response.setMaxSeqLen(10_000L);
        response.setMaxBatchTokensSize(10_000L);
        return response;
    }

    private static ServerStatus server(WorkerEndpoint endpoint, String requestId) {
        ServerStatus status = new ServerStatus();
        status.setRole(endpoint.getStatus().getRole());
        status.setServerIp(endpoint.getIp());
        status.setHttpPort(endpoint.getHttpPort());
        status.setGrpcPort(8081);
        status.setRequestId(requestId);
        status.setSuccess(true);
        status.setSelectedEngineIndex(endpoint.getStatus().getEngineIndex(), endpoint.getStatus().getMultiEngineNum());
        return status;
    }

    private static TaskInfo task(String requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(TaskPhase.RUNNING);
        task.setInputLength(128);
        return task;
    }
}
