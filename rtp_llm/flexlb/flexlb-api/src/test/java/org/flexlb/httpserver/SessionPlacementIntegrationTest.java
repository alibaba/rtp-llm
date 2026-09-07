package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.balance.session.SessionPlacementStore;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Crosses the real service publication, store and selector address boundary. */
class SessionPlacementIntegrationTest {
    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "PDFUSION"})
    void deliveredEndpointIsRecognizedByNextRealSelection(RoleType role) {
        FlexlbConfig config = new FlexlbConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        var affinity = new RoutingConfig.SessionAffinityConfig();
        affinity.setTtlMs(1_800_000L);
        affinity.setMaxExtraTtftMs(40L);
        config.getRouter().getRoles().getPrefill().setSessionAffinity(affinity);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        ConfigService configs = mock(ConfigService.class);
        when(configs.loadBalanceConfig()).thenReturn(config);
        SessionPlacementStore store = new SessionPlacementStore(configs);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        EndpointRegistry registry = new EndpointRegistry(configs,
                mock(EndpointEventProjector.class), reporter,
                new RouteDeliveryStrategy(mock(RequestRegistry.class), new DeliveryMetrics(reporter)),
                new PlacementAvailability());
        try {
            publish(registry, role, "10.0.0.2", 8088);
            CacheAwareService cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
            EngineHealthReporter health = mock(EngineHealthReporter.class);
            CostBasedPrefillStrategy strategy = new CostBasedPrefillStrategy(
                    new WorkerDirectory(registry), cache, health, store);
            RouteService routes = mock(RouteService.class);
            when(routes.route(any())).thenAnswer(invocation -> {
                BalanceContext context = invocation.getArgument(0);
                context.setConfig(config);
                try (SelectedRole selected = strategy.select(context, role, null).value()) {
                    Response response = new Response();
                    response.setSuccess(true);
                    response.setCode(200);
                    response.setServerStatus(List.of(selected.serverStatus()));
                    return CompletableFuture.completedFuture(response);
                }
            });
            ActiveRequestCounter active = mock(ActiveRequestCounter.class);
            when(active.acquire()).thenReturn(mock(ActiveRequestCounter.RequestToken.class));
            FlexlbServiceImpl service = new FlexlbServiceImpl(routes,
                    mock(LBStatusConsistencyService.class), health, active,
                    mock(FlexlbGrpcForwarder.class), configs, reporter,
                    mock(ServerScheduleLatencyRecorder.class),
                    mock(RequestSchedulerReporter.class), store);
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                    mock(StreamObserver.class);
            service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                    .setRequestId(100_100L).setModel("session-integration")
                    .setSessionRoutingHint(FlexlbScheduleProtocol.SessionRoutingHintPB.newBuilder()
                            .setSchemaVersion(1).setSessionId("real-store-session")
                            .setState(FlexlbScheduleProtocol.SessionStatePB.SESSION_STATE_NEW))
                    .build(), observer);
            verify(observer).onCompleted();
            var delivered = ArgumentCaptor.forClass(
                    FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
            verify(observer).onNext(delivered.capture());
            assertTrue(delivered.getValue().getSuccess(), delivered.getValue().getErrorMessage());
            String address = store.find("session-integration", "real-store-session",
                    affinity.getTtlMs()).orElseThrow().ipPort();
            assertEquals("10.0.0.2:8088", address);
            assertTrue(registry.endpointAddressSnapshot(role).contains(address));

            publish(registry, role, "10.0.0.3", 8099);
            Request request = new Request();
            request.setRequestId(100_101L);
            request.setModel("session-integration");
            request.setSeqLen(100L);
            request.setMaxNewTokens(8);
            request.setSessionSchemaVersion(1);
            request.setInferenceSessionId("real-store-session");
            request.setInferenceSessionState(Request.SessionState.ESTABLISHED);
            BalanceContext next = new BalanceContext();
            next.setRequest(request);
            next.setConfig(config);
            try (SelectedRole selected = strategy.select(next, role, null).value()) {
                assertEquals("SESSION_AFFINITY", next.getSessionAffinityReason());
                assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
                assertEquals(8088, selected.serverStatus().getHttpPort());
            }
        } finally {
            registry.close();
        }
    }

    private static void publish(EndpointRegistry registry, RoleType role, String ip, int port) {
        WorkerStatus worker = WorkerStatus.createDiscovered(
                role, "session-integration", ip, port, port + 1, null);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(role);
        status.setAlive(true);
        status.setStatusVersion(1L);
        status.setLatestFinishedVersion(0L);
        status.setAvailableKvCacheTokens(1_000_000L);
        status.setTotalKvCacheTokens(2_000_000L);
        status.setMaxSeqLen(1_000_000L);
        status.setMaxBatchTokensSize(1_000_000L);
        status.setRunningTaskInfo(Map.of());
        status.setFinishedTaskInfo(Map.of());
        worker.lock.lock();
        try {
            registry.publishPreparedEndpoint(worker.getIpPort(), worker,
                    worker.prepareNewStatus(worker.freezeStatusResponse(status)));
            worker.recordSuccessfulPoll(true);
        } finally {
            worker.lock.unlock();
        }
    }
}
