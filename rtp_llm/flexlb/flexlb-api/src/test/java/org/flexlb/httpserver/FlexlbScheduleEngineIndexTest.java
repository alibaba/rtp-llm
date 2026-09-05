package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.strategy.LoadBalanceStrategy;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.CacheMetadataUpdateOrchestrator;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.flexlb.service.optimizer.OptimizerClient;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FlexlbScheduleEngineIndexTest {

    private EndpointRegistry endpoints;
    private final Map<LoadBalanceStrategyEnum, LoadBalanceStrategy> originalStrategies =
            new EnumMap<>(LoadBalanceStrategyEnum.class);

    @BeforeEach
    void saveStrategies() {
        for (LoadBalanceStrategyEnum strategy : LoadBalanceStrategyEnum.values()) {
            try {
                originalStrategies.put(strategy,
                        LoadBalanceStrategyFactory.getLoadBalanceStrategy(strategy));
            } catch (RuntimeException missingStrategy) {
                // The factory has no lookup API for absent registrations.
            }
        }
    }

    @AfterEach
    void cleanUp() {
        if (endpoints != null) {
            endpoints.close();
        }
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        LoadBalanceStrategyFactory.clear();
        originalStrategies.forEach(LoadBalanceStrategyFactory::register);
    }

    @Test
    void scheduleIdentifiesEngineZeroWhenItsSiblingHasNoKvCapacity() throws Exception {
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = schedule(2, 0);

        assertTrue(response.getSuccess(), response.getErrorMessage());
        assertEquals(1, response.getServerStatusCount());
        FlexlbScheduleProtocol.FlexlbServerStatusPB selected = response.getServerStatus(0);
        assertEquals("127.0.0.1", selected.getServerIp());
        assertEquals(8080, selected.getHttpPort());
        assertEquals(8081, selected.getGrpcPort());
        assertTrue(selected.hasEngineIndex());
        assertEquals(0, selected.getEngineIndex());
    }

    @Test
    void scheduleIdentifiesEngineOneWhenItsSiblingHasNoKvCapacity() throws Exception {
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = schedule(2, 1);

        assertTrue(response.getSuccess(), response.getErrorMessage());
        assertEquals(1, response.getServerStatusCount());
        FlexlbScheduleProtocol.FlexlbServerStatusPB selected = response.getServerStatus(0);
        assertEquals("127.0.0.1", selected.getServerIp());
        assertEquals(8080, selected.getHttpPort());
        assertEquals(8081, selected.getGrpcPort());
        assertTrue(selected.hasEngineIndex());
        assertEquals(1, selected.getEngineIndex());
    }

    @Test
    void scheduleOmitsEngineIndexForSingleEngineFrontend() throws Exception {
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = schedule(1, 0);

        assertTrue(response.getSuccess(), response.getErrorMessage());
        assertEquals(1, response.getServerStatusCount());
        FlexlbScheduleProtocol.FlexlbServerStatusPB selected = response.getServerStatus(0);
        assertEquals("127.0.0.1", selected.getServerIp());
        assertEquals(8080, selected.getHttpPort());
        assertEquals(8081, selected.getGrpcPort());
        assertFalse(selected.hasEngineIndex());
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB schedule(
            int engineCount, int availableEngine) throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduler(new DirectSchedulerConfig());
        config.getRouter().getRoles().getDecode()
                .setSelector(new RoutingConfig.RandomDecodeSelectorConfig());
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        BatchSchedulerReporter batchReporter = mock(BatchSchedulerReporter.class);
        endpoints = new EndpointRegistry(configService, () -> null, batchReporter);
        for (int index = 0; index < engineCount; index++) {
            WorkerStatus worker = new WorkerStatus();
            worker.setIp("127.0.0.1");
            worker.setPort(8080);
            worker.setEngineIndex(index);
            worker.setMultiEngineNum(engineCount);
            worker.setRole(RoleType.DECODE);
            worker.setAlive(true);
            worker.getTotalKvCacheTokens().set(10000);
            worker.getAvailableKvCacheTokens().set(index == availableEngine ? 10000 : 0);
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap()
                    .put(worker.getLogicalIpPort(), worker);
            endpoints.ensureEndpoint(RoleType.DECODE, worker.getLogicalIpPort(), worker);
        }
        new RandomStrategy(new EngineWorkerStatus(endpoints), configService,
                new ResourceMeasureFactory(List.of(new DecodeResourceMeasure(configService))));
        RouteService routes = new RouteService(configService,
                new DefaultRouter(configService, context -> GroupRoutingDecision.none(), endpoints),
                null, new RecentCacheKeyTraceReporter());
        ModelMetaConfig models = new ModelMetaConfig();
        CacheAwareService cache = new CacheAwareService(null, null, null,
                new CacheMetadataUpdateOrchestrator(
                        new CacheMatchConfiguration(models, config), null, null), null);
        FlexlbServiceImpl service = new FlexlbServiceImpl(routes,
                mock(LBStatusConsistencyService.class), mock(EngineHealthReporter.class),
                new ActiveRequestCounter(), null, configService, batchReporter,
                mock(ServerScheduleLatencyRecorder.class), mock(PrioritySchedulerReporter.class),
                cache, new OptimizerClient(null, null, models, configService, null, null));
        CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> result =
                new CompletableFuture<>();
        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId("5001").setSeqLen(16).build(), new StreamObserver<>() {
                    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB response;

                    @Override
                    public void onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB value) {
                        response = value;
                    }

                    @Override
                    public void onError(Throwable error) {
                        result.completeExceptionally(error);
                    }

                    @Override
                    public void onCompleted() {
                        result.complete(response);
                    }
                });
        return FlexlbScheduleProtocol.FlexlbScheduleResponsePB.parseFrom(
                result.get(5, TimeUnit.SECONDS).toByteArray());
    }
}
