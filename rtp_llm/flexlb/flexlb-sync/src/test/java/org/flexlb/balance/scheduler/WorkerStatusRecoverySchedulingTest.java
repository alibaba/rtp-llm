package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.runner.GrpcWorkerStatusRunner;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks that a real status poll resumes generation requests waiting for a vision worker.
 */
class WorkerStatusRecoverySchedulingTest {
    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"VIT", "ENCODER"})
    void sameGenerationStatusRecoveryWakesGlobalPlacement(RoleType visionRole) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        var configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        var batchReporter = mock(BatchSchedulerReporter.class);
        var requestReporter = mock(RequestSchedulerReporter.class);
        var engineReporter = mock(EngineHealthReporter.class);
        var requests = new RequestRegistry(configService, batchReporter, requestReporter);
        var availability = new PlacementAvailability();
        var endpoints =
                new EndpointRegistry(
                        configService,
                        new EndpointEventProjector(requests),
                        batchReporter,
                        new RouteDeliveryStrategy(requests, new DeliveryMetrics(batchReporter)),
                        availability);
        var directory = new WorkerDirectory(endpoints, Runnable::run);
        var cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any()))
                .thenReturn(CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC));
        var model =
                new ModelMetaConfig(
                        """
                        {"service_id":"test","role_endpoints":[{"group":"g1",
                         "pd_fusion_endpoint":{"address":"127.0.0.1:8080"},
                         "%s_endpoint":{"address":"127.0.0.1:8090"}}]}
                        """
                                .formatted(visionRole == RoleType.VIT ? "vit" : "encoder"));
        var router =
                new DefaultRouter(
                        new CostBasedPrefillStrategy(directory, cache, engineReporter),
                        new DecodeSelector(directory),
                        new RandomStrategy(directory),
                        new EncoderStrategy(directory),
                        configService,
                        model);
        var scheduler =
                new RequestScheduler(
                        configService,
                        router,
                        endpoints,
                        batchReporter,
                        mock(EvictionManager.class),
                        requests,
                        availability);
        var runtime =
                new SchedulerRuntime(
                        requests, endpoints, batchReporter, requestReporter, scheduler);
        try {
            publishWorker(directory, endpoints, RoleType.PDFUSION, 8080, false);
            WorkerStatus vision = publishWorker(directory, endpoints, visionRole, 8090, true);
            String address = vision.routingAddress();
            var generation = endpoints.get(visionRole, address);
            assertNull(directory.captureEndpoint(visionRole, address));

            var pending = scheduler.submit(RequestLifecycleTestSupport.context(config, 1L));
            awaitCondition(() -> scheduler.getBlockedRequestCount() == 1);
            assertFalse(pending.isDone());

            var placement = PlacementKey.exact(visionRole, "g1", address);
            long beforeRecovery = availability.lastChangedSequence(placement);
            pollSameStatus(directory, engineReporter, cache, vision);
            assertSame(generation, endpoints.get(visionRole, address));
            assertTrue(vision.hasFreshSchedulingStatus(System.nanoTime() / 1000));

            var response = pending.get(2, TimeUnit.SECONDS);
            assertTrue(availability.lastChangedSequence(placement) > beforeRecovery);
            assertTrue(response.isSuccess(), response.getErrorMessage());
            assertEquals(
                    List.of(RoleType.PDFUSION, visionRole),
                    response.getServerStatus().stream().map(ServerStatus::getRole).toList());
            assertEquals(0, scheduler.getBlockedRequestCount());
        } finally {
            runtime.shutdown();
        }
    }

    private static WorkerStatus publishWorker(
            WorkerDirectory directory,
            EndpointRegistry endpoints,
            RoleType role,
            int port,
            boolean stale) {
        WorkerStatus worker =
                WorkerStatus.createDiscovered(role, "g1", "127.0.0.1", port, port + 1, "test");
        String address = worker.routingAddress();
        directory.currentOrDiscover(role, address, () -> worker);
        var response = new WorkerStatusResponse();
        response.setRole(role);
        response.setStatusVersion(1L);
        response.setAlive(true);
        response.setTotalKvCacheTokens(100_000L);
        response.setAvailableKvCacheTokens(100_000L);
        response.setMaxSeqLen(10_000L);
        response.setMaxBatchTokensSize(10_000L);
        long freshnessUs = TimeUnit.MINUTES.toMicros(1);
        worker.configureStatusFreshness(freshnessUs);
        worker.lock.lock();
        try {
            endpoints.publishPreparedEndpoint(
                    address,
                    worker,
                    worker.prepareNewStatus(worker.freezeStatusResponse(response)));
            worker.recordSuccessfulPoll(true);
            long nowUs = System.nanoTime() / 1000;
            worker.recordStatusApplied(stale ? nowUs - freshnessUs - 1 : nowUs);
        } finally {
            worker.lock.unlock();
        }
        return worker;
    }

    private static void pollSameStatus(
            WorkerDirectory directory,
            EngineHealthReporter reporter,
            CacheAwareService cache,
            WorkerStatus worker) {
        RoleType role = worker.getRole();
        var response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(role.getCode())
                        .setAlive(true)
                        .setStatusVersion(worker.appliedStatusCursor().statusVersion())
                        .build();
        var grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        new GrpcWorkerStatusRunner(
                        "test",
                        worker.routingAddress(),
                        worker.getSite(),
                        role,
                        null,
                        worker,
                        worker.tryBeginStatusPoll(),
                        directory,
                        reporter,
                        grpc,
                        5_000L,
                        cache,
                        Runnable::run)
                .run();
    }
}
