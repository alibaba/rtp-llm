package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.CacheMetadataUpdateOrchestrator;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.consistency.MasterStatusService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that successful scheduling updates standby metadata without allowing metadata failure to
 * replace the response.
 */
class FlexlbScheduleLocalStandbyTest {

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION"})
    void successfulScheduleWarmsIndexWithClientProvidedKeys(RoleType role) throws Exception {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        when(configuration.isLocalStandbyEnabled()).thenReturn(true);
        when(configuration.getLocalStandbyConfig()).thenReturn(new LocalStandbyConfig());
        when(configuration.getServiceRoutes()).thenReturn(List.of());
        WorkerStatus worker = mock(WorkerStatus.class);
        when(worker.getLogicalIpPort()).thenReturn("127.0.0.1:8080@1");
        WorkerStatusProvider workers = mock(WorkerStatusProvider.class);
        when(workers.getWorkerStatuses(role, "default")).thenReturn(List.of(worker));
        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        LocalStandbyCacheManager manager =
                spy(new LocalStandbyCacheManager(configuration, workers, metrics));
        CompletableFuture<Void> indexed = new CompletableFuture<>();
        doAnswer(
                        invocation -> {
                            invocation.callRealMethod();
                            indexed.complete(null);
                            return null;
                        })
                .when(manager)
                .addRoutedRequestBlocks(any(), any());
        LocalStandbyCacheMatchProvider provider =
                new LocalStandbyCacheMatchProvider(configuration, manager, metrics);
        CacheAwareService cache =
                new CacheAwareService(
                        metrics,
                        null,
                        null,
                        new CacheMetadataUpdateOrchestrator(configuration, null, provider));
        AbstractRequestScheduler router = mock(AbstractRequestScheduler.class);
        CompletableFuture<Response> routed = new CompletableFuture<>();
        when(router.submit(any())).thenReturn(routed);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        try {
            service(router, cache).schedule(request(), observer);
            verify(observer, never()).onCompleted();
            assertEquals(0, manager.mappingCount());

            routed.complete(response(role, true));
            verify(observer).onCompleted();
            verify(observer)
                    .onNext(org.mockito.ArgumentMatchers.argThat(result -> result.getSuccess()));
            indexed.get(5, TimeUnit.SECONDS);
            assertEquals(3, manager.mappingCount());
            var prediction =
                    provider.asyncLocalStandbyMatch(
                                    new CacheMatchQuery(
                                            "followup",
                                            List.of(11L, 22L, 33L),
                                            4096,
                                            role,
                                            "default"))
                            .get(5, TimeUnit.SECONDS);
            assertEquals(
                    3, prediction.exactHostMatch(worker.getLogicalIpPort()).localMatchBlocks());
            assertEquals(4096, prediction.blockSize());
        } finally {
            provider.shutdown();
            manager.shutdown();
        }
    }

    @Test
    void failedScheduleDoesNotUpdateIndex() {
        AbstractRequestScheduler router = mock(AbstractRequestScheduler.class);
        when(router.submit(any()))
                .thenReturn(CompletableFuture.completedFuture(response(RoleType.PREFILL, false)));
        CacheAwareService cache = mock(CacheAwareService.class);
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service(router, cache).schedule(request(), observer);

        verify(observer).onCompleted();
        verify(cache, never()).updateFromRoutedRequest(any(), any());
    }

    @Test
    void metadataUpdateFailurePreservesSuccessfulSchedule() {
        AbstractRequestScheduler router = mock(AbstractRequestScheduler.class);
        when(router.submit(any()))
                .thenReturn(CompletableFuture.completedFuture(response(RoleType.PREFILL, true)));
        CacheAwareService cache = mock(CacheAwareService.class);
        doThrow(new IllegalStateException("metadata update unavailable"))
                .when(cache)
                .updateFromRoutedRequest(any(), any());
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service(router, cache).schedule(request(), observer);

        verify(observer)
                .onNext(org.mockito.ArgumentMatchers.argThat(result -> result.getSuccess()));
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());
    }

    private FlexlbServiceImpl service(AbstractRequestScheduler router, CacheAwareService cache) {
        ConfigService config = mock(ConfigService.class);
        FlexlbConfig routingConfig = new FlexlbConfig();
        routingConfig.setDispatcher(org.flexlb.config.ConfigTestSupport.nonBatch());
        when(config.loadBalanceConfig()).thenReturn(routingConfig);
        var service =
                FlexlbServiceTestSupport.create(
                        router,
                        mock(org.flexlb.balance.scheduler.RequestRepository.class),
                        mock(MasterStatusService.class),
                        mock(EngineHealthReporter.class),
                        mock(FlexlbGrpcForwarder.class),
                        config,
                        mock(DeliveryMetricsReporter.class),
                        mock(ServerScheduleLatencyRecorder.class),
                        mock(RequestSchedulerReporter.class));
        var model = mock(org.flexlb.config.ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.PDFUSION));
        service.configureCache(cache, model);
        return service;
    }

    private FlexlbScheduleProtocol.FlexlbScheduleRequestPB request() {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestIdStr("warmup")
                .setSeqLen(12288)
                .setCacheKeyBlockSize(4096)
                .addAllBlockCacheKeys(List.of(11L, 22L, 33L))
                .build();
    }

    private Response response(RoleType role, boolean success) {
        ServerStatus worker = new ServerStatus();
        worker.setSuccess(true);
        worker.setRole(role);
        worker.setServerIp("127.0.0.1");
        worker.setHttpPort(8080);
        worker.setGrpcPort(8081);
        worker.setGroup("default");
        worker.setRequestId("warmup");
        worker.setSelectedEngineIndex(1, 2);
        Response response = new Response();
        response.setSuccess(success);
        response.setServerStatus(List.of(worker));
        return response;
    }
}
