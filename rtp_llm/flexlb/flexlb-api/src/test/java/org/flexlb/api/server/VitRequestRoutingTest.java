package org.flexlb.api.server;

import io.grpc.stub.StreamObserver;
import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.api.protocol.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.api.protocol.FlexlbScheduleProtocol.FlexlbScheduleResponsePB;
import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.schedule.RequestSchedulerTestRuntime;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.master.MasterStatusService;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * VIT routing precedes generation admission at the real API entry.
 */
class VitRequestRoutingTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void vitAndStalePreselectionDoNotRegisterAndSameIdCanLaterSchedulePd(boolean queued)
            throws Exception {
        var config = org.flexlb.mock.TestFlexlbConfigs.create();
        if (!queued) config.setScheduler(ConfigTestSupport.direct());
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var selector = mock(RequestRouteSelector.class);
        var batches = mock(DeliveryMetricsReporter.class);
        var reports = mock(RequestSchedulerReporter.class);
        var engine = mock(EngineGrpcClient.class);
        try (var runtime =
                new RequestSchedulerTestRuntime(
                        service,
                        new org.flexlb.balance.schedule.EngineCancellation(engine),
                        batches,
                        reports,
                        engine)) {
            runtime.bindRouter(selector);
            var api =
                    FlexlbServiceTestSupport.create(
                            runtime.entry(),
                            runtime.requestRegistry(),
                            mock(MasterStatusService.class),
                            mock(ServiceMetricsReporter.class),
                            mock(FlexlbGrpcForwarder.class),
                            service,
                            batches,
                            mock(ServerScheduleLatencyRecorder.class),
                            reports);
            api.configureRouting(selector);
            var selected = new Response();
            selected.setSuccess(true);
            selected.setCode(200);
            when(selector.routeVit(any())).thenReturn(selected);
            var vision =
                    FlexlbScheduleRequestPB.newBuilder()
                            .setRequestId(918L)
                            .setVitRouteOnly(true)
                            .build();
            assertTrue(schedule(api, vision).getSuccess());
            assertEquals(0, runtime.requestRegistry().liveRequestCount());
            verify(selector, never()).select(any(), any());

            var stale = FlexlbScheduleRequestPB.newBuilder().setRequestId(918L);
            stale.getSelectedVitBuilder().setRole("VIT").setWorkerGeneration(1L);
            assertEquals(
                    StrategyErrorType.VIT_ROUTE_STALE.getErrorCode(),
                    schedule(api, stale.build()).getCode());
            assertEquals(0, runtime.requestRegistry().liveRequestCount());
            verify(selector, never()).select(any(), any());

            when(selector.select(any(), isNull()))
                    .thenReturn(
                            org.flexlb.balance.schedule.RequestResult.rejected(
                                    Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
            var pd = FlexlbScheduleRequestPB.newBuilder().setRequestId(918L).build();
            assertEquals(
                    StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                    schedule(api, pd).getCode());
            verify(selector).select(any(), isNull());
        }
    }

    private static FlexlbScheduleResponsePB schedule(
            FlexlbServiceImpl api, FlexlbScheduleRequestPB request) throws Exception {
        var result = new CompletableFuture<FlexlbScheduleResponsePB>();
        api.schedule(
                request,
                new StreamObserver<>() {
                    @Override
                    public void onNext(FlexlbScheduleResponsePB response) {
                        result.complete(response);
                    }

                    @Override
                    public void onError(Throwable failure) {
                        result.completeExceptionally(failure);
                    }

                    @Override
                    public void onCompleted() {}
                });
        return result.get(5L, TimeUnit.SECONDS);
    }
}
