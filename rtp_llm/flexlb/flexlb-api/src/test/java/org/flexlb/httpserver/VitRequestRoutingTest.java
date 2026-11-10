package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.scheduler.RequestRouteSelector;
import org.flexlb.balance.scheduler.RequestSchedulerTestRuntime;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.consistency.MasterStatusService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleResponsePB;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
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
                new RequestSchedulerTestRuntime(service, engine, null, batches, reports)) {
            runtime.bindRouter(selector);
            var api =
                    FlexlbServiceTestSupport.create(
                            runtime.entry(),
                            runtime.requestRegistry(),
                            mock(MasterStatusService.class),
                            mock(EngineHealthReporter.class),
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

            var missing = FlexlbScheduleRequestPB.newBuilder().setRequestId(918L).build();
            assertEquals(
                    StrategyErrorType.GENERATE_INPUT_REQUIRED.getErrorCode(),
                    schedule(api, missing).getCode());
            assertEquals(0, runtime.requestRegistry().liveRequestCount());
            when(selector.select(any(), isNull()))
                    .thenReturn(
                            RoutingResult.rejected(
                                    Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
            var pd =
                    missing.toBuilder()
                            .setGenerateInput(
                                    EngineRpcService.GenerateInputPB.newBuilder()
                                            .setRequestId(918L)
                                            .build()
                                            .toByteString())
                            .build();
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
