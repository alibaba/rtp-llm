package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.MasterElectService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbCancelRequestPB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.RouteService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Exercises DIRECT admission and the independent Cancel path on the real Netty server.
 */
@Timeout(15)
class DirectScheduleOverloadNettyTest {
    @Test
    void saturatedDirectPlacementRejectsAndCancelStillCompletes() throws Exception {
        var config = mock(ConfigService.class);
        var routingConfig =
                ConfigService.parse(
                        """
                {"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "scheduler":{"type":"DIRECT"},"dispatcher":{"type":"NON_BATCH"},
                 "grpcServer":{"executorCoreSize":1,"executorMaxSize":1,"executorQueueSize":1,
                               "shutdownQuietPeriodMs":10}}
                """);
        when(config.loadBalanceConfig()).thenReturn(routingConfig);
        var routes = mock(RouteService.class);
        var health = mock(EngineHealthReporter.class);
        var service =
                new FlexlbServiceImpl(
                        routes,
                        mock(MasterElectService.class),
                        health,
                        mock(FlexlbGrpcForwarder.class),
                        config,
                        mock(BatchSchedulerReporter.class),
                        mock(ServerScheduleLatencyRecorder.class),
                        mock(RequestSchedulerReporter.class));
        var eventLoop = new NioEventLoopGroup(1);
        var grpc =
                new FlexlbGrpcServer(
                        service,
                        config,
                        new MockEnvironment(),
                        eventLoop,
                        NoOpFlexMonitor.getInstance(),
                        new GrpcServerTimingInterceptor(),
                        new GrpcQosHeaderInterceptor()) {
                    @Override
                    protected NettyServerBuilder createServerBuilder(int port) {
                        return NettyServerBuilder.forPort(0);
                    }
                };
        var slowRouteStarted = new CountDownLatch(1);
        var releaseSlowRoute = new CountDownLatch(1);
        var cancellationReceived = new CountDownLatch(1);
        var admissionQueueDrained = new CountDownLatch(1);
        grpc.start();
        var server = (Server) ReflectionTestUtils.getField(grpc, "server");
        var executor = (ThreadPoolExecutor) ReflectionTestUtils.getField(grpc, "grpcExecutor");
        var channel =
                NettyChannelBuilder.forAddress("127.0.0.1", server.getPort())
                        .usePlaintext()
                        .build();
        try {
            doAnswer(
                            invocation -> {
                                BalanceContext context = invocation.getArgument(0);
                                if (context.getRequestId().equals("101")) {
                                    // Fill the queue after the ingress check, while its worker is
                                    // running this callback.
                                    executor.execute(admissionQueueDrained::countDown);
                                }
                                return null;
                            })
                    .when(health)
                    .reportArriveDelayTime(any());
            when(routes.route(any()))
                    .thenAnswer(
                            invocation -> {
                                BalanceContext context = invocation.getArgument(0);
                                if (context.getRequestId().equals("102")) {
                                    assertFalse(eventLoop.next().inEventLoop());
                                    assertTrue(
                                            Thread.currentThread()
                                                    .getName()
                                                    .startsWith("flexlb-grpc-executor"));
                                    assertNotNull(Context.current().getDeadline());
                                    Context.current()
                                            .addListener(
                                                    ignored -> cancellationReceived.countDown(),
                                                    Runnable::run);
                                    slowRouteStarted.countDown();
                                    assertTrue(releaseSlowRoute.await(6, TimeUnit.SECONDS));
                                }
                                var response = new Response();
                                response.setSuccess(true);
                                return CompletableFuture.completedFuture(response);
                            });

            var client = FlexlbServiceGrpc.newFutureStub(channel);
            assertOverloaded(client, 101);
            assertTrue(admissionQueueDrained.await(2, TimeUnit.SECONDS));
            var slow = client.withDeadlineAfter(10, TimeUnit.SECONDS).schedule(request(102));
            assertTrue(slowRouteStarted.await(2, TimeUnit.SECONDS));
            executor.execute(() -> {});
            assertOverloaded(client, 103);
            assertFalse(
                    client.withDeadlineAfter(2, TimeUnit.SECONDS)
                            .cancel(
                                    FlexlbCancelRequestPB.newBuilder()
                                            .setRequestIdStr("104")
                                            .build())
                            .get(2, TimeUnit.SECONDS)
                            .getFound());

            assertTrue(slow.cancel(true));
            assertTrue(cancellationReceived.await(2, TimeUnit.SECONDS));
            releaseSlowRoute.countDown();
            verify(routes, timeout(2000)).cancelRequest("102", 0L, CancelReason.CLIENT_CANCELLED);
            verify(routes, never())
                    .route(
                            argThat(
                                    ctx ->
                                            ctx.getRequestId().equals("101")
                                                    || ctx.getRequestId().equals("103")));
        } finally {
            releaseSlowRoute.countDown();
            channel.shutdownNow().awaitTermination(2, TimeUnit.SECONDS);
            grpc.shutdown();
        }
    }

    private static void assertOverloaded(FlexlbServiceGrpc.FlexlbServiceFutureStub client, int id) {
        var failed = client.withDeadlineAfter(5, TimeUnit.SECONDS).schedule(request(id));
        var error = assertThrows(ExecutionException.class, () -> failed.get(2, TimeUnit.SECONDS));
        assertEquals(
                Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(error.getCause()).getCode());
    }

    private static FlexlbScheduleRequestPB request(int id) {
        return FlexlbScheduleRequestPB.newBuilder().setRequestIdStr(Integer.toString(id)).build();
    }
}
