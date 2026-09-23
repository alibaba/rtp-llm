package org.flexlb.httpserver;

import io.grpc.ForwardingServerCallListener;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleResponsePB;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.RouteService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class FlexlbGrpcDrainTest {
    @Test
    void acceptedRpcCanEnterServiceAndFinishAfterQuietPeriod() throws Exception {
        var accepted = new CountDownLatch(1);
        var resume = new CountDownLatch(1);
        var timing = new GrpcServerTimingInterceptor();
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(ConfigService.parse("""
                {"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "grpcServer":{"shutdownQuietPeriodMs":100}}
                """));
        var routes = mock(RouteService.class);
        var success = new Response();
        success.setSuccess(true);
        success.setCode(200);
        when(routes.route(any())).thenReturn(CompletableFuture.completedFuture(success));
        var service = new FlexlbServiceImpl(routes, mock(LBStatusConsistencyService.class),
                mock(EngineHealthReporter.class), mock(FlexlbGrpcForwarder.class), config,
                mock(BatchSchedulerReporter.class), mock(ServerScheduleLatencyRecorder.class),
                mock(RequestSchedulerReporter.class));
        ServerInterceptor pauseBeforeSchedule = new ServerInterceptor() {
            @Override
            public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
                    ServerCall<ReqT, RespT> call, io.grpc.Metadata headers,
                    ServerCallHandler<ReqT, RespT> next) {
                return new ForwardingServerCallListener.SimpleForwardingServerCallListener<>(
                        next.startCall(call, headers)) {
                    @Override
                    public void onHalfClose() {
                        accepted.countDown();
                        try {
                            assertTrue(resume.await(5, TimeUnit.SECONDS));
                        } catch (InterruptedException error) {
                            Thread.currentThread().interrupt();
                            throw new AssertionError(error);
                        }
                        super.onHalfClose();
                    }
                };
            }
        };
        var server = NettyServerBuilder.forPort(0).addService(ServerInterceptors.intercept(
                service, pauseBeforeSchedule, timing)).build().start();
        var grpc = new FlexlbGrpcServer(service, config, new MockEnvironment(), null, null, timing, null);
        ReflectionTestUtils.setField(grpc, "server", server);
        var channel = NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        var drainer = new Thread(grpc::drain);
        try {
            var pending = FlexlbServiceGrpc.newFutureStub(channel).withDeadlineAfter(8, TimeUnit.SECONDS)
                    .schedule(FlexlbScheduleRequestPB.newBuilder().setRequestId("101").build());
            assertTrue(accepted.await(3, TimeUnit.SECONDS));
            drainer.start();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            while (!server.isShutdown() && System.nanoTime() < deadline) {
                Thread.sleep(10);
            }
            assertTrue(server.isShutdown());
            assertFalse(pending.isDone());
            assertTrue(drainer.isAlive());
            resume.countDown();
            assertTrue(pending.get(3, TimeUnit.SECONDS).getSuccess());
            verify(routes, times(1)).route(any());
            drainer.join(3000);
            assertFalse(drainer.isAlive());
        } finally {
            resume.countDown();
            channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            drainer.join(3000);
        }
    }

    @Test
    void normalIdleDoesNotShutdownAndDrainStartsWithAFullQuietPeriod() throws Exception {
        try (var fixture = new Fixture(300)) {
            Thread.sleep(700); // Idle before TERM does not count toward drain.
            assertFalse(fixture.server.isShutdown());
            long started = System.nanoTime();
            fixture.startDrain();
            fixture.drainer.join(100);
            assertTrue(fixture.drainer.isAlive());
            fixture.drainer.join(3000);
            assertFalse(fixture.drainer.isAlive());
            assertTrue(System.nanoTime() - started >= TimeUnit.MILLISECONDS.toNanos(300));
        }
    }

    @Test
    void everyArrivalResetsQuietPeriodAndAcceptedRpcMustCompleteEvenAfterInterrupt() throws Exception {
        try (var fixture = new Fixture(500)) {
            // Establish the connection before starting the shutdown clock.
            var warmup = fixture.request(1);
            fixture.completeNext();
            warmup.get(2, TimeUnit.SECONDS);
            fixture.startDrain();
            for (int i = 0; i < 5; i++) {
                Thread.sleep(150);
                var late = fixture.request(10 + i);
                fixture.completeNext();
                late.get(2, TimeUnit.SECONDS);
                assertFalse(fixture.server.isShutdown());
            }
            var pending = fixture.request(99);
            var response = fixture.responses.poll(2, TimeUnit.SECONDS);
            assertNotNull(response);
            Thread.sleep(200);
            assertFalse(fixture.server.isShutdown(), "latest Schedule must restart the quiet period");
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            while (!fixture.server.isShutdown() && System.nanoTime() < deadline) {
                Thread.sleep(10);
            }
            assertTrue(fixture.server.isShutdown());
            fixture.drainer.interrupt();
            fixture.drainer.join(150);
            assertTrue(fixture.drainer.isAlive(), "pending RPC must survive even after the quiet period");
            assertFalse(pending.isDone());
            response.onNext(FlexlbScheduleResponsePB.newBuilder().setSuccess(true).build());
            response.onCompleted();
            assertTrue(pending.get(2, TimeUnit.SECONDS).getSuccess());
            fixture.drainer.join(3000);
            assertFalse(fixture.drainer.isAlive());
            assertTrue(fixture.interruptedOnReturn.get());
        }
    }

    private static class Fixture implements AutoCloseable {
        final LinkedBlockingQueue<StreamObserver<FlexlbScheduleResponsePB>> responses = new LinkedBlockingQueue<>();
        final AtomicBoolean interruptedOnReturn = new AtomicBoolean();
        final FlexlbGrpcServer grpc;
        final Server server;
        final ManagedChannel channel;
        Thread drainer;

        Fixture(long quietMs) throws Exception {
            var timing = new GrpcServerTimingInterceptor();
            var configService = mock(ConfigService.class);
            when(configService.loadBalanceConfig()).thenReturn(ConfigService.parse("""
                    {"requestLifecycle":{"request":{"timeoutMs":60000}},
                     "grpcServer":{"shutdownQuietPeriodMs":%d}}
                    """.formatted(quietMs)));
            grpc = new FlexlbGrpcServer(null, configService, new MockEnvironment(),
                    null, null, timing, null);
            server = NettyServerBuilder.forPort(0).addService(ServerInterceptors.intercept(
                    new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                        @Override
                        public void schedule(FlexlbScheduleRequestPB request,
                                             StreamObserver<FlexlbScheduleResponsePB> observer) {
                            responses.add(observer);
                        }
                    }, timing)).build().start();
            ReflectionTestUtils.setField(grpc, "server", server);
            channel = NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        }

        com.google.common.util.concurrent.ListenableFuture<FlexlbScheduleResponsePB> request(long id) {
            return FlexlbServiceGrpc.newFutureStub(channel).withDeadlineAfter(10, TimeUnit.SECONDS)
                    .schedule(FlexlbScheduleRequestPB.newBuilder().setRequestId(Long.toString(id)).build());
        }

        void completeNext() throws Exception {
            var response = responses.poll(2, TimeUnit.SECONDS);
            assertNotNull(response);
            response.onNext(FlexlbScheduleResponsePB.newBuilder().setSuccess(true).build());
            response.onCompleted();
        }

        void startDrain() {
            drainer = new Thread(() -> {
                grpc.drain();
                interruptedOnReturn.set(Thread.currentThread().isInterrupted());
            });
            drainer.start();
        }

        public void close() throws Exception {
            channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            if (drainer != null) {
                drainer.join(3000);
            }
        }
    }
}
