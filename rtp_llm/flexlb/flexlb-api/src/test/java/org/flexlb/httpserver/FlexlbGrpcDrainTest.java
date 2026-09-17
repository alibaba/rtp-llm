package org.flexlb.httpserver;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.ServerInterceptors;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.config.ConfigService;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleResponsePB;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FlexlbGrpcDrainTest {
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
                    .schedule(FlexlbScheduleRequestPB.newBuilder().setRequestId(id).build());
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
