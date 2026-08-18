package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.lang.reflect.Field;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class FlexlbGrpcForwarderAsyncTest {

    private static final String MASTER_HTTP_ADDRESS = "10.0.0.2:7001";
    private static final String MASTER_CHANNEL_KEY = "10.0.0.2:7003";

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void forwardsScheduleAsynchronouslyAndSetsOneHop() throws Exception {
        AtomicReference<FlexlbScheduleProtocol.FlexlbScheduleRequestPB> forwarded =
                new AtomicReference<>();
        try (RpcFixture fixture = RpcFixture.start((request, observer) -> {
            forwarded.set(request);
            observer.onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                    .setSuccess(true)
                    .setCode(200)
                    .build());
            observer.onCompleted();
        })) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);

            FlexlbGrpcForwarder.MasterForwardResult result = await(
                    forwarder.forwardScheduleToMaster(request(101L)));

            assertNotNull(result.response());
            assertTrue(result.response().getSuccess());
            assertEquals(1, forwarded.get().getForwardHop());
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "200");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void unavailableIsTerminalAndDoesNotDiscardTheChannel() throws Exception {
        try (RpcFixture fixture = RpcFixture.start((request, observer) ->
                observer.onError(Status.UNAVAILABLE
                        .withDescription("leader unavailable")
                        .asRuntimeException()))) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);

            FlexlbGrpcForwarder.MasterForwardResult result = await(
                    forwarder.forwardScheduleToMaster(request(102L)));

            assertTrue(result.masterFound());
            assertEquals("UNAVAILABLE", result.failure());
            assertFalse(fixture.channel.isShutdown());
            assertTrue(channels(forwarder).containsKey(MASTER_CHANNEL_KEY));
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void inboundCancellationCancelsTheForwardedRpc() throws Exception {
        CountDownLatch masterReceivedRequest = new CountDownLatch(1);
        try (RpcFixture fixture = RpcFixture.start((request, observer) ->
                masterReceivedRequest.countDown())) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);
            Context.CancellableContext inbound = Context.current().withCancellation();

            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> pending =
                    inbound.call(() -> forwarder.forwardScheduleToMaster(request(103L)));
            assertTrue(masterReceivedRequest.await(2, TimeUnit.SECONDS));
            assertFalse(pending.toCompletableFuture().isDone());

            inbound.cancel(null);
            FlexlbGrpcForwarder.MasterForwardResult result = await(pending);

            assertEquals("CANCELLED", result.failure());
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void inboundDeadlineBoundsTheForwardedRpc() throws Exception {
        CountDownLatch masterReceivedRequest = new CountDownLatch(1);
        ScheduledExecutorService deadlineTimer = Executors.newSingleThreadScheduledExecutor();
        try (RpcFixture fixture = RpcFixture.start((request, observer) ->
                masterReceivedRequest.countDown())) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);
            Context.CancellableContext inbound = Context.current()
                    .withDeadlineAfter(500, TimeUnit.MILLISECONDS, deadlineTimer);

            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> pending =
                    inbound.call(() -> forwarder.forwardScheduleToMaster(request(104L)));
            assertTrue(masterReceivedRequest.await(2, TimeUnit.SECONDS));
            FlexlbGrpcForwarder.MasterForwardResult result = await(pending);

            assertEquals("DEADLINE_EXCEEDED", result.failure());
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
            forwarder.shutdown();
            inbound.cancel(null);
        } finally {
            deadlineTimer.shutdownNow();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void monitoringFailureCannotLoseOrDuplicateTheResponse() throws Exception {
        try (RpcFixture fixture = RpcFixture.start((request, observer) -> {
            observer.onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                    .setSuccess(true)
                    .setCode(200)
                    .build());
            observer.onCompleted();
        })) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            doThrow(new RuntimeException("monitor unavailable"))
                    .when(reporter)
                    .reportForwardToMasterResult("10.0.0.2", "200");
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);

            FlexlbGrpcForwarder.MasterForwardResult result = await(
                    forwarder.forwardScheduleToMaster(request(105L)));

            assertNotNull(result.response());
            assertTrue(result.response().getSuccess());
            verify(reporter, times(1))
                    .reportForwardToMasterResult("10.0.0.2", "200");
            forwarder.shutdown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void forwardsCancelWithoutBlockingAndSetsOneHop() throws Exception {
        AtomicReference<FlexlbScheduleProtocol.FlexlbCancelRequestPB> forwarded =
                new AtomicReference<>();
        AtomicReference<StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB>>
                masterObserver = new AtomicReference<>();
        CountDownLatch masterReceivedRequest = new CountDownLatch(1);
        try (RpcFixture fixture = RpcFixture.startCancel((request, observer) -> {
            forwarded.set(request);
            masterObserver.set(observer);
            masterReceivedRequest.countDown();
        })) {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            FlexlbGrpcForwarder forwarder = forwarder(fixture.channel, reporter);
            AtomicReference<CompletionStage<FlexlbGrpcForwarder.CancelForwardResult>>
                    pending = new AtomicReference<>();

            assertTimeoutPreemptively(Duration.ofSeconds(1), () ->
                    pending.set(forwarder.forwardCancelToMaster(
                            FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                                    .setRequestId(106L)
                                    .build())));
            assertTrue(masterReceivedRequest.await(2, TimeUnit.SECONDS));
            assertFalse(pending.get().toCompletableFuture().isDone());
            assertEquals(1, forwarded.get().getForwardHop());

            masterObserver.get().onNext(
                    FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                            .setFound(false)
                            .build());
            masterObserver.get().onCompleted();
            FlexlbGrpcForwarder.CancelForwardResult result = pending.get()
                    .toCompletableFuture()
                    .get(5, TimeUnit.SECONDS);

            assertNotNull(result.response());
            assertFalse(result.response().getFound());
            verify(reporter, times(1)).reportForwardToMasterResult(
                    "10.0.0.2", "CANCEL_NOT_FOUND");
            forwarder.shutdown();
        }
    }

    private static FlexlbGrpcForwarder forwarder(
            ManagedChannel channel,
            EngineHealthReporter reporter) throws Exception {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        when(consistency.getMasterHostIpPort()).thenReturn(MASTER_HTTP_ADDRESS);
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        FlexlbGrpcForwarder forwarder = new FlexlbGrpcForwarder(
                consistency,
                mock(ConfigService.class),
                reporter,
                mock(EventLoopGroup.class),
                mock(Executor.class));
        channels(forwarder).put(MASTER_CHANNEL_KEY, channel);
        return forwarder;
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request(long requestId) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(requestId)
                .build();
    }

    private static FlexlbGrpcForwarder.MasterForwardResult await(
            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> result) throws Exception {
        return result.toCompletableFuture().get(5, TimeUnit.SECONDS);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, ManagedChannel> channels(
            FlexlbGrpcForwarder forwarder) throws Exception {
        Field field = FlexlbGrpcForwarder.class.getDeclaredField("channels");
        field.setAccessible(true);
        return (Map<String, ManagedChannel>) field.get(forwarder);
    }

    @FunctionalInterface
    private interface ScheduleHandler {
        void schedule(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer);
    }

    @FunctionalInterface
    private interface CancelHandler {
        void cancel(
                FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
                StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer);
    }

    private static final class RpcFixture implements AutoCloseable {
        private final Server server;
        private final ManagedChannel channel;

        private RpcFixture(Server server, ManagedChannel channel) {
            this.server = server;
            this.channel = channel;
        }

        static RpcFixture start(ScheduleHandler handler) throws Exception {
            return startService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                @Override
                public void schedule(
                        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
                    handler.schedule(request, observer);
                }
            });
        }

        static RpcFixture startCancel(CancelHandler handler) throws Exception {
            return startService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                @Override
                public void cancel(
                        FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
                        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer) {
                    handler.cancel(request, observer);
                }
            });
        }

        private static RpcFixture startService(
                FlexlbServiceGrpc.FlexlbServiceImplBase service) throws Exception {
            Server server = NettyServerBuilder.forPort(0)
                    .directExecutor()
                    .addService(service)
                    .build()
                    .start();
            ManagedChannel channel = NettyChannelBuilder
                    .forAddress("127.0.0.1", server.getPort())
                    .directExecutor()
                    .usePlaintext()
                    .build();
            return new RpcFixture(server, channel);
        }

        @Override
        public void close() throws Exception {
            channel.shutdownNow();
            channel.awaitTermination(5, TimeUnit.SECONDS);
            server.shutdownNow();
            server.awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
