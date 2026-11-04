package org.flexlb.httpserver;

import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbCancelRequestPB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbCancelResponsePB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleResponsePB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.GetRequestStateRequestPB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.GetRequestStateResponsePB;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Timeout(value = 20, unit = TimeUnit.SECONDS)
class FlexlbGrpcForwarderRecoveryTest {
    @Test
    void delayedStreamRejectionPanicsChannelAndNextIndependentRequestRecovers() throws Exception {
        for (String operation : List.of("schedule", "cancel", "state")) {
            try (Fixture fixture = new Fixture(true)) {
                var initial =
                        fixture.forwarder.forwardScheduleToMaster(request(1)).toCompletableFuture();
                assertTrue(
                        fixture.rejected.await(5, TimeUnit.SECONDS),
                        "exercise the real DelayedStream executor handoff");
                ManagedChannel poisoned = channels(fixture.forwarder).get(fixture.channelKey);
                assertNotNull(poisoned);
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                while (poisoned.getState(false) != ConnectivityState.TRANSIENT_FAILURE) {
                    var observed = poisoned.getState(false);
                    if (observed == ConnectivityState.TRANSIENT_FAILURE) {
                        break;
                    }
                    CountDownLatch changed = new CountDownLatch(1);
                    poisoned.notifyWhenStateChanged(observed, changed::countDown);
                    long remaining = deadline - System.nanoTime();
                    assertTrue(
                            remaining > 0 && changed.await(remaining, TimeUnit.NANOSECONDS),
                            "wait for gRPC to install its Panic picker before the next RPC");
                }

                // The rejected drain may strand its own RPC. A subsequent RPC observes the
                // channel's real gRPC Panic status; recovery must not retry either request.
                switch (operation) {
                    case "schedule" -> {
                        var failure =
                                fixture.forwarder
                                        .forwardScheduleToMaster(request(2))
                                        .toCompletableFuture()
                                        .get(5, TimeUnit.SECONDS);
                        assertEquals("INTERNAL", failure.failure());
                        Status status = Status.fromThrowable(failure.error());
                        assertEquals("Panic! This is a bug!", status.getDescription());
                        assertNotNull(status.getCause());
                    }
                    case "cancel" -> {
                        var failure =
                                fixture.forwarder
                                        .forwardCancelToMaster(
                                                FlexlbCancelRequestPB.newBuilder()
                                                        .setRequestId(2)
                                                        .build())
                                        .toCompletableFuture()
                                        .get(5, TimeUnit.SECONDS);
                        assertEquals("INTERNAL", failure.failure());
                    }
                    case "state" ->
                            assertNull(
                                    fixture.forwarder.forwardGetRequestStateToMaster(
                                            GetRequestStateRequestPB.newBuilder()
                                                    .setRequestId(2)
                                                    .build()));
                    default -> throw new AssertionError(operation);
                }
                assertFalse(channels(fixture.forwarder).containsValue(poisoned));
                initial.cancel(true);

                var recovered =
                        fixture.forwarder
                                .forwardScheduleToMaster(request(3))
                                .toCompletableFuture()
                                .get(5, TimeUnit.SECONDS);
                assertNotNull(recovered.response());
                assertTrue(recovered.response().getSuccess());
                assertNotSame(poisoned, channels(fixture.forwarder).get(fixture.channelKey));
                assertEquals(
                        List.of(3L), fixture.received, "failed requests must never be replayed");
            }
        }
    }

    @Test
    void ordinaryAndRemotePanicTextInternalResponsesKeepHealthyChannel() throws Exception {
        try (Fixture fixture = new Fixture(false)) {
            ManagedChannel original = null;
            for (long id : List.of(10L, 12L)) {
                var failure =
                        fixture.forwarder
                                .forwardScheduleToMaster(request(id))
                                .toCompletableFuture()
                                .get(5, TimeUnit.SECONDS);
                assertEquals("INTERNAL", failure.failure());
                Status status = Status.fromThrowable(failure.error());
                assertEquals(
                        id == 10 ? "ordinary application failure" : "Panic! This is a bug!",
                        status.getDescription());
                assertNull(status.getCause(), "a remote status carries no local Panic cause");
                if (original == null) {
                    original = channels(fixture.forwarder).get(fixture.channelKey);
                    assertNotNull(original);
                }
                assertSame(original, channels(fixture.forwarder).get(fixture.channelKey));
                var success =
                        fixture.forwarder
                                .forwardScheduleToMaster(request(id + 1))
                                .toCompletableFuture()
                                .get(5, TimeUnit.SECONDS);
                assertTrue(success.response().getSuccess());
                assertSame(original, channels(fixture.forwarder).get(fixture.channelKey));
            }
            assertEquals(List.of(10L, 11L, 12L, 13L), fixture.received);
        }
    }

    @Test
    void lateFailureFromRetiredChannelCannotRemoveItsReplacement() throws Exception {
        try (Fixture fixture = new Fixture(false)) {
            assertTrue(
                    fixture.forwarder
                            .forwardScheduleToMaster(request(20))
                            .toCompletableFuture()
                            .get(5, TimeUnit.SECONDS)
                            .response()
                            .getSuccess());
            ManagedChannel old = channels(fixture.forwarder).get(fixture.channelKey);
            var retire =
                    FlexlbGrpcForwarder.class.getDeclaredMethod(
                            "retirePanickedChannel",
                            String.class,
                            ManagedChannel.class,
                            Status.class);
            retire.setAccessible(true);
            Status panic =
                    Status.INTERNAL
                            .withDescription("Panic! This is a bug!")
                            .withCause(
                                    new RejectedExecutionException(
                                            "late callback from old channel"));
            assertEquals(true, retire.invoke(fixture.forwarder, fixture.masterHost, old, panic));
            assertTrue(
                    fixture.forwarder
                            .forwardScheduleToMaster(request(21))
                            .toCompletableFuture()
                            .get(5, TimeUnit.SECONDS)
                            .response()
                            .getSuccess());
            ManagedChannel replacement = channels(fixture.forwarder).get(fixture.channelKey);
            assertNotSame(old, replacement);

            // The first test exercises actual gRPC Panic. Here only the delayed
            // old-channel notification ordering is controlled at the cleanup boundary.
            assertEquals(false, retire.invoke(fixture.forwarder, fixture.masterHost, old, panic));
            assertSame(replacement, channels(fixture.forwarder).get(fixture.channelKey));
            assertFalse(replacement.isShutdown());
            assertTrue(
                    fixture.forwarder
                            .forwardScheduleToMaster(request(22))
                            .toCompletableFuture()
                            .get(5, TimeUnit.SECONDS)
                            .response()
                            .getSuccess());
            assertEquals(List.of(20L, 21L, 22L), fixture.received);
        }
    }

    private static FlexlbScheduleRequestPB request(long id) {
        return FlexlbScheduleRequestPB.newBuilder().setRequestId(id).build();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, ManagedChannel> channels(FlexlbGrpcForwarder forwarder)
            throws Exception {
        Field field = FlexlbGrpcForwarder.class.getDeclaredField("channels");
        field.setAccessible(true);
        return (Map<String, ManagedChannel>) field.get(forwarder);
    }

    private static final class Fixture implements AutoCloseable {
        private final ExecutorService callbacks = Executors.newSingleThreadExecutor();
        private final NioEventLoopGroup eventLoops = new NioEventLoopGroup(1);
        private final CountDownLatch rejected = new CountDownLatch(1);
        private final List<Long> received = new CopyOnWriteArrayList<>();
        private final Server server;
        private final String channelKey;
        private final String masterHost;
        private final FlexlbGrpcForwarder forwarder;

        private Fixture(boolean rejectDrain) throws Exception {
            server =
                    NettyServerBuilder.forPort(0)
                            .addService(
                                    new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                                        @Override
                                        public void schedule(
                                                FlexlbScheduleRequestPB request,
                                                StreamObserver<FlexlbScheduleResponsePB> observer) {
                                            received.add(request.getRequestId());
                                            if (request.getRequestId() == 10
                                                    || request.getRequestId() == 12) {
                                                observer.onError(
                                                        Status.INTERNAL
                                                                .withDescription(
                                                                        request.getRequestId() == 10
                                                                                ? "ordinary application failure"
                                                                                : "Panic! This is a bug!")
                                                                .asRuntimeException());
                                            } else {
                                                observer.onNext(
                                                        FlexlbScheduleResponsePB.newBuilder()
                                                                .setSuccess(true)
                                                                .setCode(200)
                                                                .build());
                                                observer.onCompleted();
                                            }
                                        }

                                        @Override
                                        public void cancel(
                                                FlexlbCancelRequestPB request,
                                                StreamObserver<FlexlbCancelResponsePB> observer) {
                                            received.add(request.getRequestId());
                                            observer.onNext(
                                                    FlexlbCancelResponsePB.getDefaultInstance());
                                            observer.onCompleted();
                                        }

                                        @Override
                                        public void getRequestState(
                                                GetRequestStateRequestPB request,
                                                StreamObserver<GetRequestStateResponsePB>
                                                        observer) {
                                            received.add(request.getRequestId());
                                            observer.onNext(
                                                    GetRequestStateResponsePB.getDefaultInstance());
                                            observer.onCompleted();
                                        }
                                    })
                            .build()
                            .start();
            channelKey = "127.0.0.1:" + server.getPort();
            masterHost =
                    "127.0.0.1:" + (server.getPort() - FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET);
            var consistency = mock(LBStatusConsistencyService.class);
            when(consistency.getMasterHostIpPort()).thenReturn(masterHost);
            when(consistency.getLocalHostIp()).thenReturn("127.0.0.2");
            var config = mock(ConfigService.class);
            when(config.loadBalanceConfig()).thenReturn(new FlexlbConfig());
            AtomicBoolean rejectOnce = new AtomicBoolean(rejectDrain);
            Executor executor =
                    task -> {
                        if (task.getClass().getName().startsWith("io.grpc.internal.DelayedStream$")
                                && rejectOnce.compareAndSet(true, false)) {
                            rejected.countDown();
                            throw new RejectedExecutionException(
                                    "controlled DelayedStream rejection");
                        }
                        callbacks.execute(task);
                    };
            forwarder =
                    new FlexlbGrpcForwarder(
                            consistency,
                            config,
                            mock(EngineHealthReporter.class),
                            eventLoops,
                            executor);
        }

        @Override
        public void close() throws Exception {
            forwarder.shutdown();
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            eventLoops.shutdownGracefully(0, 5, TimeUnit.SECONDS).sync();
            callbacks.shutdownNow();
            callbacks.awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
