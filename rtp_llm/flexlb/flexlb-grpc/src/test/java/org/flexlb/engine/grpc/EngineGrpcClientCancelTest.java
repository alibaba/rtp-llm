package org.flexlb.engine.grpc;

import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import org.flexlb.engine.grpc.EngineGrpcClient.CancelReply;
import org.flexlb.engine.grpc.EngineRpcService.CancelRequestPB;
import org.flexlb.engine.grpc.EngineRpcService.CancelResponsePB;
import org.flexlb.engine.grpc.EngineRpcService.CancelStatusPB;
import org.flexlb.engine.grpc.EngineRpcService.RequestCancelReasonPB;
import org.flexlb.engine.grpc.EngineRpcService.StatusVersionPB;
import org.flexlb.engine.grpc.EngineRpcService.WorkerStatusPB;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;

/**
 * Checks cancellation sends the original request and preserves the meaning of remote replies.
 * Failure and timeout must not trigger status probes or replay the cancellation.
 */
class EngineGrpcClientCancelTest {
    @ParameterizedTest
    @EnumSource(
            value = RequestCancelReasonPB.class,
            names = {"REQUEST_CANCEL_REASON_UNSPECIFIED", "UNRECOGNIZED"},
            mode = EnumSource.Mode.EXCLUDE)
    void preservesRequestAndSendsCancelWithoutAStatusQuery(RequestCancelReasonPB reason)
            throws Exception {
        try (var engine = new Engine()) {
            var request = request(reason);
            assertEquals(
                    CancelReply.ACCEPTED,
                    engine.client.cancelAsync("127.0.0.1", 9090, request, 1000).get());
            assertEquals(request, engine.received);
            assertEquals(0, engine.statusCalls.get());
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @ParameterizedTest
    @CsvSource({
        "CANCEL_STATUS_ACCEPTED,false,ACCEPTED",
        "CANCEL_STATUS_NOT_FOUND,false,NOT_FOUND",
        "CANCEL_STATUS_TOMBSTONED,false,REQUEST_FENCED",
        "CANCEL_STATUS_TOMBSTONED,true,REQUEST_CLEANED",
        "CANCEL_STATUS_UNSPECIFIED,false,FAILED"
    })
    void preservesStrengthOfRemoteEvidence(
            CancelStatusPB status, boolean downstreamCleaned, CancelReply expected)
            throws Exception {
        try (var engine = new Engine()) {
            engine.response =
                    CancelResponsePB.newBuilder()
                            .setStatus(status)
                            .setDecodeCleanupComplete(downstreamCleaned)
                            .build();
            assertEquals(expected, engine.cancel().get(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void unknownWireStatusIsNotCleanupProof() throws Exception {
        try (var engine = new Engine()) {
            engine.response =
                    CancelResponsePB.newBuilder()
                            .setStatusValue(999)
                            .setDecodeCleanupComplete(true)
                            .build();
            assertEquals(CancelReply.FAILED, engine.cancel().get(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void callerDisconnectDoesNotCancelPendingCleanup() throws Exception {
        try (var engine = new Engine();
                var caller = Context.current().withCancellation()) {
            engine.delayCancel = true;
            Context previous = caller.attach();
            Context activeCaller = Context.current();
            CompletableFuture<CancelReply> result;
            try {
                result = engine.cancel();
                assertSame(activeCaller, Context.current());
                assertFalse(result.isDone());
                caller.cancel(new IllegalStateException("frontend disconnected"));
                assertFalse(engine.cancelContext.isCancelled());
                engine.replyCancel();
                assertSame(activeCaller, Context.current());
            } finally {
                caller.detach(previous);
            }
            assertEquals(CancelReply.ACCEPTED, result.get(2, TimeUnit.SECONDS));
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = Status.Code.class,
            names = {"UNAVAILABLE", "UNIMPLEMENTED", "PERMISSION_DENIED"})
    void cancelFailureDoesNotProbeOrReplay(Status.Code code) throws Exception {
        try (var engine = new Engine()) {
            engine.cancelFailure = Status.fromCode(code).withDescription("Cancel failed");
            ExecutionException failure =
                    assertThrows(
                            ExecutionException.class,
                            () -> engine.cancel().get(2, TimeUnit.SECONDS));
            assertEquals(code, Status.fromThrowable(failure.getCause()).getCode());
            assertEquals(0, engine.statusCalls.get());
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @Test
    void cancelDeadlineCompletesExceptionally() throws Exception {
        try (var engine = new Engine()) {
            engine.delayCancel = true;
            var result =
                    engine.client.cancelAsync(
                            "127.0.0.1",
                            9090,
                            request(RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED),
                            50);
            ExecutionException failure =
                    assertThrows(ExecutionException.class, () -> result.get(2, TimeUnit.SECONDS));
            assertEquals(
                    Status.Code.DEADLINE_EXCEEDED,
                    Status.fromThrowable(failure.getCause()).getCode());
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @Test
    void unroutableCancelIsAnExceptionalFuture() throws Exception {
        try (var engine = new Engine()) {
            var result =
                    assertDoesNotThrow(
                            () ->
                                    engine.client.cancelAsync(
                                            null,
                                            0,
                                            request(
                                                    RequestCancelReasonPB
                                                            .REQUEST_CANCEL_REASON_SHUTDOWN),
                                            500));
            assertThrows(ExecutionException.class, () -> result.get(2, TimeUnit.SECONDS));
            assertEquals(0, engine.statusCalls.get());
            assertEquals(0, engine.cancelCalls.get());
        }
    }

    private static CancelRequestPB request(RequestCancelReasonPB reason) {
        return CancelRequestPB.newBuilder().setRequestId(71L).setReason(reason).build();
    }

    /**
     * Runs an in-process engine with controllable cancellation replies and records whether the client issues extra RPCs.
     */
    private static final class Engine extends RpcServiceGrpc.RpcServiceImplBase
            implements AutoCloseable {
        final AtomicInteger statusCalls = new AtomicInteger();
        final AtomicInteger cancelCalls = new AtomicInteger();
        final Server server;
        final EngineGrpcClient client;

        boolean delayCancel;

        Context cancelContext;
        Status cancelFailure;
        CancelRequestPB received;
        StreamObserver<CancelResponsePB> pendingCancel;
        CancelResponsePB response =
                CancelResponsePB.newBuilder()
                        .setStatus(CancelStatusPB.CANCEL_STATUS_ACCEPTED)
                        .build();

        Engine() throws Exception {
            String name = InProcessServerBuilder.generateName();
            server =
                    InProcessServerBuilder.forName(name)
                            .directExecutor()
                            .addService(this)
                            .build()
                            .start();
            client =
                    new EngineGrpcClient(
                            mock(EngineAddressResolver.class),
                            mock(ThreadPoolExecutor.class),
                            mock(EventLoopGroup.class),
                            mock(GrpcReporter.class),
                            20,
                            5000,
                            org.mockito.Mockito.mock(
                                    org.flexlb.config.CacheMatchConfiguration.class)) {
                        @Override
                        protected ManagedChannel createChannel(String key) {
                            return InProcessChannelBuilder.forName(name).directExecutor().build();
                        }
                    };
        }

        @Override
        public void getWorkerStatus(
                StatusVersionPB request, StreamObserver<WorkerStatusPB> observer) {
            statusCalls.incrementAndGet();
            observer.onError(Status.UNIMPLEMENTED.asRuntimeException());
        }

        @Override
        public void cancel(CancelRequestPB request, StreamObserver<CancelResponsePB> observer) {
            cancelCalls.incrementAndGet();
            received = request;
            cancelContext = Context.current();
            pendingCancel = observer;
            if (cancelFailure != null) {
                observer.onError(cancelFailure.asRuntimeException());
                return;
            }
            if (!delayCancel) {
                replyCancel();
            }
        }

        @Override
        public void close() throws Exception {
            try {
                client.shutdownChannelPool();
            } finally {
                server.shutdownNow();
                server.awaitTermination(2, TimeUnit.SECONDS);
            }
        }

        CompletableFuture<CancelReply> cancel() {
            return client.cancelAsync(
                    "127.0.0.1",
                    9090,
                    request(RequestCancelReasonPB.REQUEST_CANCEL_REASON_CLIENT_CANCELLED),
                    1000);
        }

        void replyCancel() {
            pendingCancel.onNext(response);
            pendingCancel.onCompleted();
        }
    }
}
