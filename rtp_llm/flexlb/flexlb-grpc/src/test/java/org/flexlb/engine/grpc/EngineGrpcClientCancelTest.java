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
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

class EngineGrpcClientCancelTest {
    @ParameterizedTest
    @EnumSource(value = RequestCancelReasonPB.class, names = {"REQUEST_CANCEL_REASON_UNSPECIFIED", "UNRECOGNIZED"},
            mode = EnumSource.Mode.EXCLUDE)
    void preservesRequestAndChecksCapabilityOnlyForOrdinaryCleanup(RequestCancelReasonPB reason) throws Exception {
        try (var engine = new Engine()) {
            var request = request(reason);
            assertEquals(CancelReply.ACCEPTED, engine.client.cancelAsync("127.0.0.1", 9090, request, 1000).get());
            assertEquals(request, engine.received);
            assertEquals(reason == RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED ? 0 : 1,
                    engine.statusCalls.get());
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @ParameterizedTest
    @CsvSource({"CANCEL_STATUS_ACCEPTED,false,ACCEPTED", "CANCEL_STATUS_NOT_FOUND,false,NOT_FOUND",
            "CANCEL_STATUS_TOMBSTONED,false,REQUEST_FENCED", "CANCEL_STATUS_TOMBSTONED,true,REQUEST_CLEANED",
            "CANCEL_STATUS_UNSPECIFIED,false,FAILED"})
    void preservesStrengthOfRemoteEvidence(CancelStatusPB status, boolean downstreamCleaned, CancelReply expected) throws Exception {
        try (var engine = new Engine()) {
            engine.response = CancelResponsePB.newBuilder().setStatus(status).setDecodeCleanupComplete(downstreamCleaned).build();
            assertEquals(expected, engine.cancel().get(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void unknownWireStatusIsNotCleanupProof() throws Exception {
        try (var engine = new Engine()) {
            engine.response = CancelResponsePB.newBuilder().setStatusValue(999).setDecodeCleanupComplete(true).build();
            assertEquals(CancelReply.FAILED, engine.cancel().get(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void unsupportedCleanupDoesNotSendCancel() throws Exception {
        try (var engine = new Engine()) {
            engine.supportsCleanup = false;
            assertEquals(CancelReply.UNSUPPORTED, engine.cancel().get(2, TimeUnit.SECONDS));
            assertEquals(0, engine.cancelCalls.get());
            var priority = request(RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED);
            assertEquals(CancelReply.ACCEPTED, engine.client.cancelAsync("127.0.0.1", 9090, priority, 1000).get());
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @Test
    void callerDisconnectDoesNotCancelPendingCapabilityCheckOrCleanup() throws Exception {
        try (var engine = new Engine(); var caller = Context.current().withCancellation()) {
            engine.delayStatus = true;
            Context previous = caller.attach();
            Context activeCaller = Context.current();
            CompletableFuture<CancelReply> result;
            try {
                result = engine.cancel();
                assertSame(activeCaller, Context.current());
                assertFalse(result.isDone());
                caller.cancel(new IllegalStateException("frontend disconnected"));
                engine.replyStatus();
                assertSame(activeCaller, Context.current());
            } finally {
                caller.detach(previous);
            }
            assertEquals(CancelReply.ACCEPTED, result.get(2, TimeUnit.SECONDS));
            assertFalse(engine.cancelContextCancelled);
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @Test
    void failedCapabilityCheckCompletesExceptionallyWithoutSendingCancel() throws Exception {
        try (var engine = new Engine()) {
            engine.statusFailure = Status.PERMISSION_DENIED;
            ExecutionException failure = assertThrows(ExecutionException.class, () -> engine.cancel().get(2, TimeUnit.SECONDS));
            assertEquals(Status.Code.PERMISSION_DENIED, Status.fromThrowable(failure.getCause()).getCode());
            assertEquals(0, engine.cancelCalls.get());
        }
    }

    @Test
    void ambiguousCancelFailureIsNotReplayed() throws Exception {
        try (var engine = new Engine()) {
            engine.cancelFailure = Status.UNAVAILABLE.withDescription("Connection reset");
            ExecutionException failure = assertThrows(ExecutionException.class, () -> engine.cancel().get(2, TimeUnit.SECONDS));
            assertEquals(Status.Code.UNAVAILABLE, Status.fromThrowable(failure.getCause()).getCode());
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @Test
    void cancelDeadlineCompletesExceptionally() throws Exception {
        try (var engine = new Engine()) {
            engine.delayCancel = true;
            var result = engine.client.cancelAsync("127.0.0.1", 9090,
                    request(RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED), 50);
            ExecutionException failure = assertThrows(ExecutionException.class, () -> result.get(2, TimeUnit.SECONDS));
            assertEquals(Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(failure.getCause()).getCode());
            assertEquals(1, engine.cancelCalls.get());
        }
    }

    @Test
    void unroutableCancelIsAnExceptionalFuture() throws Exception {
        try (var engine = new Engine()) {
            var result = assertDoesNotThrow(() -> engine.client.cancelAsync(null, 0,
                    request(RequestCancelReasonPB.REQUEST_CANCEL_REASON_SHUTDOWN), 500));
            assertThrows(ExecutionException.class, () -> result.get(2, TimeUnit.SECONDS));
            assertEquals(0, engine.statusCalls.get());
            assertEquals(0, engine.cancelCalls.get());
        }
    }

    private static CancelRequestPB request(RequestCancelReasonPB reason) {
        return CancelRequestPB.newBuilder().setRequestId(71L).setReason(reason).build();
    }

    private static final class Engine extends RpcServiceGrpc.RpcServiceImplBase implements AutoCloseable {
        final AtomicInteger statusCalls = new AtomicInteger();
        final AtomicInteger cancelCalls = new AtomicInteger();
        final Server server;
        final EngineGrpcClient client;
        boolean supportsCleanup = true;
        boolean delayStatus;
        boolean delayCancel;
        boolean cancelContextCancelled;
        Status statusFailure;
        Status cancelFailure;
        CancelRequestPB received;
        StreamObserver<WorkerStatusPB> pendingStatus;
        CancelResponsePB response = CancelResponsePB.newBuilder().setStatus(CancelStatusPB.CANCEL_STATUS_ACCEPTED).build();

        Engine() throws Exception {
            String name = InProcessServerBuilder.generateName();
            server = InProcessServerBuilder.forName(name).directExecutor().addService(this).build().start();
            client = new EngineGrpcClient(mock(CustomNameResolver.class), mock(ThreadPoolExecutor.class),
                    mock(EventLoopGroup.class), mock(GrpcReporter.class), 20, 5000) {
                @Override protected ManagedChannel createChannel(String key) {
                    return InProcessChannelBuilder.forName(name).directExecutor().build();
                }
            };
        }

        CompletableFuture<CancelReply> cancel() {
            return client.cancelAsync("127.0.0.1", 9090,
                    request(RequestCancelReasonPB.REQUEST_CANCEL_REASON_CLIENT_CANCELLED), 1000);
        }

        @Override public void getWorkerStatus(StatusVersionPB request, StreamObserver<WorkerStatusPB> observer) {
            statusCalls.incrementAndGet();
            if (statusFailure != null) { observer.onError(statusFailure.asRuntimeException()); return; }
            pendingStatus = observer;
            if (!delayStatus) { replyStatus(); }
        }

        void replyStatus() {
            pendingStatus.onNext(WorkerStatusPB.newBuilder().setSupportsRequestCleanup(supportsCleanup).build());
            pendingStatus.onCompleted();
        }

        @Override public void cancel(CancelRequestPB request, StreamObserver<CancelResponsePB> observer) {
            cancelCalls.incrementAndGet();
            received = request;
            cancelContextCancelled = Context.current().isCancelled();
            if (cancelFailure != null) { observer.onError(cancelFailure.asRuntimeException()); return; }
            if (!delayCancel) { observer.onNext(response); observer.onCompleted(); }
        }

        @Override public void close() throws Exception {
            try { client.shutdownChannelPool(); }
            finally { server.shutdownNow(); server.awaitTermination(2, TimeUnit.SECONDS); }
        }
    }
}
