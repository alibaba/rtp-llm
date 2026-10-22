package org.flexlb.mockengine;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

@Timeout(15)
class MockRemoteDecodeProtocolTest {
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private final AtomicInteger starts = new AtomicInteger();
    private final AtomicInteger completions = new AtomicInteger();
    private final CountDownLatch cancelled = new CountDownLatch(1);
    private final CountDownLatch started = new CountDownLatch(1);
    private volatile LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> outputs;
    private Server server;
    private ManagedChannel channel;

    private MockRemoteDecodeStream connect(CompletableFuture<EngineRpcService.GenerateOutputsPB> output,
                                          CompletableFuture<Throwable> failure) throws Exception {
        server = ServerBuilder.forPort(0).addService(new RpcServiceGrpc.RpcServiceImplBase() {
            @Override
            public StreamObserver<EngineRpcService.GenerateRequestPB> remoteGenerate(
                    StreamObserver<EngineRpcService.GenerateOutputsPB> response) {
                return new MockRemoteDecodeCall((input, clientId, stopped) -> new MockRemoteDecodeCall.Lease() {
                    @Override
                    public boolean start(LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue) {
                        outputs = queue;
                        starts.incrementAndGet();
                        started.countDown();
                        return true;
                    }

                    @Override
                    public void cancel() {
                        cancelled.countDown();
                    }

                    @Override
                    public void completePrefill() { completions.incrementAndGet(); }

                    @Override
                    public void completed() {
                        completions.incrementAndGet();
                    }
                }, response, executor);
            }
        }).build().start();
        channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        return new MockRemoteDecodeStream(channel,
                EngineRpcService.GenerateInputPB.newBuilder().setRequestId(42).build(),
                "prefill-process-generation", 10_000, output::complete, failure::complete, () -> {});
    }

    @AfterEach
    void close() throws Exception {
        if (channel != null) channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
        if (server != null) server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
        executor.shutdownNow();
        executor.awaitTermination(3, TimeUnit.SECONDS);
    }

    @Test
    void allocationDoesNotStartDecodeAndClosingReleasesReservation() throws Exception {
        try (MockRemoteDecodeStream stream = connect(new CompletableFuture<>(), new CompletableFuture<>())) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            assertEquals(0, starts.get());
            stream.close();
            assertTrue(cancelled.await(3, TimeUnit.SECONDS));
            assertEquals(0, completions.get());
        }
    }

    @Test
    void realSocketRoundTripPreservesTerminalAndCompletesOnlyOnce() throws Exception {
        CompletableFuture<EngineRpcService.GenerateOutputsPB> output = new CompletableFuture<>();
        CompletableFuture<Throwable> failure = new CompletableFuture<>();
        try (MockRemoteDecodeStream stream = connect(output, failure)) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            stream.load().get(3, TimeUnit.SECONDS);
            assertEquals(0, starts.get());
            stream.generate(7);
            assertTrue(started.await(3, TimeUnit.SECONDS));
            EngineRpcService.GenerateOutputsPB terminal = EngineRpcService.GenerateOutputsPB.newBuilder()
                    .setRequestId(42).setFlattenOutput(EngineRpcService.FlattenOutputPB.newBuilder()
                            .addFinished(true)).build();
            outputs.offer(terminal);
            assertEquals(terminal, output.get(3, TimeUnit.SECONDS));
            assertEquals(1, starts.get());
            assertEquals(1, completions.get());
            assertFalse(failure.isDone());
            assertEquals(1, cancelled.getCount());
        }
    }

    @Test
    void decodeProcessDisappearanceFailsThePendingStream() throws Exception {
        CompletableFuture<Throwable> failure = new CompletableFuture<>();
        try (MockRemoteDecodeStream stream = connect(new CompletableFuture<>(), failure)) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            server.shutdownNow();
            assertNotNull(failure.get(3, TimeUnit.SECONDS));
            assertEquals(0, starts.get());
            assertEquals(0, completions.get());
        }
    }
    @Test
    void prefillCompletionSettlesReservationWithoutGeneratingOrCancelling() throws Exception {
        CompletableFuture<Throwable> failure = new CompletableFuture<>();
        try (var stream = connect(new CompletableFuture<>(), failure)) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            stream.load().get(3, TimeUnit.SECONDS);
            stream.completePrefill().get(3, TimeUnit.SECONDS);
            assertEquals(0, starts.get());
            assertEquals(1, completions.get());
            assertEquals(1, cancelled.getCount());
            assertFalse(failure.isDone());
        }
    }

    @Test
    void prefillCompletionIdentityMismatchCancelsLeaseWithExplicitError() {
        var received = new java.util.ArrayList<EngineRpcService.GenerateOutputsPB>();
        var error = new CompletableFuture<Throwable>();
        AtomicInteger leases = new AtomicInteger();
        var call = new MockRemoteDecodeCall((input, client, stopped) -> {
            leases.incrementAndGet();
            return new MockRemoteDecodeCall.Lease() {
                public boolean start(LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue) { return true; }
                public void cancel() { leases.decrementAndGet(); }
                public void completed() { leases.decrementAndGet(); }
            };
        }, new StreamObserver<>() {
            public void onNext(EngineRpcService.GenerateOutputsPB value) { received.add(value); }
            public void onError(Throwable value) { error.complete(value); }
            public void onCompleted() { fail("identity mismatch cannot succeed"); }
        }, Runnable::run);
        call.onNext(EngineRpcService.GenerateRequestPB.newBuilder().setStage(EngineRpcService.RemoteStage.ALLOCATE)
                .setRequestId(42).setClientId("owner").setInput(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(42)).build());
        assertTrue(received.getFirst().getSupportsPrefillCompletion());
        call.onNext(EngineRpcService.GenerateRequestPB.newBuilder().setRequestId(42).setStage(EngineRpcService.RemoteStage.LOAD).build());
        call.onNext(EngineRpcService.GenerateRequestPB.newBuilder().setRequestId(42).setClientId("wrong-owner")
                .setStage(EngineRpcService.RemoteStage.PREFILL_COMPLETE).build());
        assertEquals(io.grpc.Status.Code.INVALID_ARGUMENT, io.grpc.Status.fromThrowable(error.join()).getCode());
        assertEquals(0, leases.get());
    }

    @Test
    void unnegotiatedPrefillCompletionFailsAndClosesTheRemoteLease() throws Exception {
        server = ServerBuilder.forPort(0).addService(new RpcServiceGrpc.RpcServiceImplBase() {
            @Override public StreamObserver<EngineRpcService.GenerateRequestPB> remoteGenerate(
                    StreamObserver<EngineRpcService.GenerateOutputsPB> response) {
                return new StreamObserver<>() {
                    public void onNext(EngineRpcService.GenerateRequestPB request) {
                        response.onNext(EngineRpcService.GenerateOutputsPB.getDefaultInstance());
                    }
                    public void onError(Throwable error) { cancelled.countDown(); }
                    public void onCompleted() { response.onCompleted(); }
                };
            }
        }).build().start();
        channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        try (var stream = new MockRemoteDecodeStream(channel,
                EngineRpcService.GenerateInputPB.newBuilder().setRequestId(42).build(), "owner", 0,
                value -> fail("no generation expected"), error -> {}, () -> fail("cannot succeed"))) {
            stream.allocated().get(3, TimeUnit.SECONDS);
            stream.load().get(3, TimeUnit.SECONDS);
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> stream.completePrefill().get(3, TimeUnit.SECONDS));
            assertEquals(io.grpc.Status.Code.UNIMPLEMENTED, io.grpc.Status.fromThrowable(failure).getCode());
            assertTrue(cancelled.await(3, TimeUnit.SECONDS));
        }
    }

}
