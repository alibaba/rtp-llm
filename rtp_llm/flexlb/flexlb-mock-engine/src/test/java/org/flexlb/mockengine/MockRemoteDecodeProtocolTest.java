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
}
