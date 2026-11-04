package org.flexlb.httpserver;

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerServiceDefinition;
import io.grpc.Status;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

@Timeout(30)
class GrpcExecutorOverloadTest {
    private static final MethodDescriptor<byte[], byte[]> METHOD =
            MethodDescriptor.<byte[], byte[]>newBuilder()
                    .setType(MethodDescriptor.MethodType.UNARY)
                    .setFullMethodName("overload.Test/Call")
                    .setRequestMarshaller(new BytesMarshaller())
                    .setResponseMarshaller(new BytesMarshaller())
                    .build();

    /** Deliver a real Netty message AFTER the RPC has started and the executor fills. */
    @Test
    void acceptedMessageAndCloseStillRunWhenExecutorIsFull() throws Exception {
        try (Fixture f = new Fixture()) {
            // Flush headers independently of the body to deterministically fill the
            // executor between startCall and messagesAvailable. The server is unary.
            var call =
                    f.channel.newCall(
                            METHOD.toBuilder()
                                    .setType(MethodDescriptor.MethodType.BIDI_STREAMING)
                                    .build(),
                            CallOptions.DEFAULT.withDeadlineAfter(10, TimeUnit.SECONDS));
            var result = new CompletableFuture<Status>();
            call.start(listener(result), new Metadata());
            call.request(1);
            assertTrue(f.started.await(5, TimeUnit.SECONDS));
            f.saturate();
            call.sendMessage(new byte[1024 * 1024]);
            call.halfClose();
            assertEquals(Status.Code.OK, result.get(5, TimeUnit.SECONDS).getCode());
            assertEquals(1024 * 1024, f.receivedBytes.get());
            assertTrue(f.policy.getCallerRunsCount().get() > 0);
            // Close transport as well, so only leaked retained buffers can remain.
            f.channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            f.release.countDown();
            f.server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            awaitNoPinned(f.allocator);
        }
    }

    @Test
    void newCallIsRejectedWithoutEnteringServiceAndRecoversAfterSaturation() throws Exception {
        try (Fixture f = new Fixture()) {
            f.saturate();
            var result = new CompletableFuture<Status>();
            var call = f.newCall();
            call.start(listener(result), new Metadata());
            call.request(1);
            call.sendMessage(new byte[1024 * 1024]);
            call.halfClose();
            assertEquals(Status.Code.RESOURCE_EXHAUSTED, result.get(5, TimeUnit.SECONDS).getCode());
            assertEquals(0, f.calls.get());
            f.release.countDown();
            f.executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
            var recovered = new CompletableFuture<Status>();
            var next = f.newCall();
            next.start(listener(recovered), new Metadata());
            next.request(1);
            next.sendMessage(new byte[32]);
            next.halfClose();
            assertEquals(Status.Code.OK, recovered.get(5, TimeUnit.SECONDS).getCode());
            assertEquals(1, f.calls.get());
            f.channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            f.server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            awaitNoPinned(f.allocator);
        }
    }

    @Test
    void shutdownRaceDoesNotSilentlyDiscardCleanup() {
        var policy = new FlexlbGrpcServer.CountingCallerRunsHandler();
        var executor =
                new ThreadPoolExecutor(
                        1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>(1), policy);
        executor.shutdown();
        AtomicInteger cleanup = new AtomicInteger();
        executor.execute(cleanup::incrementAndGet);
        assertEquals(1, cleanup.get());
        assertEquals(1, policy.getCallerRunsCount().get());
    }

    private static ClientCall.Listener<byte[]> listener(CompletableFuture<Status> result) {
        return new ClientCall.Listener<>() {
            @Override
            public void onClose(Status status, Metadata trailers) {
                result.complete(status);
            }
        };
    }

    private static void awaitNoPinned(PooledByteBufAllocator allocator)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            // Observe one actual zero. A later HTTP/2 PING may legitimately allocate another
            // small buffer between this read and a repeated assert; that is not retained payload.
            if (allocator.pinnedDirectMemory() == 0) return;
            Thread.sleep(10);
        }
        fail("retained direct buffers after RPC cleanup: " + allocator.pinnedDirectMemory());
    }

    private static class BytesMarshaller implements MethodDescriptor.Marshaller<byte[]> {
        @Override
        public InputStream stream(byte[] value) {
            return new ByteArrayInputStream(value);
        }

        @Override
        public byte[] parse(InputStream stream) {
            try {
                return stream.readAllBytes();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private static class Fixture implements AutoCloseable {
        final FlexlbGrpcServer.CountingCallerRunsHandler policy =
                new FlexlbGrpcServer.CountingCallerRunsHandler();
        final ThreadPoolExecutor executor =
                new ThreadPoolExecutor(
                        1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>(1), policy);
        final PooledByteBufAllocator allocator = new PooledByteBufAllocator(true);
        final CountDownLatch started = new CountDownLatch(1);
        final CountDownLatch release = new CountDownLatch(1);
        final AtomicInteger calls = new AtomicInteger();
        final AtomicInteger receivedBytes = new AtomicInteger();
        final Server server;
        final ManagedChannel channel;

        Fixture() throws Exception {
            var service =
                    ServerServiceDefinition.builder("overload.Test")
                            .addMethod(
                                    METHOD,
                                    (ServerCall<byte[], byte[]> call, Metadata headers) -> {
                                        calls.incrementAndGet();
                                        call.request(1);
                                        started.countDown();
                                        return new ServerCall.Listener<byte[]>() {
                                            @Override
                                            public void onMessage(byte[] message) {
                                                receivedBytes.set(message.length);
                                            }

                                            @Override
                                            public void onHalfClose() {
                                                call.sendHeaders(new Metadata());
                                                call.sendMessage(new byte[1]);
                                                call.close(Status.OK, new Metadata());
                                            }
                                        };
                                    })
                            .build();
            server =
                    NettyServerBuilder.forPort(0)
                            .executor(executor)
                            .withChildOption(ChannelOption.ALLOCATOR, allocator)
                            .intercept(new GrpcOverloadInterceptor(executor))
                            .addService(service)
                            .build()
                            .start();
            channel =
                    NettyChannelBuilder.forAddress("127.0.0.1", server.getPort())
                            .usePlaintext()
                            .build();
            GrpcReceivingGuardRepro.awaitReady(channel);
        }

        ClientCall<byte[], byte[]> newCall() {
            return channel.newCall(
                    METHOD, CallOptions.DEFAULT.withDeadlineAfter(10, TimeUnit.SECONDS));
        }

        void saturate() throws Exception {
            CountDownLatch running = new CountDownLatch(1);
            executor.execute(
                    () -> {
                        running.countDown();
                        try {
                            release.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    });
            assertTrue(running.await(5, TimeUnit.SECONDS));
            executor.execute(() -> {});
            assertEquals(0, executor.getQueue().remainingCapacity());
        }

        @Override
        public void close() throws Exception {
            release.countDown();
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            executor.shutdown();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }
}
