package org.flexlb.engine.grpc.config;

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerServiceDefinition;
import io.grpc.Status;
import io.grpc.internal.GrpcUtil;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Timeout(30)
class GrpcCallbackOverloadTest {
    private static final MethodDescriptor<byte[], byte[]> METHOD =
            MethodDescriptor.<byte[], byte[]>newBuilder()
                    .setType(MethodDescriptor.MethodType.SERVER_STREAMING)
                    .setFullMethodName("overload.Client/Call")
                    .setRequestMarshaller(new BytesMarshaller())
                    .setResponseMarshaller(new BytesMarshaller())
                    .build();

    @Test
    void engineResponseAndCloseSurviveFullCallbackExecutor() throws Exception {
        try (Fixture f = new Fixture(false)) {
            f.responseAfterSaturation();
        }
    }

    @Test
    void forwarderResponseAndCloseSurviveFullCallbackExecutor() throws Exception {
        try (Fixture f = new Fixture(true)) {
            f.responseAfterSaturation();
        }
    }

    @Test
    void racingShutdownDoesNotDiscardCallback() {
        var executor =
                new ThreadPoolExecutor(
                        1,
                        1,
                        0,
                        TimeUnit.SECONDS,
                        new LinkedBlockingQueue<>(1),
                        new GrpcCallbackCallerRunsPolicy());
        executor.shutdown();
        AtomicInteger cleaned = new AtomicInteger();
        executor.execute(cleaned::incrementAndGet);
        assertEquals(1, cleaned.get());
    }

    @Test
    void cancelAfterParsedRequestReachesServerAndReleasesBuffers() throws Exception {
        try (Fixture f = new Fixture(false)) {
            f.call.cancel("frontend cancellation", null);
            assertEquals(Status.Code.CANCELLED, f.result.get(5, TimeUnit.SECONDS).getCode());
            assertTrue(
                    f.cancelled.await(5, TimeUnit.SECONDS),
                    "server must observe RST_STREAM cancellation");
            f.assertNoPinnedWhileConnected();
            f.assertNoPinnedAfterTransportClose();
        }
    }

    @Test
    void cancelWithResponseWaitingInCallbackQueueStillCleansBuffers() throws Exception {
        try (Fixture f = new Fixture(true)) {
            CountDownLatch blocked = new CountDownLatch(1);
            f.executor.execute(
                    () -> {
                        blocked.countDown();
                        try {
                            f.release.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    });
            assertTrue(blocked.await(5, TimeUnit.SECONDS));
            f.accepted.get().sendMessage(new byte[1024 * 1024]);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (f.executor.getQueue().isEmpty() && System.nanoTime() < deadline) {
                Thread.sleep(5);
            }
            assertFalse(
                    f.executor.getQueue().isEmpty(), "message callback must actually be waiting");
            f.call.cancel("frontend cancellation before response parse", null);
            assertTrue(f.cancelled.await(5, TimeUnit.SECONDS));
            f.release.countDown();
            assertEquals(Status.Code.CANCELLED, f.result.get(5, TimeUnit.SECONDS).getCode());
            f.assertNoPinnedWhileConnected();
            f.assertNoPinnedAfterTransportClose();
        }
    }

    private static final class Fixture implements AutoCloseable {
        final ThreadPoolExecutor executor;
        final PooledByteBufAllocator allocator = new PooledByteBufAllocator(true);
        final PooledByteBufAllocator serverAllocator = new PooledByteBufAllocator(true);
        final CountDownLatch headers = new CountDownLatch(1);
        final CountDownLatch warmup = new CountDownLatch(1);
        final CountDownLatch release = new CountDownLatch(1);
        final CountDownLatch cancelled = new CountDownLatch(1);
        final AtomicReference<ServerCall<byte[], byte[]>> accepted = new AtomicReference<>();
        final CompletableFuture<Status> result = new CompletableFuture<>();
        final AtomicInteger receivedBytes = new AtomicInteger();
        final Server server;
        final ManagedChannel channel;
        final ClientCall<byte[], byte[]> call;

        Fixture(boolean forwarder) throws Exception {
            ConfigService config = mock(ConfigService.class);
            FlexlbConfig settings = new FlexlbConfig();
            when(config.loadBalanceConfig()).thenReturn(settings);
            ChannelConfiguration factory = new ChannelConfiguration(config);
            executor =
                    forwarder
                            ? factory.forwarderChannelExecutor()
                            : factory.managedChannelThreadPoolExecutor();
            // Keep the production factory's handler while making deterministic saturation
            // inexpensive.
            executor.setCorePoolSize(1);
            executor.setMaximumPoolSize(1);
            var service =
                    ServerServiceDefinition.builder("overload.Client")
                            .addMethod(
                                    METHOD,
                                    (ServerCall<byte[], byte[]> call, Metadata metadata) -> {
                                        call.request(1);
                                        return new ServerCall.Listener<>() {
                                            @Override
                                            public void onHalfClose() {
                                                accepted.set(call);
                                                call.sendHeaders(new Metadata());
                                                call.sendMessage(new byte[1]);
                                            }

                                            @Override
                                            public void onCancel() {
                                                cancelled.countDown();
                                            }
                                        };
                                    })
                            .build();
            server =
                    NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0))
                            .addService(service)
                            .withChildOption(ChannelOption.ALLOCATOR, serverAllocator)
                            .build()
                            .start();
            channel =
                    NettyChannelBuilder.forAddress("127.0.0.1", server.getPort())
                            .usePlaintext()
                            .executor(executor)
                            .proxyDetector(GrpcUtil.NOOP_PROXY_DETECTOR)
                            .withOption(ChannelOption.ALLOCATOR, allocator)
                            .build();
            call =
                    channel.newCall(
                            METHOD, CallOptions.DEFAULT.withDeadlineAfter(10, TimeUnit.SECONDS));
            call.start(
                    new ClientCall.Listener<>() {
                        @Override
                        public void onHeaders(Metadata metadata) {
                            headers.countDown();
                        }

                        @Override
                        public void onMessage(byte[] message) {
                            receivedBytes.set(message.length);
                            if (message.length == 1) {
                                warmup.countDown();
                            }
                        }

                        @Override
                        public void onClose(Status status, Metadata trailers) {
                            result.complete(status);
                        }
                    },
                    new Metadata());
            call.request(2);
            call.sendMessage(new byte[1]);
            call.halfClose();
            assertTrue(
                    headers.await(5, TimeUnit.SECONDS),
                    "headers missing, result=" + result.getNow(null) + " executor=" + executor);
            assertTrue(warmup.await(5, TimeUnit.SECONDS));
            awaitIdle();
        }

        void awaitIdle() throws Exception {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while ((executor.getActiveCount() != 0 || !executor.getQueue().isEmpty())
                    && System.nanoTime() < deadline) {
                Thread.sleep(5);
            }
            assertEquals(0, executor.getActiveCount());
            assertTrue(executor.getQueue().isEmpty());
        }

        void responseAfterSaturation() throws Exception {
            CountDownLatch blocked = new CountDownLatch(1);
            executor.execute(
                    () -> {
                        blocked.countDown();
                        try {
                            release.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    });
            assertTrue(blocked.await(5, TimeUnit.SECONDS));
            while (executor.getQueue().remainingCapacity() > 0) {
                executor.execute(() -> {});
            }
            accepted.get().sendMessage(new byte[1024 * 1024]);
            accepted.get().close(Status.OK, new Metadata());
            assertEquals(Status.Code.OK, result.get(5, TimeUnit.SECONDS).getCode());
            assertEquals(1024 * 1024, receivedBytes.get());
            assertNoPinnedWhileConnected();
            assertNoPinnedAfterTransportClose();
        }

        void assertNoPinnedWhileConnected() throws Exception {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while ((allocator.pinnedDirectMemory() != 0
                            || serverAllocator.pinnedDirectMemory() != 0)
                    && System.nanoTime() < deadline) {
                Thread.sleep(5);
            }
            assertEquals(
                    0,
                    allocator.pinnedDirectMemory(),
                    "cancel must clean response buffers before TCP close");
            assertEquals(
                    0,
                    serverAllocator.pinnedDirectMemory(),
                    "cancel must clean request buffers before TCP close");
            assertFalse(channel.isShutdown());
            assertFalse(server.isShutdown());
        }

        void assertNoPinnedAfterTransportClose() throws Exception {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (allocator.pinnedDirectMemory() != 0 && System.nanoTime() < deadline) {
                Thread.sleep(5);
            }
            assertEquals(
                    0,
                    allocator.pinnedDirectMemory(),
                    "response buffers must be released after close");
        }

        @Override
        public void close() throws Exception {
            release.countDown();
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    private static final class BytesMarshaller implements MethodDescriptor.Marshaller<byte[]> {
        public InputStream stream(byte[] value) {
            return new ByteArrayInputStream(value);
        }

        public byte[] parse(InputStream stream) {
            try {
                return stream.readAllBytes();
            } catch (IOException error) {
                throw new RuntimeException(error);
            }
        }
    }
}
