package org.flexlb.engine.grpc.config;

import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerServiceDefinition;
import io.grpc.netty.NettyServerBuilder;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelOption;
import io.netty.channel.FixedRecvByteBufAllocator;
import io.netty.handler.codec.http2.DefaultHttp2Headers;
import io.netty.handler.codec.http2.DefaultHttp2HeadersEncoder;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Timeout(20)
class GrpcPartialMessageCancelTest {
    @Test
    void http2CancelReleasesIncompleteGrpcBodyBeforeBusinessHandler() throws Exception {
        var allocator = new PooledByteBufAllocator(true);
        var started = new CountDownLatch(1);
        var cancelled = new CountDownLatch(1);
        var messages = new AtomicInteger();
        MethodDescriptor.Marshaller<byte[]> marshaller =
                new MethodDescriptor.Marshaller<>() {
                    public InputStream stream(byte[] bytes) {
                        return new ByteArrayInputStream(bytes);
                    }

                    public byte[] parse(InputStream stream) {
                        try {
                            return stream.readAllBytes();
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    }
                };
        var method =
                MethodDescriptor.<byte[], byte[]>newBuilder()
                        .setType(MethodDescriptor.MethodType.UNARY)
                        .setFullMethodName("partial.Test/Call")
                        .setRequestMarshaller(marshaller)
                        .setResponseMarshaller(marshaller)
                        .build();
        var service =
                ServerServiceDefinition.builder("partial.Test")
                        .addMethod(
                                method,
                                (ServerCall<byte[], byte[]> call, Metadata headers) -> {
                                    call.request(1);
                                    started.countDown();
                                    return new ServerCall.Listener<>() {
                                        @Override
                                        public void onMessage(byte[] message) {
                                            messages.incrementAndGet();
                                        }

                                        @Override
                                        public void onCancel() {
                                            cancelled.countDown();
                                        }
                                    };
                                })
                        .build();
        Server server =
                NettyServerBuilder.forPort(0)
                        .addService(service)
                        .withChildOption(ChannelOption.ALLOCATOR, allocator)
                        .withChildOption(
                                ChannelOption.RCVBUF_ALLOCATOR, new FixedRecvByteBufAllocator(4096))
                        .build()
                        .start();
        try (Socket socket = new Socket("127.0.0.1", server.getPort())) {
            var output = new DataOutputStream(socket.getOutputStream());
            output.write("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".getBytes(StandardCharsets.US_ASCII));
            frame(output, 4, 0, 0, new byte[0]); // SETTINGS
            ByteBuf encoded = Unpooled.buffer();
            new DefaultHttp2HeadersEncoder()
                    .encodeHeaders(
                            1,
                            new DefaultHttp2Headers()
                                    .method("POST")
                                    .scheme("http")
                                    .authority("localhost")
                                    .path("/partial.Test/Call")
                                    .set("content-type", "application/grpc")
                                    .set("te", "trailers"),
                            encoded);
            byte[] hpack = new byte[encoded.readableBytes()];
            encoded.readBytes(hpack);
            encoded.release();
            frame(output, 1, 4, 1, hpack); // HEADERS, END_HEADERS only
            output.flush();
            assertTrue(started.await(5, TimeUnit.SECONDS));
            // Promise 1MiB but transmit only 32KiB. No END_STREAM and no complete message.
            ByteArrayOutputStream partial = new ByteArrayOutputStream();
            var body = new DataOutputStream(partial);
            body.writeByte(0);
            body.writeInt(1024 * 1024);
            body.write(new byte[32763]);
            byte[] bytes = partial.toByteArray();
            frame(output, 0, 0, 1, java.util.Arrays.copyOfRange(bytes, 0, 16384));
            frame(output, 0, 0, 1, java.util.Arrays.copyOfRange(bytes, 16384, 32768));
            output.flush();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (allocator.pinnedDirectMemory() < 16384 && System.nanoTime() < deadline) {
                Thread.sleep(5);
            }
            long held = allocator.pinnedDirectMemory();
            assertTrue(held >= 16384, "incomplete body must actually retain network buffers");
            assertEquals(0, messages.get());
            frame(output, 3, 0, 1, new byte[] {0, 0, 0, 8}); // RST_STREAM CANCEL
            output.flush();
            assertTrue(cancelled.await(5, TimeUnit.SECONDS));
            deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (allocator.pinnedDirectMemory() != 0 && System.nanoTime() < deadline) {
                Thread.sleep(5);
            }
            assertEquals(
                    0,
                    allocator.pinnedDirectMemory(),
                    "RST_STREAM must release all retained partial body buffers while TCP stays open");
            assertFalse(socket.isClosed());
            assertFalse(server.isShutdown());
        } finally {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (allocator.pinnedDirectMemory() != 0 && System.nanoTime() < deadline) {
            Thread.sleep(5);
        }
        assertEquals(0, allocator.pinnedDirectMemory());
    }

    private static void frame(DataOutputStream out, int type, int flags, int stream, byte[] payload)
            throws IOException {
        out.writeByte(payload.length >>> 16);
        out.writeByte(payload.length >>> 8);
        out.writeByte(payload.length);
        out.writeByte(type);
        out.writeByte(flags);
        out.writeInt(stream);
        out.write(payload);
    }
}
