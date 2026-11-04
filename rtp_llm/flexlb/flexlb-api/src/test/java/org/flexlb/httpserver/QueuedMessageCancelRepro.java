package org.flexlb.httpserver;

import io.grpc.ServerServiceDefinition;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.ServerCalls;
import io.grpc.stub.StreamObserver;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelOption;
import io.netty.channel.FixedRecvByteBufAllocator;
import io.netty.handler.codec.http2.DefaultHttp2Headers;
import io.netty.handler.codec.http2.DefaultHttp2HeadersEncoder;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class QueuedMessageCancelRepro {
    public static void main(String[] args) throws Exception {
        var pool = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>());
        var release = new CountDownLatch(1);
        var guard = new GrpcReceivingGuard(1, 5000, pool);
        var allocator = new PooledByteBufAllocator(true);
        var service =
                ServerServiceDefinition.builder("receiving.Test")
                        .addMethod(
                                GrpcReceivingGuardRepro.METHOD,
                                ServerCalls.asyncUnaryCall(
                                        (byte[] x, StreamObserver<byte[]> o) -> {}))
                        .build();
        var server =
                NettyServerBuilder.forPort(0)
                        .directExecutor()
                        .callExecutor(guard)
                        .addStreamTracerFactory(guard)
                        .intercept(guard)
                        .addService(service)
                        .withChildOption(ChannelOption.ALLOCATOR, allocator)
                        .withChildOption(
                                ChannelOption.RCVBUF_ALLOCATOR, new FixedRecvByteBufAllocator(4096))
                        .build()
                        .start();
        try (Socket socket = new Socket("127.0.0.1", server.getPort())) {
            var out = new DataOutputStream(socket.getOutputStream());
            out.write("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".getBytes(StandardCharsets.US_ASCII));
            GrpcReceivingGuardRepro.frame(out, 4, 0, 0, new byte[0]);
            var h = Unpooled.buffer();
            new DefaultHttp2HeadersEncoder()
                    .encodeHeaders(
                            1,
                            new DefaultHttp2Headers()
                                    .method("POST")
                                    .scheme("http")
                                    .authority("localhost")
                                    .path("/receiving.Test/Call")
                                    .set("content-type", "application/grpc")
                                    .set("te", "trailers"),
                            h);
            byte[] hb = new byte[h.readableBytes()];
            h.readBytes(hb);
            h.release();
            GrpcReceivingGuardRepro.frame(out, 1, 4, 1, hb);
            out.flush();
            GrpcReceivingGuardRepro.until(
                    () -> pool.getCompletedTaskCount() >= 1,
                    "interceptor must request messages first");
            Thread.sleep(50);
            var blocked = new CountDownLatch(1);
            pool.execute(
                    () -> {
                        blocked.countDown();
                        try {
                            release.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    });
            blocked.await();
            var body = new ByteArrayOutputStream();
            var b = new DataOutputStream(body);
            b.writeByte(0);
            b.writeInt(32768);
            b.write(new byte[32768]);
            byte[] wire = body.toByteArray();
            for (int i = 0; i < wire.length; i += 16384)
                GrpcReceivingGuardRepro.frame(
                        out,
                        0,
                        0,
                        1,
                        Arrays.copyOfRange(wire, i, Math.min(wire.length, i + 16384)));
            out.flush();
            GrpcReceivingGuardRepro.until(
                    () -> allocator.pinnedDirectMemory() >= 32768,
                    "queued full response must retain actual direct");
            GrpcReceivingGuardRepro.frame(out, 3, 0, 1, new byte[] {0, 0, 0, 8});
            out.flush();
            Thread.sleep(100);
            System.out.println(
                    "AFTER_RST receiving="
                            + guard.receiving()
                            + " pinned="
                            + allocator.pinnedDirectMemory()
                            + " queue="
                            + pool.getQueue().size());
            GrpcReceivingGuardRepro.check(
                    guard.receiving() == 1,
                    "must retain budget while full-message producer remains queued");
            release.countDown();
            GrpcReceivingGuardRepro.until(
                    () -> allocator.pinnedDirectMemory() == 0, "producer cleanup");
            GrpcReceivingGuardRepro.until(
                    () -> guard.receiving() == 0, "budget cleanup after queue drained");
            System.out.println(
                    "QUEUED_PRODUCER_CLEANED receiving="
                            + guard.receiving()
                            + " pinned="
                            + allocator.pinnedDirectMemory());
        } finally {
            release.countDown();
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            guard.close();
            pool.shutdownNow();
            pool.awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
