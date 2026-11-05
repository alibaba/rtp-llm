package org.flexlb.httpserver;

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.ServerServiceDefinition;
import io.grpc.Status;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.ServerCalls;
import io.grpc.stub.StreamObserver;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelOption;
import io.netty.channel.FixedRecvByteBufAllocator;
import io.netty.handler.codec.http2.DefaultHttp2Headers;
import io.netty.handler.codec.http2.DefaultHttp2HeadersEncoder;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class GrpcReceivingGuardRepro {
    static final MethodDescriptor<byte[], byte[]> METHOD =
            MethodDescriptor.<byte[], byte[]>newBuilder()
                    .setType(MethodDescriptor.MethodType.UNARY)
                    .setFullMethodName("receiving.Test/Call")
                    .setRequestMarshaller(new Bytes())
                    .setResponseMarshaller(new Bytes())
                    .build();

    static class Bytes implements MethodDescriptor.Marshaller<byte[]> {
        public InputStream stream(byte[] v) {
            return new ByteArrayInputStream(v);
        }

        public byte[] parse(InputStream i) {
            try {
                return i.readAllBytes();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    static void check(boolean x, String m) {
        if (!x) throw new AssertionError(m);
    }

    static void until(java.util.function.BooleanSupplier p, String m) throws Exception {
        long e = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < e) {
            if (p.getAsBoolean())
                return; // A sampled allocator zero can legitimately change on a new PING.
            Thread.sleep(2);
        }
        throw new AssertionError(m);
    }

    // Exclude lazy TCP/HTTP2 setup from the later sub-second saturation assertion.
    static void awaitReady(ManagedChannel channel) throws Exception {
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (System.nanoTime() < until) {
            ConnectivityState state = channel.getState(true);
            if (state == ConnectivityState.READY) return;
            check(state != ConnectivityState.SHUTDOWN, "test channel shutdown before READY");
            CountDownLatch changed = new CountDownLatch(1);
            channel.notifyWhenStateChanged(state, changed::countDown);
            changed.await(
                    Math.min(TimeUnit.SECONDS.toNanos(1), Math.max(1, until - System.nanoTime())),
                    TimeUnit.NANOSECONDS);
        }
        throw new AssertionError("test channel did not become HTTP/2 READY");
    }

    static void frame(DataOutputStream o, int type, int flags, int stream, byte[] b)
            throws Exception {
        o.writeByte(b.length >>> 16);
        o.writeByte(b.length >>> 8);
        o.writeByte(b.length);
        o.writeByte(type);
        o.writeByte(flags);
        o.writeInt(stream);
        o.write(b);
    }

    static Socket partial(int port) throws Exception {
        Socket s = new Socket("127.0.0.1", port);
        var o = new DataOutputStream(s.getOutputStream());
        o.write("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".getBytes(StandardCharsets.US_ASCII));
        frame(o, 4, 0, 0, new byte[0]);
        ByteBuf h = Unpooled.buffer();
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
        byte[] b = new byte[h.readableBytes()];
        h.readBytes(b);
        h.release();
        frame(o, 1, 4, 1, b);
        var bo = new ByteArrayOutputStream();
        var q = new DataOutputStream(bo);
        q.writeByte(0);
        q.writeInt(1024 * 1024);
        q.write(new byte[32763]);
        b = bo.toByteArray();
        frame(o, 0, 0, 1, Arrays.copyOfRange(b, 0, 16384));
        frame(o, 0, 0, 1, Arrays.copyOfRange(b, 16384, b.length));
        o.flush();
        return s;
    }

    static ClientCall<byte[], byte[]> full(ManagedChannel c, CompletableFuture<Status> done) {
        var call = c.newCall(METHOD, CallOptions.DEFAULT);
        call.start(
                new ClientCall.Listener<>() {
                    @Override
                    public void onClose(Status s, Metadata t) {
                        done.complete(s);
                    }
                },
                new Metadata());
        call.request(1);
        call.sendMessage(new byte[8192]);
        call.halfClose();
        return call;
    }

    public static void main(String[] args) throws Exception {
        var pool = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>());
        var blocked = new CountDownLatch(1);
        var release = new CountDownLatch(1);
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
        var guard = new GrpcReceivingGuard(2, 500, pool);
        var allocator = new PooledByteBufAllocator(true);
        AtomicInteger entered = new AtomicInteger();
        AtomicInteger cancelled = new AtomicInteger();
        var service =
                ServerServiceDefinition.builder("receiving.Test")
                        .addMethod(
                                METHOD,
                                ServerCalls.asyncUnaryCall(
                                        (byte[] b, StreamObserver<byte[]> response) -> {
                                            entered.incrementAndGet();
                                            ((ServerCallStreamObserver<byte[]>) response)
                                                    .setOnCancelHandler(cancelled::incrementAndGet);
                                        }))
                        .build();
        NettyServerBuilder builder = NettyServerBuilder.forPort(0);
        if (args.length > 0 && args[0].equals("late")) builder.executor(pool);
        else builder.directExecutor();
        Server server =
                builder.callExecutor(guard)
                        .addStreamTracerFactory(guard)
                        .intercept(guard)
                        .addService(service)
                        .withChildOption(ChannelOption.ALLOCATOR, allocator)
                        .withChildOption(
                                ChannelOption.RCVBUF_ALLOCATOR, new FixedRecvByteBufAllocator(4096))
                        .build()
                        .start();
        var channel =
                NettyChannelBuilder.forAddress("127.0.0.1", server.getPort())
                        .usePlaintext()
                        .build();
        List<Socket> sockets = new ArrayList<>();
        try {
            sockets.add(partial(server.getPort()));
            sockets.add(partial(server.getPort()));
            until(() -> guard.receiving() == 2, "headers must reserve before business executor");
            until(
                    () -> allocator.pinnedDirectMemory() >= 32768,
                    "partial bytes must actually pin before callback admission");
            long began = System.nanoTime();
            var rejected = new CompletableFuture<Status>();
            full(channel, rejected);
            Status status = rejected.get(400, TimeUnit.MILLISECONDS);
            long ms = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - began);
            check(status.getCode() == Status.Code.RESOURCE_EXHAUSTED, "denial status " + status);
            check(entered.get() == 0, "blocked business must not enter");
            System.out.println(
                    "BLOCKED_EXECUTOR_REJECT status="
                            + status.getCode()
                            + " latencyMs="
                            + ms
                            + " pinned="
                            + allocator.pinnedDirectMemory()
                            + " active="
                            + guard.receiving());
            until(
                    () -> guard.receiving() == 0,
                    "watchdog terminal must release permit with business executor blocked");
            until(
                    () -> allocator.pinnedDirectMemory() == 0,
                    "watchdog must release partial buffers before TCP close");
            check(guard.timedOut() == 2, "two partial timeouts");
            for (Socket s : sockets) check(!s.isClosed(), "raw TCP still open");
            System.out.println(
                    "NO_DEADLINE_PARTIAL_TIMEOUT active="
                            + guard.receiving()
                            + " pinned="
                            + allocator.pinnedDirectMemory()
                            + " businessEntered="
                            + entered.get());
            release.countDown();
            List<ClientCall<byte[], byte[]>> calls = new ArrayList<>();
            List<CompletableFuture<Status>> dones = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                var done = new CompletableFuture<Status>();
                dones.add(done);
                calls.add(full(channel, done));
                int want = i + 1;
                until(() -> entered.get() == want, "decoded request must reach handler");
                check(
                        guard.receiving() == 0,
                        "half-close must release receiving budget while business is pending");
            }
            Thread.sleep(700);
            for (var f : dones)
                check(!f.isDone(), "body watchdog must not cut admitted business queue");
            System.out.println(
                    "BUSINESS_QUEUE_NOT_CAPPED pending="
                            + dones.size()
                            + " receiving="
                            + guard.receiving());
            for (var c : calls) c.cancel("test cancel", null);
            for (var f : dones)
                check(
                        f.get(5, TimeUnit.SECONDS).getCode() == Status.Code.CANCELLED,
                        "cancel status");
            until(() -> cancelled.get() == 8, "all business cancels observed");
            until(() -> guard.receiving() == 0, "no receiving permit leak after cancel");
            System.out.println(
                    "TERMINAL_CANCEL_CLEANUP cancelled="
                            + cancelled.get()
                            + " receiving="
                            + guard.receiving());
        } finally {
            release.countDown();
            for (Socket s : sockets) s.close();
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            guard.close();
            pool.shutdownNow();
            pool.awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
