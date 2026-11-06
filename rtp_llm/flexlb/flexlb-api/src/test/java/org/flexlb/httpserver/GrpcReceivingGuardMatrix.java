package org.flexlb.httpserver;

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerServiceDefinition;
import io.grpc.Status;
import io.grpc.netty.NettyChannelBuilder;
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
import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class GrpcReceivingGuardMatrix {
    static void ck(boolean b, String m) {
        GrpcReceivingGuardRepro.check(b, m);
    }

    static void waitFor(java.util.function.BooleanSupplier p, String m) throws Exception {
        GrpcReceivingGuardRepro.until(p, m);
    }

    static final class Raw implements AutoCloseable {
        Socket socket;
        DataOutputStream out;

        Raw(int port, String timeout) throws Exception {
            socket = new Socket("127.0.0.1", port);
            out = new DataOutputStream(socket.getOutputStream());
            out.write("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".getBytes(StandardCharsets.US_ASCII));
            frame(4, 0, new byte[0]);
            var h = Unpooled.buffer();
            var headers =
                    new DefaultHttp2Headers()
                            .method("POST")
                            .scheme("http")
                            .authority("localhost")
                            .path("/receiving.Test/Call")
                            .set("content-type", "application/grpc")
                            .set("te", "trailers");
            if (timeout != null) headers.set("grpc-timeout", timeout);
            new DefaultHttp2HeadersEncoder().encodeHeaders(1, headers, h);
            byte[] b = new byte[h.readableBytes()];
            h.readBytes(b);
            h.release();
            frame(1, 4, b);
            out.flush();
        }

        void frame(int type, int flags, byte[] b) throws Exception {
            GrpcReceivingGuardRepro.frame(out, type, flags, type == 4 ? 0 : 1, b);
        }

        void message(byte[] b, boolean end) throws Exception {
            var ba = new ByteArrayOutputStream();
            var d = new DataOutputStream(ba);
            d.writeByte(0);
            d.writeInt(b.length);
            d.write(b);
            byte[] wire = ba.toByteArray();
            for (int i = 0; i < wire.length; i += 16384)
                frame(
                        0,
                        end && i + 16384 >= wire.length ? 1 : 0,
                        Arrays.copyOfRange(wire, i, Math.min(wire.length, i + 16384)));
            out.flush();
        }

        void end() throws Exception {
            frame(0, 1, new byte[0]);
            out.flush();
        }

        void reset() throws Exception {
            frame(3, 0, new byte[] {0, 0, 0, 8});
            out.flush();
        }

        public void close() throws Exception {
            socket.close();
        }
    }

    static MethodDescriptor<byte[], byte[]> method(String name) {
        return GrpcReceivingGuardRepro.METHOD.toBuilder()
                .setFullMethodName("receiving.Test/" + name)
                .setRequestMarshaller(
                        new GrpcReceivingGuardRepro.Bytes() {
                            @Override
                            public byte[] parse(InputStream s) {
                                byte[] b = super.parse(s);
                                if (b.length > 0 && b[0] == (byte) 255)
                                    throw new IllegalArgumentException("parse failure");
                                if (b.length > 0 && b[0] == (byte) 254)
                                    throw new AssertionError("parse Error");
                                return b;
                            }
                        })
                .build();
    }

    static CompletableFuture<Status> full(ManagedChannel ch, MethodDescriptor<byte[], byte[]> m) {
        var done = new CompletableFuture<Status>();
        var c = ch.newCall(m, CallOptions.DEFAULT);
        c.start(
                new ClientCall.Listener<>() {
                    @Override
                    public void onClose(Status s, Metadata t) {
                        done.complete(s);
                    }
                },
                new Metadata());
        c.request(1);
        c.sendMessage(new byte[1]);
        c.halfClose();
        return done;
    }

    public static void main(String[] args) throws Exception {
        var pool = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>());
        var control = Executors.newSingleThreadExecutor();
        var guard = new GrpcReceivingGuard(2, 1, 180, pool, control);
        var alloc = new PooledByteBufAllocator(true);
        var entered = new AtomicInteger();
        var cancelled = new AtomicInteger();
        var call = method("Call");
        var cancel = method("Cancel");
        var service =
                ServerServiceDefinition.builder("receiving.Test")
                        .addMethod(
                                call,
                                ServerCalls.asyncUnaryCall(
                                        (byte[] b, StreamObserver<byte[]> o) -> {
                                            entered.incrementAndGet();
                                            o.onNext(new byte[1]);
                                            o.onCompleted();
                                        }))
                        .addMethod(
                                cancel,
                                ServerCalls.asyncUnaryCall(
                                        (byte[] b, StreamObserver<byte[]> o) -> {
                                            cancelled.incrementAndGet();
                                            o.onNext(new byte[1]);
                                            o.onCompleted();
                                        }))
                        .build();
        var server =
                NettyServerBuilder.forPort(0)
                        .directExecutor()
                        .callExecutor(guard)
                        .addStreamTracerFactory(guard)
                        .intercept(guard)
                        .addService(service)
                        .maxInboundMessageSize(65536)
                        .withChildOption(ChannelOption.ALLOCATOR, alloc)
                        .withChildOption(
                                ChannelOption.RCVBUF_ALLOCATOR, new FixedRecvByteBufAllocator(4096))
                        .build()
                        .start();
        var ch =
                NettyChannelBuilder.forAddress("127.0.0.1", server.getPort())
                        .usePlaintext()
                        .build();
        var release = new CountDownLatch(1);
        List<Raw> sockets = new ArrayList<>();
        try {
            GrpcReceivingGuardRepro.awaitReady(ch);
            // Initialize transport/HPACK and dependency classes before exercising sub-200 ms
            // deadlines.
            for (int i = 0; i < 3; i++)
                ck(full(ch, call).get(3, TimeUnit.SECONDS).isOk(), "warmup");
            entered.set(0);
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
            sockets.add(new Raw(server.getPort(), null));
            sockets.add(new Raw(server.getPort(), null));
            waitFor(() -> guard.receiving() == 2, "headers only reserve");
            ck(
                    full(ch, cancel).get(1, TimeUnit.SECONDS).isOk(),
                    "Cancel control lane must bypass full Schedule receiving and blocked pool");
            ck(cancelled.get() == 1, "control entered");
            for (int i = 0; i < 20; i++) {
                Raw r = new Raw(server.getPort(), null);
                sockets.add(r);
                r.message(new byte[32768], false);
            }
            waitFor(() -> guard.rejected() >= 20, "reject across different TCP connections");
            waitFor(() -> guard.receiving() == 0, "header only watchdog budget cleanup");
            waitFor(
                    () -> alloc.pinnedDirectMemory() == 0,
                    "rejected multiconnection buffers reclaimed with TCP still open");
            ck(entered.get() == 0, "timer before interceptor must block business entry");
            System.out.println(
                    "MULTI_CONNECTION_CONTROL rejected="
                            + guard.rejected()
                            + " receiving="
                            + guard.receiving()
                            + " pinned="
                            + alloc.pinnedDirectMemory()
                            + " control="
                            + cancelled.get());
            long before = System.nanoTime();
            try (Raw r = new Raw(server.getPort(), "60m")) {
                waitFor(() -> guard.receiving() == 1, "deadline header admitted");
                waitFor(
                        () -> guard.receiving() == 0,
                        "short client deadline closes body independently of blocked pool");
            }
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - before);
            ck(elapsed < 170, "shorter client deadline respected: " + elapsed);
            System.out.println("SHORT_DEADLINE elapsedMs=" + elapsed);
            release.countDown();
            Thread.sleep(30);
            int prior = entered.get();
            try (Raw r = new Raw(server.getPort(), null)) {
                waitFor(() -> guard.receiving() == 1, "full nohalf header admitted");
                r.message(new byte[100], false);
                Thread.sleep(40);
                ck(
                        guard.receiving() == 1,
                        "decoded full message without halfclose retains receiving permit");
                waitFor(() -> guard.receiving() == 0, "full message but no halfclose timeout");
                ck(entered.get() == prior, "no halfclose no business handler");
            }
            System.out.println("FULL_MESSAGE_NO_HALFCLOSE_TIMEOUT receiving=" + guard.receiving());
            try (Raw r = new Raw(server.getPort(), null)) {
                r.message(new byte[] {(byte) 255}, true);
                waitFor(() -> guard.receiving() == 0, "parse exception releases permit");
            }
            Thread.sleep(30);
            ck(entered.get() == prior, "parse exception must not invoke business");
            System.out.println("PARSE_EXCEPTION_CLEAN receiving=" + guard.receiving());
            try (Raw r = new Raw(server.getPort(), null)) {
                waitFor(() -> guard.receiving() == 1, "Error header admitted");
                byte[] errorBody = new byte[32768];
                errorBody[0] = (byte) 254;
                r.message(errorBody, true);
                waitFor(() -> guard.receiving() == 0, "parse Error terminal listener resumes");
                waitFor(
                        () -> alloc.pinnedDirectMemory() == 0,
                        "parse Error actual pooled bytes reclaimed while TCP open");
                ck(entered.get() == prior, "parse Error must not invoke business");
            }
            System.out.println(
                    "PARSE_ERROR_TCP_OPEN_CLEAN receiving="
                            + guard.receiving()
                            + " pinned="
                            + alloc.pinnedDirectMemory());
            try (Raw r = new Raw(server.getPort(), null)) {
                waitFor(() -> guard.receiving() == 1, "unary header admitted");
                r.message(new byte[10], false);
                r.message(new byte[10], false);
                waitFor(
                        () -> guard.receiving() == 0,
                        "multiple unary messages protocol error cleanup");
            }
            ck(entered.get() == prior, "illegal unary messages no business");
            System.out.println("UNARY_SECOND_MESSAGE_CLEAN receiving=" + guard.receiving());
            try (Raw r = new Raw(server.getPort(), null)) {
                waitFor(() -> guard.receiving() == 1, "cancel header admitted");
                r.reset();
                waitFor(() -> guard.receiving() == 0, "headers then RST cleanup");
            }
            System.out.println("HEADERS_CANCEL_CLEAN receiving=" + guard.receiving());
            for (int i = 0; i < 20; i++) {
                try (Raw r = new Raw(server.getPort(), null)) {
                    waitFor(() -> guard.receiving() == 1, "race admitted");
                    r.message(new byte[100], false);
                    Thread.sleep(150 + i % 6 * 8);
                    try {
                        r.end();
                    } catch (IOException ignored) {
                    }
                    waitFor(() -> guard.receiving() == 0, "timer/halfclose race budget cleanup");
                }
            }
            for (int i = 0; i < 200; i++)
                ck(full(ch, call).get(2, TimeUnit.SECONDS).isOk(), "normal full call " + i);
            waitFor(() -> guard.pendingTimers() == 0, "cancelled timers removed from DelayQueue");
            ck(guard.receiving() == 0, "normal calls no permits retained");
            System.out.println(
                    "RACE_AND_TIMER_QUEUE races=20 normal=200 timers="
                            + guard.pendingTimers()
                            + " receiving="
                            + guard.receiving());
        } finally {
            release.countDown();
            for (Raw r : sockets) r.close();
            ch.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            guard.close();
            pool.shutdownNow();
            control.shutdownNow();
            pool.awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
