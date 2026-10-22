package org.flexlb.mockengine;

import io.grpc.*;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.buffer.ByteBuf;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.lang.reflect.Field;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;

// The reflection follows the pinned gRPC 1.65.0 unary framer.
// Observe the real retained buffer; a quiet leak detector alone is not a release assertion.
class RpcCancelledResponseBufferTest {
    static Object field(Object value, String name) throws Exception {
        for (Class<?> type = value.getClass(); type != null; type = type.getSuperclass()) {
            try { Field f = type.getDeclaredField(name); f.setAccessible(true); return f.get(value); }
            catch (NoSuchFieldException ignored) { }
        }
        throw new NoSuchFieldException(value.getClass() + "." + name);
    }
    static Object unwrap(ServerCall<?, ?> value) throws Exception {
        Object current = value;
        for (int depth = 0; depth < 10; depth++) {
            if (current.getClass().getName().equals("io.grpc.internal.ServerCallImpl")) return current;
            current = field(current, "delegate");
        }
        throw new IllegalStateException("unexpected call wrapper");
    }

    @ParameterizedTest
    @CsvSource({"false, false", "true, false", "false, true", "true, true"})
    void cancellationAfterUnaryOnNextMustStillDrainFramer(boolean cacheMethod, boolean completeWithError) throws Exception {
        CountDownLatch framed = new CountDownLatch(1), finish = new CountDownLatch(1), returned = new CountDownLatch(1);
        AtomicReference<ServerCall<?, ?>> call = new AtomicReference<>();
        AtomicReference<Object> framer = new AtomicReference<>();
        AtomicReference<ByteBuf> buffer = new AtomicReference<>();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        AtomicReference<Context> serverContext = new AtomicReference<>();
        try (var pools = new MockRpcExecutors(2, 16)) {
            var engine = pools.engine();
            var service = new RpcServiceGrpc.RpcServiceImplBase() {
                private <T> void respond(T response, StreamObserver<T> observer) {
                    try {
                        serverContext.set(Context.current());
                        observer.onNext(response);
                        Object stream = field(unwrap(call.get()), "stream");
                        Object f = field(stream, "framer"); framer.set(f);
                        buffer.set((ByteBuf) field(field(f, "buffer"), "bytebuf"));
                        framed.countDown();
                        try { finish.await(); } catch (InterruptedException expectedOnCancellation) { }
                        if (completeWithError) observer.onError(Status.CANCELLED.asRuntimeException());
                        else observer.onCompleted();
                    } catch (Throwable error) { failure.set(error); framed.countDown(); }
                    finally { returned.countDown(); }
                }
                @Override public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                        StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                    respond(EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).build(), observer);
                }
                @Override public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                        StreamObserver<EngineRpcService.CacheStatusPB> observer) {
                    respond(EngineRpcService.CacheStatusPB.newBuilder().setVersion(1).build(), observer);
                }
            };
            ServerInterceptor capture = new ServerInterceptor() {
                @Override public <Q, S> ServerCall.Listener<Q> interceptCall(ServerCall<Q, S> c,
                        Metadata headers, ServerCallHandler<Q, S> next) {
                    call.set(c); return next.startCall(c, headers);
                }
            };
            var builder = NettyServerBuilder.forPort(0).executor(pools.controlExecutor());
            builder.callExecutor(engine).intercept(engine);
            Server server = builder.intercept(capture).addService(service).build().start();
            ManagedChannel channel = NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
            try {
                var stub = RpcServiceGrpc.newFutureStub(channel).withDeadlineAfter(10, TimeUnit.SECONDS);
                var pending = cacheMethod ? stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance())
                        : stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
                assertTrue(framed.await(5, TimeUnit.SECONDS));
                assertNull(failure.get()); assertNotNull(buffer.get()); assertEquals(1, buffer.get().refCnt());
                pending.cancel(true);
                long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
                while (!serverContext.get().isCancelled() && System.nanoTime() < until) Thread.sleep(1);
                assertTrue(serverContext.get().isCancelled());
                finish.countDown(); assertTrue(returned.await(3, TimeUnit.SECONDS)); assertNull(failure.get());
                long drainedBy = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
                while (buffer.get().refCnt() != 0 && System.nanoTime() < drainedBy) Thread.sleep(1);
                assertEquals(0, buffer.get().refCnt(), "onCompleted must release/transfer the unary framer buffer after cancellation");
            } finally {
                finish.countDown();
                if (buffer.get() != null && buffer.get().refCnt() > 0) {
                    framer.get().getClass().getMethod("dispose").invoke(framer.get());
                }
                channel.shutdownNow(); server.shutdownNow();
                assertTrue(channel.awaitTermination(3, TimeUnit.SECONDS));
                assertTrue(server.awaitTermination(3, TimeUnit.SECONDS)); engine.stop();
            }
        }
    }
}
