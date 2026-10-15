package org.flexlb.mockengine;

import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;

class MockRpcExecutorsTest {
    static final class Endpoint implements AutoCloseable {
        final MockRpcExecutors.Engine engine;
        final Server server;
        final ManagedChannel channel;
        final RpcServiceGrpc.RpcServiceFutureStub stub;
        Endpoint(MockRpcExecutors pools, RpcServiceGrpc.RpcServiceImplBase service) throws Exception {
            engine = pools.engine();
            server = NettyServerBuilder.forPort(0).executor(pools.controlExecutor())
                    .callExecutor(engine).intercept(engine).addService(service).build().start();
            channel = NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
            // Separate connection establishment from the bounded lifecycle assertions.
            long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (channel.getState(true) != io.grpc.ConnectivityState.READY && System.nanoTime() < end) Thread.sleep(5);
            assertEquals(io.grpc.ConnectivityState.READY, channel.getState(false));
            stub = RpcServiceGrpc.newFutureStub(channel).withDeadlineAfter(15, TimeUnit.SECONDS);
        }
        @Override public void close() throws Exception {
            channel.shutdownNow(); server.shutdownNow();
            assertTrue(channel.awaitTermination(3, TimeUnit.SECONDS));
            assertTrue(server.awaitTermination(3, TimeUnit.SECONDS));
            engine.stop();
        }
    }
    static class Service extends RpcServiceGrpc.RpcServiceImplBase {
        final CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1), cancelled = new CountDownLatch(1);
        final AtomicInteger statuses = new AtomicInteger();
        final AtomicReference<String> methodThread = new AtomicReference<>();
        @Override public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                io.grpc.stub.StreamObserver<EngineRpcService.CacheStatusPB> observer) {
            Context context = Context.current();
            methodThread.set(Thread.currentThread().getName());
            context.addListener(ignored -> cancelled.countDown(), Runnable::run);
            entered.countDown();
            try { release.await(); } catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
            if (!context.isCancelled()) {
                observer.onNext(EngineRpcService.CacheStatusPB.getDefaultInstance()); observer.onCompleted();
            }
        }
        @Override public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                io.grpc.stub.StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
            statuses.incrementAndGet();
            observer.onNext(EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).build()); observer.onCompleted();
        }
    }
    static void await(java.util.function.BooleanSupplier condition) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean() && System.nanoTime() < end) Thread.sleep(5);
        assertTrue(condition.getAsBoolean());
    }
    @Test void engineCompetitionDoesNotBlockAnotherEngineOrNetty() throws Exception {
        try (var pools = new MockRpcExecutors(1, 64)) {
            var a = new Service(); var b = new Service();
            try (var first = new Endpoint(pools, a); var second = new Endpoint(pools, b)) {
                try {
                    var cache = first.stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
                    assertTrue(a.entered.await(2, TimeUnit.SECONDS));
                    var status = first.stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
                    await(() -> first.engine.inFlight() == 2);
                    assertFalse(status.isDone());
                    assertTrue(second.stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()).get(2, TimeUnit.SECONDS).getAlive());
                    assertTrue(a.methodThread.get().startsWith("mock-rpc-"));
                    a.release.countDown(); cache.get(2, TimeUnit.SECONDS);
                    assertTrue(status.get(2, TimeUnit.SECONDS).getAlive());
                } finally { a.release.countDown(); }
            }
        }
    }
    @Test void overloadRejectsOnlyNewCallsWithExplicitStatus() throws Exception {
        try (var pools = new MockRpcExecutors(1, 2)) {
            var service = new Service();
            try (var endpoint = new Endpoint(pools, service)) {
                try {
                    var cache = endpoint.stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
                    assertTrue(service.entered.await(2, TimeUnit.SECONDS));
                    var queued = endpoint.stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
                    await(() -> endpoint.engine.inFlight() == 2);
                    var rejected = endpoint.stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
                    Exception error = assertThrows(Exception.class, () -> rejected.get(2, TimeUnit.SECONDS));
                    assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(error.getCause()).getCode());
                    assertEquals(1L, endpoint.engine.snapshot().get("rejected_calls_total"));
                    service.release.countDown(); cache.get(2, TimeUnit.SECONDS);
                    assertTrue(queued.get(2, TimeUnit.SECONDS).getAlive());
                    await(() -> endpoint.engine.inFlight() == 0);
                } finally { service.release.countDown(); }
            }
        }
    }
    @Test void expiredQueuedCallReclaimsAdmissionAndNeverInvokesBusinessWork() throws Exception {
        try (var pools = new MockRpcExecutors(1, 2)) {
            var service = new Service();
            try (var endpoint = new Endpoint(pools, service)) {
                try {
                    var cache = endpoint.stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
                    assertTrue(service.entered.await(2, TimeUnit.SECONDS));
                    var expired = endpoint.stub.withDeadlineAfter(100, TimeUnit.MILLISECONDS).getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
                    assertThrows(Exception.class, () -> expired.get(2, TimeUnit.SECONDS));
                    await(() -> endpoint.engine.inFlight() == 1);
                    var live = endpoint.stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
                    await(() -> endpoint.engine.inFlight() == 2);
                    service.release.countDown(); cache.get(2, TimeUnit.SECONDS); live.get(2, TimeUnit.SECONDS);
                    assertEquals(1, service.statuses.get());
                } finally { service.release.countDown(); }
            }
        }
    }
    @Test void cancellationIsDeliveredWhileEngineWorkersAreBusy() throws Exception {
        try (var pools = new MockRpcExecutors(1, 2)) {
            var service = new Service();
            try (var endpoint = new Endpoint(pools, service)) {
                var cache = endpoint.stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
                assertTrue(service.entered.await(2, TimeUnit.SECONDS)); cache.cancel(true);
                assertTrue(service.cancelled.await(2, TimeUnit.SECONDS));
                await(() -> endpoint.engine.inFlight() == 0);
                assertTrue(endpoint.stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()).get(2, TimeUnit.SECONDS).getAlive());
            } finally { service.release.countDown(); }
        }
    }
    @Test void shutdownDrainsBusyAndQueuedCallsAndIsIdempotent() throws Exception {
        var pools = new MockRpcExecutors(1, 64); var service = new Service();
        try (var endpoint = new Endpoint(pools, service)) {
            endpoint.stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
            assertTrue(service.entered.await(2, TimeUnit.SECONDS));
            var calls = new ArrayList<java.util.concurrent.Future<?>>();
            for (int i=0; i<32; i++) calls.add(endpoint.stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()));
            await(() -> endpoint.engine.inFlight() > 1);
            endpoint.channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            endpoint.server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            pools.close(); assertEquals(0, endpoint.engine.inFlight());
            assertTrue(calls.stream().allMatch(java.util.concurrent.Future::isDone)); pools.close();
        } finally { service.release.countDown(); pools.close(); }
    }
    @Test void retainedBurstDrainsWithoutRejectingLifecycleCallbacks() throws Exception {
        try (var pools = new MockRpcExecutors(1, 4096)) {
            var service = new Service();
            try (var endpoint = new Endpoint(pools, service)) {
                try {
                    var cache = endpoint.stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
                    assertTrue(service.entered.await(2, TimeUnit.SECONDS));
                    var calls = new ArrayList<com.google.common.util.concurrent.ListenableFuture<EngineRpcService.WorkerStatusPB>>();
                    for (int i=0; i<2048; i++) calls.add(endpoint.stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()));
                    await(() -> endpoint.engine.inFlight() > 1024);
                    service.release.countDown(); cache.get(5, TimeUnit.SECONDS);
                    for (var call : calls) assertTrue(call.get(10, TimeUnit.SECONDS).getAlive());
                    await(() -> endpoint.engine.inFlight() == 0);
                } finally { service.release.countDown(); }
            }
        }
    }
    @Test void streamingCancellationReleasesResourcesWhileAnotherHandlerBlocks() throws Exception {
        try (var pools = new MockRpcExecutors(1, 8)) {
            var owned = new AtomicInteger();
            var started = new CountDownLatch(1); var cleaned = new CountDownLatch(1);
            var service = new Service() {
                @Override public void generateStreamCall(EngineRpcService.GenerateInputPB request,
                        io.grpc.stub.StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
                    owned.incrementAndGet();
                    Context.current().addListener(ignored -> { owned.decrementAndGet(); cleaned.countDown(); }, Runnable::run);
                    started.countDown();
                }
            };
            try (var endpoint = new Endpoint(pools, service)) {
                try {
                    var stream = new AtomicReference<io.grpc.stub.ClientCallStreamObserver<EngineRpcService.GenerateInputPB>>();
                    RpcServiceGrpc.newStub(endpoint.channel).generateStreamCall(EngineRpcService.GenerateInputPB.getDefaultInstance(),
                            new io.grpc.stub.ClientResponseObserver<EngineRpcService.GenerateInputPB, EngineRpcService.GenerateOutputsPB>() {
                                @Override public void beforeStart(io.grpc.stub.ClientCallStreamObserver<EngineRpcService.GenerateInputPB> call) { stream.set(call); }
                                @Override public void onNext(EngineRpcService.GenerateOutputsPB value) { }
                                @Override public void onError(Throwable error) { }
                                @Override public void onCompleted() { }
                            });
                    assertTrue(started.await(2, TimeUnit.SECONDS));
                    var cache = endpoint.stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
                    assertTrue(service.entered.await(2, TimeUnit.SECONDS));
                    stream.get().cancel("client left", null);
                    assertTrue(cleaned.await(2, TimeUnit.SECONDS));
                    assertEquals(0, owned.get());
                    assertEquals(1, service.release.getCount(), "cleanup must not wait for the cache handler");
                    service.release.countDown(); cache.get(2, TimeUnit.SECONDS);
                } finally { service.release.countDown(); }
            }
        }
    }

    @Test void incompleteShutdownIsReportedAndCanBeRetried() throws Exception {
        var pools = new MockRpcExecutors(1, 2);
        var exited = new CountDownLatch(1);
        var service = new Service() {
            @Override public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                    io.grpc.stub.StreamObserver<EngineRpcService.CacheStatusPB> observer) {
                entered.countDown();
                try {
                    while (release.getCount() > 0) {
                        try { release.await(); } catch (InterruptedException ignored) { }
                    }
                } finally { exited.countDown(); }
            }
        };
        try (var endpoint = new Endpoint(pools, service)) {
            endpoint.stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
            assertTrue(service.entered.await(2, TimeUnit.SECONDS));
            endpoint.channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            endpoint.server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            long begin = System.nanoTime();
            assertThrows(IllegalStateException.class, pools::close);
            assertTrue(System.nanoTime() - begin < TimeUnit.SECONDS.toNanos(8));
            service.release.countDown();
            assertTrue(exited.await(2, TimeUnit.SECONDS));
            pools.close(); pools.close();
        } finally { service.release.countDown(); pools.close(); }
    }

}
