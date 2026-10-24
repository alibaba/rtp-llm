package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class MockRpcExecutorsTest {
    @Test
    void transportKeepsMethodsOffNettyAndUsesGeneratedProtobufContract() throws Exception {
        var pools = new MockRpcExecutors(2);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var methodThread = new java.util.concurrent.atomic.AtomicReference<String>();
        var response = EngineRpcService.CacheStatusPB.newBuilder().setVersion(7).putCacheKeys(9L, true).build();
        var service = new RpcServiceGrpc.RpcServiceImplBase() {
            @Override
            public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                       io.grpc.stub.StreamObserver<EngineRpcService.CacheStatusPB> observer) {
                methodThread.set(Thread.currentThread().getName());
                entered.countDown();
                try { release.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                observer.onNext(response); observer.onCompleted();
            }
            @Override
            public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                        io.grpc.stub.StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                observer.onNext(EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).build());
                observer.onCompleted();
            }
        };
        var server = io.grpc.netty.NettyServerBuilder.forPort(0)
                .executor(pools.executor())
                .addService(service).build().start();
        var channel = io.grpc.netty.NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        try {
            var stub = RpcServiceGrpc.newFutureStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
            var cache = stub.getCacheStatus(EngineRpcService.CacheVersionPB.newBuilder()
                    .setLatestCacheVersion(7).setNeedCacheKeys(true).build());
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            assertTrue(methodThread.get().startsWith("mock-rpc-"));
            assertTrue(stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance())
                    .get(1, TimeUnit.SECONDS).getAlive());
            release.countDown();
            assertEquals(response, cache.get(1, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            pools.close();
        }
    }
    @Test
    void sharedPoolPreservesCompetitionAndQueuesCallbacksWithoutRunningOnCaller() throws Exception {
        var pools = new MockRpcExecutors(1);
        var entered = new CountDownLatch(1);
        var unblock = new CountDownLatch(1);
        var status = new CountDownLatch(4096);
        try {
            var cache = pools.executor();
            cache.execute(() -> {
                entered.countDown();
                try { unblock.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            });
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            for (int i = 0; i < 4096; i++) pools.executor().execute(status::countDown);
            assertFalse(status.await(100, TimeUnit.MILLISECONDS));
            assertEquals(4096, status.getCount(), "callbacks must not run on the submitting thread");
            unblock.countDown();
            assertTrue(status.await(1, TimeUnit.SECONDS));
        } finally { unblock.countDown(); pools.close(); }
    }

    @Test
    void transportDrainsCallbackBurstBeyondFormerQueueLimit() throws Exception {
        var pools = new MockRpcExecutors(1);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var queued = new CountDownLatch(2048);
        var service = new RpcServiceGrpc.RpcServiceImplBase() {
            @Override
            public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                       io.grpc.stub.StreamObserver<EngineRpcService.CacheStatusPB> observer) {
                entered.countDown();
                try { release.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                observer.onNext(EngineRpcService.CacheStatusPB.getDefaultInstance());
                observer.onCompleted();
            }
            @Override
            public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                        io.grpc.stub.StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                observer.onNext(EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).build());
                observer.onCompleted();
            }
        };
        var server = io.grpc.netty.NettyServerBuilder.forPort(0)
                .executor(command -> {
                    pools.executor().execute(command);
                    if (entered.getCount() == 0) queued.countDown();
                }).addService(service).build().start();
        var channel = io.grpc.netty.NettyChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext().build();
        try {
            var stub = RpcServiceGrpc.newFutureStub(channel).withDeadlineAfter(10, TimeUnit.SECONDS);
            var cache = stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance());
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var calls = new java.util.ArrayList<com.google.common.util.concurrent.ListenableFuture<EngineRpcService.WorkerStatusPB>>();
            for (int i = 0; i < 2048; i++)
                calls.add(stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()));
            assertTrue(queued.await(5, TimeUnit.SECONDS), "transport must queue the entire callback burst");
            assertTrue(calls.stream().noneMatch(java.util.concurrent.Future::isDone));
            release.countDown();
            cache.get(5, TimeUnit.SECONDS);
            for (var call : calls) assertTrue(call.get(5, TimeUnit.SECONDS).getAlive());
        } finally {
            release.countDown();
            channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            pools.close();
        }
    }
}
