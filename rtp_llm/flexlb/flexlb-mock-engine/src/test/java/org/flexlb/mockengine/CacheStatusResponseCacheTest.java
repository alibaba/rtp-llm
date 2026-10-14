package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class CacheStatusResponseCacheTest {
    @Test
    void realTransportDispatchesCacheOutsideStatusPoolAndKeepsProtobufContract() throws Exception {
        var pools = new MockRpcExecutors(1, 4);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var response = EngineRpcService.CacheStatusPB.newBuilder().setVersion(7).putCacheKeys(9L, true).build();
        var service = new RpcServiceGrpc.RpcServiceImplBase() {
            @Override
            public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                       io.grpc.stub.StreamObserver<EngineRpcService.CacheStatusPB> observer) {
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
                .executor(pools.executor(RpcServiceGrpc.getGetWorkerStatusMethod().getFullMethodName()))
                .callExecutor(pools).addService(CacheStatusResponseCache.bind(service)).build().start();
        var channel = io.grpc.netty.NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        try {
            var stub = RpcServiceGrpc.newFutureStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
            var cache = stub.getCacheStatus(EngineRpcService.CacheVersionPB.newBuilder()
                    .setLatestCacheVersion(7).setNeedCacheKeys(true).build());
            assertTrue(entered.await(1, TimeUnit.SECONDS));
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
    void snapshotsKeepMembershipAndDoNotInvalidateForReferencesOrLruTouches() {
        var device = new MockLruBlockCache(16, 0);
        device.admit(List.of(1L, 2L, 3L));
        var original = device.snapshotKeys();
        var lease = device.acquireWithReuse(1, List.of(1L));
        assertSame(original, device.snapshotKeys());
        device.release(lease);
        assertSame(original, device.snapshotKeys());
        device.admit(List.of(1L, 2L, 3L));
        assertSame(original, device.snapshotKeys());
        device.evict(List.of(3L));
        assertEquals(Set.of(1L, 2L, 3L), original);
        assertEquals(Set.of(1L, 2L), device.snapshotKeys());
        device.clear();
        assertTrue(device.snapshotKeys().isEmpty());

        var memory = new MockMemoryBlockCache(16, ignored -> {});
        memory.write(List.of(1L, 2L, 3L));
        var host = memory.keySnapshot();
        try (var read = memory.pinRead(List.of(1L), 0)) {
            assertSame(host, memory.keySnapshot());
        }
        memory.write(List.of(1L, 2L, 3L));
        assertSame(host, memory.keySnapshot());
        memory.pinRead(List.of(1L, 2L, 3L), 0).consume();
        assertEquals(Set.of(1L, 2L, 3L), host);
        assertTrue(memory.keySnapshot().isEmpty());
    }

    @Test
    void fullResponseAndEncodingAreReusableButCapacityAndMembershipStayFresh() throws Exception {
        var cache = new CacheStatusResponseCache();
        var device = Set.of(1L, 2L);
        var host = Set.of(2L, 3L);
        var first = cache.response(100, 200, 128, 7, device, host);
        assertSame(first, cache.response(100, 200, 128, 7, device, host));
        assertEquals(Set.of(1L, 2L, 3L), first.getCacheKeysMap().keySet());
        var changed = cache.response(90, 200, 128, 7, device, host);
        assertEquals(90, changed.getAvailableKvCache());
        assertEquals(first.getCacheKeysMap(), changed.getCacheKeysMap());
        assertNotSame(first, changed);
        assertEquals(Set.of(1L, 2L), cache.response(90, 200, 128, 8, device, Set.of()).getCacheKeysMap().keySet());
        assertTrue(cache.response(90, 200, 128, 8, null, Set.of()).getCacheKeysMap().isEmpty());
        var marshaller = new CacheStatusResponseCache.Marshaller(
                RpcServiceGrpc.getGetCacheStatusMethod().getResponseMarshaller());
        var a = marshaller.stream(first);
        var b = marshaller.stream(first);
        a.read();
        assertEquals(first, EngineRpcService.CacheStatusPB.parseFrom(b));
        assertEquals(changed, marshaller.parse(marshaller.stream(changed)));
    }

    @Test
    void blockedCacheWorkDoesNotOccupyStatusOrCallerAndQueuesAreBounded() throws Exception {
        var pools = new MockRpcExecutors(1, 1);
        var entered = new CountDownLatch(1);
        var unblock = new CountDownLatch(1);
        var status = new CountDownLatch(1);
        try {
            var cache = pools.executor(RpcServiceGrpc.getGetCacheStatusMethod().getFullMethodName());
            cache.execute(() -> {
                entered.countDown();
                try { unblock.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            });
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            cache.execute(() -> {});
            assertThrows(java.util.concurrent.RejectedExecutionException.class, () -> cache.execute(() -> fail("caller must not run")));
            pools.executor(RpcServiceGrpc.getGetWorkerStatusMethod().getFullMethodName()).execute(status::countDown);
            assertTrue(status.await(1, TimeUnit.SECONDS));
        } finally { unblock.countDown(); pools.close(); }
    }
}
