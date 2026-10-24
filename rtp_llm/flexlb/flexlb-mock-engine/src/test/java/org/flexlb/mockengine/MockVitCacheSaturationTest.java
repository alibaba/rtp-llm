package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.ManagedChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(15)
class MockVitCacheSaturationTest {
    private static MockVitService service(long gpu, long cpu) {
        return service(gpu, cpu, 8192);
    }
    private static MockVitService service(long gpu, long cpu, long hash) {
        var document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        ((ObjectNode)document.path("roles").path("vit"))
                .put("vit_concurrency", 1).put("vit_max_queue_size", 4)
                .put("mm_cache_gpu_max_bytes", gpu).put("mm_cache_cpu_max_bytes", cpu)
                .put("mm_hash_key_cache_max_bytes", hash);
        return new MockVitService(new MockVitService.Config(8, 1, 4, cpu, hash, 1 << 20, 1 << 20, 0), null, document);
    }
    @SuppressWarnings("unchecked")
    private static MockVitEmbeddingCache<MockVitImage.Result> cache(MockVitService service) throws Exception {
        var field = service.getClass().getDeclaredField("embeddingCache"); field.setAccessible(true);
        return (MockVitEmbeddingCache<MockVitImage.Result>)field.get(service);
    }
    @Test void absentHashSidecarDoesNotReplaceEmbeddingGenerationOnConsumption() throws Exception {
        try (var service = service(128, 128, 0)) {
            var image = MockVitServiceTest.image(56, 56, 1);
            service.acquire(image, 3000).get(3, TimeUnit.SECONDS);
            String generation = cache(service).get(MockVitImage.key(image)).generation();
            service.acquire(image, 3000).get(3, TimeUnit.SECONDS);
            assertEquals(generation, cache(service).get(MockVitImage.key(image)).generation());
            assertEquals(1, service.computations.get());
        }
    }
    @Test void waitingColdConsumerDoesNotHoldServiceAdmissionMonitor() throws Exception {
        var finish = new CompletableFuture<Void>();
        try (var service = service(256, 512)) {
            var a = MockVitServiceTest.image(56, 56, 1);
            var b = MockVitServiceTest.image(56, 56, 2);
            var c = MockVitServiceTest.image(56, 56, 3);
            var d = MockVitServiceTest.image(56, 56, 4);
            for (var image : List.of(a, b, c)) service.acquire(image, 3000).get(3, TimeUnit.SECONDS);
            var copying = new CompletableFuture<Void>();
            cache(service).copier = (operation, value) -> {
                if (operation == MockVitEmbeddingCache.Copy.STORE) { copying.complete(null); finish.join(); }
            };
            var producer = service.acquire(d, 3000);
            try {
                copying.get(2, TimeUnit.SECONDS);
                var entered = new CompletableFuture<Void>();
                var cold = CompletableFuture.supplyAsync(() -> {
                    entered.complete(null); return service.acquire(a, 1000).join();
                });
                entered.get(1, TimeUnit.SECONDS);
                var hot = CompletableFuture.supplyAsync(() -> service.acquire(c, 1000).join());
                assertNotNull(hot.get(300, TimeUnit.MILLISECONDS));
                var error = assertThrows(java.util.concurrent.ExecutionException.class, () -> cold.get(2, TimeUnit.SECONDS));
                assertEquals(io.grpc.Status.Code.DEADLINE_EXCEEDED, io.grpc.Status.fromThrowable(error).getCode());
            } finally { finish.complete(null); producer.get(3, TimeUnit.SECONDS); }
        } finally { finish.complete(null); }
    }
    @Test void rpcResultsAndHashesSurviveSaturationPromotionAndOptionalCopyFailure() throws Exception {
        try (var service = service(128, 128)) {
            var server = NettyServerBuilder.forPort(0).addService(service).build().start();
            var channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
            try {
                var stub = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                var a = MockVitServiceTest.image(56, 56, 1); var b = MockVitServiceTest.image(56, 56, 2);
                var c = MockVitServiceTest.image(56, 56, 3);
                var first = stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(a));
                var second = stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(b));
                var probe = EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys(MockVitImage.key(a)).build();
                var metadata = stub.getMultimodalHashes(probe).getEntries(0);
                assertEquals("cpu", metadata.getEmbeddingTier());
                assertEquals("cpu", cache(service).tier(MockVitImage.key(a)), "metadata must not promote or touch residency");
                assertEquals(first, stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(a)));
                assertEquals(2, service.computations.get());
                stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(c));
                var evicted = stub.getMultimodalHashes(probe.toBuilder().clearKeys().addKeys(MockVitImage.key(b)).build()).getEntries(0);
                assertTrue(evicted.getHashHit()); assertFalse(evicted.getEmbeddingHit());
                assertEquals(128, service.snapshot().get("gpu_pool_used_bytes"));
                assertEquals(128, service.snapshot().get("cpu_pool_used_bytes"));
                // A subsequent cache copy error cannot turn the successful RPC into an error or stale hash output.
                cache(service).copier = (operation, value) -> {
                    if (operation == MockVitEmbeddingCache.Copy.STORE) throw new IllegalStateException("injected pool copy error");
                };
                var d = MockVitServiceTest.image(56, 56, 4);
                var copied = stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(d));
                assertEquals(List.of(4L, 8L), copied.getMultimodalEmbedding().getShapeList());
                assertFalse(cache(service).contains(MockVitImage.key(d)));
                assertEquals(copied, stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(d)));
                assertTrue(service.snapshot().get("cache_transfer_errors") > 0);
                assertEquals(128, first.getMultimodalEmbedding().getFp32Data().size());
                assertEquals(128, second.getMultimodalEmbedding().getFp32Data().size());
            } finally {
                channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
                server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            }
        }
    }
    @Test void deadlineDoesNotRecycleStorageBeforeItsReleaseEventOrInvalidateOtherReaders() throws Exception {
        var released = new CompletableFuture<Void>();
        var service = service(0, 128);
        var server = NettyServerBuilder.forPort(0).addService(service).build().start();
        var channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        try {
            var a = MockVitServiceTest.image(56, 56, 1); var b = MockVitServiceTest.image(56, 56, 2);
            service.acquire(a, 3000).get(3, TimeUnit.SECONDS);
            MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS)
                    .remoteMultimodalEmbedding(MockVitServiceTest.inputs(a));
            cache(service).releaseEvents = value -> List.of(released);
            var stub = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(150, TimeUnit.MILLISECONDS);
            var error = assertThrows(io.grpc.StatusRuntimeException.class,
                    () -> stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(b)));
            assertEquals(io.grpc.Status.Code.DEADLINE_EXCEEDED, error.getStatus().getCode());
            assertEquals(128, service.snapshot().get("cpu_pool_used_bytes"));
            assertEquals(1, service.snapshot().get("active"), "running insertion retains its task until copy completion");
            assertFalse(cache(service).contains(MockVitImage.key(b)), "unpublished entry must not be returned as a hit");
            released.complete(null);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            while (service.snapshot().get("active") != 0 && System.nanoTime() < deadline) Thread.sleep(1);
            assertEquals(0, service.snapshot().get("active"));
            assertEquals(128, service.snapshot().get("cpu_pool_used_bytes"));
            assertTrue(cache(service).contains(MockVitImage.key(b)));
        } finally {
            released.complete(null);
            channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            service.close();
        }
    }
}
