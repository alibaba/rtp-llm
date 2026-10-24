package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(10)
class MockVitProtocolMetricsTest {
    private static <T> T field(MockVitService service, String name, Class<T> type) throws Exception {
        var field = service.getClass().getDeclaredField(name); field.setAccessible(true); return type.cast(field.get(service));
    }
    private static void await(java.util.function.BooleanSupplier done) throws Exception {
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!done.getAsBoolean() && System.nanoTime() < until) Thread.sleep(1);
        assertTrue(done.getAsBoolean());
    }
    @Test void backgroundFailureCarriesDomainAndReportedMarkerAndIsCountedOnceForSharedWaiters() throws Exception {
        try (var service = MockVitTestSupport.service(MockVitService.Config.defaults())) {
            var errors = new AtomicInteger(); var qps = new AtomicInteger();
            service.setSchedulerMetrics(values -> {
                if (values.containsKey("py_rtp_vit_error_qps")) errors.incrementAndGet();
                if (values.containsKey("py_rtp_vit_qps")) qps.incrementAndGet();
            });
            var pool = field(service, "preprocessor", MockVitPreprocessor.class);
            var release = new CountDownLatch(1);
            pool.dispatch = (executor, task) -> executor.execute(() -> {
                try { release.await(); task.run(); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
            });
            var invalid = MockVitServiceTest.image(56, 56, 1).toBuilder().setMultimodalUrl("invalid://image").build();
            var ack = new CompletableFuture<EngineRpcService.EmptyPB>();
            service.asyncSubmitEmbedding(MockVitServiceTest.inputs(invalid), new io.grpc.stub.StreamObserver<>() {
                public void onNext(EngineRpcService.EmptyPB value) { ack.complete(value); }
                public void onError(Throwable error) { ack.completeExceptionally(error); }
                public void onCompleted() {}
            });
            try {
                ack.get(1, TimeUnit.SECONDS);
                var first = service.acquire(invalid, 3000); var second = service.acquire(invalid, 3000);
                release.countDown();
                for (var work : java.util.List.of(first, second)) {
                    var error = assertThrows(ExecutionException.class, () -> work.get(3, TimeUnit.SECONDS)).getCause();
                    assertTrue(MockVitErrors.reported(error));
                    assertTrue(MockRpcErrors.downstream(error).getErrorCode() != 0);
                }
                assertEquals(1, errors.get()); assertEquals(1, qps.get());
            } finally { release.countDown(); }
        }
    }
    @Test void cacheAccessAndResidencyAreEventMetricsAndHitsDoNotReportComputeQps() throws Exception {
        try (var service = MockVitTestSupport.service(MockVitService.Config.defaults())) {
            var values = new ConcurrentHashMap<String, Number>();
            var events = new ConcurrentHashMap<String, AtomicInteger>();
            service.setSchedulerMetrics(samples -> samples.forEach((key, value) -> {
                values.put(key, value); events.computeIfAbsent(key, ignored -> new AtomicInteger()).incrementAndGet();
            }));
            var image = MockVitServiceTest.image(56, 56, 1);
            service.acquire(image, 3000).get(3, TimeUnit.SECONDS);
            @SuppressWarnings("unchecked") var cache = (MockVitEmbeddingCache<MockVitImage.Result>)field(service, "embeddingCache", MockVitEmbeddingCache.class);
            cache.copier = (operation, value) -> {
                if (operation == MockVitEmbeddingCache.Copy.READ) assertTrue(events.containsKey("py_rtp_vit_embedding_cache_hit_qps"), "claim metric precedes consumption");
            };
            service.acquire(image, 3000).get(3, TimeUnit.SECONDS);
            assertEquals(1, events.get("py_rtp_vit_qps").get());
            assertEquals(1, events.get("py_rtp_vit_success_qps").get());
            assertEquals(1, events.get("py_rtp_vit_embedding_cache_miss_qps").get());
            assertEquals(1, events.get("py_rtp_vit_embedding_cache_hit_qps").get());
            assertEquals(4, values.get("py_rtp_vit_embedding_cache_tokens").longValue());
            assertEquals(1024, values.get("py_rtp_vit_embedding_cache_bytes").longValue());
            cache.copier = (operation, value) -> { if (operation == MockVitEmbeddingCache.Copy.READ) throw new IllegalStateException("read failed"); };
            for (int i = 0; i < 2; i++) assertThrows(ExecutionException.class, () -> service.acquire(image, 3000).get());
            assertEquals(1, events.get("py_rtp_vit_error_qps").get(), "same entry error is reported once");
            cache.remove(MockVitImage.key(image));
            assertEquals(0, values.get("py_rtp_vit_embedding_cache_bytes").longValue());
        }
    }
    @Test void hashMetadataRowLimitIsCheckedOnPureCacheProbes() throws Exception {
        var document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        ((ObjectNode)document.path("roles").path("vit")).put("mm_hash_key_cache_max_bytes", 8 << 20);
        var defaults = MockVitService.Config.defaults();
        try (var service = new MockVitService(defaults, null, document)) {
            var cache = field(service, "hashCache", MockVitHashCache.class);
            var tensor = EngineRpcService.TensorPB.newBuilder().setInt32Data(com.google.protobuf.ByteString.copyFrom(new byte[1048577 * 4])).build();
            cache.put("big", new MockVitImage.Result(EngineRpcService.TensorPB.getDefaultInstance(), tensor, 1048577), "g");
            var result = new CompletableFuture<EngineRpcService.MultimodalHashResponsePB>();
            service.getMultimodalHashes(EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys("big").build(), new io.grpc.stub.StreamObserver<>() {
                public void onNext(EngineRpcService.MultimodalHashResponsePB value) { result.complete(value); }
                public void onError(Throwable error) { result.completeExceptionally(error); }
                public void onCompleted() {}
            });
            var error = assertThrows(ExecutionException.class, result::get).getCause();
            assertEquals(io.grpc.Status.Code.RESOURCE_EXHAUSTED, io.grpc.Status.fromThrowable(error).getCode());
        }
    }
    @Test void httpFailureRetainsDownloadDomainAndReportingMarker() throws Exception {
        var origin = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        origin.createContext("/missing", exchange -> { exchange.sendResponseHeaders(404, -1); exchange.close(); });
        origin.start();
        try (var service = MockVitTestSupport.service(MockVitService.Config.defaults())) {
            var image = MockVitServiceTest.image(56, 56, 1).toBuilder()
                    .setMultimodalUrl("http://127.0.0.1:" + origin.getAddress().getPort() + "/missing").build();
            var error = assertThrows(ExecutionException.class, () -> service.acquire(image, 3000).get(3, TimeUnit.SECONDS)).getCause();
            assertEquals(906, MockRpcErrors.downstream(error).getErrorCode());
            assertTrue(MockVitErrors.reported(error));
        } finally { origin.stop(0); }
    }
    @Test void callerCancellationAndLaterComputeTimeoutShareOneErrorReport() throws Exception {
        try (var service = MockVitTestSupport.service(MockVitService.Config.defaults())) {
            var errors = new AtomicInteger();
            service.setSchedulerMetrics(values -> { if (values.containsKey("py_rtp_vit_error_qps")) errors.incrementAndGet(); });
            var gpu = field(service, "gpuScheduler", MockVitScheduler.class);
            var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
            gpu.submit(() -> { entered.countDown(); release.await(); return null; });
            try {
                assertTrue(entered.await(1, TimeUnit.SECONDS));
                var context = io.grpc.Context.current().withCancellation();
                var input = MockVitServiceTest.inputs(MockVitServiceTest.image(56, 56, 1)).toBuilder().setRequestId(55).setTimeoutMs(100).build();
                context.run(() -> service.remoteMultimodalEmbedding(input, new io.grpc.stub.StreamObserver<>() {
                    public void onNext(EngineRpcService.MultimodalOutputPB value) {}
                    public void onCompleted() {}
                    public void onError(Throwable error) {}
                }));
                await(() -> service.snapshot().get("active") == 1);
                context.cancel(null);
                await(() -> service.snapshot().get("active") == 0);
                assertEquals(1, errors.get());
            } finally { release.countDown(); }
        }
    }
}
