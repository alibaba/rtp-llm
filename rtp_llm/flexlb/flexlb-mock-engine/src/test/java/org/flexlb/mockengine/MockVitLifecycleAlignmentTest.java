package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.Status;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(15)
class MockVitLifecycleAlignmentTest {
    private static MockVitService service(boolean cache, int concurrency, int queue) {
        var document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        ((ObjectNode)document.path("roles").path("vit")).put("vit_concurrency", concurrency)
                .put("vit_max_queue_size", queue).put("mm_preprocess_max_workers", 1).put("mm_cache_gpu_max_bytes", 0)
                .put("mm_cache_cpu_max_bytes", cache ? 8192 : 0).put("mm_hash_key_cache_max_bytes", 8192);
        return new MockVitService(new MockVitService.Config(8, concurrency, queue, cache ? 8192 : 0,
                8192, 1 << 20, 1 << 20, 0), null, document);
    }
    private static EngineRpcService.MultimodalInputsPB input(long id, int color) throws Exception {
        return MockVitServiceTest.inputs(MockVitServiceTest.image(56, 56, color)).toBuilder().setRequestId(id).build();
    }
    private static <T> T field(MockVitService service, String name, Class<T> type) throws Exception {
        var field = service.getClass().getDeclaredField(name); field.setAccessible(true); return type.cast(field.get(service));
    }
    private static CompletableFuture<EngineRpcService.EmptyPB> submit(MockVitService service, EngineRpcService.MultimodalInputsPB input) {
        var result = new CompletableFuture<EngineRpcService.EmptyPB>();
        service.asyncSubmitEmbedding(input, new io.grpc.stub.StreamObserver<>() {
            public void onNext(EngineRpcService.EmptyPB value) { result.complete(value); }
            public void onError(Throwable error) { result.completeExceptionally(error); }
            public void onCompleted() {}
        }); return result;
    }
    private static CompletableFuture<EngineRpcService.MultimodalHashResponsePB> hashes(MockVitService service, EngineRpcService.MultimodalHashRequestPB input) {
        var result = new CompletableFuture<EngineRpcService.MultimodalHashResponsePB>();
        service.getMultimodalHashes(input, new io.grpc.stub.StreamObserver<>() {
            public void onNext(EngineRpcService.MultimodalHashResponsePB value) { result.complete(value); }
            public void onError(Throwable error) { result.completeExceptionally(error); }
            public void onCompleted() {}
        }); return result;
    }
    private static void await(java.util.function.BooleanSupplier done) throws Exception {
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!done.getAsBoolean() && System.nanoTime() < until) Thread.sleep(1);
        assertTrue(done.getAsBoolean());
    }
    @Test void disabledEmbeddingCacheDoesNotDeduplicateEvenWithinOneBatch() throws Exception {
        try (var service = service(false, 2, 2)) {
            var image = MockVitServiceTest.image(56, 56, 1);
            var first = service.acquire(image, 3000); var second = service.acquire(image, 3000);
            first.get(3, TimeUnit.SECONDS); second.get(3, TimeUnit.SECONDS);
            assertEquals(2, service.computations.get());
            assertEquals(0, service.snapshot().get("embedding_bytes"));
        }
    }
    @Test void asyncAckDoesNotInheritSubmittingRpcDeadlineOrBackgroundFailure() throws Exception {
        try (var service = service(true, 1, 3)) {
            var gpu = field(service, "gpuScheduler", MockVitScheduler.class);
            var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
            var blocked = gpu.submit(() -> { entered.countDown(); release.await(); return null; });
            try {
                assertTrue(entered.await(1, TimeUnit.SECONDS));
                var input = input(11, 1).toBuilder().setTimeoutMs(1).build();
                submit(service, input).get(1, TimeUnit.SECONDS);
                Thread.sleep(30); release.countDown(); blocked.get(1, TimeUnit.SECONDS);
                await(() -> service.computations.get() == 1);
                var invalid = input(12, 2).toBuilder().setMultimodalInputs(0,
                        input(12, 2).getMultimodalInputs(0).toBuilder().setMultimodalUrl("invalid://image")).build();
                submit(service, invalid).get(1, TimeUnit.SECONDS);
                await(() -> service.snapshot().get("active") == 0 && service.snapshot().get("queued") == 0);
            } finally { release.countDown(); }
        }
    }
    @Test void requestIdSetOwnsSharedQueuedWorkAndRepeatedClaimsDoNotPinIt() throws Exception {
        try (var service = service(true, 1, 4)) {
            var gpu = field(service, "gpuScheduler", MockVitScheduler.class);
            var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
            gpu.submit(() -> { entered.countDown(); release.await(); return null; });
            try {
                assertTrue(entered.await(1, TimeUnit.SECONDS));
                submit(service, input(10, 1)).get(); await(() -> service.snapshot().get("active") == 1);
                var queued = input(20, 2); submit(service, queued).get();
                submit(service, queued).get(); // same request id is one owner
                submit(service, queued.toBuilder().setRequestId(30).build()).get();
                assertEquals(1, service.snapshot().get("queued"));
                service.cancelRequest(20); assertEquals(1, service.snapshot().get("queued"));
                service.cancelRequest(30); assertEquals(0, service.snapshot().get("queued"));
                assertEquals(1, service.snapshot().get("pending"));
                service.cancelRequest(10); assertEquals(1, service.snapshot().get("active"), "running work retains ownership");
            } finally { release.countDown(); }
        }
    }
    @Test void hashMissingInputsAndWholeBatchOverloadStartNoPartialWork() throws Exception {
        try (var service = service(true, 1, 0)) {
            var a = input(1, 1); var b = input(1, 2);
            var request = EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys(MockVitImage.key(a.getMultimodalInputs(0)))
                    .addKeys(MockVitImage.key(b.getMultimodalInputs(0))).setInputs(a).build();
            var invalid = assertThrows(ExecutionException.class, () -> hashes(service, request).get()).getCause();
            assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(invalid).getCode());
            assertEquals(0, service.snapshot().get("pending")); assertEquals(0, service.computations.get());
            var full = request.toBuilder().setInputs(a.toBuilder().addAllMultimodalInputs(b.getMultimodalInputsList())).build();
            var overload = assertThrows(ExecutionException.class, () -> hashes(service, full).get()).getCause();
            assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(overload).getCode());
            assertEquals(0, service.snapshot().get("pending")); assertEquals(0, service.computations.get());
        }
    }
    @Test void hashResponsePreservesDuplicateKeysAndFreshFillHasNoGeneration() throws Exception {
        try (var service = service(true, 1, 2)) {
            var input = input(1, 1); String key = MockVitImage.key(input.getMultimodalInputs(0));
            var request = EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys(key).addKeys(key).setInputs(input).build();
            var filled = hashes(service, request).get(3, TimeUnit.SECONDS);
            assertEquals(2, filled.getEntriesCount()); assertEquals("", filled.getEntries(0).getEntryGeneration());
            var hit = hashes(service, request.toBuilder().clearInputs().build()).get();
            assertEquals(2, hit.getEntriesCount()); assertFalse(hit.getEntries(0).getEntryGeneration().isEmpty());
            assertEquals(1, service.computations.get());
        }
    }
    @Test void partialExecutorFailureSettlesUnsubmittedClaimsAndPreservesRunningWork() throws Exception {
        try (var service = service(true, 1, 3)) {
            var original = service.submitAsync; var calls = new AtomicInteger();
            var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
            var pool = field(service, "preprocessor", MockVitPreprocessor.class);
            var blocking = pool.submit(() -> { entered.countDown(); release.await(); return null; });
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            service.submitAsync = task -> {
                if (calls.incrementAndGet() == 2) throw new RejectedExecutionException("second submit failed");
                original.accept(task);
                try { await(() -> service.snapshot().get("active") == 1); } catch (Exception error) { throw new RuntimeException(error); }
            };
            try {
                var batch = input(1, 1).toBuilder().addAllMultimodalInputs(input(1, 2).getMultimodalInputsList()).build();
                assertThrows(ExecutionException.class, () -> submit(service, batch).get());
                assertEquals(1, service.snapshot().get("pending")); assertEquals(0, service.snapshot().get("queued"));
            } finally { release.countDown(); blocking.get(1, TimeUnit.SECONDS); }
            await(() -> service.snapshot().get("pending") == 0);
        }
    }
    @Test void rejectedBatchDoesNotCancelEarlierWorkWithTheSameRequestId() throws Exception {
        try (var service = service(true, 1, 2)) {
            var gpu = field(service, "gpuScheduler", MockVitScheduler.class);
            var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
            gpu.submit(() -> { entered.countDown(); release.await(); return null; });
            try {
                assertTrue(entered.await(1, TimeUnit.SECONDS));
                submit(service, input(1, 1)).get(); await(() -> service.snapshot().get("active") == 1);
                submit(service, input(77, 2)).get();
                var batch = input(77, 3).toBuilder().addAllMultimodalInputs(input(77, 4).getMultimodalInputsList()).build();
                assertThrows(ExecutionException.class, () -> submit(service, batch).get());
                assertEquals(1, service.snapshot().get("queued")); assertEquals(2, service.snapshot().get("pending"));
            } finally { release.countDown(); }
            await(() -> service.computations.get() == 2);
        }
    }
    @Test void asyncSubmitWithDisabledEmbeddingCacheDoesNotPopulateHashesUntilConsumption() throws Exception {
        try (var service = service(false, 1, 2)) {
            var input = input(1, 1); String key = MockVitImage.key(input.getMultimodalInputs(0));
            submit(service, input).get(); await(() -> service.snapshot().get("active") == 0 && service.computations.get() == 1);
            var probe = EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys(key).build();
            assertFalse(hashes(service, probe).get().getEntries(0).getHashHit());
            assertTrue(hashes(service, probe.toBuilder().setInputs(input).build()).get().getEntries(0).getHashHit());
            assertTrue(hashes(service, probe).get().getEntries(0).getHashHit());
            assertEquals(2, service.computations.get());
        }
    }
    @Test void shutdownSettlesQueuedWaitersWhileRunningWorkDrains() throws Exception {
        var service = service(true, 1, 2);
        var gpu = field(service, "gpuScheduler", MockVitScheduler.class);
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        gpu.submit(() -> { entered.countDown(); release.await(); return null; });
        CompletableFuture<Void> closing = null;
        try {
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            submit(service, input(1, 1)).get(); await(() -> service.snapshot().get("active") == 1);
            var queued = input(2, 2); submit(service, queued).get();
            var waiter = hashes(service, EngineRpcService.MultimodalHashRequestPB.newBuilder()
                    .addKeys(MockVitImage.key(queued.getMultimodalInputs(0))).setInputs(queued).build());
            closing = CompletableFuture.runAsync(service::close);
            var error = assertThrows(ExecutionException.class, () -> waiter.get(1, TimeUnit.SECONDS)).getCause();
            assertEquals(Status.Code.CANCELLED, Status.fromThrowable(error).getCode());
            assertFalse(closing.isDone());
        } finally {
            release.countDown();
            if (closing != null) closing.get(3, TimeUnit.SECONDS); else service.close();
        }
        assertEquals(0, service.snapshot().get("pending"));
    }
}
