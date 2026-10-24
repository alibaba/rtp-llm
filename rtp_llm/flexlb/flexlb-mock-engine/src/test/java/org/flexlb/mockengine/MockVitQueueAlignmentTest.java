package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(15)
class MockVitQueueAlignmentTest {
    private static MockVitService service() {
        var document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        ((ObjectNode)document.path("roles").path("vit")).put("vit_concurrency", 1).put("vit_max_queue_size", 0)
                .put("mm_max_queue_size", 1).put("gpu_max_batch_size", 4).put("gpu_max_batch_images", 2)
                .put("use_local_preprocess", false).put("mm_preprocess_max_workers", 1).put("gpu_batch_wait_ms", 0).put("mm_cache_cpu_max_bytes", 4096).put("mm_hash_key_cache_max_bytes", 8192);
        return new MockVitService(new MockVitService.Config(8, 1, 0, 4096, 8192, 1 << 20, 1 << 20, 0),
                MockV41ImageTest.config(), document);
    }
    private static MockVitScheduler gpu(MockVitService service) throws Exception {
        var field = MockVitService.class.getDeclaredField("gpuScheduler"); field.setAccessible(true);
        return (MockVitScheduler)field.get(service);
    }
    private static EngineRpcService.MultimodalInputsPB typed(int count) {
        var images = EngineRpcService.V41RequestInputsPB.newBuilder().setSchemaVersion(1);
        for (int i = 0; i < count; i++) images.addImages(MockV41ImageTest.image(String.format("%064x", i + 1))
                .toBuilder().setStart(i * 5 + 1));
        return EngineRpcService.MultimodalInputsPB.newBuilder().setTimeoutMs(3000).setV41Inputs(images).build();
    }
    private static CompletableFuture<EngineRpcService.MultimodalOutputPB> rpc(MockVitService service, EngineRpcService.MultimodalInputsPB input) {
        var result = new CompletableFuture<EngineRpcService.MultimodalOutputPB>();
        service.remoteMultimodalEmbedding(input, new StreamObserver<>() {
            public void onNext(EngineRpcService.MultimodalOutputPB value) { result.complete(value); }
            public void onError(Throwable error) { result.completeExceptionally(error); }
            public void onCompleted() { }
        });
        return result;
    }
    private static void await(java.util.function.BooleanSupplier condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) Thread.sleep(1);
        assertTrue(condition.getAsBoolean());
    }
    @Test void typedBypassesAsyncAdmissionAndKeepsAllImagesInOneSchedulerRequest() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var service = service();
        try {
            var gpu = gpu(service);
            var running = gpu.submit(() -> { entered.countDown(); release.await(); return null; });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var url = service.acquire(MockVitServiceTest.image(56, 56, 11), 3000);
            await(() -> gpu.queued() == 1);
            var blockedUrl = service.acquire(MockVitServiceTest.image(56, 56, 12), 3000);
            assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(
                    assertThrows(java.util.concurrent.ExecutionException.class, blockedUrl::get).getCause()).getCode());
            release.countDown(); running.get(3, TimeUnit.SECONDS); url.get(3, TimeUnit.SECONDS);
            var first = rpc(service, typed(2)).get(3, TimeUnit.SECONDS);
            assertEquals(java.util.List.of(5, 5), first.getSplitSizeList());
            assertEquals(3, gpu.batches(), "two typed images occupy one scheduler request");
            assertEquals(3, service.computations.get());
            assertEquals(first, rpc(service, typed(2)).get(3, TimeUnit.SECONDS));
            assertEquals(4, gpu.batches(), "model-cache hits still enter the GPU scheduler");
            assertEquals(3, service.computations.get(), "forward reuses trusted model-cache values");
        } finally { release.countDown(); service.close(); }
    }
    @Test void typedIsAdmittedWhileAsyncCapacityIsOccupied() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var prepared = new CountDownLatch(1); var prepareRelease = new CountDownLatch(1);
        var service = service();
        try {
            var gpu = gpu(service);
            var running = gpu.submit(() -> { entered.countDown(); release.await(); return null; });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            // Keep async work in preprocessing rather than filling the GPU waiting slot.
            var field = MockVitService.class.getDeclaredField("preprocessor"); field.setAccessible(true);
            var preprocessor = (MockVitPreprocessor)field.get(service);
            var preprocessing = preprocessor.submit(() -> { prepared.countDown(); prepareRelease.await(); return null; });
            assertTrue(prepared.await(2, TimeUnit.SECONDS));
            var url = service.acquire(MockVitServiceTest.image(56, 56, 11), 3000);
            await(() -> service.snapshot().get("active") == 1);
            var typed = rpc(service, typed(2));
            await(() -> gpu.queued() == 1);
            assertFalse(typed.isCompletedExceptionally(), "typed must not consume async C+Q admission");
            release.countDown(); running.get(3, TimeUnit.SECONDS);
            typed.get(3, TimeUnit.SECONDS); prepareRelease.countDown();
            preprocessing.get(3, TimeUnit.SECONDS); url.get(3, TimeUnit.SECONDS);
        } finally { release.countDown(); prepareRelease.countDown(); service.close(); }
    }
    @Test void typedRejectsWholeOversizeOrMalformedRequestBeforeForward() throws Exception {
        try (var service = service()) {
            var oversize = assertThrows(java.util.concurrent.ExecutionException.class, () -> rpc(service, typed(3)).get()).getCause();
            assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(oversize).getCode());
            assertEquals(902, MockRpcErrors.downstream(oversize).getErrorCode());
            var malformed = typed(2).toBuilder();
            malformed.getV41InputsBuilder().getImagesBuilder(1).setProcessorIdentity("wrong");
            assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(
                    assertThrows(java.util.concurrent.ExecutionException.class, () -> rpc(service, malformed.build()).get()).getCause()).getCode());
            assertEquals(0, service.computations.get()); assertEquals(0, gpu(service).batches());
        }
    }
    @Test void cancelledTypedQueueEntryIsSkippedWithoutForward() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var service = service();
        try (var context = Context.current().withCancellation()) {
            var gpu = gpu(service);
            var running = gpu.submit(() -> { entered.countDown(); release.await(); return null; });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            context.run(() -> rpc(service, typed(2)));
            assertEquals(1, gpu.queued()); context.cancel(null);
            assertEquals(1, gpu.queued(), "cancelled entries remain counted until consumed");
            release.countDown(); running.get(3, TimeUnit.SECONDS);
            await(() -> gpu.queued() == 0);
            assertEquals(0, service.computations.get());
            rpc(service, typed(2)).get(3, TimeUnit.SECONDS);
            assertEquals(2, service.computations.get(), "later requests can reacquire capacity");
        } finally { release.countDown(); service.close(); }
    }
    @Test void typedQueueOverloadHasRetryableProcessErrorButAsyncAdmissionDoesNot() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var service = service();
        try {
            var gpu = gpu(service);
            var running = gpu.submit(() -> { entered.countDown(); release.await(); return null; });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var waiting = rpc(service, typed(1));
            var overload = assertThrows(java.util.concurrent.ExecutionException.class, () -> rpc(service, typed(1)).get()).getCause();
            assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(overload).getCode());
            assertEquals(903, MockRpcErrors.downstream(overload).getErrorCode());
            assertTrue(MockVitClient.retryable(overload));
            release.countDown(); running.get(3, TimeUnit.SECONDS); waiting.get(3, TimeUnit.SECONDS);
        } finally { release.countDown(); service.close(); }
    }

    @Test void typedModelCacheIsAnItemLruAndDoesNotPopulateUrlCaches() throws Exception {
        try (var service = service()) {
            var metadata = (java.util.Map<?, ?>)service.effectiveConfiguration().get("model");
            int capacity = ((Number)metadata.get("V41_IMAGE_ENCODE_CACHE_ITEMS")).intValue();
            var first = typed(1);
            for (int i = 0; i <= capacity; i++) {
                var next = first.toBuilder();
                next.getV41InputsBuilder().getImagesBuilder(0).setContentSha256(String.format("%064x", i + 1));
                rpc(service, next.build()).get(3, TimeUnit.SECONDS);
            }
            assertEquals(capacity + 1, service.computations.get());
            rpc(service, first).get(3, TimeUnit.SECONDS);
            assertEquals(capacity + 2, service.computations.get(), "oldest item was evicted by item count");
            var untrusted = first.toBuilder();
            untrusted.getV41InputsBuilder().getImagesBuilder(0).clearContentSha256();
            rpc(service, untrusted.build()).get(3, TimeUnit.SECONDS);
            rpc(service, untrusted.build()).get(3, TimeUnit.SECONDS);
            assertEquals(capacity + 4, service.computations.get(), "empty hashes must never share encoded values");
            assertEquals(0L, service.snapshot().get("embedding_bytes"));
            assertEquals(0L, service.snapshot().get("hash_bytes"));
        }
    }

}
