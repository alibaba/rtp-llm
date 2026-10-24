package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

class MockVitSchedulerTest {
    @Test void expiredAsyncWaiterNeverStartsGpuWork() throws Exception {
        var document = (com.fasterxml.jackson.databind.node.ObjectNode)MockEngineConfigFixture.load().snapshot();
        var limits = (com.fasterxml.jackson.databind.node.ObjectNode)document.path("roles").path("vit");
        limits.put("vit_concurrency", 1).put("vit_max_queue_size", 2)
                .put("mm_cache_cpu_max_bytes", 4096).put("mm_hash_key_cache_max_bytes", 8192);
        try (var service = new MockVitService(new MockVitService.Config(8, 1, 2, 4096, 8192, 1 << 20, 1 << 20, 200), null, document)) {
            var first = service.acquire(MockVitServiceTest.image(56, 56, 101), 3000);
            var expired = service.acquire(MockVitServiceTest.image(56, 56, 202), 10);
            first.get(3, TimeUnit.SECONDS);
            assertThrows(java.util.concurrent.ExecutionException.class, () -> expired.get(3, TimeUnit.SECONDS));
            assertEquals(1, service.computations.get());
            assertEquals(0L, service.snapshot().get("pending"));
        }
    }
    @Test void cancellingLastWaiterReclaimsQueuedAsyncWork() throws Exception {
        var document = (com.fasterxml.jackson.databind.node.ObjectNode)MockEngineConfigFixture.load().snapshot();
        var limits = (com.fasterxml.jackson.databind.node.ObjectNode)document.path("roles").path("vit");
        limits.put("vit_concurrency", 1).put("vit_max_queue_size", 2).put("mm_cache_cpu_max_bytes", 4096).put("mm_hash_key_cache_max_bytes", 8192);
        try (var service = new MockVitService(new MockVitService.Config(8, 1, 2, 4096, 8192, 1 << 20, 1 << 20, 300), null, document);
             var cancelled = io.grpc.Context.current().withCancellation()) {
            var completed = new java.util.concurrent.CompletableFuture<org.flexlb.engine.grpc.EngineRpcService.MultimodalOutputPB>();
            var first = MockVitServiceTest.image(56, 56, 123);
            var second = MockVitServiceTest.image(56, 56, 456);
            var firstRequest = org.flexlb.engine.grpc.EngineRpcService.MultimodalInputsPB.newBuilder()
                    .setRequestId(1).setTimeoutMs(3000).addMultimodalInputs(first).build();
            service.remoteMultimodalEmbedding(firstRequest, new io.grpc.stub.StreamObserver<>() {
                public void onNext(org.flexlb.engine.grpc.EngineRpcService.MultimodalOutputPB value) { completed.complete(value); }
                public void onError(Throwable error) { completed.completeExceptionally(error); }
                public void onCompleted() { }
            });
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (service.snapshot().get("active") == 0 && System.nanoTime() < deadline) Thread.sleep(1);
            assertEquals(1L, service.snapshot().get("active"));
            cancelled.run(() -> service.remoteMultimodalEmbedding(firstRequest.toBuilder().setRequestId(2)
                    .clearMultimodalInputs().addMultimodalInputs(second).build(), new io.grpc.stub.StreamObserver<>() {
                public void onNext(org.flexlb.engine.grpc.EngineRpcService.MultimodalOutputPB value) { fail("cancelled waiter received a response"); }
                public void onError(Throwable error) { }
                public void onCompleted() { }
            }));
            assertEquals(1L, service.snapshot().get("queued"));
            cancelled.cancel(null);
            assertEquals(0L, service.snapshot().get("queued"));
            completed.get(3, TimeUnit.SECONDS);
            assertEquals(1, service.computations.get(), "the cancelled queued image must never compute");
            service.acquire(second, 3000).get(3, TimeUnit.SECONDS);
            assertEquals(2, service.computations.get(), "cancel must remove its stale pending entry");
        }
    }

    @Test void gpuCancellationKeepsItsQueueEntryUntilTheConsumerSkipsIt() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        try (var scheduler = new MockVitScheduler(1, 1, 200, 0, 0)) {
            var first = scheduler.submit(() -> { entered.countDown(); release.await(); return null; });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var queued = scheduler.submit(() -> null);
            assertTrue(scheduler.submit(() -> null).isCompletedExceptionally());
            assertTrue(queued.cancel(false));
            var replacement = scheduler.submit(() -> null);
            assertTrue(replacement.isCompletedExceptionally(), "real Queue.qsize includes cancelled entries until pulled");
            release.countDown();
            first.get(2, TimeUnit.SECONDS);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (scheduler.queued() != 0 && System.nanoTime() < deadline) Thread.sleep(1);
            scheduler.submit(() -> null).get(2, TimeUnit.SECONDS);
            assertEquals(2, scheduler.batches());
        } finally { release.countDown(); }
    }

    @Test void zeroCollectionWindowDoesNotMergeAlreadyWaitingRequests() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        try (var scheduler = new MockVitScheduler(8, 4, 200, 0, 0)) {
            var running = scheduler.submit(() -> { entered.countDown(); release.await(); return null; });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var second = scheduler.submit(() -> null);
            var third = scheduler.submit(() -> null);
            assertFalse(running.cancel(false), "real Future cannot cancel a forward already claimed RUNNING");
            release.countDown();
            running.get(2, TimeUnit.SECONDS); second.get(2, TimeUnit.SECONDS); third.get(2, TimeUnit.SECONDS);
            assertEquals(3, scheduler.batches(), "real _collect_batch stops when remaining <= 0");
        } finally { release.countDown(); }
    }

    @Test void aForwardFailureFailsItsWholeBatch() throws Exception {
        try (var scheduler = new MockVitScheduler(8, 2, 200, 100, 0)) {
            var first = scheduler.submit(() -> null);
            var second = scheduler.submit(() -> { throw new IllegalStateException("forward failed"); });
            assertThrows(java.util.concurrent.ExecutionException.class, () -> first.get(2, TimeUnit.SECONDS));
            assertThrows(java.util.concurrent.ExecutionException.class, () -> second.get(2, TimeUnit.SECONDS));
            scheduler.submit(() -> null).get(2, TimeUnit.SECONDS);
            assertEquals(2, scheduler.batches(), "subsequent batches continue");
        }
    }

    @Test void forwardBatchRespectsRequestAndImageLimits() throws Exception {
        try (var scheduler = new MockVitScheduler(8, 4, 2, 100, 0)) {
            var first = scheduler.submit(() -> null);
            var second = scheduler.submit(() -> null);
            first.get(2, TimeUnit.SECONDS); second.get(2, TimeUnit.SECONDS);
            assertEquals(1, scheduler.batches());
        }
    }

    private static MockVitScheduler.Item item(long patches, java.util.concurrent.Callable<MockVitImage.Result> forward) {
        return new MockVitScheduler.Item(forward, 1, new MockVitWorkEstimate(patches, 0, 0, 0, 0));
    }

    @Test void costChunksContinueAtTheTailEvenWhenNewcomerQueueIsFull() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var order = new java.util.concurrent.CopyOnWriteArrayList<Integer>();
        try (var scheduler = new MockVitScheduler(1, 4, 8, 0, 0, new MockVitWorkEstimate(10, 0, 0, 0, 0))) {
            var chunks = scheduler.submitItems(java.util.List.of(
                    item(8, () -> { entered.countDown(); release.await(); order.add(1); return null; }),
                    item(8, () -> { order.add(3); return null; })));
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var newcomer = scheduler.submitItems(java.util.List.of(item(1, () -> { order.add(2); return null; })));
            assertTrue(scheduler.submitItems(java.util.List.of(item(1, () -> null))).isCompletedExceptionally());
            release.countDown();
            assertEquals(2, chunks.get(2, TimeUnit.SECONDS).size());
            newcomer.get(2, TimeUnit.SECONDS);
            assertEquals(java.util.List.of(1, 2, 3), order);
            assertEquals(3, scheduler.batches());
        } finally { release.countDown(); }
    }

    @Test void overBudgetIndivisibleCostRunsAloneButOversizeImagesReject() throws Exception {
        var samples = new java.util.concurrent.CopyOnWriteArrayList<Number>();
        try (var scheduler = new MockVitScheduler(8, 8, 4, 50, 0, new MockVitWorkEstimate(10, 0, 0, 0, 0))) {
            scheduler.setMetrics(values -> { if (values.containsKey("py_rtp_vit_embedding_batch_size")) samples.add(values.get("py_rtp_vit_embedding_batch_size")); });
            var huge = scheduler.submitItems(java.util.List.of(item(100, () -> null)));
            var small = scheduler.submitItems(java.util.List.of(item(1, () -> null)));
            huge.get(2, TimeUnit.SECONDS); small.get(2, TimeUnit.SECONDS);
            assertEquals(java.util.List.of(1, 1), samples);
            assertTrue(scheduler.submitItems(java.util.List.of(new MockVitScheduler.Item(() -> null, 5, MockVitWorkEstimate.ZERO))).isCompletedExceptionally());
            assertTrue(scheduler.submitItems(java.util.List.of(new MockVitScheduler.Item(() -> null, 1, null))).isCompletedExceptionally());
        }
    }

    @Test void requestTimeoutDuringForwardSuppressesContinuationWithoutInterruptingForward() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        try (var scheduler = new MockVitScheduler(1, 4, 8, 0, 0, new MockVitWorkEstimate(10, 0, 0, 0, 0))) {
            var chunks = scheduler.submitItems(java.util.List.of(
                    item(8, () -> { entered.countDown(); release.await(); calls.incrementAndGet(); return null; }),
                    item(8, () -> { calls.incrementAndGet(); return null; })));
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            assertFalse(chunks.cancel(false));
            scheduler.cancelRequest(chunks);
            release.countDown();
            assertEquals(1, chunks.get(2, TimeUnit.SECONDS).size(), "the running chunk finishes, caller has already timed out");
            assertEquals(1, calls.get());
            assertEquals(0, scheduler.queued());
        } finally { release.countDown(); }
    }

    @Test void failedChunkDoesNotPublishPartialResultsOrRunItsContinuation() throws Exception {
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        try (var scheduler = new MockVitScheduler(4, 4, 8, 0, 0, new MockVitWorkEstimate(10, 0, 0, 0, 0))) {
            var chunks = scheduler.submitItems(java.util.List.of(
                    item(8, () -> { throw new IllegalStateException("failed"); }),
                    item(8, () -> { calls.incrementAndGet(); return null; })));
            assertThrows(java.util.concurrent.ExecutionException.class, () -> chunks.get(2, TimeUnit.SECONDS));
            scheduler.submitItems(java.util.List.of(item(1, () -> null))).get(2, TimeUnit.SECONDS);
            assertEquals(0, calls.get());
        }
    }

    @Test void attentionSegmentIsMaximumAndZeroBudgetDimensionsAreUnbounded() {
        var a = new MockVitWorkEstimate(3, 4, 5, 6, 7);
        assertEquals(new MockVitWorkEstimate(6, 8, 10, 6, 14), a.plus(a));
        assertTrue(a.plus(a).fits(new MockVitWorkEstimate(0, 0, 0, 6, 0)));
        assertFalse(a.plus(a).fits(new MockVitWorkEstimate(0, 0, 0, 5, 0)));
    }

    @Test void realEmbeddingTimeoutAndPerInputOverrideAreSeparateFromRpcDeadline() throws Exception {
        var document = (com.fasterxml.jackson.databind.node.ObjectNode)MockEngineConfigFixture.load().snapshot();
        var vit = (com.fasterxml.jackson.databind.node.ObjectNode)document.path("roles").path("vit");
        vit.put("vit_concurrency", 1).put("vit_max_queue_size", 4).put("mm_timeout_ms", 10);
        var config = new MockVitService.Config(8, 1, 4, vit.path("mm_cache_cpu_max_bytes").longValue(),
                vit.path("mm_hash_key_cache_max_bytes").longValue(), 1 << 20, 1 << 20, 100);
        try (var service = new MockVitService(config, null, document)) {
            var first = service.acquire(MockVitServiceTest.image(28, 28, 0x111111), 3000);
            var failure = assertThrows(java.util.concurrent.ExecutionException.class, () -> first.get(2, TimeUnit.SECONDS));
            assertEquals(io.grpc.Status.Code.DEADLINE_EXCEEDED, io.grpc.Status.fromThrowable(failure.getCause()).getCode());
            var image = MockVitServiceTest.image(28, 28, 0x222222).toBuilder();
            image.setMmPreprocessConfig(image.getMmPreprocessConfig().toBuilder().setMmTimeoutMs(1000));
            assertNotNull(service.acquire(image.build(), 3000).get(2, TimeUnit.SECONDS));
        }
    }

    @Test void cacheTiersDemoteAndPromoteWithoutRecomputing() throws Exception {
        var document = (com.fasterxml.jackson.databind.node.ObjectNode)MockEngineConfigFixture.load().snapshot();
        var limits = (com.fasterxml.jackson.databind.node.ObjectNode)document.path("roles").path("vit");
        limits.put("vit_concurrency", 2).put("vit_max_queue_size", 4).put("mm_hash_key_cache_max_bytes", 8192);
        limits.put("mm_cache_gpu_max_bytes", 128).put("mm_cache_cpu_max_bytes", 128);
        try (var service = new MockVitService(new MockVitService.Config(8, 2, 4, 128, 8192, 1 << 20, 1 << 20, 0), null, document)) {
            var a = MockVitServiceTest.image(56, 56, 123);
            var b = MockVitServiceTest.image(56, 56, 456);
            service.acquire(a, 2000).get(2, TimeUnit.SECONDS);
            service.acquire(b, 2000).get(2, TimeUnit.SECONDS);
            assertTrue(service.snapshot().get("gpu_embedding_bytes") > 0);
            assertTrue(service.snapshot().get("embedding_bytes") > 0);
            service.acquire(a, 2000).get(2, TimeUnit.SECONDS);
            assertEquals(2, service.computations.get());
            assertTrue(service.snapshot().get("gpu_embedding_bytes") <= 128);
            assertTrue(service.snapshot().get("embedding_bytes") <= 128);
        }
    }
}
