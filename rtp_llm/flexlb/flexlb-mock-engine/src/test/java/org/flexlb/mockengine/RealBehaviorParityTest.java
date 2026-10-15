package org.flexlb.mockengine;

import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.*;

import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

/** Behavioral vectors from DecodeRpcServer, CoordinatorCacheManager and PrefillBatchRpcServer. */
@Timeout(10)
class RealBehaviorParityTest {
    @TempDir Path directory;

    private static MockLruBlockCache cache(JavaMockEngineCluster.FastRpcService service) throws Exception {
        var field = service.getClass().getDeclaredField("cache");
        field.setAccessible(true);
        return (MockLruBlockCache) field.get(service);
    }

    private static MockPerformanceModel.RequestShape shape(int length, long timeout) {
        var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(90)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setTimeoutMs((int) timeout)).build();
        return new MockPerformanceModel.RequestShape(input, length, 2, List.of(), 0, 0, false);
    }

    @Test void temporaryKvPressureWaitsPastLegacyWindowAndWakesOnRelease() throws Exception {
        var model = performanceModel(directory, "1");
        try (var c = MockEngineTestCluster.create(model, 62000, 1, 1);
             var worker = Executors.newSingleThreadExecutor()) {
            var d = c.decode(0); var pool = cache(d);
            int demand = pool.totalBlocks() / 2;
            var held = pool.acquire(demand, List.of());
            var pending = worker.submit(() -> d.reserveDecodeLease(90, shape(demand * 1024, 3000)));
            Thread.sleep(180);
            assertFalse(pending.isDone(), "temporary capacity is not a 100ms stage failure");
            pool.release(held);
            assertTrue(pending.get(2, TimeUnit.SECONDS));
            d.cancel(90);
            assertEquals(pool.totalBlocks(), pool.availableBlocks());
        }
    }

    @Test void permanentKvPressureFailsWithoutRetryingAndCancellationStopsWait() throws Exception {
        try (var c = MockEngineTestCluster.create(performanceModel(directory, "1"), 62000, 1, 1);
             var worker = Executors.newSingleThreadExecutor()) {
            var d = c.decode(0); var pool = cache(d);
            long start = System.nanoTime();
            assertFalse(d.reserveDecodeLease(90, shape((pool.totalBlocks() + 1) * 1024, 3000)));
            assertTrue(System.nanoTime() - start < TimeUnit.SECONDS.toNanos(1));
            var held = pool.acquire(pool.totalBlocks() - pool.reserveBlocks(), List.of());
            try (var context = Context.current().withCancellation()) {
                var pending = worker.submit(() -> context.call(() -> d.reserveDecodeLease(90, shape(1024, 0))));
                Thread.sleep(50); assertFalse(pending.isDone());
                context.cancel(null);
                assertFalse(pending.get(1, TimeUnit.SECONDS));
            } finally { pool.release(held); }
            assertEquals(pool.totalBlocks(), pool.availableBlocks());
        }
    }

    @Test void localRequestDeadlineProducesTypedFailureAndReleasesBothPools() throws Exception {
        try (var c = MockEngineTestCluster.create(performanceModel(directory, "500"), 62000, 1, 1)) {
            var input = inputWithDecode(90, 1024, c.decode(0).getGrpcPort(), 8);
            input = input.toBuilder().setGenerateConfig(input.getGenerateConfig().toBuilder().setTimeoutMs(40)).build();
            Output result = new Output();
            c.prefill(0).generateStreamCall(input, result);
            assertTrue(result.done.await(2, TimeUnit.SECONDS));
            assertNotNull(result.failure);
            assertEquals(Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(result.failure).getCode());
            assertEquals(603, MockRpcErrors.downstream(result.failure).getErrorCode());
            for (var service : c.services().values()) assertEquals(0, cache(service).heldBlocks());
            Thread.sleep(550);
            for (var service : c.services().values()) assertTrue(service.getSnapshot().get("active_decode_requests").equals(0));
        }
    }

    @Test void deadlineDuringLocalAllocateKeepsTimeoutCauseAndCannotAcquireLate() throws Exception {
        try (var c = MockEngineTestCluster.create(performanceModel(directory, "1"), 62000, 1, 1)) {
            var pool = cache(c.decode(0));
            var held = pool.acquire(pool.totalBlocks() - pool.reserveBlocks(), List.of());
            try {
                var input = inputWithDecode(90, 1024, c.decode(0).getGrpcPort(), 8);
                input = input.toBuilder().setGenerateConfig(input.getGenerateConfig().toBuilder().setTimeoutMs(50)).build();
                var result = new Output(); c.prefill(0).generateStreamCall(input, result);
                assertTrue(result.done.await(2, TimeUnit.SECONDS));
                assertEquals(Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(result.failure).getCode());
                assertEquals(603, MockRpcErrors.downstream(result.failure).getErrorCode());
                assertEquals(0, cache(c.prefill(0)).heldBlocks());
                assertEquals(0, c.decode(0).getAcceptedCount());
            } finally { pool.release(held); }
        }
    }

    @Test void priorityCancelCanStopLocalAllocateWithoutWaitingForCapacity() throws Exception {
        try (var c = MockEngineTestCluster.create(performanceModel(directory, "1"), 62000, 1, 1);
             var worker = Executors.newSingleThreadExecutor()) {
            var pool = cache(c.decode(0));
            var held = pool.acquire(pool.totalBlocks() - pool.reserveBlocks(), List.of());
            try {
                var pending = worker.submit(() -> enqueue(c.prefill(0), batch(90, slot(0,
                        inputWithDecode(90, 1024, c.decode(0).getGrpcPort(), 8)))));
                var field = c.prefill(0).getClass().getDeclaredField("prefillRegistrations");
                field.setAccessible(true);
                long limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
                while (!((java.util.Set<?>) field.get(c.prefill(0))).contains(90L) && System.nanoTime() < limit) Thread.sleep(1);
                assertTrue(c.prefill(0).cancelRequest(90).found());
                var ack = pending.get(1, TimeUnit.SECONDS);
                assertEquals(0, ack.getSuccessesCount());
                assertEquals(8429, ack.getErrors(0).getErrorInfo().getErrorCode());
                assertEquals(0, c.decode(0).getAcceptedCount());
                assertEquals(0, cache(c.prefill(0)).heldBlocks());
            } finally { pool.release(held); }
        }
    }

    @Test void crashCannotCarryOldRequestDeadlineIntoTheNextGeneration() throws Exception {
        try (var c = MockEngineTestCluster.create(performanceModel(directory, "100"), 62000, 1, 1)) {
            var old = input(90, 32).toBuilder().setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                    .setMaxNewTokens(2).setTimeoutMs(60)).build();
            c.prefill(0).generateStreamCall(old, new Output());
            c.prefill(0).crashNow(); c.prefill(0).setStopped(false);
            var fresh = old.toBuilder().setGenerateConfig(old.getGenerateConfig().toBuilder().setTimeoutMs(1000)).build();
            var output = new Output(); c.prefill(0).generateStreamCall(fresh, output);
            assertTrue(output.done.await(2, TimeUnit.SECONDS)); assertNull(output.failure);
            assertTrue(output.frames.stream().noneMatch(EngineRpcService.GenerateOutputsPB::hasErrorInfo));
            assertTrue(output.frames.getLast().getFlattenOutput().getFinished(0));
        }
    }

    @Test void nativePrefixReuseAlwaysLeavesATokenAndHonorsRequestReuse() throws Exception {
        var model = performanceModel(directory, "1"); model.nativeTokenCacheKeys = true;
        var pool = new MockLruBlockCache(10);
        var input = input(90, 1024).toBuilder().setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                .setMaxNewTokens(2).setReuseCache(true)).build();
        pool.admit(model.shape(input, pool).blockKeys());
        assertEquals(0, model.shape(input, pool).hitTokens());
        var extended = input.toBuilder().addTokenIds(9).build();
        assertEquals(1024, model.shape(extended, pool).hitTokens());
        var disabled = extended.toBuilder().setGenerateConfig(extended.getGenerateConfig().toBuilder().setReuseCache(false)).build();
        assertEquals(0, model.shape(disabled, pool).hitTokens());
        assertTrue(MockCachePolicy.reuse(input, true, true));
        assertFalse(MockCachePolicy.reuse(input, true, false));
    }

    @Test void localTierPolicyDoesNotUseWireTierFlagsAndDecodeHonorsReuse() throws Exception {
        var model = performanceModel(directory, "1"); model.nativeTokenCacheKeys = true;
        try (var c = MockEngineTestCluster.create(model, 62000, 1, 1)) {
            var input = input(90, 1025).toBuilder().setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                    .setMaxNewTokens(1).setReuseCache(true).setEnableDeviceCache(false)).build();
            var output = new Output(); c.prefill(0).generateStreamCall(input, output);
            assertTrue(output.done.await(2, TimeUnit.SECONDS)); assertNull(output.failure);
            assertFalse(cache(c.prefill(0)).snapshotKeys().isEmpty(), "deployment enables device storage despite wire flag=false");
            var pool = cache(c.decode(0)); pool.admit(List.of(7L));
            var disabled = new MockPerformanceModel.RequestShape(input.toBuilder().setGenerateConfig(
                    input.getGenerateConfig().toBuilder().setReuseCache(false)).build(), 1025, 1, List.of(7L), 0, 0, true);
            assertTrue(c.decode(0).reserveDecodeLease(90, disabled));
            assertEquals(0, pool.referencedKeyBlocks());
            assertEquals(2, pool.heldBlocks()); c.decode(0).cancel(90);
        }
    }

    @Test void typedDownstreamErrorsSurviveAndBareTransportUsesFallback() {
        var error = MockRpcErrors.error(Status.RESOURCE_EXHAUSTED, 602, "physical capacity");
        assertEquals(602, MockRpcErrors.downstream(error).getErrorCode());
        assertEquals("physical capacity", MockRpcErrors.downstream(error).getErrorMessage());
        assertEquals(603, MockRpcErrors.downstream(MockRpcErrors.error(Status.DEADLINE_EXCEEDED, 603, "deadline")).getErrorCode());
        assertEquals(8211, MockRpcErrors.downstream(Status.RESOURCE_EXHAUSTED.asRuntimeException()).getErrorCode());
    }

    @Test void invalidBatchNeverPartiallyAdmitsDuplicateOrWrongDpRank() throws Exception {
        try (var c = MockEngineTestCluster.create(performanceModel(directory, "1"), 62000, 1, 1)) {
            var input = inputWithDecode(90, 32, c.decode(0).getGrpcPort(), 2);
            var duplicate = enqueue(c.prefill(0), batch(1, slot(0, input, input)));
            assertEquals(0, duplicate.getSuccessesCount()); assertEquals(2, duplicate.getErrorsCount());
            assertEquals(6, duplicate.getErrors(0).getErrorInfo().getErrorCode());
            var rank = enqueue(c.prefill(0), batch(2, slot(9, input)));
            assertEquals(0, rank.getSuccessesCount()); assertEquals(3, rank.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, cache(c.decode(0)).heldBlocks());
            assertTrue(c.prefill(0).getRequestStates().isEmpty());
        }
    }

    @Test void oneTokenSettlesDecodeReservationWithoutRunningDecode() throws Exception {
        try (var c = MockEngineTestCluster.create(performanceModel(directory, "1"), 62000, 1, 1)) {
            var output = new Output();
            c.prefill(0).generateStreamCall(inputWithDecode(90, 32, c.decode(0).getGrpcPort(), 1), output);
            assertTrue(output.done.await(2, TimeUnit.SECONDS)); assertNull(output.failure);
            assertEquals(0, c.stats().decodeAdmitted.sum());
            assertEquals(0, cache(c.decode(0)).heldBlocks());
        }
    }

    static final class Output implements StreamObserver<EngineRpcService.GenerateOutputsPB> {
        final CountDownLatch done = new CountDownLatch(1);
        final List<EngineRpcService.GenerateOutputsPB> frames = new CopyOnWriteArrayList<>();
        volatile Throwable failure;
        public void onNext(EngineRpcService.GenerateOutputsPB value) { frames.add(value); }
        public void onError(Throwable error) { failure = error; done.countDown(); }
        public void onCompleted() { done.countDown(); }
    }
}
