package org.flexlb.mockengine;

import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.*;

import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class DecodeLinkBreakTest {
    @TempDir Path tempDir;

    private static class Output implements StreamObserver<EngineRpcService.GenerateOutputsPB> {
        final List<EngineRpcService.GenerateOutputsPB> frames = new CopyOnWriteArrayList<>();
        final CountDownLatch done = new CountDownLatch(1);
        volatile Throwable error;
        public void onNext(EngineRpcService.GenerateOutputsPB frame) { frames.add(frame); }
        public void onError(Throwable t) { error = t; done.countDown(); }
        public void onCompleted() { done.countDown(); }
        void broken() throws Exception {
            assertTrue(done.await(2, TimeUnit.SECONDS), "P stream must terminate without client deadline");
            assertNull(error);
            var errors = frames.stream().filter(EngineRpcService.GenerateOutputsPB::hasErrorInfo).toList();
            assertEquals(1, errors.size());
            assertEquals(8209, errors.get(0).getErrorInfo().getErrorCodeValue());
            assertFalse(errors.get(0).getErrorInfo().getErrorMessage().isBlank());
            assertFalse(frames.stream().anyMatch(f -> f.getFlattenOutput().getFinishedList().contains(true)));
        }
    }

    private Output submit(MockEngineTestCluster c, long id) {
        var ack = enqueue(c.prefill(0), batch(id, slot(0,
                inputWithDecode(id, 512, c.decode(0).getGrpcPort(), 20))));
        assertEquals(1, ack.getSuccessesCount(), ack.toString());
        Output out = new Output();
        c.prefill(0).fetchResponse(EngineRpcService.FetchRequestPB.newBuilder().setRequestId(id).build(), out);
        return out;
    }

    private void death(int kind, boolean beforeHandoff) throws Exception {
        try (var c = MockEngineTestCluster.create(
                performanceModel(tempDir, beforeHandoff ? "1000" : "1", 1, 100), 62100, 1, 1)) {
            Output out = submit(c, 1);
            if (!beforeHandoff) {
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
                while (out.frames.isEmpty() && System.nanoTime() < deadline) Thread.sleep(1);
                assertFalse(out.frames.isEmpty(), "must reach P->D handoff before death");
                c.awaitNoInflight(c.prefill(0), 2000);
            }
            switch (kind) {
                case 0 -> c.decode(0).setStopped(true);
                case 1 -> c.decode(0).crashNow();
                case 2 -> c.decode(0).drainAndShutdown();
                default -> throw new AssertionError();
            }
            out.broken();
            c.awaitAllInflightZero(2000);
            assertEquals(0, ((Number) c.prefill(0).getSnapshot().get("response_buffers")).intValue());
            c.decode(0).setStopped(true); // repeated teardown must be harmless
            assertEquals(1, out.frames.stream().filter(EngineRpcService.GenerateOutputsPB::hasErrorInfo).count());
        }
    }

    @Test void stopAfterHandoff() throws Exception { death(0, false); }
    @Test void crashAfterHandoff() throws Exception { death(1, false); }
    @Test void forcedDrainAfterHandoff() throws Exception { death(2, false); }
    @Test void stopDuringPrefill() throws Exception { death(0, true); }
    @Test void deathDuringPrefill() throws Exception { death(1, true); }

    @Test void handoffRacingStopAlwaysTerminates() throws Exception {
        for (int i = 0; i < 25; i++) {
            try (var c = MockEngineTestCluster.create(
                    performanceModel(tempDir, "1", 1, 100), 62100, 1, 1)) {
                Output out = submit(c, 1);
                c.decode(0).setStopped(true);
                out.broken();
            }
        }
    }

    @Test void decodeCancelRacingDeathDoesNotLoseTerminal() throws Exception {
        ExecutorService workers = Executors.newFixedThreadPool(2);
        try (var c = MockEngineTestCluster.create(
                performanceModel(tempDir, "1", 1, 100), 62100, 1, 1)) {
            for (int i = 0; i < 25; i++) {
                long id = 100 + i;
                c.decode(0).setStopped(false);
                Output out = submit(c, id);
                c.awaitNoInflight(c.prefill(0), 2000);
                CyclicBarrier start = new CyclicBarrier(2);
                Future<?> cancel = workers.submit(() -> {
                    start.await(); c.decode(0).cancel(id); return null;
                });
                Future<?> stop = workers.submit(() -> {
                    start.await(); c.decode(0).setStopped(true); return null;
                });
                cancel.get(2, TimeUnit.SECONDS);
                stop.get(2, TimeUnit.SECONDS);
                assertTrue(out.done.await(2, TimeUnit.SECONDS));
                assertNull(out.error);
                var errors = out.frames.stream().filter(EngineRpcService.GenerateOutputsPB::hasErrorInfo).toList();
                assertEquals(1, errors.size());
                int code = errors.get(0).getErrorInfo().getErrorCodeValue();
                assertTrue(code == 8209 || code == EngineRpcService.ErrorCodePB.CANCELLED_VALUE);
            }
        } finally { workers.shutdownNow(); }
    }

    @Test void completionAndCancellationRaceCannotAppendAfterLinkBreak() throws Exception {
        ExecutorService workers = Executors.newFixedThreadPool(3);
        try {
            for (int i = 0; i < 200; i++) {
                MockResponseQueue queue = new MockResponseQueue();
                CyclicBarrier start = new CyclicBarrier(3);
                var success = EngineRpcService.GenerateOutputsPB.newBuilder()
                        .setFlattenOutput(EngineRpcService.FlattenOutputPB.newBuilder().addFinished(true)).build();
                var cancel = EngineRpcService.GenerateOutputsPB.newBuilder().setErrorInfo(
                        EngineRpcService.RpcErrorPB.newBuilder().setErrorCodeValue(1)).build();
                var broken = EngineRpcService.GenerateOutputsPB.newBuilder().setErrorInfo(
                        EngineRpcService.RpcErrorPB.newBuilder().setErrorCodeValue(8209)).build();
                var futures = List.of(success, cancel, broken).stream().map(frame -> workers.submit(() -> {
                    start.await();
                    queue.offer(frame);
                    return null;
                })).toList();
                for (var f : futures) f.get(2, TimeUnit.SECONDS);
                assertEquals(1, queue.size());
                assertFalse(queue.offer(EngineRpcService.GenerateOutputsPB.getDefaultInstance()));
                queue.poll();
                assertFalse(queue.offer(success), "consuming terminal must not reopen stream");
            }
        } finally { workers.shutdownNow(); }
    }
}
