package org.flexlb.mock.engine.service;

import io.grpc.stub.ServerCallStreamObserver;
import java.util.ArrayDeque;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies frame coalescing and delivery lifecycle for the response stream.
 */
class GenerateResponseStreamTest {
    private final ManualDeadline deadlines = new ManualDeadline();

    @AfterEach void closeDeadlines() { deadlines.shutdownNow(); }

    @Test void lateFetchRetainsOnlyProgressAndBuildsBoundedFramesOnDemand() {
        var tasks = new ArrayDeque<Runnable>();
        var output = output(tasks::add);
        for (int length = 1; length <= 100000; length++)
            output.advance(7, 42, length, AuxInfoPB.getDefaultInstance(), length == 100000);
        output.finish(null);
        assertTrue(tasks.isEmpty()); // No consumer and no queued token messages.
        var observer = new Observer(); output.attach(observer, () -> {});
        assertEquals(1, tasks.size()); tasks.remove().run();
        assertEquals(0, observer.frames); assertFalse(observer.completed);
        observer.ready = true; observer.onReady.run(); tasks.remove().run();
        assertEquals(100000, observer.received);
        assertEquals((100000 + GenerateResponseStream.MAX_FRAME_TOKENS - 1) / GenerateResponseStream.MAX_FRAME_TOKENS, observer.frames);
        assertEquals(1, observer.finals); assertTrue(observer.completed); assertTrue(output.discardable());
    }

    @Test void unwritableConsumerCoalescesAllUpdatesIntoOneSenderTask() {
        var tasks = new ArrayDeque<Runnable>(); var observer = new Observer();
        var output = output(tasks::add); output.attach(observer, () -> {});
        for (int n = 1; n <= 10000; n++) output.advance(7, 42, n, AuxInfoPB.getDefaultInstance(), n == 10000);
        assertEquals(1, tasks.size()); tasks.remove().run();
        assertEquals(0, observer.frames);
        output.finish(null); tasks.remove().run();
        assertFalse(observer.completed);
        observer.ready = true; observer.onReady.run(); tasks.remove().run();
        assertEquals(10000, observer.received); assertEquals(1, observer.finals); assertTrue(observer.completed);
    }

    @Test void blockedGrpcWriteDoesNotBlockProgressPublication() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1); var finished = new CountDownLatch(1);
        var observer = new Observer() {
            @Override public void onNext(GenerateOutputsPB value) {
                if (frames == 0) {
                    entered.countDown();
                    try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
                    catch (InterruptedException error) { throw new RuntimeException(error); }
                }
                super.onNext(value);
            }
            @Override public void onCompleted() { super.onCompleted(); finished.countDown(); }
        };
        observer.ready = true;
        try (var threads = Executors.newVirtualThreadPerTaskExecutor()) {
            var output = output(threads); output.attach(observer, () -> {});
            output.advance(7, 42, 1, AuxInfoPB.getDefaultInstance(), false);
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(1), () -> {
                for (int n = 2; n <= 10000; n++) output.advance(7, 42, n, AuxInfoPB.getDefaultInstance(), n == 10000);
                output.finish(null);
            });
            release.countDown(); assertTrue(finished.await(2, TimeUnit.SECONDS));
            assertEquals(10000, observer.received); assertEquals(1, observer.finals);
        } finally { release.countDown(); }
    }

    @Test void decodeFinalMarkerDoesNotRepeatThePrefillToken() {
        var tasks = new ArrayDeque<Runnable>(); var observer = new Observer();
        observer.ready = true; observer.received = 1;
        var output = output(tasks::add); output.startAt(1); output.attach(observer, () -> {});
        output.advance(7, 42, 1, AuxInfoPB.getDefaultInstance(), true); output.finish(null);
        tasks.remove().run();
        assertEquals(1, observer.received); assertEquals(1, observer.frames);
        assertEquals(1, observer.finals); assertTrue(observer.completed);
    }

    @Test void compactDecodeCoalescesProgressWithoutBuildingTokenTensors() {
        var tasks = new ArrayDeque<Runnable>();
        var receivedFrames = new java.util.ArrayList<GenerateOutputsPB>();
        var observer = new Observer() {
            @Override public void onNext(GenerateOutputsPB value) { receivedFrames.add(value); }
        };
        var stream = output(tasks::add);
        stream.startAt(1);
        stream.attach(observer, () -> {});
        stream.advance(7, 42, 4, AuxInfoPB.getDefaultInstance(), false);
        stream.advance(7, 42, 7, AuxInfoPB.getDefaultInstance(), true);
        stream.finish(null);
        tasks.remove().run();
        assertTrue(receivedFrames.isEmpty());
        observer.ready = true;
        observer.onReady.run();
        tasks.remove().run();
        assertEquals(1, receivedFrames.size());
        var frame = receivedFrames.getFirst().getFlattenOutput();
        assertFalse(frame.hasOutputIds());
        assertEquals(7, frame.getAuxInfo(0).getOutputLen());
        assertEquals(6, frame.getAuxInfo(0).getStepOutputLen());
        assertTrue(frame.getFinished(0));
        assertTrue(observer.completed);
    }

    @Test void completedUnwritableOutputExpiresAndRemovesRegistryEntry() {
        var tasks = new ArrayDeque<Runnable>(); var observer = new Observer();
        var registry = new java.util.HashMap<Long, GenerateResponseStream>();
        var output = new GenerateResponseStream(tasks::add, deadlines, 10000, expired -> registry.remove(7L, expired));
        registry.put(7L, output);
        output.attach(observer, () -> fail("Output expiry must not cancel computation"));
        output.advance(7, 42, 100, AuxInfoPB.getDefaultInstance(), true);
        tasks.remove().run();
        assertNull(deadlines.expiration); // No output timer during computation.
        output.finish(null); tasks.remove().run();
        assertNull(observer.failure); assertEquals(1, registry.size());
        deadlines.expiration.run(); tasks.remove().run();
        assertTrue(registry.isEmpty());
        assertEquals(io.grpc.Status.Code.DEADLINE_EXCEEDED, io.grpc.Status.fromThrowable(observer.failure).getCode());
        assertEquals(0, observer.frames); assertTrue(output.discardable());
        observer.ready = true; observer.onReady.run(); assertTrue(tasks.isEmpty());
    }

    @Test void expirationCleansRegistryEvenWhileAWriteIsBlocked() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1); var failed = new CountDownLatch(1);
        var removed = new java.util.concurrent.atomic.AtomicBoolean();
        var observer = new Observer() {
            @Override public void onNext(GenerateOutputsPB value) {
                entered.countDown();
                try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { throw new RuntimeException(error); }
                super.onNext(value);
            }
            @Override public void onError(Throwable error) { super.onError(error); failed.countDown(); }
        };
        observer.ready = true;
        try (var threads = Executors.newVirtualThreadPerTaskExecutor()) {
            var output = new GenerateResponseStream(threads, deadlines, 10000, expired -> removed.set(true));
            output.attach(observer, () -> fail("Delivery expiry must not cancel computation"));
            output.advance(7, 42, 1, AuxInfoPB.getDefaultInstance(), true);
            assertTrue(entered.await(2, TimeUnit.SECONDS)); output.finish(null);
            deadlines.expiration.run();
            assertTrue(removed.get()); assertNull(observer.failure); // No concurrent observer calls.
            release.countDown(); assertTrue(failed.await(2, TimeUnit.SECONDS));
            assertEquals(io.grpc.Status.Code.DEADLINE_EXCEEDED, io.grpc.Status.fromThrowable(observer.failure).getCode());
            assertFalse(observer.completed);
        } finally { release.countDown(); }
    }

    @Test void deliveredOutputCancelsItsDeadlineAndCannotExpireLater() {
        var tasks = new ArrayDeque<Runnable>(); var observer = new Observer(); observer.ready = true;
        var output = new GenerateResponseStream(tasks::add, deadlines, 10000, expired -> fail("Already delivered"));
        output.attach(observer, () -> {});
        output.advance(7, 42, 1, AuxInfoPB.getDefaultInstance(), true); output.finish(null);
        tasks.remove().run();
        assertTrue(observer.completed); assertTrue(deadlines.getQueue().isEmpty());
        deadlines.expiration.run(); assertNull(observer.failure); assertTrue(tasks.isEmpty());
    }

    @Test void cancellationAndErrorsDiscardUndeliveredProgress() {
        var tasks = new ArrayDeque<Runnable>(); var observer = new Observer();
        var output = output(tasks::add); output.attach(observer, () -> {});
        output.advance(7, 42, 10000, AuxInfoPB.getDefaultInstance(), false);
        var error = new IllegalStateException("failed"); output.finish(error); tasks.remove().run();
        assertSame(error, observer.failure); assertEquals(0, observer.frames); assertTrue(output.discardable());
        var second = output(tasks::add); var cancelled = new Observer();
        var notified = new java.util.concurrent.atomic.AtomicBoolean();
        second.attach(cancelled, () -> notified.set(true)); cancelled.cancelled = true; cancelled.onCancel.run();
        tasks.remove().run(); assertTrue(notified.get()); assertEquals(0, cancelled.frames);
    }

    private GenerateResponseStream output(Executor executor) {
        return new GenerateResponseStream(executor, deadlines, 10000, ignored -> {});
    }

    /**
     * Lets tests trigger delivery expiry without waiting for the scheduled deadline.
     */
    private static class ManualDeadline extends ScheduledThreadPoolExecutor {
        Runnable expiration;
        ManualDeadline() { super(1); setRemoveOnCancelPolicy(true); }
        @Override public ScheduledFuture<?> schedule(Runnable action, long delay, TimeUnit unit) {
            assertEquals(TimeUnit.MINUTES.toNanos(10), unit.toNanos(delay));
            expiration = action;
            return super.schedule(action, 1, TimeUnit.DAYS);
        }
    }

    /**
     * Records responses and lets tests control stream readiness and cancellation.
     */
    private static class Observer extends ServerCallStreamObserver<GenerateOutputsPB> {
        volatile boolean ready, cancelled, completed;
        Runnable onReady, onCancel;
        Throwable failure;
        int received, frames, finals;
        @Override public void onNext(GenerateOutputsPB value) {
            if (!value.hasFlattenOutput()) return;
            var flat = value.getFlattenOutput();
            int width = flat.getAuxInfo(0).getStepOutputLen();
            if (flat.hasOutputIds()) {
                assertEquals(width, flat.getOutputIds().getShape(2));
                assertEquals(width * 4, flat.getOutputIds().getInt32Data().size());
            }
            assertTrue(width <= GenerateResponseStream.MAX_FRAME_TOKENS);
            received += width; frames++;
            assertEquals(received, flat.getAuxInfo(0).getOutputLen());
            if (flat.getFinished(0)) finals++;
        }
        @Override public void onError(Throwable error) { failure = error; }
        @Override public void onCompleted() { completed = true; }
        @Override public boolean isCancelled() { return cancelled; }
        @Override public void setOnCancelHandler(Runnable handler) { onCancel = handler; }
        @Override public boolean isReady() { return ready; }
        @Override public void setOnReadyHandler(Runnable handler) { onReady = handler; }
        @Override public void setCompression(String compression) {}
        @Override public void request(int count) {}
        @Override public void setMessageCompression(boolean enabled) {}
        @Override public void disableAutoInboundFlowControl() {}
    }
}
