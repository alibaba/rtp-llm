package org.flexlb.mock.engine.service;

import io.grpc.stub.ServerCallStreamObserver;
import java.util.ArrayDeque;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GeneratedOutputTest {
    private static class Observer extends ServerCallStreamObserver<GenerateOutputsPB> {
        volatile boolean ready, cancelled, completed;
        Runnable onReady, onCancel;
        Throwable failure;
        int received, frames, finals;
        @Override public void onNext(GenerateOutputsPB value) {
            if (!value.hasFlattenOutput()) return;
            var flat = value.getFlattenOutput(); int width = Math.toIntExact(flat.getOutputIds().getShape(2));
            assertTrue(width <= GeneratedOutput.MAX_FRAME_TOKENS);
            received += width; frames++;
            assertEquals(received, flat.getAuxInfo(0).getOutputLen());
            assertEquals(width * 4, flat.getOutputIds().getInt32Data().size());
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

    @Test void lateFetchRetainsOnlyProgressAndBuildsBoundedFramesOnDemand() {
        var tasks = new ArrayDeque<Runnable>();
        var output = new GeneratedOutput(tasks::add, 10000);
        for (int length = 1; length <= 100000; length++)
            output.advance(7, 42, length, AuxInfoPB.getDefaultInstance(), length == 100000);
        output.finish(null);
        assertTrue(tasks.isEmpty()); // No consumer and no queued token messages.
        var observer = new Observer(); output.attach(observer, () -> {});
        assertEquals(1, tasks.size()); tasks.remove().run();
        assertEquals(0, observer.frames); assertFalse(observer.completed);
        observer.ready = true; observer.onReady.run(); tasks.remove().run();
        assertEquals(100000, observer.received);
        assertEquals((100000 + GeneratedOutput.MAX_FRAME_TOKENS - 1) / GeneratedOutput.MAX_FRAME_TOKENS, observer.frames);
        assertEquals(1, observer.finals); assertTrue(observer.completed); assertTrue(output.discardable());
    }

    @Test void unwritableConsumerCoalescesAllUpdatesIntoOneSenderTask() {
        var tasks = new ArrayDeque<Runnable>(); var observer = new Observer();
        var output = new GeneratedOutput(tasks::add, 10000); output.attach(observer, () -> {});
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
            var output = new GeneratedOutput(threads, 10000); output.attach(observer, () -> {});
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
        var output = new GeneratedOutput(tasks::add, 10000); output.startAt(1); output.attach(observer, () -> {});
        output.advance(7, 42, 1, AuxInfoPB.getDefaultInstance(), true); output.finish(null);
        tasks.remove().run();
        assertEquals(1, observer.received); assertEquals(1, observer.frames);
        assertEquals(1, observer.finals); assertTrue(observer.completed);
    }

    @Test void cancellationAndErrorsDiscardUndeliveredProgress() {
        var tasks = new ArrayDeque<Runnable>(); var observer = new Observer();
        var output = new GeneratedOutput(tasks::add, 10000); output.attach(observer, () -> {});
        output.advance(7, 42, 10000, AuxInfoPB.getDefaultInstance(), false);
        var error = new IllegalStateException("failed"); output.finish(error); tasks.remove().run();
        assertSame(error, observer.failure); assertEquals(0, observer.frames); assertTrue(output.discardable());
        var second = new GeneratedOutput(tasks::add, 10000); var cancelled = new Observer();
        var notified = new java.util.concurrent.atomic.AtomicBoolean();
        second.attach(cancelled, () -> notified.set(true)); cancelled.cancelled = true; cancelled.onCancel.run();
        tasks.remove().run(); assertTrue(notified.get()); assertEquals(0, cancelled.frames);
    }
}
