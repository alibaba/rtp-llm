package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RtpSchedulerTest {
    @TempDir Path directory;
    RtpConfig config(String role, String expression) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration(role));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2);
        ((ObjectNode) root.path("component").path("config").path("compute")).put("expression", expression);
        var path = directory.resolve(role + ".json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }
    static RtpScheduler.Input input(long id, boolean retained) {
        return new RtpScheduler.Input(Long.toString(id), id, 0, 5, 3, 0, System.nanoTime(), Long.MAX_VALUE, List.of(), false, retained);
    }
    static class Listener implements RtpScheduler.Listener {
        final CompletableFuture<RtpScheduler.Request> allocated = new CompletableFuture<>();
        final CompletableFuture<RtpScheduler.Request> output = new CompletableFuture<>();
        final CompletableFuture<RequestFailure> terminal = new CompletableFuture<>();
        boolean writable = true, cancelled;
        public boolean writable() { return writable; }
        public boolean cancelled() { return cancelled; }
        public void allocated(RtpScheduler.Request request) { allocated.complete(request); }
        public void output(RtpScheduler.Request request, int committed, boolean finished) { output.complete(request); }
        public void terminal(RtpScheduler.Request request, RequestFailure failure) { terminal.complete(failure); }
    }

    @Test void retainedPrefillReleasesExecutionSlotButKeepsKvUntilConsumed() throws Exception {
        var lock = new Object(); var first = new Listener(); var second = new Listener();
        try (var cache = new DeviceCache(8, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(lock, config("prefill", "10"), cache)) {
            var retained = scheduler.submit(input(1, true), first);
            scheduler.submit(input(2, false), second);
            first.output.get(2, TimeUnit.SECONDS); assertNull(second.terminal.get(2, TimeUnit.SECONDS));
            synchronized (lock) {
                assertEquals(new QueueSnapshot(1, 0), scheduler.queues());
                assertEquals(RequestState.Phase.READY, retained.snapshot().phase());
                assertFalse(first.terminal.isDone()); assertEquals(5, cache.snapshot(0, false).availableBlocks());
                scheduler.consume(retained, () -> {});
            }
            assertNull(first.terminal.get(2, TimeUnit.SECONDS)); assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void decodeRequeuesRoundRobinWithoutProtocolObjects() throws Exception {
        var lock = new Object(); var order = new ArrayList<String>();
        class Recording extends Listener {
            @Override public void output(RtpScheduler.Request request, int committed, boolean finished) {
                order.add(request.input.identity()); super.output(request, committed, finished);
            }
        }
        var first = new Recording(); var second = new Recording();
        try (var cache = new DeviceCache(8, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(lock, config("decode", "10"), cache)) {
            var a = scheduler.submit(input(1, false), first); var b = scheduler.submit(input(2, false), second);
            first.allocated.get(2, TimeUnit.SECONDS); second.allocated.get(2, TimeUnit.SECONDS);
            synchronized (lock) { scheduler.generate(a, 3); scheduler.generate(b, 3); }
            assertNull(first.terminal.get(2, TimeUnit.SECONDS)); assertNull(second.terminal.get(2, TimeUnit.SECONDS));
            synchronized (lock) { assertEquals(List.of("1", "2", "1", "2"), order); }
            assertEquals(new QueueSnapshot(0, 0), scheduler.queues()); assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void closedOutputCannotCommitEvenBeforeItsCancellationCallbackRuns() throws Exception {
        var lock = new Object(); var listener = new Listener();
        try (var cache = new DeviceCache(8, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(lock, config("decode", "10"), cache)) {
            var request = scheduler.submit(input(1, false), listener);
            listener.allocated.get(2, TimeUnit.SECONDS);
            synchronized (lock) { scheduler.generate(request, 3); listener.cancelled = true; }
            assertEquals(RequestFailure.Reason.CANCELLED, listener.terminal.get(2, TimeUnit.SECONDS).reason());
            synchronized (lock) { assertEquals(1, request.snapshot().generated()); assertEquals(0, request.snapshot().iterations()); }
            assertFalse(listener.output.isDone()); assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void capacityTransferAndOutputWaitsAreNotForwardAndOldOwnerCannotStopReplacement() throws Exception {
        var lock = new Object(); var listener = new Listener(); listener.writable = false;
        try (var cache = new DeviceCache(8, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(lock, config("decode", "10"), cache)) {
            var occupied = cache.acquire(8, List.of()).lease();
            var old = scheduler.submit(input(1, false), listener);
            assertFalse(listener.allocated.isDone()); assertEquals(new QueueSnapshot(1, 0), scheduler.queues());
            occupied.close(); listener.allocated.get(2, TimeUnit.SECONDS);
            synchronized (lock) {
                assertEquals(RequestState.Phase.TRANSFER, old.snapshot().phase());
                scheduler.generate(old, 3); assertEquals(RequestState.Phase.OUTPUT, old.snapshot().phase());
                assertEquals(new QueueSnapshot(1, 0), scheduler.queues());
                listener.writable = true; scheduler.resume(old);
                assertEquals(new QueueSnapshot(0, 1), scheduler.queues());
                scheduler.finish(old, new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel"));
            }
            assertEquals(RequestFailure.Reason.CANCELLED, listener.terminal.get(2, TimeUnit.SECONDS).reason());
            var next = new Listener(); var replacement = scheduler.submit(input(1, false), next);
            next.allocated.get(2, TimeUnit.SECONDS); scheduler.finish(old, null);
            assertFalse(next.terminal.isDone());
            scheduler.generate(replacement, 3); assertNull(next.terminal.get(2, TimeUnit.SECONDS));
            assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }
}
