package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.compute.SimulatedCompute;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.cache.DeviceCache;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RtpSchedulerTest {
    @TempDir Path directory;
    private RtpConfig config(String role, String expression, int batchSize) throws Exception {
        return config(role, expression, batchSize, 0);
    }
    private RtpConfig config(String role, String expression, int batchSize, int tokenBudget) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration(role));
        var config = (ObjectNode) root.path("component").path("config");
        config.put("seq_size_per_block", 2).put("max_context_batch_size", batchSize)
            .put("max_generate_batch_size", batchSize).put("max_batch_tokens_size", tokenBudget);
        ((ObjectNode) config.path("compute")).put("expression", expression);
        var path = directory.resolve(role + ".json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }
    private static Scheduler.Input input(long id, boolean retained) {
        return new Scheduler.Input(Long.toString(id), id, 0, 5, 3, 0, System.nanoTime(), Long.MAX_VALUE, List.of(), false, retained);
    }
    private static class Listener implements Scheduler.Listener {
        final CompletableFuture<Scheduler.Request> allocated = new CompletableFuture<>();
        final CompletableFuture<Scheduler.Output> output = new CompletableFuture<>();
        final CompletableFuture<RequestFailure> terminal = new CompletableFuture<>();
        volatile boolean writable = true, cancelled;
        public boolean writable() { return writable; }
        public boolean cancelled() { return cancelled; }
        public void allocated(Scheduler.Request request) { allocated.complete(request); }
        public void output(Scheduler.Request request, Scheduler.Output output) { this.output.complete(output); }
        public void terminal(Scheduler.Request request, RequestFailure failure) { terminal.complete(failure); }
    }
    private static void drain(Scheduler scheduler) {
        for (var notices = scheduler.takeNotifications(); !notices.isEmpty(); notices = scheduler.takeNotifications())
            notices.forEach(Runnable::run);
    }
    private static void complete(Scheduler scheduler, Scheduler.Work work, RtpConfig config) {
        var sample = config.compute().plan(config.role(), config.id(), work.batch());
        scheduler.completed(work, new SimulatedCompute.Outcome(SimulatedCompute.State.COMPLETED, sample, 10_000_000));
    }
    private static void generate(Scheduler scheduler, Scheduler.Request request) {
        scheduler.dependencyCompleted(request, new Scheduler.Loaded(3));
        scheduler.dependencyCompleted(request, Scheduler.Signal.GENERATE);
    }

    @Test void retainedPrefillKeepsKvThroughReadUntilTransferAcknowledgement() throws Exception {
        var config = config("prefill", "1", 2); var listener = new Listener();
        try (var cache = new DeviceCache(8, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var request = scheduler.submit(input(1, true), listener);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(new QueueSnapshot(0, 0, 0, 1), scheduler.queues());
            assertFalse(listener.terminal.isDone()); assertEquals(5, cache.snapshot(0, false).availableBlocks());
            assertThrows(RequestFailure.class, () -> scheduler.dependencyCompleted(request, Scheduler.Signal.TRANSFER_ACKNOWLEDGED));
            var read = scheduler.readReady(request);
            scheduler.dependencyCompleted(request, Scheduler.Signal.TRANSFER_ACKNOWLEDGED);
            assertEquals(5, cache.snapshot(0, false).availableBlocks()); assertFalse(listener.terminal.isDone());
            read.close(); drain(scheduler);
            assertNull(listener.terminal.get(1, TimeUnit.SECONDS)); assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void decodeBatchRetainsRunningAndCancellationCannotDamageAnotherMember() throws Exception {
        var config = config("decode", "1", 2); var first = new Listener(); var second = new Listener();
        try (var cache = new DeviceCache(12, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var a = scheduler.submit(input(1, false), first); var b = scheduler.submit(input(2, false), second);
            scheduler.maintain(); drain(scheduler); generate(scheduler, a); generate(scheduler, b);
            var batch = scheduler.schedule();
            assertEquals(2, batch.batch().dps().getFirst().requests().size());
            assertEquals(new QueueSnapshot(0, 2, 0, 0), scheduler.queues());
            long available = cache.snapshot(0, false).availableBlocks();
            scheduler.finish(a, new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel one"));
            assertEquals(available, cache.snapshot(0, false).availableBlocks());
            assertFalse(first.terminal.isDone()); assertFalse(scheduler.cancelled(batch));
            complete(scheduler, batch, config); drain(scheduler);
            assertEquals(RequestFailure.Reason.CANCELLED, first.terminal.get(1, TimeUnit.SECONDS).reason());
            assertEquals(1, a.snapshot().generated()); assertEquals(2, b.snapshot().generated());
            assertEquals(new QueueSnapshot(0, 1, 0, 0), scheduler.queues());
            assertTrue(scheduler.scrapeMetrics().contains("mock_engine_forward_seconds_count{role=\"decode\",} 1.0"));
            var next = scheduler.schedule();
            assertEquals(List.of("2"), next.batch().dps().getFirst().requests().stream().map(r -> r.id()).toList());
            complete(scheduler, next, config); drain(scheduler);
            assertNull(second.terminal.get(1, TimeUnit.SECONDS)); assertEquals(12, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void closingEngineDefersCleanupUntilOutstandingReadReturns() throws Exception {
        Object lock = new Object(); var config = config("prefill", "1", 1); var listener = new Listener();
        try (var cache = new DeviceCache(8, 0, DeviceCache.EvictionPolicy.LRU);
             var engine = new EngineLoop(lock, new RtpScheduler(lock, config, cache), config.compute(), config.role(), config.id())) {
            var request = engine.submit(input(1, true), listener);
            listener.output.get(2, TimeUnit.SECONDS);
            var read = engine.readReady(request);
            engine.close();
            assertEquals(5, cache.snapshot(0, false).availableBlocks()); assertFalse(listener.terminal.isDone());
            read.close();
            assertEquals(RequestFailure.Reason.CLOSED, listener.terminal.get(2, TimeUnit.SECONDS).reason());
            assertEquals(8, cache.snapshot(0, false).availableBlocks());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void transferAndGenerationAreIndependentRequirementsAndOldHandlesCannotStopNewRequests() throws Exception {
        var config = config("decode", "1", 2); var listener = new Listener();
        try (var cache = new DeviceCache(8, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var old = scheduler.submit(input(1, false), listener);
            scheduler.maintain(); drain(scheduler);
            assertEquals(new QueueSnapshot(0, 0, 1, 0), scheduler.queues());
            assertThrows(RequestFailure.class, () -> scheduler.dependencyCompleted(old, Scheduler.Signal.GENERATE));
            scheduler.dependencyCompleted(old, new Scheduler.Loaded(3)); assertNull(scheduler.schedule());
            scheduler.finish(old, new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel")); drain(scheduler);
            var replacement = scheduler.submit(input(1, false), new Listener());
            scheduler.maintain(); scheduler.finish(old, null);
            scheduler.dependencyCompleted(old, Scheduler.Signal.GENERATE);
            assertFalse(replacement.snapshot().terminating());
            assertEquals(0, replacement.snapshot().generated()); assertNull(scheduler.schedule());
        }
    }

    @Test void prefillAdmissionWaitsForCapacityAndBatchesAccordingToConfiguredLimit() throws Exception {
        var config = config("prefill", "1", 2);
        try (var cache = new DeviceCache(12, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var held = cache.acquire(12, List.of()).lease();
            var first = new Listener(); scheduler.submit(input(1, false), first); scheduler.submit(input(2, false), new Listener());
            assertNull(scheduler.schedule()); assertFalse(first.terminal.isDone());
            held.close();
            var batch = scheduler.schedule(); assertEquals(2, batch.batch().dps().getFirst().requests().size());
            complete(scheduler, batch, config); drain(scheduler);
            assertNull(first.terminal.get(1, TimeUnit.SECONDS)); assertEquals(12, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void slowOutputIsBoundedPerRequestAndDoesNotBlockAnotherForward() throws Exception {
        Object lock = new Object(); var blocked = new CountDownLatch(1); var release = new CountDownLatch(1);
        var config = config("prefill", "2", 2); var next = new Listener();
        var slow = new Listener() {
            @Override public void output(Scheduler.Request request, Scheduler.Output output) {
                assertFalse(Thread.holdsLock(lock)); blocked.countDown();
                try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new RuntimeException(error); }
                super.output(request, output);
            }
        };
        try (var cache = new DeviceCache(12, 0, DeviceCache.EvictionPolicy.LRU);
             var engine = new EngineLoop(lock, new RtpScheduler(lock, config, cache), config.compute(), config.role(), config.id())) {
            engine.submit(input(1, false), slow);
            assertTrue(blocked.await(2, TimeUnit.SECONDS));
            engine.submit(input(2, false), next);
            assertNull(next.terminal.get(2, TimeUnit.SECONDS)); assertFalse(slow.terminal.isDone());
            release.countDown(); assertNull(slow.terminal.get(2, TimeUnit.SECONDS));
        } finally { release.countDown(); }
    }

    @Test void outputFailureIsTerminalWithoutLeakingKv() throws Exception {
        var config = config("decode", "1", 1); var listener = new Listener() {
            @Override public void output(Scheduler.Request request, Scheduler.Output output) { throw new IllegalStateException("closed"); }
        };
        try (var cache = new DeviceCache(8, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var request = scheduler.submit(input(1, false), listener);
            scheduler.maintain(); drain(scheduler); generate(scheduler, request);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(RequestFailure.Reason.CANCELLED, listener.terminal.get(1, TimeUnit.SECONDS).reason());
            assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void tokenBudgetDoesNotReserveKvForUnselectedRequests() throws Exception {
        var config = config("prefill", "1", 2, 5);
        try (var cache = new DeviceCache(12, 0, DeviceCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            scheduler.submit(input(1, false), new Listener());
            var waiting = scheduler.submit(input(2, false), new Listener());
            var work = scheduler.schedule();
            assertEquals(1, work.batch().dps().getFirst().requests().size());
            assertFalse(waiting.allocated()); assertEquals(9, cache.snapshot(0, false).availableBlocks());
            complete(scheduler, work, config); drain(scheduler);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(12, cache.snapshot(0, false).availableBlocks());
            var oversized = new Scheduler.Input("large", 3, 0, 6, 1, 0, System.nanoTime(), Long.MAX_VALUE, List.of(), false, false);
            assertEquals(RequestFailure.Reason.INVALID,
                assertThrows(RequestFailure.class, () -> scheduler.submit(oversized, new Listener())).reason());
            assertTrue(scheduler.requests().isEmpty());
        }
    }

    @Test void cancelledOutputFailureDoesNotFailTheEngine() throws Exception {
        Object lock = new Object(); var blocked = new CountDownLatch(1); var release = new CountDownLatch(1);
        var config = config("prefill", "1", 2);
        var listener = new Listener() {
            @Override public void output(Scheduler.Request request, Scheduler.Output output) {
                blocked.countDown();
                try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new RuntimeException(error); }
                throw new IllegalStateException("Transport closed while sending");
            }
        };
        try (var cache = new DeviceCache(12, 0, DeviceCache.EvictionPolicy.LRU);
             var engine = new EngineLoop(lock, new RtpScheduler(lock, config, cache), config.compute(), config.role(), config.id())) {
            var request = engine.submit(input(1, false), listener);
            assertTrue(blocked.await(2, TimeUnit.SECONDS));
            engine.finish(request, new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel"));
            release.countDown();
            assertEquals(RequestFailure.Reason.CANCELLED, listener.terminal.get(2, TimeUnit.SECONDS).reason());
            var next = new Listener(); engine.submit(input(2, false), next);
            assertNull(next.terminal.get(2, TimeUnit.SECONDS));
            assertEquals(12, cache.snapshot(0, false).availableBlocks());
        } finally { release.countDown(); }
    }
}
