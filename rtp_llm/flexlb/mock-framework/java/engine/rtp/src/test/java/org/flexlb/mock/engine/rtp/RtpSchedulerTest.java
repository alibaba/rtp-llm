package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.compute.SimulatedCompute;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.cache.KvCache;
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
        config.put("seq_size_per_block", 2).put("max_generate_batch_size", batchSize).put("max_batch_tokens_size", tokenBudget);
        ((ObjectNode) config.path("compute").path("forward")).put("expression", expression);
        var path = directory.resolve(role + ".json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }
    private static Scheduler.Input input(long id, boolean retained) {
        return new Scheduler.Input(Long.toString(id), id, 0, 5, 3, 0, System.nanoTime(), Long.MAX_VALUE, List.of(), false, retained, 0, false);
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
        scheduler.computeCompleted(work, new SimulatedCompute.Outcome(SimulatedCompute.State.COMPLETED, sample, 10_000_000));
    }
    private static void loaded(Scheduler scheduler, Scheduler.Request request) {
        try (var use = scheduler.beginTransfer(request)) {
            scheduler.transferCompleted(request, new RtpTransferResult(false, null));
        }
    }
    private static void generate(Scheduler scheduler, Scheduler.Request request) {
        loaded(scheduler, request);
        scheduler.control(request, RtpControl.GENERATE);
    }

    @Test void retainedPrefillKeepsKvThroughReadUntilTransferAcknowledgement() throws Exception {
        var config = config("prefill", "1", 2); var listener = new Listener();
        try (var cache = new KvCache(config.blockSize(), 8, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var request = scheduler.input(input(1, true), listener);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(new QueueSnapshot(0, 0, 0, 1), scheduler.queues());
            assertFalse(listener.terminal.isDone()); assertEquals(5, cache.snapshot(0, false).availableBlocks());
            assertThrows(RequestFailure.class, () -> scheduler.control(request, RtpControl.TRANSFER_ACKNOWLEDGED));
            var read = scheduler.beginTransfer(request);
            scheduler.transferCompleted(request, new RtpTransferResult(false, null));
            scheduler.control(request, RtpControl.TRANSFER_ACKNOWLEDGED);
            assertEquals(5, cache.snapshot(0, false).availableBlocks()); assertFalse(listener.terminal.isDone());
            read.close(); drain(scheduler);
            assertNull(listener.terminal.get(1, TimeUnit.SECONDS)); assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void decodeBatchRetainsRunningAndCancellationCannotDamageAnotherMember() throws Exception {
        var config = config("decode", "1", 2); var first = new Listener(); var second = new Listener();
        try (var cache = new KvCache(config.blockSize(), 12, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var a = scheduler.input(input(1, false), first); var b = scheduler.input(input(2, false), second);
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
        try (var cache = new KvCache(config.blockSize(), 8, 0, KvCache.EvictionPolicy.LRU);
             var engine = new EngineLoop(lock, new RtpScheduler(lock, config, cache), cache, config.compute(), config.role(), config.id())) {
            var request = engine.input(input(1, true), listener);
            listener.output.get(2, TimeUnit.SECONDS);
            var read = engine.beginTransfer(request);
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
        try (var cache = new KvCache(config.blockSize(), 8, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var old = scheduler.input(input(1, false), listener);
            scheduler.maintain(); drain(scheduler);
            assertEquals(new QueueSnapshot(0, 0, 1, 0), scheduler.queues());
            assertThrows(RequestFailure.class, () -> scheduler.control(old, RtpControl.GENERATE));
            loaded(scheduler, old); assertNull(scheduler.schedule());
            scheduler.finish(old, new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel")); drain(scheduler);
            var replacement = scheduler.input(input(1, false), new Listener());
            scheduler.maintain(); scheduler.finish(old, null);
            scheduler.control(old, RtpControl.GENERATE);
            assertFalse(replacement.snapshot().terminating());
            assertEquals(0, replacement.snapshot().generated()); assertNull(scheduler.schedule());
        }
    }

    @Test void prefillAdmissionWaitsForCapacityAndBatchesAccordingToConfiguredLimit() throws Exception {
        var config = config("prefill", "1", 2);
        try (var cache = new KvCache(config.blockSize(), 12, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var held = cache.acquire(12L * config.blockSize(), List.of()).lease();
            var first = new Listener(); scheduler.input(input(1, false), first); scheduler.input(input(2, false), new Listener());
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
        try (var cache = new KvCache(config.blockSize(), 12, 0, KvCache.EvictionPolicy.LRU);
             var engine = new EngineLoop(lock, new RtpScheduler(lock, config, cache), cache, config.compute(), config.role(), config.id())) {
            engine.input(input(1, false), slow);
            assertTrue(blocked.await(2, TimeUnit.SECONDS));
            engine.input(input(2, false), next);
            assertNull(next.terminal.get(2, TimeUnit.SECONDS)); assertFalse(slow.terminal.isDone());
            release.countDown(); assertNull(slow.terminal.get(2, TimeUnit.SECONDS));
        } finally { release.countDown(); }
    }

    @Test void outputFailureIsTerminalWithoutLeakingKv() throws Exception {
        var config = config("decode", "1", 1); var listener = new Listener() {
            @Override public void output(Scheduler.Request request, Scheduler.Output output) { throw new IllegalStateException("closed"); }
        };
        try (var cache = new KvCache(config.blockSize(), 8, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var request = scheduler.input(input(1, false), listener);
            scheduler.maintain(); drain(scheduler); generate(scheduler, request);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(RequestFailure.Reason.CANCELLED, listener.terminal.get(1, TimeUnit.SECONDS).reason());
            assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void tokenBudgetDoesNotReserveKvForUnselectedRequests() throws Exception {
        var config = config("prefill", "1", 2, 5);
        try (var cache = new KvCache(config.blockSize(), 12, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            scheduler.input(input(1, false), new Listener());
            var waiting = scheduler.input(input(2, false), new Listener());
            var work = scheduler.schedule();
            assertEquals(1, work.batch().dps().getFirst().requests().size());
            assertEquals(Scheduler.ReportedPhase.RECEIVED, scheduler.status(0).tasks().stream().filter(task -> task.requestId() == waiting.input().requestId()).findFirst().orElseThrow().phase()); assertEquals(9, cache.snapshot(0, false).availableBlocks());
            complete(scheduler, work, config); drain(scheduler);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(12, cache.snapshot(0, false).availableBlocks());
            var oversized = new Scheduler.Input("large", 3, 0, 6, 1, 0, System.nanoTime(), Long.MAX_VALUE, List.of(), false, false, 0, false);
            assertEquals(RequestFailure.Reason.INVALID,
                assertThrows(RequestFailure.class, () -> scheduler.input(oversized, new Listener())).reason());
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
        try (var cache = new KvCache(config.blockSize(), 12, 0, KvCache.EvictionPolicy.LRU);
             var engine = new EngineLoop(lock, new RtpScheduler(lock, config, cache), cache, config.compute(), config.role(), config.id())) {
            var request = engine.input(input(1, false), listener);
            assertTrue(blocked.await(2, TimeUnit.SECONDS));
            engine.finish(request, new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel"));
            release.countDown();
            assertEquals(RequestFailure.Reason.CANCELLED, listener.terminal.get(2, TimeUnit.SECONDS).reason());
            var next = new Listener(); engine.input(input(2, false), next);
            assertNull(next.terminal.get(2, TimeUnit.SECONDS));
            assertEquals(12, cache.snapshot(0, false).availableBlocks());
        } finally { release.countDown(); }
    }
    @Test void terminalStatusIsCommittedBeforeTheExternalTerminalCallbackReturns() throws Exception {
        var config = config("prefill", "0", 1);
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var listener = new Listener() {
            @Override public void terminal(Scheduler.Request request, RequestFailure failure) {
                entered.countDown();
                try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new RuntimeException(error); }
                super.terminal(request, failure);
            }
        };
        var lock = new Object();
        try (var cache = new KvCache(config.blockSize(), 8, 0, KvCache.EvictionPolicy.LRU);
             var loop = new EngineLoop(lock, new RtpScheduler(lock, config, cache), cache, config.compute(), config.role(), config.id())) {
            loop.input(input(1, false), listener);
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var status = loop.status(0);
            assertTrue(status.tasks().isEmpty()); assertEquals(0, status.running()); assertEquals(0, status.waiting());
            assertEquals(1, status.finishedVersion()); assertEquals(1, status.finished().getFirst().requestId());
            assertThrows(UnsupportedOperationException.class, () -> status.finished().clear());
            assertTrue(loop.status(status.finishedVersion()).finished().isEmpty());
        } finally { release.countDown(); }
    }

    @Test void statusIsReadOnlyAndWaitingTimeFreezesAtResourceAdmission() throws Exception {
        var config = config("prefill", "1", 1);
        try (var cache = new KvCache(config.blockSize(), 8, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            scheduler.input(input(1, true), new Listener());
            var before = scheduler.status(0);
            assertEquals(Scheduler.ReportedPhase.RECEIVED, before.tasks().getFirst().phase());
            assertEquals(8, cache.snapshot(0, false).availableBlocks(), "A status query must not allocate");
            var work = scheduler.schedule();
            complete(scheduler, work, config); drain(scheduler);
            var ready = scheduler.status(0).tasks().getFirst();
            assertEquals(Scheduler.ReportedPhase.KV_ALLOCATED, ready.phase());
            Thread.sleep(20);
            var later = scheduler.status(0).tasks().getFirst();
            assertEquals(ready.waitingMillis(), later.waitingMillis());
            assertTrue(later.executionMillis() >= ready.executionMillis() + 10);
            assertEquals(Scheduler.ReportedPhase.RECEIVED, before.tasks().getFirst().phase(), "Snapshots must not change later");
        }
    }

    @Test void speculativeWindowIsReservedBeforeAcceptanceAndGrowthFailureEndsOnlyThatRequest() throws Exception {
        for (int capacity : List.of(1, 2)) {
            var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("decode"));
            var settings = (ObjectNode) root.path("component").path("config");
            settings.put("seq_size_per_block", 128);
            ((ObjectNode) settings.path("compute")).putObject("decode").put("draft_tokens", 4).put("acceptance_rate", 0);
            var path = directory.resolve("spec-" + capacity + ".json"); json.writeValue(path.toFile(), root);
            var config = RtpConfig.load(path);
            try (var cache = new KvCache(config.blockSize(), capacity, 0, KvCache.EvictionPolicy.LRU)) {
                var scheduler = new RtpScheduler(new Object(), config, cache); var listener = new Listener();
                try {
                    var request = scheduler.input(new Scheduler.Input("r", 1, 0, 123, 10, 0, System.nanoTime(),
                        Long.MAX_VALUE, List.of(), false, false, 0, true), listener);
                    scheduler.maintain(); drain(scheduler); generate(scheduler, request);
                    var work = scheduler.schedule();
                    if (capacity == 1) {
                        assertNull(work); drain(scheduler);
                        assertEquals(RequestFailure.Reason.CAPACITY, listener.terminal.get(1, TimeUnit.SECONDS).reason());
                        assertEquals(1, cache.snapshot(-1, false).freeBlocks());
                    } else {
                        assertNotNull(work); assertEquals(0, cache.snapshot(-1, false).availableBlocks());
                        assertEquals(5, work.batch().dps().getFirst().requests().getFirst().stepTokens());
                        complete(scheduler, work, config); drain(scheduler);
                        assertEquals(2, request.snapshot().generated());
                        assertEquals(0, cache.snapshot(-1, false).availableBlocks(), "Rejected drafts do not shrink the reserved window");
                        scheduler.finish(request, new RequestFailure(RequestFailure.Reason.CANCELLED, "test")); drain(scheduler);
                        assertEquals(2, cache.snapshot(-1, false).freeBlocks());
                    }
                } finally { scheduler.close(); }
            }
        }
    }

    @Test void decodeGrowthFailureReleasesOnlyItsLeaseAndUnblocksWaitingAdmission() throws Exception {
        var config = config("decode", "1", 2);
        try (var cache = new KvCache(config.blockSize(), 7, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var first = new Listener(); var second = new Listener(); var waiting = new Listener();
            var a = scheduler.input(input(1, false), first);
            var b = scheduler.input(input(2, false), second);
            scheduler.maintain(); drain(scheduler); generate(scheduler, a); generate(scheduler, b);
            var c = scheduler.input(input(3, false), waiting);
            scheduler.maintain(); drain(scheduler);
            assertFalse(waiting.allocated.isDone());
            assertEquals(1, cache.snapshot(0, false).availableBlocks());

            // The first Decode round still fits; the next round crosses the block boundary.
            var initial = scheduler.schedule();
            assertEquals(2, initial.batch().dps().getFirst().requests().size());
            complete(scheduler, initial, config); drain(scheduler);
            assertEquals(2, a.snapshot().generated());

            // A grows into the last free block; B cannot grow and must release only B's lease.
            var work = scheduler.schedule(); drain(scheduler);
            assertNotNull(work); assertEquals(1, work.batch().dps().getFirst().requests().size());
            assertEquals(RequestFailure.Reason.CAPACITY, second.terminal.get(1, TimeUnit.SECONDS).reason());
            assertEquals(3, cache.snapshot(0, false).availableBlocks());
            assertFalse(first.terminal.isDone());
            scheduler.maintain(); drain(scheduler);
            assertSame(c, waiting.allocated.get(1, TimeUnit.SECONDS));
            assertEquals(0, cache.snapshot(0, false).availableBlocks());

            complete(scheduler, work, config); drain(scheduler);
            assertEquals(3, a.snapshot().generated());
            assertNull(first.terminal.get(1, TimeUnit.SECONDS));
            assertEquals(4, cache.snapshot(0, false).availableBlocks());
            scheduler.finish(c, new RequestFailure(RequestFailure.Reason.CANCELLED, "test")); drain(scheduler);
            assertEquals(7, cache.snapshot(0, false).freeBlocks());
            assertTrue(scheduler.requests().isEmpty());
            assertEquals(3, scheduler.status(0).finished().size());
        }
    }

    @Test void prefillCapacityWaitExpiresOrCancelsWithoutRevivingAfterRelease() throws Exception {
        var config = config("prefill", "1", 2); var lock = new Object();
        try (var cache = new KvCache(config.blockSize(), 3, 0, KvCache.EvictionPolicy.LRU);
             var owner = cache.acquire(6, List.of()).lease();
             var engine = new EngineLoop(lock, new RtpScheduler(lock, config, cache), cache, config.compute(), config.role(), config.id())) {
            var expired = new Listener(); var cancelled = new Listener();
            long now = System.nanoTime();
            engine.input(new Scheduler.Input("1", 1, 0, 5, 3, 0, now,
                now + TimeUnit.MILLISECONDS.toNanos(100), List.of(), false, false, 0, false), expired);
            var request = engine.input(input(2, false), cancelled);
            engine.finish(request, new RequestFailure(RequestFailure.Reason.CANCELLED, "test"));
            assertEquals(RequestFailure.Reason.CANCELLED, cancelled.terminal.get(2, TimeUnit.SECONDS).reason());
            assertEquals(RequestFailure.Reason.TIMEOUT, expired.terminal.get(2, TimeUnit.SECONDS).reason());
            assertFalse(expired.output.isDone()); assertFalse(cancelled.output.isDone());
            assertEquals(0, cache.snapshot(0, false).availableBlocks());

            owner.close();
            var next = new Listener(); engine.input(input(3, false), next);
            assertNull(next.terminal.get(2, TimeUnit.SECONDS));
            assertFalse(expired.output.isDone()); assertFalse(cancelled.output.isDone());
            assertEquals(3, cache.snapshot(0, false).freeBlocks());
            assertEquals(3, engine.status(0).finished().size());
        }
    }

    @Test void oversizedPrefillDoesNotPreventSmallerRequestFromRunning() throws Exception {
        var config = config("prefill", "1", 2);
        try (var cache = new KvCache(config.blockSize(), 2, 0, KvCache.EvictionPolicy.LRU);
             var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var oversized = new Listener(); var small = new Listener();
            scheduler.input(input(1, false), oversized);
            scheduler.input(new Scheduler.Input("2", 2, 0, 1, 3, 0, System.nanoTime(),
                Long.MAX_VALUE, List.of(), false, false, 0, false), small);
            var work = scheduler.schedule(); drain(scheduler);
            assertEquals(RequestFailure.Reason.CAPACITY, oversized.terminal.get(1, TimeUnit.SECONDS).reason());
            assertNotNull(work); assertEquals(1, work.batch().dps().getFirst().requests().size());
            complete(scheduler, work, config); drain(scheduler);
            assertNull(small.terminal.get(1, TimeUnit.SECONDS));
            assertTrue(scheduler.requests().isEmpty());
            assertEquals(2, cache.snapshot(0, false).freeBlocks());
        }
    }

}
