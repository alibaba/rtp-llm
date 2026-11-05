package org.flexlb.mock.engine.scheduler;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.cache.KvCache;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.RtpEngineLoop;
import org.flexlb.mock.engine.forward.SimulatedCompute;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler.RtpControl;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler.RtpTransferResult;
import org.flexlb.mock.engine.scheduler.EngineScheduler.QueueSnapshot;
import org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure;
import org.flexlb.mock.protocol.CacheSequence;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class RtpSchedulerTest {
    @TempDir Path directory;
    private RtpConfig config(String role, String expression, int batchSize) throws Exception {
        return config(role, expression, batchSize, 0);
    }
    private RtpConfig config(String role, String expression, int batchSize, int tokenBudget) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration(role));
        var config = (ObjectNode) root.path("component").path("config");
        config.put("seq_size_per_block", 2).put("concurrency_limit", batchSize).put("max_batch_tokens_size", tokenBudget);
        ((ObjectNode) config.path("mock.compute").path("forward")).put("expression", expression);
        var path = directory.resolve(role + ".json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }
    private static EngineScheduler.Input input(long id, boolean retained) {
        return new EngineScheduler.Input(Long.toString(id), id, 0, 5, 3, 0, System.nanoTime(), Long.MAX_VALUE, new CacheSequence(2, Collections.nCopies(5, 0), List.of()), false, retained, 0, false);
    }
    private static class Listener implements EngineScheduler.Listener {
        final CompletableFuture<EngineScheduler.Request> allocated = new CompletableFuture<>();
        final CompletableFuture<EngineScheduler.Output> output = new CompletableFuture<>();
        final CompletableFuture<RequestFailure> terminal = new CompletableFuture<>();
        volatile boolean writable = true, cancelled;
        public boolean writable() { return writable; }
        public boolean cancelled() { return cancelled; }
        public void allocated(EngineScheduler.Request request) { allocated.complete(request); }
        public void output(EngineScheduler.Request request, EngineScheduler.Output output) { this.output.complete(output); }
        public void terminal(EngineScheduler.Request request, RequestFailure failure) { terminal.complete(failure); }
    }
    private static void drain(EngineScheduler scheduler) {
        for (var notices = scheduler.takeNotifications(); !notices.isEmpty(); notices = scheduler.takeNotifications())
            notices.forEach(Runnable::run);
    }
    private static void complete(EngineScheduler scheduler, EngineScheduler.Work work, RtpConfig config) {
        var sample = config.compute().plan(config.role(), config.id(), work.batch());
        scheduler.computeCompleted(work, new SimulatedCompute.Outcome(SimulatedCompute.State.COMPLETED, sample, 10_000_000));
    }
    private static void loaded(EngineScheduler scheduler, EngineScheduler.Request request) {
        try (var use = scheduler.beginTransfer(request)) {
            scheduler.transferCompleted(request, new RtpTransferResult(false, null));
        }
    }
    private static void generate(EngineScheduler scheduler, EngineScheduler.Request request) {
        loaded(scheduler, request);
        scheduler.control(request, new RtpControl.Generate(42));
    }

    @Test void retainedPrefillKeepsKvThroughReadUntilTransferAcknowledgement() throws Exception {
        var config = config("prefill", "1", 2); var listener = new Listener();
        try (var cache = new KvCache(config.blockSize(), 8, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            var request = scheduler.input(input(1, true), listener);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(new QueueSnapshot(0, 0, 0, 1), scheduler.queues());
            assertFalse(listener.terminal.isDone()); assertEquals(5, cache.snapshot(0, false).availableBlocks());
            assertThrows(RequestFailure.class, () -> scheduler.control(request, RtpControl.TRANSFER_ACKNOWLEDGED));
            var read = scheduler.beginTransfer(request);
            scheduler.control(request, RtpControl.TRANSFER_ACKNOWLEDGED);
            assertEquals(5, cache.snapshot(0, false).availableBlocks()); assertFalse(listener.terminal.isDone());
            read.close(); drain(scheduler);
            assertNull(listener.terminal.get(1, TimeUnit.SECONDS)); assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void decodeBatchRetainsRunningAndCancellationCannotDamageAnotherMember() throws Exception {
        var config = config("decode", "1", 2); var first = new Listener(); var second = new Listener();
        try (var cache = new KvCache(config.blockSize(), 12, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
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
            String observations = scheduler.metrics().scrape();
            assertTrue(observations.lines().anyMatch(line -> line.startsWith("rtp_llm_generate_batch_size{") && line.endsWith(" 2.0")), observations);
            assertTrue(observations.lines().anyMatch(line -> line.startsWith("rtp_llm_execute_token_size{") && line.endsWith(" 2.0")), observations);
            assertEquals(1, a.snapshot().generated()); assertEquals(2, b.snapshot().generated());
            assertEquals(new QueueSnapshot(0, 1, 0, 0), scheduler.queues());
            assertTrue(scheduler.metrics().scrape().lines().anyMatch(line -> line.startsWith("rtp_llm_model_forward_seconds_count{")
                && line.contains("role=\"decode\"") && line.endsWith(" 1.0")));
            var next = scheduler.schedule();
            assertEquals(List.of("2"), next.batch().dps().getFirst().requests().stream().map(r -> r.id()).toList());
            complete(scheduler, next, config); drain(scheduler);
            assertNull(second.terminal.get(1, TimeUnit.SECONDS)); assertEquals(12, cache.snapshot(0, false).availableBlocks());
            String terminalMetrics = scheduler.metrics().scrape();
            for (String name : List.of("rtp_llm_latency_us", "rtp_llm_wait_latency_us", "rtp_llm_first_token_latency_us",
                    "rtp_llm_input_token_length", "rtp_llm_output_token_length", "rtp_llm_iterate_count", "rtp_llm_query_batch_size"))
                assertTrue(terminalMetrics.contains(name + "{"), terminalMetrics);
        }
    }

    @Test void closingEngineDefersCleanupUntilOutstandingReadReturns() throws Exception {
        Object lock = new Object(); var config = config("prefill", "1", 1); var listener = new Listener();
        try (var cache = new KvCache(config.blockSize(), 8, 0);
             var engine = new RtpEngineLoop(lock, AbstractRtpScheduler.factory(config), new org.flexlb.mock.engine.scheduler.TaskHistory(), cache, config.compute(), config.role(), config.id())) {
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
        try (var cache = new KvCache(config.blockSize(), 8, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            var old = scheduler.input(input(1, false), listener);
            scheduler.maintain(); drain(scheduler);
            assertEquals(new QueueSnapshot(0, 0, 1, 0), scheduler.queues());
            assertThrows(RequestFailure.class, () -> scheduler.control(old, new RtpControl.Generate(42)));
            loaded(scheduler, old); assertNull(scheduler.schedule());
            scheduler.finish(old, new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel")); drain(scheduler);
            var replacement = scheduler.input(input(1, false), new Listener());
            scheduler.maintain(); scheduler.finish(old, null);
            scheduler.control(old, new RtpControl.Generate(42));
            assertFalse(replacement.snapshot().terminating());
            assertEquals(0, replacement.snapshot().generated()); assertNull(scheduler.schedule());
        }
    }

    @Test void prefillAdmissionWaitsForCapacityAndBatchesAccordingToConfiguredLimit() throws Exception {
        var config = config("prefill", "1", 2, 100);
        try (var cache = new KvCache(config.blockSize(), 12, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
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
            @Override public void output(EngineScheduler.Request request, EngineScheduler.Output output) {
                assertFalse(Thread.holdsLock(lock)); blocked.countDown();
                try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new RuntimeException(error); }
                super.output(request, output);
            }
        };
        try (var cache = new KvCache(config.blockSize(), 12, 0);
             var engine = new RtpEngineLoop(lock, AbstractRtpScheduler.factory(config), new org.flexlb.mock.engine.scheduler.TaskHistory(), cache, config.compute(), config.role(), config.id())) {
            engine.input(input(1, false), slow);
            assertTrue(blocked.await(2, TimeUnit.SECONDS));
            engine.input(input(2, false), next);
            assertNull(next.terminal.get(2, TimeUnit.SECONDS)); assertFalse(slow.terminal.isDone());
            release.countDown(); assertNull(slow.terminal.get(2, TimeUnit.SECONDS));
        } finally { release.countDown(); }
    }

    @Test void outputFailureIsTerminalWithoutLeakingKv() throws Exception {
        var config = config("decode", "1", 1); var listener = new Listener() {
            @Override public void output(EngineScheduler.Request request, EngineScheduler.Output output) { throw new IllegalStateException("closed"); }
        };
        try (var cache = new KvCache(config.blockSize(), 8, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            var request = scheduler.input(input(1, false), listener);
            scheduler.maintain(); drain(scheduler); generate(scheduler, request);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(RequestFailure.Reason.CANCELLED, listener.terminal.get(1, TimeUnit.SECONDS).reason());
            assertEquals(8, cache.snapshot(0, false).availableBlocks());
        }
    }

    @Test void tokenBudgetDoesNotReserveKvForUnselectedRequests() throws Exception {
        var config = config("prefill", "1", 2, 5);
        try (var cache = new KvCache(config.blockSize(), 12, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            scheduler.input(input(1, false), new Listener());
            var waiting = scheduler.input(input(2, false), new Listener());
            var work = scheduler.schedule();
            assertEquals(1, work.batch().dps().getFirst().requests().size());
            assertEquals(EngineScheduler.ReportedPhase.RECEIVED, scheduler.tasks().stream().filter(task -> task.requestId() == waiting.input().requestId()).findFirst().orElseThrow().phase()); assertEquals(9, cache.snapshot(0, false).availableBlocks());
            complete(scheduler, work, config); drain(scheduler);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(12, cache.snapshot(0, false).availableBlocks());
            var oversized = new EngineScheduler.Input("large", 3, 0, 6, 1, 0, System.nanoTime(), Long.MAX_VALUE, new CacheSequence(2, Collections.nCopies(6, 0), List.of()), false, false, 0, false);
            scheduler.input(oversized, new Listener());
            var singleton = scheduler.schedule(); assertNotNull(singleton);
            assertEquals(1, singleton.batch().dps().getFirst().requests().size());
            complete(scheduler, singleton, config); drain(scheduler);
            assertTrue(scheduler.requests().isEmpty());
        }
    }

    @Test void cancelledOutputFailureDoesNotFailTheEngine() throws Exception {
        Object lock = new Object(); var blocked = new CountDownLatch(1); var release = new CountDownLatch(1);
        var config = config("prefill", "1", 2);
        var listener = new Listener() {
            @Override public void output(EngineScheduler.Request request, EngineScheduler.Output output) {
                blocked.countDown();
                try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new RuntimeException(error); }
                throw new IllegalStateException("Transport closed while sending");
            }
        };
        try (var cache = new KvCache(config.blockSize(), 12, 0);
             var engine = new RtpEngineLoop(lock, AbstractRtpScheduler.factory(config), new org.flexlb.mock.engine.scheduler.TaskHistory(), cache, config.compute(), config.role(), config.id())) {
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
            @Override public void terminal(EngineScheduler.Request request, RequestFailure failure) {
                entered.countDown();
                try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new RuntimeException(error); }
                super.terminal(request, failure);
            }
        };
        var lock = new Object();
        try (var cache = new KvCache(config.blockSize(), 8, 0);
             var loop = new RtpEngineLoop(lock, AbstractRtpScheduler.factory(config), new org.flexlb.mock.engine.scheduler.TaskHistory(), cache, config.compute(), config.role(), config.id())) {
            loop.input(input(1, false), listener);
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var status = loop.status(0);
            assertTrue(status.tasks().isEmpty()); assertEquals(0, status.running()); assertEquals(0, status.waiting());
            assertEquals(1, status.finishedVersion()); assertEquals(1, status.finished().getFirst().requestId());
            assertThrows(UnsupportedOperationException.class, () -> status.finished().clear());
            assertTrue(loop.status(status.finishedVersion()).finished().isEmpty());
        } finally { release.countDown(); }
    }

    @Test void liveStatusKeepsNativeTimingSentinelsUntilDequeue() throws Exception {
        var config = config("prefill", "1", 1);
        try (var cache = new KvCache(config.blockSize(), 8, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            scheduler.input(input(1, true), new Listener());
            var before = scheduler.tasks();
            assertEquals(EngineScheduler.ReportedPhase.RECEIVED, before.getFirst().phase());
            assertEquals(8, cache.snapshot(0, false).availableBlocks(), "A status query must not allocate");
            var work = scheduler.schedule();
            complete(scheduler, work, config); drain(scheduler);
            var ready = scheduler.tasks().getFirst();
            assertEquals(EngineScheduler.ReportedPhase.KV_ALLOCATED, ready.phase());
            Thread.sleep(20);
            var later = scheduler.tasks().getFirst();
            assertEquals(ready.waitingMillis(), later.waitingMillis());
            assertEquals(-1, ready.executionMillis());
            assertEquals(-1, later.executionMillis());
            assertEquals(EngineScheduler.ReportedPhase.RECEIVED, before.getFirst().phase(), "Snapshots must not change later");
        }
    }

    @Test void multiTokenStepReservesKvBeforeForward() throws Exception {
        for (int capacity : List.of(1, 2)) {
            var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("decode"));
            var settings = (ObjectNode) root.path("component").path("config");
            settings.put("seq_size_per_block", 128);
            ((ObjectNode) settings.path("mock.compute")).putObject("decode").put("tokens.per.step", 6);
            var path = directory.resolve("wide-step-" + capacity + ".json"); json.writeValue(path.toFile(), root);
            var config = RtpConfig.load(path);
            try (var cache = new KvCache(config.blockSize(), capacity, 0)) {
                var scheduler = new RtpFifoScheduler(new Object(), config, cache); var listener = new Listener();
                try {
                    var request = scheduler.input(new EngineScheduler.Input("r", 1, 0, 123, 10, 0, System.nanoTime(),
                        Long.MAX_VALUE, new CacheSequence(128, Collections.nCopies(123, 0), List.of()), false, false, 0, true), listener);
                    scheduler.maintain(); drain(scheduler); generate(scheduler, request);
                    var work = scheduler.schedule();
                    if (capacity == 1) {
                        assertNull(work); drain(scheduler);
                        assertEquals(RequestFailure.Reason.CAPACITY, listener.terminal.get(1, TimeUnit.SECONDS).reason());
                        assertEquals(1, cache.snapshot(-1, false).freeBlocks());
                    } else {
                        assertNotNull(work); assertEquals(0, cache.snapshot(-1, false).availableBlocks());
                        assertEquals(1, work.batch().dps().getFirst().requests().getFirst().stepTokens());
                        complete(scheduler, work, config); drain(scheduler);
                        assertEquals(7, request.snapshot().generated());
                        assertEquals(0, cache.snapshot(-1, false).availableBlocks());
                        scheduler.finish(request, new RequestFailure(RequestFailure.Reason.CANCELLED, "test")); drain(scheduler);
                        assertEquals(2, cache.snapshot(-1, false).freeBlocks());
                    }
                } finally { scheduler.close(); }
            }
        }
    }

    @Test void decodeGrowthFailureReleasesOnlyItsLeaseAndUnblocksWaitingAdmission() throws Exception {
        var config = config("decode", "1", 2);
        var finishedLog = new org.flexlb.mock.engine.scheduler.TaskHistory();
        try (var cache = new KvCache(config.blockSize(), 7, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache, finishedLog)) {
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
            assertEquals(3, finishedLog.snapshot(0, List.of(), 0, 0).finished().size());
        }
    }

    @Test void prefillCapacityWaitExpiresOrCancelsWithoutRevivingAfterRelease() throws Exception {
        var config = config("prefill", "1", 2); var lock = new Object();
        try (var cache = new KvCache(config.blockSize(), 3, 0);
             var owner = cache.acquire(6, List.of()).lease();
             var engine = new RtpEngineLoop(lock, AbstractRtpScheduler.factory(config), new org.flexlb.mock.engine.scheduler.TaskHistory(), cache, config.compute(), config.role(), config.id())) {
            var expired = new Listener(); var cancelled = new Listener();
            long now = System.nanoTime();
            engine.input(new EngineScheduler.Input("1", 1, 0, 5, 3, 0, now,
                now + TimeUnit.MILLISECONDS.toNanos(100), new CacheSequence(2, Collections.nCopies(5, 0), List.of()), false, false, 0, false), expired);
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
        try (var cache = new KvCache(config.blockSize(), 2, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            var oversized = new Listener(); var small = new Listener();
            assertEquals(RequestFailure.Reason.CAPACITY,
                assertThrows(RequestFailure.class, () -> scheduler.input(input(1, false), oversized)).reason());
            scheduler.input(new EngineScheduler.Input("2", 2, 0, 1, 3, 0, System.nanoTime(),
                Long.MAX_VALUE, new CacheSequence(2, Collections.nCopies(1, 0), List.of()), false, false, 0, false), small);
            var work = scheduler.schedule(); drain(scheduler);
            assertFalse(oversized.terminal.isDone(), "Rejected requests were never admitted");
            assertNotNull(work); assertEquals(1, work.batch().dps().getFirst().requests().size());
            complete(scheduler, work, config); drain(scheduler);
            assertNull(small.terminal.get(1, TimeUnit.SECONDS));
            assertTrue(scheduler.requests().isEmpty());
            assertEquals(2, cache.snapshot(0, false).freeBlocks());
        }
    }

    private static EngineScheduler.Input sized(long id, int tokens) {
        return new EngineScheduler.Input(Long.toString(id), id, 81, tokens, 3, 0, System.nanoTime(),
            Long.MAX_VALUE, new CacheSequence(2, Collections.nCopies(tokens, 0), List.of()), false, false, 0, false);
    }
    private static List<EngineScheduler.InputItem> batch(EngineScheduler.Input... inputs) {
        return Arrays.stream(inputs).map(input -> new EngineScheduler.InputItem(input, new Listener())).toList();
    }
    private static List<Long> ids(EngineScheduler.Work work) {
        return work.batch().dps().getFirst().requests().stream().map(item -> Long.valueOf(item.id())).toList();
    }
    private RtpConfig admissionLimits(RtpConfig config, int inited, int uncached) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("prefill"));
        ((ObjectNode) root.path("component")).set("config", config.business());
        ((ObjectNode) root.path("component").path("config"))
            .put("max_inited_kv_cache_streams", inited).put("max_batch_tokens_without_cache", uncached);
        var path = directory.resolve("limits.json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }
    @Test void ordinarySchedulerIgnoresGroupMetadataAndRejectsExplicitGroupsWithoutAdmission() throws Exception {
        var config = config("prefill", "1", 4, 100);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            assertThrows(RequestFailure.class, () -> scheduler.input(batch(sized(9, 5))));
            assertTrue(scheduler.requests().isEmpty());
            assertEquals(20, cache.snapshot(0, false).freeBlocks());
            scheduler.input(sized(1, 5), new Listener());
            scheduler.input(sized(2, 5), new Listener());
            var work = scheduler.schedule();
            assertEquals(List.of(1L, 2L), ids(work));
            complete(scheduler, work, config); drain(scheduler);
            assertTrue(scheduler.requests().isEmpty());
            assertEquals(20, cache.snapshot(0, false).freeBlocks());
        }
    }

    @Test void explicitBatchesWithTheSameReportedIdNeverMergeEvenWhenSingleton() throws Exception {
        var config = config("prefill", "1", 4, 100);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpPrefillGroupScheduler(new Object(), config, cache)) {
            assertEquals(2, scheduler.input(batch(sized(1, 5), sized(2, 5))).size());
            scheduler.input(batch(sized(3, 5)));
            var first = scheduler.schedule(); assertEquals(List.of(1L, 2L), ids(first));
            complete(scheduler, first, config); drain(scheduler);
            var next = scheduler.schedule(); assertEquals(List.of(3L), ids(next));
            complete(scheduler, next, config); drain(scheduler);
            assertTrue(scheduler.requests().isEmpty()); assertEquals(20, cache.snapshot(0, false).freeBlocks());
        }
    }
    @Test void groupCapacityScanKeepsResidualMembersAndCanAdmitLaterSmallerMembers() throws Exception {
        var config = config("prefill", "1", 4, 100);
        try (var cache = new KvCache(2, 4, 0);
             var scheduler = new RtpPrefillGroupScheduler(new Object(), config, cache)) {
            scheduler.input(batch(sized(1, 5), sized(2, 5), sized(3, 1)));
            var first = scheduler.schedule(); assertEquals(List.of(1L, 3L), ids(first));
            assertEquals(1, scheduler.queues().waiting());
            complete(scheduler, first, config); drain(scheduler);
            var residual = scheduler.schedule(); assertEquals(List.of(2L), ids(residual));
            complete(scheduler, residual, config); drain(scheduler);
            assertEquals(4, cache.snapshot(0, false).freeBlocks());
        }
    }
    @Test void ordinaryAndGroupWorkAlternateAtEmptyForwardBoundariesWithoutMixing() throws Exception {
        var config = config("prefill", "1", 4, 100);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpPrefillGroupScheduler(new Object(), config, cache)) {
            scheduler.input(batch(sized(1, 5), sized(2, 5)));
            scheduler.input(sized(3, 5), new Listener());
            var normal = scheduler.schedule(); assertEquals(List.of(3L), ids(normal));
            scheduler.input(sized(4, 5), new Listener());
            complete(scheduler, normal, config); drain(scheduler);
            var group = scheduler.schedule(); assertEquals(List.of(1L, 2L), ids(group));
            complete(scheduler, group, config); drain(scheduler);
            var tail = scheduler.schedule(); assertEquals(List.of(4L), ids(tail));
            complete(scheduler, tail, config); drain(scheduler);
        }
    }
    @Test void oversizedGroupFallsBackToOrdinaryAdmissionAndCanMergeWithSingles() throws Exception {
        var config = config("prefill", "1", 2, 100);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpPrefillGroupScheduler(new Object(), config, cache)) {
            scheduler.input(batch(sized(1, 5), sized(2, 5), sized(3, 5)));
            scheduler.input(sized(4, 5), new Listener());
            var first = scheduler.schedule(); assertEquals(List.of(1L, 2L), ids(first));
            complete(scheduler, first, config); drain(scheduler);
            var tail = scheduler.schedule(); assertEquals(List.of(3L, 4L), ids(tail));
            complete(scheduler, tail, config); drain(scheduler);
        }
    }
    @Test void initializedKvLimitProtectsRetainedSourcesAndCountsAsAnOversizedGroupLimit() throws Exception {
        var config = admissionLimits(config("prefill", "1", 4, 100), 1, 0);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpPrefillGroupScheduler(new Object(), config, cache)) {
            var held = scheduler.input(input(9, true), new Listener());
            var first = scheduler.schedule(); complete(scheduler, first, config); drain(scheduler);
            scheduler.input(batch(sized(1, 5), sized(2, 5)));
            assertNull(scheduler.schedule());
            scheduler.finish(held, new RequestFailure(RequestFailure.Reason.CANCELLED, "test")); drain(scheduler);
            var next = scheduler.schedule(); assertEquals(List.of(1L), ids(next));
            complete(scheduler, next, config); drain(scheduler);
            var tail = scheduler.schedule(); assertEquals(List.of(2L), ids(tail));
            complete(scheduler, tail, config); drain(scheduler);
        }
    }
    @Test void prefillRectangleLimitsGroupsAndUncachedQuotaAppliesToOrdinaryAdmission() throws Exception {
        var config = admissionLimits(config("prefill", "1", 4, 100), 0, 1);
        try (var cache = new KvCache(2, 100, 0);
             var scheduler = new RtpPrefillGroupScheduler(new Object(), config, cache)) {
            scheduler.input(batch(sized(1, 60), sized(2, 30)));
            var first = scheduler.schedule(); assertEquals(List.of(1L), ids(first));
            complete(scheduler, first, config); drain(scheduler);
            var second = scheduler.schedule(); assertEquals(List.of(2L), ids(second));
            complete(scheduler, second, config); drain(scheduler);
            scheduler.input(batch(sized(3, 10), sized(4, 10)));
            var group = scheduler.schedule(); assertEquals(List.of(3L, 4L), ids(group));
            complete(scheduler, group, config); drain(scheduler);
            scheduler.input(sized(5, 10), new Listener()); scheduler.input(sized(6, 10), new Listener());
            var tail = scheduler.schedule(); assertEquals(List.of(5L), ids(tail));
            complete(scheduler, tail, config); drain(scheduler);
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
        }
    }
    @Test void rejectedAndCancelledGroupMembersDoNotCancelOtherMembers() throws Exception {
        var config = config("prefill", "1", 4, 100);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpPrefillGroupScheduler(new Object(), config, cache)) {
            var admitted = scheduler.input(batch(sized(1, 5), sized(1, 5), sized(2, 5)));
            assertEquals(RequestFailure.Reason.DUPLICATE, admitted.get(1).failure().reason());
            scheduler.finish(admitted.get(0).request(), new RequestFailure(RequestFailure.Reason.CANCELLED, "test")); drain(scheduler);
            var next = scheduler.schedule(); assertEquals(List.of(2L), ids(next));
            complete(scheduler, next, config); drain(scheduler);
            assertEquals(20, cache.snapshot(0, false).freeBlocks()); assertTrue(scheduler.requests().isEmpty());
        }
    }

    @Test void oneTokenGenerationCanShareAGroupWithMultiTokenGeneration() throws Exception {
        var config = config("prefill", "1", 4, 100);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpPrefillGroupScheduler(new Object(), config, cache)) {
            var one = new EngineScheduler.Input("1", 1, 81, 5, 1, 0, System.nanoTime(),
                Long.MAX_VALUE, new CacheSequence(2, Collections.nCopies(5, 0), List.of()), false, false, 0, false);
            scheduler.input(batch(one, sized(2, 5)));
            var work = scheduler.schedule(); assertEquals(List.of(1L, 2L), ids(work));
            complete(scheduler, work, config); drain(scheduler);
            assertEquals(20, cache.snapshot(0, false).freeBlocks());
        }
    }

    @Test void cpSinglePrefillPolicyAppliesToOrdinaryAndExplicitGroupAdmission() throws Exception {
        for (String mode : List.of("fifo", "group")) for (boolean single : List.of(true, false)) {
            var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("prefill"));
            ((ObjectNode) root.path("component").path("config"))
                .put("seq_size_per_block", 2).put("max_batch_tokens_size", 100)
                .put("cp_rotate_method", "ALL_GATHER").put("tp_size", 2)
                .put("cp_force_single_prefill", single).put("mock.scheduler.mode", mode);
            var config = RtpConfig.load(root);
            try (var cache = new KvCache(2, 20, 0);
                 var scheduler = AbstractRtpScheduler.factory(config).create(new Object(), cache, new TaskHistory())) {
                if (mode.equals("group")) scheduler.input(batch(sized(1, 5), sized(2, 5)));
                else { scheduler.input(sized(1, 5), new Listener()); scheduler.input(sized(2, 5), new Listener()); }
                assertEquals(single ? 1 : 2, ids(scheduler.schedule()).size());
            }
        }
    }

    @Test void uncachedAdmissionQuotaIncludesCpPadding() throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("prefill"));
        ((ObjectNode) root.path("component").path("config"))
            .put("seq_size_per_block", 2).put("max_batch_tokens_size", 100).put("max_batch_tokens_without_cache", 6)
            .put("cp_rotate_method", "ALL_GATHER").put("tp_size", 2).put("cp_force_single_prefill", false);
        var config = RtpConfig.load(root);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            scheduler.input(sized(1, 5), new Listener()); scheduler.input(sized(2, 5), new Listener());
            assertEquals(List.of(1L), ids(scheduler.schedule()));
        }
    }

    @Test void automaticTokenBudgetDoesNotSilentlyDisableBatching() throws Exception {
        var config = config("prefill", "1", 2);
        try (var cache = new KvCache(2, 20, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            scheduler.input(sized(1, 5), new Listener()); scheduler.input(sized(2, 5), new Listener());
            assertEquals(List.of(1L, 2L), ids(scheduler.schedule()));
        }
    }

    private RtpConfig boundaryConfig(String role, int sequence, int width) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration(role));
        var values = (ObjectNode) root.path("component").path("config");
        values.put("seq_size_per_block", 2).put("max_seq_len", sequence).put("concurrency_limit", 2)
            .put("max_inited_kv_cache_streams", 1);
        ((ObjectNode) values.path("mock.compute")).putObject("decode").put("tokens.per.step", width);
        return RtpConfig.load(root);
    }

    @Test void decodeClampsRequestedLengthToContext() throws Exception {
        for (int width : List.of(1, 3)) {
            var config = boundaryConfig("decode", 10, width); var listener = new Listener();
            try (var cache = new KvCache(2, 16, 0);
                 var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
                var request = scheduler.input(new EngineScheduler.Input("r", 1, 0, 4, 100, 0, System.nanoTime(), Long.MAX_VALUE,
                    new CacheSequence(2, Collections.nCopies(4, 0), List.of()), false, false, 20, true), listener);
                scheduler.maintain(); drain(scheduler); generate(scheduler, request);
                int rounds = 0;
                while (!listener.terminal.isDone()) {
                    assertTrue(rounds++ < 10);
                    var work = scheduler.schedule(); assertNotNull(work);
                    complete(scheduler, work, config); drain(scheduler);
                }
                assertNull(listener.terminal.get());
                assertEquals(6, request.snapshot().generated());
                assertEquals(width == 1 ? 5 : 2, rounds);
                var metrics = scheduler.metrics().scrape();
                assertEquals(0, RtpMetricsTest.sample(metrics, "rtp_llm_wait_stream_size", ""));
                assertEquals(5, RtpMetricsTest.sample(metrics, "rtp_llm_forward_tokens_total", "kind=\"output\""));
                assertEquals(16, cache.snapshot(0, false).freeBlocks());
            }
        }
    }

    @Test void decodeRejectsFullPromptButAllowsFirstTokenAtContextBoundary() throws Exception {
        var config = boundaryConfig("decode", 10, 2); var listener = new Listener();
        try (var cache = new KvCache(2, 16, 0);
             var scheduler = new RtpFifoScheduler(new Object(), config, cache)) {
            var invalid = new EngineScheduler.Input("invalid", 1, 0, 10, 1, 0, System.nanoTime(), Long.MAX_VALUE,
                new CacheSequence(2, Collections.nCopies(10, 0), List.of()), false, false, 0, true);
            assertEquals(RequestFailure.Reason.INVALID,
                assertThrows(RequestFailure.class, () -> scheduler.input(invalid, new Listener())).reason());
            assertTrue(scheduler.requests().isEmpty()); assertEquals(16, cache.snapshot(0, false).freeBlocks());
            var boundary = new EngineScheduler.Input("boundary", 2, 0, 9, 100, 0, System.nanoTime(), Long.MAX_VALUE,
                new CacheSequence(2, Collections.nCopies(9, 0), List.of()), false, false, 0, true);
            var request = scheduler.input(boundary, listener);
            scheduler.maintain(); drain(scheduler); generate(scheduler, request); drain(scheduler);
            assertNull(scheduler.schedule()); assertNull(listener.terminal.get(1, TimeUnit.SECONDS));
            assertEquals(1, request.snapshot().generated()); assertEquals(0, request.snapshot().iterations());
            assertTrue(listener.output.get().finished()); assertEquals(0, listener.output.get().committed());
            assertEquals(16, cache.snapshot(0, false).freeBlocks());
        }
    }

    @Test void impossiblePromptRejectsBeforeInitializedKvQuotaButTemporaryShortageStillWaits() throws Exception {
        // Same common input path is used by individual requests and explicit groups.
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("prefill"));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2)
            .put("max_inited_kv_cache_streams", 1);
        var config = RtpConfig.load(root);
        for (boolean group : List.of(false, true)) try (var cache = new KvCache(2, 3, 0);
                var scheduler = group ? new RtpPrefillGroupScheduler(new Object(), config, cache) : new RtpFifoScheduler(new Object(), config, cache)) {
            scheduler.input(input(1, true), new Listener());
            complete(scheduler, scheduler.schedule(), config); drain(scheduler);
            assertEquals(0, cache.snapshot(0, false).availableBlocks());
            var oversized = sized(2, 7);
            if (group) {
                var result = scheduler.input(batch(oversized)).getFirst();
                assertNull(result.request()); assertEquals(RequestFailure.Reason.CAPACITY, result.failure().reason());
            } else {
                assertEquals(RequestFailure.Reason.CAPACITY,
                    assertThrows(RequestFailure.class, () -> scheduler.input(oversized, new Listener())).reason());
            }
            var waiting = scheduler.input(sized(3, 5), new Listener());
            assertNull(scheduler.schedule()); assertFalse(waiting.snapshot().terminating());
            assertEquals(2, scheduler.requests().size());
        }
    }

}
