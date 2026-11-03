package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.mock.cache.*;
import org.flexlb.mock.cache.remote.RemoteCache;
import org.flexlb.mock.compute.SimulatedCompute;
import org.flexlb.mock.engine.*;
import org.junit.jupiter.api.Test;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BooleanSupplier;
import static org.junit.jupiter.api.Assertions.*;

class RtpRemoteSchedulerTest {
    private static RtpConfig config(int batch, int inited) throws Exception {
        var root = new ObjectMapper().<ObjectNode>valueToTree(RtpNodeTest.configuration("prefill"));
        ((ObjectNode) root.path("component").path("config"))
            .put("seq_size_per_block", 2).put("reuse_cache", true).put("concurrency_limit", batch)
            .put("max_inited_kv_cache_streams", inited).put("max_batch_tokens_size", 100);
        return RtpConfig.load(root);
    }
    private static Scheduler.Input input(long id, boolean reuse, long deadline) {
        return new Scheduler.Input(Long.toString(id), id, 0, 5, 1, 0, System.nanoTime(), deadline,
            new CacheSequence(2, List.of((int) id, 2, 3, 4, 5), List.of()), reuse, false, 0, false);
    }
    private static class Listener implements Scheduler.Listener {
        final CompletableFuture<Scheduler.Output> output = new CompletableFuture<>();
        final CompletableFuture<RequestFailure> terminal = new CompletableFuture<>();
        public void output(Scheduler.Request request, Scheduler.Output value) { output.complete(value); }
        public void terminal(Scheduler.Request request, RequestFailure failure) { terminal.complete(failure); }
    }
    private static class Backend implements RemoteCache {
        final CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        volatile int calls;
        volatile boolean ignoreInterrupt;
        public Match match(List<Long> keys, int local, long deadline) { calls++; return () -> keys.size(); }
        public void read(Match match, List<Long> keys, long deadline) throws Exception {
            entered.countDown();
            while (true) {
                try { release.await(); return; }
                catch (InterruptedException error) { if (!ignoreInterrupt) throw error; }
            }
        }
        public void write(List<Long> keys, long deadline) {}
        public void close() { release.countDown(); }
    }
    private static KvCache cache(RemoteCache remote) {
        return new KvCache(2, 16, 0, KvCache.EvictionPolicy.LRU, true, 0, KvCache.EvictionPolicy.LRU, null, null, remote);
    }
    private static void await(BooleanSupplier condition) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean()) { assertTrue(System.nanoTime() < end, "Timed out"); Thread.sleep(1); }
    }
    private static void drain(Scheduler scheduler) {
        for (var callbacks = scheduler.takeNotifications(); !callbacks.isEmpty(); callbacks = scheduler.takeNotifications())
            callbacks.forEach(Runnable::run);
    }
    private static void complete(Scheduler scheduler, Scheduler.Work work, RtpConfig config) {
        var sample = config.compute().plan(config.role(), config.id(), work.batch());
        scheduler.computeCompleted(work, new SimulatedCompute.Outcome(SimulatedCompute.State.COMPLETED, sample, 1000));
        drain(scheduler);
    }

    @Test void pendingReadDoesNotBlockOtherRequestsAndOnlyReadyPrefixReducesForwardWork() throws Exception {
        var config = config(2, 0); var remote = new Backend();
        try (var cache = cache(remote); var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var loaded = new Listener(); var ordinary = new Listener();
            var request = scheduler.input(input(1, true, Long.MAX_VALUE), loaded);
            scheduler.input(input(2, false, Long.MAX_VALUE), ordinary);
            var first = scheduler.schedule();
            assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
            assertEquals("2", first.batch().dps().getFirst().requests().getFirst().id());
            assertEquals(1, scheduler.queues().loading()); assertEquals(0, request.reused());
            complete(scheduler, first, config); assertTrue(ordinary.output.isDone()); assertFalse(loaded.output.isDone());
            assertNull(scheduler.schedule()); assertEquals(1, remote.calls);
            remote.release.countDown();
            await(() -> cache.snapshot(-1, true).keys().size() == 2);
            await(() -> { scheduler.maintain(); return scheduler.queues().loading() == 0; });
            assertEquals(1, scheduler.queues().waiting());
            var second = scheduler.schedule(); var item = second.batch().dps().getFirst().requests().getFirst();
            assertEquals("1", item.id()); assertEquals(4, item.hitCacheTokens()); assertEquals(1, item.computeTokens());
            complete(scheduler, second, config); assertTrue(loaded.output.isDone());
        }
    }

    @Test void loadsCountAgainstAdmissionAndInitializedKvLimits() throws Exception {
        var config = config(1, 1); var remote = new Backend();
        try (var cache = cache(remote); var scheduler = new RtpScheduler(new Object(), config, cache)) {
            scheduler.input(input(1, true, Long.MAX_VALUE), new Listener());
            scheduler.input(input(2, true, Long.MAX_VALUE), new Listener());
            assertNull(scheduler.schedule()); assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
            assertEquals(1, remote.calls); assertEquals(new QueueSnapshot(1, 0, 1, 0), scheduler.queues());
            assertNull(scheduler.schedule()); assertEquals(1, remote.calls);
            remote.release.countDown();
            await(() -> { scheduler.maintain(); return scheduler.queues().loading() == 0; });
            var work = scheduler.schedule(); assertNotNull(work);
            assertEquals("1", work.batch().dps().getFirst().requests().getFirst().id());
            complete(scheduler, work, config);
        }
    }

    @Test void loadingMembersAlsoConsumeTheRoundsPrefillRectangleBudget() throws Exception {
        var values = new ObjectMapper().<ObjectNode>valueToTree(RtpNodeTest.configuration("prefill"));
        ((ObjectNode) values.path("component").path("config")).put("seq_size_per_block", 2)
            .put("reuse_cache", true).put("concurrency_limit", 4).put("max_batch_tokens_size", 8);
        var config = RtpConfig.load(values); var remote = new Backend();
        try (var cache = cache(remote); var scheduler = new RtpScheduler(new Object(), config, cache)) {
            scheduler.input(input(1, true, Long.MAX_VALUE), new Listener());
            scheduler.input(input(2, true, Long.MAX_VALUE), new Listener());
            assertNull(scheduler.schedule()); assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
            assertEquals(new QueueSnapshot(1, 0, 1, 0), scheduler.queues());
            assertEquals(1, remote.calls);
        }
    }

    @Test void cancellationCannotPublishIntoAReplacementWithTheSameIdentity() throws Exception {
        var config = config(2, 0); var remote = new Backend(); remote.ignoreInterrupt = true;
        try (var cache = cache(remote); var scheduler = new RtpScheduler(new Object(), config, cache)) {
            var oldListener = new Listener();
            var old = scheduler.input(input(1, true, Long.MAX_VALUE), oldListener);
            assertNull(scheduler.schedule()); assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
            scheduler.finish(old, new RequestFailure(RequestFailure.Reason.CANCELLED, "test")); drain(scheduler);
            assertTrue(oldListener.terminal.isDone()); assertEquals(0, scheduler.queues().loading());
            var replacement = scheduler.input(input(1, false, Long.MAX_VALUE), new Listener());
            var work = scheduler.schedule(); assertNotNull(work); complete(scheduler, work, config);
            remote.release.countDown(); await(() -> cache.snapshot(0, false).freeBlocks() == 16);
            assertEquals(0, replacement.reused()); assertTrue(cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void engineLoopIsWokenByReadCompletionWithoutNewTraffic() throws Exception {
        var config = config(2, 0); var remote = new Backend(); var lock = new Object();
        try (var cache = cache(remote);
             var loop = new EngineLoop(lock, new RtpScheduler(lock, config, cache), cache, config.compute(), config.role(), config.id())) {
            var listener = new Listener();
            loop.input(input(1, true, Long.MAX_VALUE), listener);
            assertTrue(remote.entered.await(2, TimeUnit.SECONDS)); assertFalse(listener.output.isDone());
            remote.release.countDown();
            assertTrue(listener.output.get(2, TimeUnit.SECONDS).finished());
            assertNull(listener.terminal.get(2, TimeUnit.SECONDS));
        }
    }

    @Test void deadlineTerminatesLoadingWithoutWaitingForAnUncooperativeRead() throws Exception {
        var config = config(2, 0); var remote = new Backend(); remote.ignoreInterrupt = true; var lock = new Object();
        try (var cache = cache(remote);
             var loop = new EngineLoop(lock, new RtpScheduler(lock, config, cache), cache, config.compute(), config.role(), config.id())) {
            var listener = new Listener();
            loop.input(input(1, true, System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(300)), listener);
            assertTrue(remote.entered.await(2, TimeUnit.SECONDS));
            assertEquals(RequestFailure.Reason.TIMEOUT, listener.terminal.get(2, TimeUnit.SECONDS).reason());
            assertFalse(listener.output.isDone()); assertTrue(cache.snapshot(-1, true).keys().isEmpty());
            remote.release.countDown(); await(() -> cache.snapshot(0, false).freeBlocks() == 16);
        }
    }

    @Test void failedReadFallsBackToLocalComputeWithoutAnotherLookup() throws Exception {
        var config = config(2, 0);
        var remote = new Backend() {
            @Override public void read(Match match, List<Long> keys, long deadline) throws Exception {
                throw new java.io.IOException("unavailable");
            }
        };
        try (var cache = cache(remote); var scheduler = new RtpScheduler(new Object(), config, cache)) {
            scheduler.input(input(1, true, Long.MAX_VALUE), new Listener());
            Scheduler.Work work = scheduler.schedule();
            if (work == null) {
                await(() -> { scheduler.maintain(); return scheduler.queues().loading() == 0; });
                work = scheduler.schedule();
            }
            assertNotNull(work);
            assertEquals(5, work.batch().dps().getFirst().requests().getFirst().computeTokens());
            assertEquals(1, remote.calls); complete(scheduler, work, config);
        }
    }

    @Test void requestTerminalDoesNotWaitForBackgroundStoreButItsBlocksStayPinned() throws Exception {
        var config = config(2, 0); var lock = new Object();
        var writing = new CountDownLatch(1); var finishWrite = new CountDownLatch(1);
        var remote = new Backend() {
            @Override public void write(List<Long> keys, long deadline) {
                writing.countDown();
                try { finishWrite.await(); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
            }
            @Override public void close() { super.close(); finishWrite.countDown(); }
        };
        remote.release.countDown();
        try (var cache = cache(remote);
             var loop = new EngineLoop(lock, new RtpScheduler(lock, config, cache), cache, config.compute(), config.role(), config.id())) {
            var listener = new Listener(); loop.input(input(1, true, Long.MAX_VALUE), listener);
            assertNull(listener.terminal.get(2, TimeUnit.SECONDS));
            assertTrue(writing.await(2, TimeUnit.SECONDS));
            assertEquals(14, cache.snapshot(0, false).availableBlocks());
            finishWrite.countDown(); await(() -> cache.snapshot(0, false).availableBlocks() == 16);
        }
    }

    @Test void remoteAdapterCannotBeCombinedWithGroupScheduling() throws Exception {
        var config = config(2, 0);
        try (var cache = cache(new Backend())) {
            assertThrows(IllegalArgumentException.class, () -> new RtpGroupScheduler(new Object(), config, cache));
        }
    }
}
