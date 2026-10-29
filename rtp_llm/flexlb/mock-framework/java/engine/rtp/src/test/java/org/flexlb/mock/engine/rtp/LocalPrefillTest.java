package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.Status;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.cache.KvCache;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LocalPrefillTest {
    @TempDir Path directory;
    static GenerateInputPB input(long id, int timeout, boolean reuse) {
        return GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(List.of(1, 2, 3, 4, 5))
            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(1).setTimeoutMs(timeout).setReuseCache(reuse)).build();
    }
    RtpConfig config(String expression) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("prefill"));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2).put("reuse_cache", true);
        ((ObjectNode) root.path("component").path("config").path("compute").path("forward")).put("expression", expression);
        var path = directory.resolve("config.json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }

    @Test void forwardsThenCommitsCompleteBlocksAndReportsCacheReuse() throws Exception {
        var config = config("totalComputeTokens");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            assertTrue(engine.snapshot(-1).finished().isEmpty());
            var first = engine.submit(input(1, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertTrue(first.status().isOk());
            assertTrue(first.output().getFlattenOutput().getAuxInfo(0).getFirstTokenCostTimeUs() >= 5000);
            assertEquals(2, cache.snapshot(-1, true).keys().size());
            var second = engine.submit(input(2, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(4, second.output().getFlattenOutput().getAuxInfo(0).getTotalReuseLen());
            assertTrue(second.output().getFlattenOutput().getAuxInfo(0).getFirstTokenCostTimeUs() >= 1000);
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            assertEquals(2, engine.snapshot(0).finishedVersion());
            assertEquals(0, engine.snapshot(2).finished().size());
        }
    }

    @Test void queuedTimeoutNeverOwnsKvAndDoesNotResetItsBudget() throws Exception {
        var config = config("500");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            var first = engine.submit(input(1, 0, true));
            var queued = engine.submit(input(2, 25, true));
            var result = queued.completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(Status.Code.DEADLINE_EXCEEDED, result.status().getCode()); assertEquals(603, result.engineCode());
            assertEquals(5, cache.snapshot(-1, false).availableBlocks());
            assertEquals(1, engine.snapshot(0).running()); assertEquals(0, engine.snapshot(0).waiting());
            var rpcBound = engine.submit(input(3, 0, true), System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(25));
            assertEquals(Status.Code.DEADLINE_EXCEEDED, rpcBound.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
            assertEquals(5, cache.snapshot(-1, false).availableBlocks());
            first.cancel();
        }
    }

    @Test void cancellationReleasesLeaseAndCannotCommitLaterOrCancelNewSameId() throws Exception {
        var config = config("500");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            var old = engine.submit(input(1, 0, true)); old.cancel();
            assertEquals(Status.Code.CANCELLED, old.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
            var fresh = engine.submit(input(1, 0, true)); old.cancel();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (engine.snapshot(0).running() == 0) {
                assertTrue(System.nanoTime() < deadline); Thread.sleep(1);
            }
            fresh.cancel();
            assertEquals(Status.Code.CANCELLED, fresh.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
        }
    }

    @Test void runningTimeoutAndShutdownReleaseAllResources() throws Exception {
        var config = config("500");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU)) {
            var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config));
            try {
                var timed = engine.submit(input(1, 25, true));
                assertEquals(Status.Code.DEADLINE_EXCEEDED, timed.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
                assertEquals(8, cache.snapshot(-1, false).availableBlocks());
                var running = engine.submit(input(2, 0, true)); var queued = engine.submit(input(3, 0, true));
                engine.close();
                for (var session : List.of(running, queued))
                    assertEquals(Status.Code.UNAVAILABLE, session.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
                assertEquals(8, cache.snapshot(-1, false).availableBlocks());
                assertTrue(cache.snapshot(-1, true).keys().isEmpty());
            } finally { engine.close(); }
        }
    }

    @Test void allocationAndFormulaFailureDoNotLeakOrBlockTheNextRequest() throws Exception {
        var config = config("sqrt(-1)");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            assertEquals(Status.Code.INTERNAL, engine.submit(input(1, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            try (var owner = cache.acquire(6, List.of()).lease()) {
                var waiting = engine.submit(input(2, 0, true)).completion().toCompletableFuture();
                assertFalse(waiting.isDone()); assertEquals(1, engine.snapshot(0).waiting());
                owner.close();
                assertEquals(Status.Code.INTERNAL, waiting.get(2, TimeUnit.SECONDS).status().getCode());
            }
            assertEquals(Status.Code.INTERNAL, engine.submit(input(3, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
        }
    }

    @Test void completionCancellationRaceHasOneTerminalAndConservesCapacity() throws Exception {
        var config = config("0");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            for (int i = 0; i < 100; i++) {
                var session = engine.submit(input(i, 0, true)); session.cancel();
                var result = session.completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
                assertTrue(result.status().isOk() || result.status().getCode() == Status.Code.CANCELLED);
                assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            }
            assertEquals(100, engine.snapshot(0).finishedVersion());
            var text = engine.scrapeMetrics();
            assertEquals(100, LocalRpcTest.sample(text, "mock_engine_requests_started_total", ""));
            assertEquals(100, LocalRpcTest.sample(text, "mock_engine_requests_finished_total", "outcome=\"success\"")
                + LocalRpcTest.sample(text, "mock_engine_requests_finished_total", "outcome=\"cancelled\""));
        }
    }

    @Test void noCacheReuseForLastPromptTokenAndPerRequestSwitchIsHonored() throws Exception {
        var config = config("0");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            var exact = input(1, 0, true).toBuilder().clearTokenIds().addAllTokenIds(List.of(1, 2, 3, 4)).build();
            engine.submit(exact).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            var reused = engine.submit(exact.toBuilder().setRequestId(2).build()).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(2, reused.output().getFlattenOutput().getAuxInfo(0).getTotalReuseLen());
            var disabled = engine.submit(input(3, 0, false)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(0, disabled.output().getFlattenOutput().getAuxInfo(0).getTotalReuseLen());
        }
    }

    @Test void disabledReuseDoesNotPublishNewCacheKeys() throws Exception {
        var config = config("0");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            assertTrue(engine.submit(input(1, 0, false)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().isOk());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
            assertEquals(8, cache.snapshot(-1, false).freeBlocks());
        }
    }

    @Test void boundedFinishedHistoryDoesNotBlockCurrentStateOrCursorRecovery() throws Exception {
        var config = config("0");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            for (int i = 0; i <= RtpConfig.REQUEST_LIMIT; i++)
                engine.submit(input(i, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            var snapshot = engine.snapshot(0);
            assertEquals(RtpConfig.REQUEST_LIMIT, snapshot.finished().size());
            assertEquals(RtpConfig.REQUEST_LIMIT + 1, snapshot.finishedVersion());
            assertTrue(engine.snapshot(snapshot.finishedVersion()).finished().isEmpty());
            assertEquals(RtpConfig.REQUEST_LIMIT, engine.snapshot(1).finished().size());
        }
    }

    @Test void activeIdsAndRequestLimitRejectWithoutAllocatingWaitingKv() throws Exception {
        var config = config("3600000");
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU); var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            engine.submit(input(0, 0, true));
            var duplicate = assertThrows(io.grpc.StatusRuntimeException.class, () -> engine.submit(input(0, 0, true)));
            assertEquals(Status.Code.ALREADY_EXISTS, duplicate.getStatus().getCode());
            for (int i = 1; i < RtpConfig.REQUEST_LIMIT; i++) engine.submit(input(i, 0, true));
            var full = assertThrows(io.grpc.StatusRuntimeException.class, () -> engine.submit(input(RtpConfig.REQUEST_LIMIT, 0, true)));
            assertEquals(Status.Code.RESOURCE_EXHAUSTED, full.getStatus().getCode());
            assertEquals(5, cache.snapshot(-1, false).availableBlocks());
            assertEquals(RtpConfig.REQUEST_LIMIT - 1, engine.snapshot(0).waiting());
            engine.close(); assertEquals(8, cache.snapshot(-1, false).availableBlocks());
        }
    }
    @Test void hostOnlyPrefillReusesLocalKvWithoutALoadingStage() throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("prefill"));
        var settings = (ObjectNode) root.path("component").path("config");
        settings.put("seq_size_per_block", 2).put("reuse_cache", true).put("enable_device_cache", false)
            .put("enable_memory_cache", true).put("memory_cache_blocks", 8);
        var path = directory.resolve("host.json"); json.writeValue(path.toFile(), root); var config = RtpConfig.load(path);
        try (var cache = new KvCache(8, 0, KvCache.EvictionPolicy.LRU, false, 8, KvCache.EvictionPolicy.LRU, null, null);
             var engine = new LocalPrefill(config, cache, RtpScheduler.factory(config))) {
            assertTrue(engine.submit(input(1, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().isOk());
            assertEquals(6, cache.hostSnapshot().free()); assertEquals(8, cache.snapshot(-1, false).freeBlocks());
            var reused = engine.submit(input(2, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(4, reused.output().getFlattenOutput().getAuxInfo(0).getTotalReuseLen());
            assertEquals(6, cache.hostSnapshot().free()); assertEquals(8, cache.snapshot(-1, false).freeBlocks());
        }
    }

}
