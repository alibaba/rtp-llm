package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.Status;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;
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
        ((ObjectNode) root.path("environment").path("compute").path("prefill")).put("expression", expression);
        var path = directory.resolve("config.json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }

    @Test void forwardsThenCommitsCompleteBlocksAndReportsCacheReuse() throws Exception {
        var config = config("totalComputeTokens");
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
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
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
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
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
            var old = engine.submit(input(1, 0, true)); old.cancel();
            assertEquals(Status.Code.CANCELLED, old.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
            var fresh = engine.submit(input(1, 0, true)); old.cancel();
            assertEquals(1, engine.snapshot(0).running()); fresh.cancel();
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
        }
    }

    @Test void runningTimeoutAndShutdownReleaseAllResources() throws Exception {
        var config = config("500");
        try (var cache = new DeviceCache(8, 0, true)) {
            var engine = new LocalPrefill(config, cache);
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
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
            assertEquals(Status.Code.INTERNAL, engine.submit(input(1, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
            var owner = cache.acquire(6, List.of()).lease();
            var rejected = engine.submit(input(2, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(602, rejected.engineCode()); owner.close();
            assertEquals(Status.Code.INTERNAL, engine.submit(input(3, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().getCode());
            assertEquals(8, cache.snapshot(-1, false).availableBlocks());
        }
    }

    @Test void completionCancellationRaceHasOneTerminalAndConservesCapacity() throws Exception {
        var config = config("0");
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
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
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
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
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
            assertTrue(engine.submit(input(1, 0, false)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS).status().isOk());
            assertTrue(cache.snapshot(-1, true).keys().isEmpty());
            assertEquals(8, cache.snapshot(-1, false).freeBlocks());
        }
    }

    @Test void boundedFinishedHistorySignalsCoverageGapInsteadOfReturningPartialHistory() throws Exception {
        var config = config("0");
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
            for (int i = 0; i <= LocalPrefill.LIMIT; i++)
                engine.submit(input(i, 0, true)).completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            var error = assertThrows(io.grpc.StatusRuntimeException.class, () -> engine.snapshot(0));
            assertEquals(Status.Code.OUT_OF_RANGE, error.getStatus().getCode());
            assertEquals(LocalPrefill.LIMIT, engine.snapshot(1).finished().size());
        }
    }

    @Test void activeIdsAndRequestLimitRejectWithoutAllocatingWaitingKv() throws Exception {
        var config = config("3600000");
        try (var cache = new DeviceCache(8, 0, true); var engine = new LocalPrefill(config, cache)) {
            engine.submit(input(0, 0, true));
            var duplicate = assertThrows(io.grpc.StatusRuntimeException.class, () -> engine.submit(input(0, 0, true)));
            assertEquals(Status.Code.ALREADY_EXISTS, duplicate.getStatus().getCode());
            for (int i = 1; i < LocalPrefill.LIMIT; i++) engine.submit(input(i, 0, true));
            var full = assertThrows(io.grpc.StatusRuntimeException.class, () -> engine.submit(input(LocalPrefill.LIMIT, 0, true)));
            assertEquals(Status.Code.RESOURCE_EXHAUSTED, full.getStatus().getCode());
            assertEquals(5, cache.snapshot(-1, false).availableBlocks());
            assertEquals(LocalPrefill.LIMIT - 1, engine.snapshot(0).waiting());
            engine.close(); assertEquals(8, cache.snapshot(-1, false).availableBlocks());
        }
    }
}
