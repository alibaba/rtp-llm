package org.flexlb.mockengine;

import io.grpc.stub.StreamObserver;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/** Open-loop CPU replay. All turns exist independently of completion, errors and scheduling. */
@EnabledIfSystemProperty(named = "flexlb.overloadReplay", matches = "true")
class PrefillOverloadRecoveryTest {
    private static final int WORKERS = 30;
    private static final int SESSIONS = 270;
    private static final int NORMAL = 258;
    private static final int BLOCK_SIZE = 512;
    private static final String CONFIG = """
            {"dispatchTimeoutMs":30000,"scheduler":{"decision":{"type":"SINGLE"},
            "ordering":{"type":"PRIORITY","defaultPriority":50,"prefillCostOrder":true,
            "preemption":{"allowedVictimStages":["PREFILL_QUEUED","DECODE_RESERVED","DECODE_ENGINE_OWNED"],
            "timeoutMs":1000}}},"dispatcher":{"maxInflightPerPrefillWorker":256},"router":{"roles":{
            "prefill":{"tokenBalance":{"engineQueueAdmissionThreshold":80}},
            "decode":{"availability":{"maxEngineRequests":1024}}}}}
            """;
    private final GlobalCacheIndex index = new GlobalCacheIndex();
    private final List<ReplayPrefillWorker> engines = new ArrayList<>();
    private final List<Integer> order = new ArrayList<>();
    private final int[] turns = new int[SESSIONS];
    private final AtomicInteger admitted = new AtomicInteger();
    private final AtomicInteger completed = new AtomicInteger();
    private final AtomicInteger rejected = new AtomicInteger();
    private final AtomicReference<Throwable> error = new AtomicReference<>();

    private static List<Long> keys(int session, int turn) {
        var keys = new ArrayList<Long>();
        for (int i = 0; i < 120; i++) {
            keys.add(i < 40 ? 900_000L + i : 10_000_000L + 10_000L * session + i);
        }
        long branch = 100_000_000L + (session + (long) (turn / 16) * SESSIONS) * 8192;
        for (int i = 0; i < 16 * (1 + turn % 16); i++) { keys.add(branch + i); }
        return keys;
    }

    private void publishCaches(AutoTpmE2EHarness h) {
        for (int i = 0; i < WORKERS; i++) {
            var worker = h.prefillEndpoint(i);
            if (engines.get(i).publish(index, worker)) { h.endpointRegistry.publishCacheAvailability(worker.getStatus()); }
        }
    }

    private void submit(AutoTpmE2EHarness h, int sequence, Map<PrefillEndpoint, Integer> workers) {
        int session = order.get(sequence % SESSIONS);
        var keys = keys(session, turns[session]++);
        int tokens = keys.size() * BLOCK_SIZE;
        int priority = switch (sequence % 20) {
            case 0 -> 39;
            case 1, 2, 3 -> 40;
            case 4, 5 -> 49;
            default -> 50;
        };
        long id = 9_000_000L + sequence;
        var ctx = h.context(id, priority, 1, 1);
        ctx.data().setRequest(ctx.data().getRequest().toBuilder().seqLen((long) tokens)
                .blockCacheKeys(keys).cacheKeyBlockSize(BLOCK_SIZE).build());
        ctx.data().setSchedulingMetadata(SchedulingMetadata.explicit(priority, ctx.data().getStartTime() + 30_000));
        h.scheduler.submit(ctx).thenAccept(response -> {
            if (!response.isSuccess()) {
                if (response.getCode() != 8431) { throw new AssertionError("unexpected code " + response.getCode()); }
                rejected.incrementAndGet();
                return;
            }
            admitted.incrementAndGet();
            var route = ctx.snapshot().route();
            var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(id);
            // The in-process mock accepts lengths and cache keys without allocating a token-id array.
            input.getGenerateConfigBuilder().setMaxNewTokens(1)
                    .setUniqueKey("{\"input_len\":" + tokens + ",\"block_cache_keys\":" + keys + "}")
                    .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                            .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE).setRoleStr("DECODE")
                            .setGrpcPort(route.decodeEp().getStatus().topologySnapshot().grpcPort()));
            h.prefillEngines.get(workers.get(route.prefillEp())).generateStreamCall(input.build(), new StreamObserver<>() {
                public void onNext(EngineRpcService.GenerateOutputsPB output) {
                    if (output.hasErrorInfo() && output.getErrorInfo().getErrorCodeValue() != 0) {
                        error.compareAndSet(null, new AssertionError(output.getErrorInfo().toString()));
                    }
                }
                public void onError(Throwable t) { error.compareAndSet(null, t); }
                public void onCompleted() { completed.incrementAndGet(); }
            });
        }).exceptionally(t -> { error.compareAndSet(null, t); return null; });
    }

    private record Sample(int completed, long input, long hit, long waiting, int pending) { }

    private Sample sample(int offered) {
        long input = 0, hit = 0, waiting = 0;
        for (var worker : engines) {
            input += worker.input.get();
            hit += worker.hits.get();
            waiting += worker.waiting.get();
        }
        return new Sample(completed.get(), input, hit, waiting, offered - admitted.get() - rejected.get());
    }

    @Test
    void recoversAfterAnOpenLoopLoadStep() throws Exception {
        int peak = 305;
        int recovery = Integer.getInteger("flexlb.recoverySeconds", 180);
        int stepSeconds = Boolean.getBoolean("flexlb.matchedFlat") ? 19 : 16;
        int stepRequests = peak * 16;
        int recoveryAt = 12 + stepSeconds;
        if (recovery < 14) { throw new IllegalArgumentException("recovery must include the measurement window"); }
        var cfg = ConfigService.parse(CONFIG);
        var model = Map.<String, Object>of("test_pool_blocks", 119580, "test_decode_pool_blocks", 119580,
                "jitter_pct", 0, "prefill", Map.of("max_batch_requests", 32, "max_batch_tokens", 256 * BLOCK_SIZE * 32));
        try (var h = new AutoTpmE2EHarness(58820, WORKERS, 50, "10 + 0.0125 * sum(computeTokens)", 1,
                true, false, cfg.queueScheduler().getDecision(), true, cfg, 10, model)) {
            var workers = new HashMap<PrefillEndpoint, Integer>();
            for (int i = 0; i < WORKERS; i++) {
                workers.put(h.prefillEndpoint(i), i);
                engines.add(new ReplayPrefillWorker(h.prefillEngines.get(i)));
            }
            for (int s = 0; s < SESSIONS; s++) {
                turns[s] = s * 120 / SESSIONS;
                order.add(s);
                var first = keys(s, turns[s]);
                MockEngineTestSupport.seedCache(engines.get(s % WORKERS).cache, first.subList(0, first.size() - 16));
            }
            Collections.shuffle(order, new Random(20261120));
            publishCaches(h);
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call ->
                    index.batchCalculatePrefixMatchLength(call.getArgument(2), call.getArgument(0)));
            h.startAutoPump(20);
            var samples = new ArrayList<Sample>();
            samples.add(sample(0));
            long start = System.nanoTime(), nextSync = start;
            int sent = 0;
            for (int sec = 0; sec < recoveryAt + recovery + 180; sec++) {
                int offset = sec - 12;
                int qps = NORMAL;
                if (offset >= 0 && sec < recoveryAt) {
                    // Stretch exactly the same requests across 19 s for the flat control.
                    qps = (offset + 1) * stepRequests / stepSeconds - offset * stepRequests / stepSeconds;
                } else if (sec >= recoveryAt + recovery) { qps = 0; }
                int ticks = qps == 0 ? 10 : qps;
                for (int j = 0; j < ticks; j++) {
                    long due = start + TimeUnit.SECONDS.toNanos(sec) + j * 1_000_000_000L / ticks;
                    LockSupport.parkNanos(due - System.nanoTime());
                    if (System.nanoTime() >= nextSync) {
                        publishCaches(h);
                        nextSync = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(200);
                    }
                    if (qps != 0) { submit(h, sent++, workers); }
                }
                long end = start + TimeUnit.SECONDS.toNanos(sec + 1);
                LockSupport.parkNanos(end - System.nanoTime());
                h.discardRecordedCalls();
                var now = sample(sent);
                samples.add(now);
                assertNull(error.get(), () -> String.valueOf(error.get()));
                System.out.printf("sec=%d sent=%d completed=%d rejected=%d masterPending=%d workerWaiting=%d "
                                + "inputTokens=%d freshTokens=%d hitTokens=%d duplicateKeys=%d%n",
                        sec + 1, sent, now.completed(), rejected.get(), sent - admitted.get() - rejected.get(),
                        now.waiting(), now.input(), now.input() - now.hit(), now.hit(), index.totalMappings() - index.totalBlocks());
                if (qps != 0) {
                    assertTrue(System.nanoTime() - end < TimeUnit.SECONDS.toNanos(1), "arrival generator missed schedule");
                } else if (now.completed() + rejected.get() == sent) { break; }
            }
            assertEquals(sent, completed.get() + rejected.get(), "all offered requests must settle");
            h.assertAutoPumpHealthy();
            checkWindow(samples, 2, 11, 0.90, 0.90);
            int window = Math.min(60, recovery);
            int last = recoveryAt + recovery;
            checkWindow(samples, last - window, last, 0.90, 0.90);
            if (window == 60) {
                double waiting = samples.subList(last - window + 1, last + 1).stream()
                        .mapToLong(Sample::waiting).average().orElseThrow();
                double pending = samples.subList(last - window + 1, last + 1).stream()
                        .mapToInt(Sample::pending).average().orElseThrow();
                assertTrue(waiting < WORKERS * 4, "worker queues did not drain: " + waiting);
                assertTrue(pending < NORMAL / 4.0, "Master backlog persisted: " + pending);
            }
        }
    }

    private static void checkWindow(List<Sample> samples, int from, int to, double minGoodput, double minHitRate) {
        var a = samples.get(from);
        var b = samples.get(to);
        double goodput = (b.completed() - a.completed()) / (double) (to - from);
        double hitRate = (b.hit() - a.hit()) / (double) (b.input() - a.input());
        System.out.printf("window=%d-%d completedQps=%.1f hit=%.3f%n", from, to, goodput, hitRate);
        assertTrue(goodput >= NORMAL * minGoodput, "goodput " + goodput);
        assertTrue(hitRate >= minHitRate, "hit rate " + hitRate);
    }
}
