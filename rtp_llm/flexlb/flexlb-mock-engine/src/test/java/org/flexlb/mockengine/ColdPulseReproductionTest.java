package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/** Opt-in paired synthetic cold-prefix experiment; no network transport or production endpoints. */
@EnabledIfSystemProperty(named = "flexlb.cold.reproduction", matches = "true")
class ColdPulseReproductionTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int PREFILL = 25;
    private static final int DECODE = 192;
    private static final long FIRST_ID = 40_000_000L;
    private static final long UNIQUE_KEYS = 1_000_000_000L;
    private record Match(long atMs, long maximumHitTokens, int bestWorkers, int acceptingWorkers) { }

    @Test
    @Timeout(1800)
    void measureColdPulseAndRecovery() throws Exception {
        var snapshotPath = Path.of(System.getProperty("flexlb.cold.snapshot"));
        var snapshot = JSON.readTree(Files.readString(snapshotPath));
        var config = ConfigService.parse(snapshot.required("config").toString());
        String scenario = System.getProperty("flexlb.cold.scenario", "cold_pulse");
        int warmSeconds = Integer.getInteger("flexlb.cold.warmSeconds", 90);
        int pulseSeconds = Integer.getInteger("flexlb.cold.pulseSeconds", 30);
        int recoverySeconds = Integer.getInteger("flexlb.cold.recoverySeconds", 900);
        int cohortsPerWorker = Integer.getInteger("flexlb.cold.cohortsPerWorker", 224);
        int poolBlocks = Integer.getInteger("flexlb.cold.poolBlocks", 4096);
        double targetQps = Double.parseDouble(System.getProperty("flexlb.cold.targetQps", "200"));
        double pulseQps = Double.parseDouble(System.getProperty("flexlb.cold.pulseQps", Double.toString(targetQps)));
        long feedbackDelayMs = Long.getLong("flexlb.cold.feedbackDelayMs", 20L);
        int input = Integer.getInteger("flexlb.cold.inputTokens", 9216);
        int prefixBlocks = Integer.getInteger("flexlb.cold.prefixBlocks", 16);
        int output = Integer.getInteger("flexlb.cold.outputTokens", 16);
        double fraction = Double.parseDouble(System.getProperty("flexlb.cold.fraction", "0.05"));
        long seed = Long.getLong("flexlb.cold.seed", 20261112L);
        int blocks = input / 512;
        assertTrue(input % 512 == 0 && prefixBlocks < blocks && prefixBlocks * cohortsPerWorker < poolBlocks);
        assertTrue(fraction >= 0 && fraction <= 1);
        assertTrue(Double.isFinite(targetQps) && targetQps > 0 && Double.isFinite(pulseQps) && pulseQps > 0);
        double computeTokensPerMs = Double.parseDouble(System.getProperty("flexlb.cold.computeTokensPerMs",
                Double.toString((input - prefixBlocks * 512) / 100.0)));
        assertTrue(Double.isFinite(computeTokensPerMs) && computeTokensPerMs > 0);
        String formula = "sum(computeTokens) / " + computeTokensPerMs;
        Path out = Path.of(System.getProperty("flexlb.cold.output"));
        Files.createDirectories(out);
        var manifest = new LinkedHashMap<String, Object>();
        manifest.put("scenario", scenario);
        manifest.put("config", snapshot.required("config"));
        manifest.put("sourceRevision", System.getProperty("flexlb.cold.sourceRevision", "unspecified"));
        manifest.put("productionBinaryCommitVerified", false);
        manifest.put("snapshotSha256", java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(snapshotPath))));
        manifest.put("topology", Map.of("prefill", PREFILL, "decode", DECODE, "schedulerInstances", 1));
        manifest.put("phaseSeconds", List.of(warmSeconds, pulseSeconds, recoverySeconds));
        manifest.put("targetQps", targetQps);
        manifest.put("pulseQps", pulseQps);
        manifest.put("feedbackDelayMs", feedbackDelayMs);
        manifest.put("fraction", fraction);
        manifest.put("seed", seed);
        manifest.put("inputTokens", input);
        manifest.put("outputTokens", output);
        manifest.put("priority", 50);
        manifest.put("prefixBlocks", prefixBlocks);
        manifest.put("cohortsPerWorker", cohortsPerWorker);
        manifest.put("poolBlocksPerWorker", poolBlocks);
        manifest.put("blockTokens", 512);
        manifest.put("prefillFormulaMs", formula);
        manifest.put("limitations", List.of("one Master scheduler; no multi-master accounting effects", "synthetic capacity about 250 warm QPS, not GPU calibrated",
                "live non-touching mock cache index, no real cache-index sync delay", "mock status is not atomic across running/finished; residual context alone is not a production bug",
                "production binary/source version not independently verified", "no frontend process, no real Worker TTL runner"));
        JSON.writerWithDefaultPrettyPrinter().writeValue(out.resolve("manifest.json").toFile(), manifest);
        var perf = Map.<String, Object>of("test_pool_blocks", poolBlocks, "jitter_pct", 0,
                "prefill", Map.of("max_batch_requests", 32, "max_batch_tokens", 131072),
                "decode", Map.of("tokens_per_step", 1, "step_base_ms", 1, "step_per_running_ms", 0));
        var configuredLifetime = config.getInternalRuntime().getEngineVisibilityMultiplier();
        try (var events = new Recorder(out.resolve("events.jsonl")); var samples = new Recorder(out.resolve("samples.jsonl"));
             var h = new AutoTpmE2EHarness(61000, PREFILL, DECODE, formula, 1, false, false,
                     config.queueScheduler().getDecision(), true, config, feedbackDelayMs, perf)) {
            config.getInternalRuntime().setEngineVisibilityMultiplier(configuredLifetime);
            var caches = new ArrayList<MockLruBlockCache>();
            var prefixes = new ArrayList<List<Long>>();
            for (int i = 0; i < PREFILL; i++) {
                var cache = (MockLruBlockCache) ReflectionTestUtils.getField(h.prefillEngines.get(i), "cache");
                assertNotNull(cache);
                caches.add(cache);
            }
            for (int i = 0; i < PREFILL * cohortsPerWorker; i++) {
                long first = 1L + i * 1024L;
                var keys = LongStream.range(first, first + prefixBlocks).boxed().toList();
                prefixes.add(keys);
                var seedLease = caches.get(i % PREFILL).acquire(keys.size(), keys);
                assertNotNull(seedLease);
                assertTrue(caches.get(i % PREFILL).admit(seedLease, keys));
            }
            var estimates = new ConcurrentHashMap<Long, Match>();
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                List<Long> keys = call.getArgument(0);
                var hits = new HashMap<String, Integer>();
                int max = 0, best = 0, accepting = 0;
                for (int i = 0; i < PREFILL; i++) {
                    var ep = h.prefillEndpoint(i);
                    if (ep == null) { continue; }
                    int hit = caches.get(i).peekPrefixHitBlocks(keys);
                    hits.put(ep.ipPort(), hit);
                    if (hit > max) { max = hit; best = 0; }
                    if (hit == max) { best++; }
                    if (ep.engineQueueAllowsAdmission()) { accepting++; }
                }
                if (keys != null && !keys.isEmpty() && keys.getLast() >= UNIQUE_KEYS) {
                    long id = FIRST_ID + (keys.getLast() - UNIQUE_KEYS) / blocks;
                    estimates.put(id, new Match(System.currentTimeMillis(), max * 512L, best, accepting));
                }
                return hits;
            });
            h.startAutoPump(config.getWorkerRegistry().getHealth().getStatusPollIntervalMs());
            var pending = new ConcurrentHashMap<Long, RequestContext>();
            var complete = new AtomicInteger(); var rejected = new AtomicInteger(); var errors = new AtomicInteger();
            var failure = new AtomicReference<Throwable>();
            var random = new SplittableRandom(seed);
            long start = System.nanoTime(); long nextSample = start;
            double nextArrival = 0;
            int offered = 0;
            int totalSeconds = warmSeconds + pulseSeconds + recoverySeconds;
            events.write(Map.of("event", "start", "atMs", System.currentTimeMillis()));
            while (System.nanoTime() - start < TimeUnit.SECONDS.toNanos(totalSeconds)) {
                double elapsed = (System.nanoTime() - start) / 1e9;
                while (nextArrival <= elapsed && nextArrival < totalSeconds) {
                    boolean pulseSlot = (scenario.equals("steady_mixed") || nextArrival >= warmSeconds && nextArrival < warmSeconds + pulseSeconds) && random.nextDouble() < fraction;
                    pulseSlot &= !scenario.equals("hot_control");
                    boolean cold = scenario.equals("steady_cold") || pulseSlot && (scenario.equals("cold_pulse") || scenario.equals("steady_mixed"));
                    int cohort = offered % prefixes.size();
                    var keys = new ArrayList<Long>();
                    if (!cold) { keys.addAll(prefixes.get(cohort)); }
                    long unique = UNIQUE_KEYS + offered * (long) blocks;
                    keys.addAll(LongStream.range(unique, unique + blocks - keys.size()).boxed().toList());
                    submit(h, FIRST_ID + offered, keys, cold, pulseSlot, cohort, input, output, nextArrival,
                            events, pending, estimates, complete, rejected, errors, failure);
                    offered++;
                    boolean pulsePhase = nextArrival >= warmSeconds && nextArrival < warmSeconds + pulseSeconds;
                    nextArrival += 1.0 / (pulsePhase ? pulseQps : targetQps);
                }
                if (System.nanoTime() >= nextSample) {
                    sample(h, caches, prefixes, prefixBlocks, samples, start, offered, complete.get(), rejected.get(), errors.get(), pending);
                    nextSample = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                }
                h.assertAutoPumpHealthy(); assertNull(failure.get()); Thread.sleep(2);
            }
            long drainStart = System.nanoTime();
            while (System.nanoTime() - drainStart < TimeUnit.SECONDS.toNanos(90)) {
                sample(h, caches, prefixes, prefixBlocks, samples, start, offered, complete.get(), rejected.get(), errors.get(), pending);
                if (pending.isEmpty() && h.requests.snapshotActive().isEmpty()) { break; }
                h.assertAutoPumpHealthy(); assertNull(failure.get()); Thread.sleep(1000);
            }
            var result = new LinkedHashMap<String, Object>();
            result.put("offered", offered); result.put("completed", complete.get()); result.put("rejected", rejected.get()); result.put("engineErrors", errors.get());
            result.put("clientInflight", pending.size()); result.put("schedulerActive", h.requests.snapshotActive().size());
            long decodeReserved = 0;
            for (int i = 0; i < DECODE; i++) { decodeReserved += h.decodeEndpoint(i).resourceSnapshot().reservedCount(); }
            result.put("decodeReserved", decodeReserved);
            assertEquals(0L, decodeReserved);
            result.put("drainSeconds", (System.nanoTime() - drainStart) / 1e9);
            JSON.writerWithDefaultPrettyPrinter().writeValue(out.resolve("result.json").toFile(), result);
            assertEquals(offered, complete.get() + rejected.get() + errors.get());
            assertEquals(0, pending.size()); assertEquals(0, h.requests.snapshotActive().size());
        }
    }

    private static void submit(AutoTpmE2EHarness h, long id, List<Long> keys, boolean cold, boolean pulseSlot,
                               int cohort, int inputTokens, int outputTokens, double plannedAtSeconds, Recorder events,
                               ConcurrentHashMap<Long, RequestContext> pending, ConcurrentHashMap<Long, Match> estimates,
                               AtomicInteger complete, AtomicInteger rejected, AtomicInteger errors, AtomicReference<Throwable> failure) {
        long start = System.nanoTime();
        var ctx = h.context(id, 50, 1, outputTokens);
        ctx.data().setRequest(Request.builder().requestId(id).seqLen(inputTokens).maxNewTokens(outputTokens).numBeams(1)
                .model("synthetic-model").rawPriority(50).blockCacheKeys(keys).cacheKeyBlockSize(512).build());
        ctx.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, h.config.resolveExpiresAtMs(ctx.data().getStartTime())));
        pending.put(id, ctx);
        events.write(Map.of("event", "arrival", "id", id, "atMs", System.currentTimeMillis(), "class", cold ? "cold" : "hot",
                "pulseSlot", pulseSlot, "cohort", cohort, "plannedAtSeconds", plannedAtSeconds, "inputTokens", inputTokens));
        h.scheduler.submit(ctx).thenAccept(response -> {
            long routeMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
            if (!response.isSuccess()) {
                rejected.incrementAndGet(); pending.remove(id); estimates.remove(id);
                events.write(Map.of("event", "route_rejected", "id", id, "atMs", System.currentTimeMillis(), "routeMs", routeMs,
                        "code", response.getCode(), "reason", String.valueOf(response.getErrorMessage())));
                return;
            }
            var route = ctx.snapshot().route();
            var p = route.prefillEp();
            int pPort = p.getStatus().topologySnapshot().grpcPort();
            int dPort = route.decodeEp().getStatus().topologySnapshot().grpcPort();
            var row = new LinkedHashMap<String, Object>();
            row.put("event", "route_accepted"); row.put("id", id); row.put("atMs", System.currentTimeMillis()); row.put("routeMs", routeMs);
            row.put("prefill", pPort - 61000); row.put("decode", dPort - 61000 - PREFILL); row.put("selectedEstimatedHitTokens", route.hitCache());
            row.put("candidateEstimate", estimates.remove(id)); events.write(row);
            var input = MockEngineTestSupport.inputWithBlockKeys(id, inputTokens, keys).toBuilder();
            input.getGenerateConfigBuilder().setMaxNewTokens(outputTokens).addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                    .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE).setRoleStr("DECODE").setGrpcPort(dPort));
            h.engineScheduler.schedule(() -> h.services.get(pPort).generateStreamCall(input.build(), new StreamObserver<>() {
                private boolean first = true;
                @Override public void onNext(EngineRpcService.GenerateOutputsPB output) {
                    if (first) {
                        first = false;
                        var info = output.getFlattenOutput().getAuxInfo(0);
                        events.write(Map.of("event", "first_token", "id", id, "atMs", System.currentTimeMillis(), "prefill", pPort - 61000,
                                "ttftMs", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start), "outputPrefixTokens", info.getPrefixLen()));
                    }
                }
                @Override public void onError(Throwable error) {
                    errors.incrementAndGet(); pending.remove(id); estimates.remove(id);
                    events.write(Map.of("event", "engine_error", "id", id, "atMs", System.currentTimeMillis(), "reason", String.valueOf(error.getMessage())));
                }
                @Override public void onCompleted() {
                    complete.incrementAndGet(); pending.remove(id); estimates.remove(id);
                    events.write(Map.of("event", "completed", "id", id, "atMs", System.currentTimeMillis(), "latencyMs", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start)));
                }
            }), 20, TimeUnit.MILLISECONDS);
        }).exceptionally(error -> { failure.compareAndSet(null, error); return null; });
    }

    private static void sample(AutoTpmE2EHarness h, List<MockLruBlockCache> caches, List<List<Long>> prefixes,
                               int prefixBlocks, Recorder samples, long start, int offered, int complete, int rejected,
                               int errors, Map<Long, RequestContext> pending) {
        var row = new LinkedHashMap<String, Object>();
        row.put("atMs", System.currentTimeMillis()); row.put("elapsedSeconds", (System.nanoTime() - start) / 1e9);
        row.put("offered", offered); row.put("completed", complete); row.put("rejected", rejected); row.put("engineErrors", errors);
        row.put("clientInflight", pending.size()); row.put("schedulerActive", h.requests.snapshotActive().size());
        row.put("globalUnplaced", pending.values().stream().filter(c -> c.snapshot().route() == null).count());
        row.put("candidatePrefill", h.endpointRegistry.prefillRoutingSnapshot(RoleType.PREFILL).size());
        var workers = new ArrayList<Map<String, Object>>();
        var fullPrefixCounts = new int[PREFILL];
        int anywhere = 0;
        for (int cohort = 0; cohort < prefixes.size(); cohort++) {
            boolean found = false;
            for (int i = 0; i < PREFILL; i++) {
                if (caches.get(i).peekPrefixHitBlocks(prefixes.get(cohort)) == prefixBlocks) { fullPrefixCounts[i]++; found = true; }
            }
            if (found) { anywhere++; }
        }
        row.put("hotPrefixesResidentAnywhere", anywhere); row.put("hotPrefixesTotal", prefixes.size());
        for (int i = 0; i < PREFILL; i++) {
            var worker = new LinkedHashMap<String, Object>(); var snap = h.prefillEngines.get(i).getSnapshot(); var ep = h.prefillEndpoint(i);
            worker.put("index", i); worker.put("residentHotPrefixes", fullPrefixCounts[i]);
            for (String key : List.of("waiting", "running", "inflight", "cache_evictions", "context_tps", "context_tps_with_cache", "hit_tokens_total")) { worker.put(key, snap.get(key)); }
            worker.put("pendingComputeTokens", ep.ownership.tokenWorkload().pendingTokens());
            worker.put("reservedRequests", ep.ownership.progress().outstandingRequests()); worker.put("localQueue", ep.ownership.progress().queueDepth());
            worker.put("accepting", ep.engineQueueAllowsAdmission());
            long observed = ep.getStatus().appliedObservation().observedAtUs();
            worker.put("statusAgeMs", observed == 0 ? null : (System.nanoTime() / 1000 - observed) / 1000); workers.add(worker);
        }
        row.put("prefill", workers);
        long reserved = 0, hardKv = 0;
        for (int i = 0; i < DECODE; i++) { reserved += h.decodeEndpoint(i).resourceSnapshot().reservedCount(); hardKv += h.decodeEndpoint(i).routingView().inflightHardKv(); }
        row.put("decodeReserved", reserved); row.put("decodeHardKv", hardKv); samples.write(row);
    }

    private static final class Recorder implements AutoCloseable {
        private final BufferedWriter writer;
        Recorder(Path path) throws Exception { writer = Files.newBufferedWriter(path); }
        synchronized void write(Object row) {
            try { writer.write(JSON.writeValueAsString(row)); writer.newLine(); writer.flush(); }
            catch (Exception e) { throw new IllegalStateException(e); }
        }
        @Override public synchronized void close() throws Exception { writer.close(); }
    }
}
