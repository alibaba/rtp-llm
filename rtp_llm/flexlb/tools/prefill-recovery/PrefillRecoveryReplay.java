package org.flexlb.mockengine;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.engine.grpc.EngineRpcService;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;

class PrefillRecoveryReplay {
    static final long MAX_LAG_MS = Long.getLong("maxLagMs", 2_000L);
    static final int P = Integer.getInteger("workers", 3), S = Integer.getInteger("sessions", 48);
    static final int blocks = Integer.getInteger("blocks", 32),
            hot = blocks - blocks / 16,
            len = blocks * 512;
    static final int pool = Integer.getInteger("pool", 1800),
            rotate = Integer.getInteger("rotate", 100);
    static final int normal = Integer.getInteger("normal", 172),
            peak = Integer.getInteger("peak", 218);
    static final int baseline = Integer.getInteger("baseline", 10),
            overload = Integer.getInteger("overload", 14);
    static final int recovery = Integer.getInteger("recovery", 18),
            budget = Integer.getInteger("budget", 30000);
    static final int observationMs = 200;
    void observeReplay() { }
    final AtomicLong completedInputTokens = new AtomicLong(), offeredInputTokens = new AtomicLong();
    final AtomicInteger offered = new AtomicInteger(),
            admitted = new AtomicInteger(),
            success = new AtomicInteger(),
            rejected = new AtomicInteger(),
            failed = new AtomicInteger();
    final AtomicLong selectedHit = new AtomicLong(),
            indexAdds = new AtomicLong(),
            doubleWarm = new AtomicLong();
    final AtomicLong crossWorkerOverlap = new AtomicLong(), sameWorkerOverlap = new AtomicLong();
    final Map<Long, int[]> filling = new HashMap<>();

    synchronized void beginFill(long key, int worker) {
        int[] owners = filling.computeIfAbsent(key, k -> new int[P]);
        for (int i = 0; i < P; i++)
            if (i != worker && owners[i] > 0) {
                crossWorkerOverlap.incrementAndGet();
                break;
            }
        if (owners[worker]++ > 0) sameWorkerOverlap.incrementAndGet();
    }

    synchronized void endFill(long key, int worker) {
        int[] owners = filling.get(key);
        owners[worker]--;
        for (int count : owners) if (count > 0) return;
        filling.remove(key);
    }

    final Map<Long, Integer> coldOwners = new ConcurrentHashMap<>();
    final int[] turns = new int[S];
    final List<Integer> sessions = new ArrayList<>();
    int currentSession;
    final Random rng = new Random(Long.getLong("seed", 20261120L));
    final List<MockLruBlockCache> caches = new ArrayList<>();
    final List<ReplayPrefillWorker> observed = new ArrayList<>();
    final org.flexlb.cache.core.GlobalCacheIndex index =
            new org.flexlb.cache.core.GlobalCacheIndex();
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    final List<AtomicLong> decodeOutputCounters = new ArrayList<>();
    AutoTpmE2EHarness h;

    void reportDecode(int second) {
        int occupied = 0, maxOccupied = 0, accepted = 0, running = 0, permits = 0;
        long maxExpected = 0, minAvailable = Long.MAX_VALUE;
        for (int i = 0; i < h.decodeEngines.size(); i++) {
            var snapshot = h.decodeEndpoint(i).resourceSnapshot();
            var usage = snapshot.routing().placementUsage();
            if (usage.totalKvTokens() != 9_918_939L * 128) throw new AssertionError(usage);
            occupied += Math.toIntExact(usage.occupiedRequests());
            maxOccupied = Math.max(maxOccupied, Math.toIntExact(usage.occupiedRequests()));
            accepted += snapshot.acceptedCount();
            running += snapshot.runningCount();
            permits += snapshot.activeDispatchPermits();
            maxExpected = Math.max(maxExpected, usage.expectedKvUsed());
            minAvailable = Math.min(minAvailable, usage.availableKvTokens());
        }
        long generated = decodeOutputCounters.stream().mapToLong(AtomicLong::get).sum();
        System.out.printf("DECODE sec=%d workers=%d totalPerWorker=%d occupied=%d maxOccupied=%d "
                        + "accepted=%d running=%d permits=%d maxExpected=%d minAvailable=%d generated=%d%n",
                second, h.decodeEngines.size(), 9_918_939L * 128, occupied, maxOccupied, accepted,
                running, permits, maxExpected, minAvailable, generated);
    }
    final ConcurrentHashMap<Integer, Long> queueBirths = new ConcurrentHashMap<>();
    String eventDirectory = System.getProperty("events");
    List<JavaMockEngineCluster.EngineEventLog> eventLogs = new ArrayList<>();

    static List<Long> keys(int session, int generation, int id) {
        var out = new ArrayList<Long>(blocks);
        long base = 1_000_000L + (generation * 100000L + session) * 8192;
        for (int k = 0; k < hot; k++)
            out.add(k < Integer.getInteger("common", 0) ? 900000L + k : base + k);
        for (int k = hot; k < blocks; k++) out.add(-1L - (long) id * blocks - k);
        return out;
    }

    static int extension = Integer.getInteger("extension", 0);

    static List<Long> sessionKeys(int s, int turn, int id) {
        if (extension == 0) return keys(s, turn / rotate, id);
        var result = new ArrayList<Long>();
        int base = Integer.getInteger("base", 32), cycle = Integer.getInteger("cycle", 16);
        for (int j = 0; j < base; j++)
            result.add(
                    j < Integer.getInteger("common", 0) ? 900000L + j : 10000000L + 10000L * s + j);
        // Open-loop immutable session trace. The next turn extends the preceding request,
        // whether that request succeeded, failed, or is still queued.
        long branch = 100000000L + (s + (long) (turn / cycle) * S) * 8192;
        for (int j = 0; j < extension * (1 + turn % cycle); j++) result.add(branch + j);
        // A bounded increase in genuinely new suffix blocks; arrival count and session choices stay
        // fixed.
        int begin = baseline * normal, end = begin + overload * peak;
        if (id >= begin && id < end && Boolean.getBoolean("moreWork")) {
            int extra = Integer.getInteger("extra", 6);
            for (int k = 0; k < extra; k++) result.add(-300000000000L - (long) id * 1024 - k);
        }
        int tail = Integer.getInteger("privateTail", 0);
        for (int k = 0; k < tail; k++) result.add(-10000000000L - (long) id * 1024 - k);
        return result;
    }

    void synchronizeIndex() {
        for (int i = 0; i < P; i++) {
            var endpoint = h.prefillEndpoint(i);
            if (observed.get(i).publish(index, endpoint))
                h.endpointRegistry.publishCacheAvailability(endpoint.getStatus());
        }
    }

    record Arrival(int ms, int priority, int tokens) {}

    final List<Arrival> arrivals = new ArrayList<>();
    final boolean realClock = System.getProperty("arrivals") != null;
    final boolean flatClock = Boolean.getBoolean("traceFlat");
    int traceSeconds;
    static final int families = Integer.getInteger("families", 2400);
    static final int privateBlocks = Integer.getInteger("privateBlocks", 14);
    long digest = 1;

    int family(int id) {
        long h = (id + Long.getLong("seed", 621L)) * 0x9E3779B97F4A7C15L;
        return Math.floorMod((int) (h ^ (h >>> 32)), families);
    }

    List<Long> template(int family, int blocks) {
        var keys = new ArrayList<Long>(blocks);
        for (int i = 0; i < blocks; i++)
            keys.add(i < 8 ? 100L + i : 10_000_000L + 8192L * family + i);
        return keys;
    }

    // Donor dictionary preserves anonymous block equality and branch shape.
    // Zipf/population mapping to the full traffic is a model, not an ID join to production.
    record Donor(int fullTokens, int[] blocks) {}

    final List<Donor> donors = new ArrayList<>();

    void loadGeometry() throws Exception {
        try (var in =
                new java.io.DataInputStream(
                        Files.newInputStream(Path.of(System.getProperty("geometry"))))) {
            int n = in.readInt();
            for (int j = 0; j < n; j++) {
                int len = in.readInt();
                int[] keys = new int[in.readInt()];
                for (int k = 0; k < keys.length; k++) keys[k] = in.readInt();
                donors.add(new Donor(len, keys));
            }
        }
        donors.sort(Comparator.comparingInt(Donor::fullTokens));
    }

    int namespace(int id) {
        long z = (id + Long.getLong("seed", 621L)) * 0x9E3779B97F4A7C15L;
        return Math.floorMod((int) (z ^ (z >>> 32)), Integer.getInteger("copies", 24));
    }

    int donorFor(int id, int tokens) {
        int lo = 0, hi = donors.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (donors.get(mid).fullTokens() < tokens) lo = mid + 1;
            else hi = mid;
        }
        return Math.clamp(lo + Math.floorMod(id * 7919, 7) - 3, 0, donors.size() - 1);
    }

    List<Long> fromDonor(int donor, int ns, int stem) {
        var shape = donors.get(donor).blocks();
        long domain = (long) (ns + 1) << 40;
        var keys = new ArrayList<Long>(stem);
        for (int i = 0; i < stem; i++)
            keys.add(domain + (i < shape.length ? shape[i] : 1000000L + 4096L * donor + i));
        return keys;
    }

    List<Long> traceKeys(int id, Arrival row) {
        int complete = row.tokens() / 512;
        int stem = Math.max(0, complete - privateBlocks);
        int ns = namespace(id);
        long time = row.ms();
        if (ns < Integer.getInteger("rotateCopies", 0)
                && time >= Integer.getInteger("shiftStart", 400000)
                && time < Integer.getInteger("shiftEnd", 520000))
            ns += Integer.getInteger("copies", 24);
        var keys =
                donors.isEmpty()
                        ? template(family(id), stem)
                        : fromDonor(donorFor(id, row.tokens()), ns, stem);
        for (int i = stem; i < complete; i++) keys.add(-10000000000L - 4096L * id - i);
        return keys;
    }

    void readTrace() throws Exception {
        try (var in =
                new java.io.BufferedReader(
                        new java.io.InputStreamReader(
                                new java.util.zip.GZIPInputStream(
                                        Files.newInputStream(
                                                Path.of(System.getProperty("arrivals"))))))) {
            if (!in.readLine().trim().equals("arrival_ms,priority,seq_tokens"))
                throw new AssertionError("header");
            String line;
            while ((line = in.readLine()) != null) {
                var v = line.split(",");
                arrivals.add(
                        new Arrival(
                                Integer.parseInt(v[0]),
                                Integer.parseInt(v[1]),
                                Integer.parseInt(v[2])));
            }
        }
        traceSeconds = 1 + arrivals.getLast().ms() / 1000;
    }

    long dueMs(int id) {
        return flatClock
                ? (long) id * 1000 * traceSeconds / arrivals.size()
                : arrivals.get(id).ms();
    }

    int advanceSession(int n) {
        int burst = Integer.getInteger("burst", 1);
        if (n % burst == 0) {
            if (Boolean.getBoolean("uniformRounds")) {
                int pos = (n / burst) % S;
                if (pos == 0 && (n == 0 || !Boolean.getBoolean("cyclic")))
                    Collections.shuffle(sessions, rng);
                currentSession = sessions.get(pos);
            } else currentSession = rng.nextInt(S);
        }
        return currentSession;
    }

    final int boundary = baseline + overload, resumeId = boundary * normal;
    final String restore = System.getProperty("restore");
    final String capture = System.getProperty("capture");
    long tailDigest = 1;

    long[] image(MockLruBlockCache cache) {
        @SuppressWarnings("unchecked")
        var resident = (Map<Long, Integer>) ReflectionTestUtils.getField(cache, "blocks");
        long[] keys = new long[resident.size()];
        int j = 0;
        for (long k : resident.keySet()) keys[j++] = k;
        return keys;
    }

    void captureAll(int i, CacheImage state) {
        if (i < P) {
            synchronized (caches.get(i)) {
                captureAll(i + 1, state);
            }
        } else {
            for (int j = 0; j < P; j++) state.lru[j] = image(caches.get(j));
        }
    }

    static final class CacheImage {
        final long[][] lru = new long[P][];
    }

    void captureImage() {
        CacheImage state = new CacheImage();
        captureAll(0, state);
        try (var out =
                new java.io.DataOutputStream(
                        new java.io.BufferedOutputStream(
                                Files.newOutputStream(Path.of(capture + ".part"))))) {
            out.writeInt(P);
            for (long[] keys : state.lru) {
                out.writeInt(keys.length);
                for (long key : keys) out.writeLong(key);
            }
        } catch (java.io.IOException ex) {
            throw new java.io.UncheckedIOException(ex);
        }
        try {
            Files.move(
                    Path.of(capture + ".part"), Path.of(capture), StandardCopyOption.ATOMIC_MOVE);
        } catch (java.io.IOException ex) {
            throw new java.io.UncheckedIOException(ex);
        }
    }

    void loadImage() throws Exception {
        try (var in =
                new java.io.DataInputStream(
                        new java.io.BufferedInputStream(Files.newInputStream(Path.of(restore))))) {
            if (in.readInt() != P) throw new AssertionError("worker count");
            for (int j = 0; j < P; j++) {
                int n = in.readInt();
                var keys = new ArrayList<Long>(n);
                for (int k = 0; k < n; k++) keys.add(in.readLong());
                var c = caches.get(j);
                @SuppressWarnings("unchecked")
                var resident = (Map<Long, Integer>) ReflectionTestUtils.getField(c, "blocks");
                synchronized (c) {
                    for (long k : keys) resident.put(k, 0);
                }
                long[] restored = image(c);
                if (restored.length != n) throw new AssertionError("duplicate resident keys");
                for (int k = 0; k < n; k++)
                    if (restored[k] != keys.get(k)) throw new AssertionError("LRU order");
            }
            if (in.read() != -1) throw new AssertionError("trailing input");
        }
        for (int n = 0; n < resumeId; n++) turns[advanceSession(n)]++;
        System.out.println(
                "RESTORED workers="
                        + P
                        + " keys="
                        + observed.stream().mapToInt(ReplayPrefillWorker::physicalKeys).sum());
    }

    // Geometric EOS per output token. The production request caps and stopping RNG identities
    // are absent from the arrival trace. Use a fixed replay cap and per-request seed for both arms.
    static final int OUTPUT_CAP = 16_384;
    static int outputTokens(long id) {
        double u = new java.util.SplittableRandom(id ^ Long.getLong("seed", 20261120L)).nextDouble();
        return Math.min(OUTPUT_CAP, 1 + (int) (Math.log1p(-u) / Math.log1p(-0.00089)));
    }

    final AtomicLong completedOutputTokens = new AtomicLong();

    void submit(int n) {
        int s = advanceSession(n);
        int turn = turns[s]++;
        var keys = realClock ? traceKeys(n, arrivals.get(n)) : sessionKeys(s, turn, n);
        if (eventDirectory != null)
            System.out.println("TRACE " + (9000000L + n) + " " + s + " " + turn);
        int len = realClock ? arrivals.get(n).tokens() : keys.size() * 512;
        int priority =
                switch (n % 20) {
                    case 0 -> 39;
                    case 1, 2, 3 -> 40;
                    case 4, 5 -> 49;
                    default -> 50;
                };
        if (realClock) priority = arrivals.get(n).priority();
        digest = digest * 31 + len;
        digest = digest * 31 + priority;
        digest = digest * 31 + keys.hashCode();
        if (n >= resumeId) {
            tailDigest = tailDigest * 31 + len;
            tailDigest = tailDigest * 31 + priority;
            tailDigest = tailDigest * 31 + keys.hashCode();
        }
        long id = 9000000L + n;
        int output = outputTokens(id);
        var ctx = h.context(id, priority, 1, OUTPUT_CAP);
        ctx.data()
                .setRequest(
                        ctx.data().getRequest().toBuilder()
                                .seqLen((long) len)
                                .blockCacheKeys(keys)
                                .cacheKeyBlockSize(512)
                                .build());
        long enqueued = System.nanoTime();
        queueBirths.put(n, enqueued);
        ctx.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(priority, ctx.data().getStartTime() + budget));
        offeredInputTokens.addAndGet(len);
        offered.incrementAndGet();
        h.scheduler
                .submit(ctx)
                .thenAccept(
                        reply -> {
                            queueBirths.remove(n);
                            if (!reply.isSuccess()) {
                                int count = rejected.incrementAndGet();
                                if (count == 1 || count % 500 == 0)
                                    System.out.println(
                                            "REJECT "
                                                    + reply.getCode()
                                                    + " "
                                                    + reply.getErrorMessage()
                                                    + " "
                                                    + ctx.data().getSchedulingDiagnostics());
                                return;
                            }
                            int i = 0;
                            while (h.prefillEndpoint(i) != ctx.snapshot().route().prefillEp()) i++;
                            admitted.incrementAndGet();
                            long estimate = ctx.snapshot().route().hitCache();
                            selectedHit.addAndGet(estimate);
                            if (!keys.isEmpty()
                                    && estimate < (keys.size() - Math.max(extension, 2)) * 512L) {
                                Integer prior =
                                        coldOwners.put(
                                                keys.get(
                                                        Math.min(
                                                                keys.size() - 1,
                                                                Integer.getInteger("common", 0)
                                                                        + 1)),
                                                i);
                                if (prior != null && prior != i) doubleWarm.incrementAndGet();
                            }
                            int missing = (int) (estimate / 512);
                            Long block = missing + 1 < keys.size() ? keys.get(missing) : null;
                            final int worker = i;
                            if (block != null) beginFill(block, worker);
                            int d = 0;
                            while (h.decodeEndpoint(d) != ctx.snapshot().route().decodeEp()) d++;
                            var input =
                                    EngineRpcService.GenerateInputPB.newBuilder().setRequestId(id);
                            input.getGenerateConfigBuilder()
                                    .setMaxNewTokens(OUTPUT_CAP)
                                    .setUniqueKey(
                                            "{\"input_len\":"
                                                    + len
                                                    + ",\"output_len\":" + output
                                                    + ",\"block_cache_keys\":"
                                                    + keys
                                                    + "}");
                            input.getGenerateConfigBuilder()
                                    .addRoleAddrs(
                                            EngineRpcService.RoleAddrPB.newBuilder()
                                                    .setRole(
                                                            EngineRpcService.RoleAddrPB.RoleType
                                                                    .DECODE)
                                                    .setRoleStr("DECODE")
                                                    .setGrpcPort(
                                                            h.decodeEngines.get(d).getGrpcPort()));
                            h.prefillEngines
                                    .get(i)
                                    .generateStreamCall(
                                            input.build(),
                                            new io.grpc.stub.StreamObserver<>() {
                                                boolean error;

                                                public void onNext(
                                                        EngineRpcService.GenerateOutputsPB out) {
                                                    if (out.hasErrorInfo()
                                                            && out.getErrorInfo()
                                                                            .getErrorCodeValue()
                                                                    != 0) {
                                                        error = true;
                                                        if (failed.get() < 3)
                                                            System.out.println(
                                                                    "WORKER_ERROR "
                                                                            + out.getErrorInfo());
                                                    }
                                                }

                                                public void onError(Throwable ex) {
                                                    if (block != null) endFill(block, worker);
                                                    failed.incrementAndGet();
                                                }

                                                public void onCompleted() {
                                                    if (block != null) endFill(block, worker);
                                                    if (error) failed.incrementAndGet();
                                                    else {
                                                        completedOutputTokens.addAndGet(output);
                                                        completedInputTokens.addAndGet(len);
                                                        success.incrementAndGet();
                                                    }
                                                }
                                            });
                        })
                .exceptionally(
                        ex -> {
                            failure.compareAndSet(null, ex);
                            return null;
                        });
    }

    void run() throws Exception {
        if (realClock) readTrace();
        if (System.getProperty("geometry") != null) loadGeometry();
        var cfg = ConfigService.parse(Files.readString(Path.of(System.getProperty("config"))));
        String formula = System.getProperty("formula", "10 + 0.0125 * sum(computeTokens)");
        System.out.printf(
                "CONFIG budget=%s decision=%s ordering=%s default=%s cost=%s victims=%s"
                    + " preemptionMs=%s inflight=%s admission=%s decode=%s rescue=%s%n",
                cfg.getDispatchTimeoutMs(),
                cfg.queueScheduler().getDecision().getType(),
                cfg.queueScheduler().getOrdering().getType(),
                cfg.priorityOrdering().getDefaultPriority(),
                cfg.priorityOrdering().isPrefillCostOrder(),
                cfg.priorityOrdering().getPreemption().getAllowedVictimStages(),
                cfg.priorityOrdering().getPreemption().getTimeoutMs(),
                cfg.getDispatcher().getMaxInflightPerPrefillWorker(),
                cfg.getRouter()
                        .getRoles()
                        .getPrefill()
                        .getTokenBalance()
                        .getEngineQueueAdmissionThreshold(),
                cfg.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests(),
                cfg.getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue());
        System.out.printf(
                "MODEL cacheMatch=batch-prepare workers=%d blocks=%d sessions=%d pool=%d generationTurns=%d normal=%d"
                    + " peak=%d statusMs=20.5 cachePrefillMs=49/41.8/40.45 cacheDecodeMs=1000 deadline=%d expression=%s%n",
                P, blocks, S, pool, rotate, normal, peak, budget, formula);
        for (int s = 0; s < S; s++) {
            turns[s] = s * rotate / S + (Boolean.getBoolean("fullCache") ? 4096 : 0);
            sessions.add(s);
        }
        var perf =
                Map.<String, Object>of(
                        "test_pool_blocks",
                        pool,
                        "test_decode_pool_blocks",
                        9_918_939,
                        "test_decode_block_size", 128,
                        "decode", Map.of("tokens_per_step", 2.874334269284765,
                                "step_ms_by_batch", List.of(List.of(1, 0.0010343))),
                        "jitter_pct",
                        0,
                        "prefill",
                        Map.of("max_batch_requests", 32, "max_batch_tokens", len * 32));
        try (var harness =
                new AutoTpmE2EHarness(
                        58820,
                        P,
                        Integer.getInteger("decode", 5),
                        formula,
                        1,
                        true,
                        false,
                        cfg.queueScheduler().getDecision(),
                        true,
                        cfg,
                        10,
                        perf, false)) {
            h = harness;
            for (var engine : h.decodeEngines) {
                decodeOutputCounters.add((AtomicLong) ReflectionTestUtils.getField(engine, "generateTokens"));
            }
            reportDecode(0);
            for (int i = 0; i < P; i++) {
                h.prefillEngines.get(i).enableBatchStartMatching();
                var worker = new ReplayPrefillWorker(h.prefillEngines.get(i));
                observed.add(worker);
                caches.add(worker.cache);
            }
            if (eventDirectory != null) {
                Files.createDirectories(Path.of(eventDirectory));
                for (int i = 0; i < P; i++) {
                    var writer =
                            JavaMockEngineCluster.EngineEventLog.open(
                                    Path.of(eventDirectory, "p" + i + ".jsonl").toString());
                    h.prefillEngines.get(i).setEngineEventLog(writer);
                    eventLogs.add(writer);
                }
            }
            if (restore != null) loadImage();
            else {
                if (Boolean.getBoolean("fullCache")) {
                    // Restore a mature LRU from older completed session epochs. Use real
                    // leases/admit
                    // for cache replacement; measured online arrivals follow this warm history.
                    int epochs = (pool * P / (S * extension * Integer.getInteger("cycle", 16))) + 3;
                    int cycle = Integer.getInteger("cycle", 16);
                    for (int i = epochs; i > 0; i--)
                        for (int s = 0; s < S; s++) {
                            int turn = (turns[s] / cycle - i) * cycle + cycle - 1;
                            var past = sessionKeys(s, turn, -(i * S + s + 1));
                            var history = caches.get(s % P);
                            var allocation = history.acquire(past.size(), past);
                            if (allocation == null) throw new AssertionError("warm allocation");
                            history.admit(allocation, past);
                        }
                }
                for (int s = 0; s < S; s++) {
                    var first = sessionKeys(s, turns[s], 0);
                    int seeded =
                            extension == 0
                                    ? hot
                                    : first.size()
                                            - extension
                                            - Integer.getInteger("privateTail", 0);
                    MockEngineTestSupport.seedCache(caches.get(s % P), first.subList(0, seeded));
                    if (Boolean.getBoolean("secondary")) {
                        var alt = caches.get((s % P + 1 + (s / P) % (P - 1)) % P);
                        MockEngineTestSupport.seedCache(alt, first.subList(0, seeded));
                    }
                }
                if (!donors.isEmpty()) {
                    for (int ns = 0; ns < Integer.getInteger("copies", 24); ns++)
                        for (int j = 0; j < donors.size(); j++) {
                            var shape = donors.get(j);
                            var keys = fromDonor(j, ns, shape.blocks().length);
                            if (keys.isEmpty()) continue;
                            // Most prior copies live on their root owner; a minority on a second
                            // worker.
                            int owner =
                                    Math.floorMod(
                                            (ns * 7919 + shape.blocks()[0] * 11939)
                                                    ^ (j % 7 == 0 ? 0x1f31 : 0),
                                            P);
                            var cache = caches.get(owner);
                            var lease = cache.acquire(keys.size(), keys);
                            if (lease == null) throw new AssertionError("donor seed");
                            cache.admit(lease, keys);
                        }
                } else if (realClock) {
                    // A stationary hot-template control. All prefixes come from synthetic history,
                    // and lengths/priorities/clocks come from the ingress trace. No outcome labels.
                    for (int f = 0; f < families; f++) {
                        var warm = template(f, Integer.getInteger("templateDepth", 768));
                        var cache = caches.get(f % P);
                        var lease = cache.acquire(warm.size(), warm);
                        if (lease == null) throw new AssertionError("template too large");
                        cache.admit(lease, warm);
                    }
                }
            }
            synchronizeIndex();
            when(h.cache.findMatchingEngines(any(), any(), any()))
                    .thenAnswer(
                            c ->
                                    index.batchCalculatePrefixMatchLength(
                                            c.getArgument(2), c.getArgument(0)));
            try (var feedback = new ReplayFeedback(this, restore != null, 10)) {
            long start = feedback.start() - (restore == null ? 0L : boundary * 1_000_000_000L),
                    nextObservation = start;
            int id = restore == null ? 0 : resumeId;
            System.out.println(
                    "sec,offered,admitted,success,rejected,workerErrors,masterPending,workerWaiting,fresh,input,hits,indexUnique,physicalKeys,evictions,indexAdds,coldWorkerSwitches,selectedHit,indexMappings,crossWorkerConcurrent,sameWorkerConcurrent");
            int stepSeconds =
                    Boolean.getBoolean("matchedFlat")
                            ? Math.round(
                                    (float) peak
                                            * overload
                                            / Integer.getInteger("flatRate", normal))
                            : overload;
            for (int sec = restore == null ? 0 : boundary;
                    sec < (realClock ? traceSeconds : baseline + stepSeconds + recovery) + 180;
                    sec++) {
                if (sec == boundary && capture != null) {
                    captureImage();
                    return;
                }
                long stepNanos =
                        Boolean.getBoolean("pacedHistory") && sec < boundary
                                ? 2_000_000_000L
                                : 1_000_000_000L;
                long extraHistory =
                        Boolean.getBoolean("pacedHistory")
                                ? Math.min(sec, boundary) * 1_000_000_000L
                                : 0L;
                long stepStart = start + sec * 1_000_000_000L + extraHistory;
                int offset = sec - baseline;
                int q =
                        sec < baseline
                                ? normal
                                : offset < stepSeconds
                                        ? (offset + 1) * peak * overload / stepSeconds
                                                - offset * peak * overload / stepSeconds
                                        : offset < stepSeconds + recovery ? normal : 0;
                int pulse = Integer.getInteger("pulse", 0);
                if (offset >= 0 && offset < stepSeconds && !Boolean.getBoolean("matchedFlat")) {
                    if (stepSeconds % 10 != 0 && pulse != 0)
                        throw new IllegalArgumentException("pulse requires complete 10s periods");
                    q += offset % 10 < 5 ? pulse : -pulse;
                }
                int beginId = id;
                if (realClock) {
                    int next = id;
                    while (next < arrivals.size() && dueMs(next) < (sec + 1) * 1000L) next++;
                    q = next - id;
                }
                long late = 0;
                for (int j = 0; j < (q > 0 ? q : 10); j++) {
                    long due =
                            realClock && q > 0
                                    ? start + dueMs(beginId + j) * 1000000L
                                    : stepStart + j * stepNanos / (q > 0 ? q : 10);
                    if (System.nanoTime() < due) LockSupport.parkNanos(due - System.nanoTime());
                    if (System.nanoTime() >= nextObservation) {
                        feedback.check();
                        observeReplay();
                        nextObservation = System.nanoTime() + observationMs * 1000000L;
                    }
                    if (q > 0) late = Math.max(late, System.nanoTime() - due);
                    if (q > 0 && id < Integer.getInteger("maxOffered", Integer.MAX_VALUE))
                        submit(id++);
                }
                long end = stepStart + stepNanos;
                if (System.nanoTime() < end) LockSupport.parkNanos(end - System.nanoTime());
                long fresh = 0, input = 0, hits = 0, waiting = 0, physical = 0, evictions = 0;
                for (var engine : h.prefillEngines) engine.checkBatchPreparation();
                for (var e : observed) {
                    fresh += e.fresh.get();
                    input += e.input.get();
                    hits += e.hits.get();
                    waiting += e.waiting.get();
                    physical += e.physicalKeys();
                    evictions += e.cache.evictions();
                }
                System.out.printf(
                        Locale.ROOT,
                        "ROW,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d%n",
                        sec + 1,
                        offered.get(),
                        admitted.get(),
                        success.get(),
                        rejected.get(),
                        failed.get(),
                        offered.get() - admitted.get() - rejected.get(),
                        waiting,
                        fresh,
                        input,
                        hits,
                        index.totalBlocks(),
                        physical,
                        evictions,
                        indexAdds.get(),
                        doubleWarm.get(),
                        selectedHit.get(),
                        index.totalMappings(),
                        crossWorkerOverlap.get(),
                        sameWorkerOverlap.get());
                var now = System.nanoTime();
                var ages =
                        queueBirths.values().stream()
                                .mapToLong(t -> TimeUnit.NANOSECONDS.toMillis(now - t))
                                .sorted()
                                .toArray();
                System.out.printf(
                        "DETAIL sec=%d workerWaiting=%s masterAgeP95Ms=%d oldestMs=%d%n",
                        sec + 1,
                        observed.stream().map(w -> w.waiting.get()).toList(),
                        ages.length == 0 ? 0 : ages[(ages.length - 1) * 95 / 100],
                        ages.length == 0 ? 0 : ages[ages.length - 1]);
                System.out.println(
                        "WALL "
                                + (sec + 1)
                                + " "
                                + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start)
                                + " lateMs "
                                + TimeUnit.NANOSECONDS.toMillis(late));
                if (TimeUnit.NANOSECONDS.toMillis(late) > MAX_LAG_MS)
                    throw new AssertionError(
                            "arrival clock late by " + TimeUnit.NANOSECONDS.toMillis(late) + "ms");
                if (q == 0 || (sec + 1) % 5 == 0) reportDecode(sec + 1);
                if (sec % 5 == 3)
                    System.out.println(
                            "DIAG "
                                    + ReflectionTestUtils.invokeMethod(
                                            h.scheduler, "getLatestQueueWaitSnapshot"));
                if (failure.get() != null) throw new AssertionError(failure.get());
                System.out.printf("GOODPUT %d %d%n", sec + 1, completedInputTokens.get());
                System.out.printf("COMPLETED_OUTPUT %d %d%n", sec + 1, completedOutputTokens.get());
                System.out.printf("OFFERED_TOKENS %d %d%n", sec + 1, offeredInputTokens.get());
                if (sec + 1 == 395) System.out.println("PREFIX_DIGEST_57600 " + tailDigest);
                if ((!realClock || id == arrivals.size())
                        && q == 0
                        && offered.get() == success.get() + failed.get() + rejected.get()) break;
            }
            if (offered.get() != success.get() + failed.get() + rejected.get())
                throw new AssertionError("unfinished " + id);
            System.out.println("INPUT_DIGEST " + id + " " + digest);
            System.out.println("TAIL_DIGEST " + tailDigest);
            // Engine completion and the Master's next status heartbeat are asynchronous.
            AutoTpmE2EHarness.await(() -> {
                feedback.check();
                for (int i = 0; i < h.decodeEngines.size(); i++) {
                    var ledger = h.decodeEndpoint(i).resourceSnapshot();
                    if (!ledger.requests().isEmpty() || ledger.activeDispatchPermits() != 0
                            || h.decodeEngines.get(i).getAvailableKvTokens() != 9_918_939L * 128) return false;
                }
                return true;
            }, 5_000, "Decode reservations or physical KV still held after completions");
            reportDecode(-1);
            }
            eventLogs.forEach(JavaMockEngineCluster.EngineEventLog::close);
        }
    }

    static long num(Map<String, Object> snap, String name) {
        return ((Number) snap.get(name)).longValue();
    }

    public static void main(String[] args) throws Exception {
        new PrefillRecoveryReplay().run();
    }
}
