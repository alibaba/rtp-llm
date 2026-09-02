package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import io.grpc.ManagedChannel;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.util.PriorityNormalizer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.math.BigInteger;
import java.net.InetAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Standalone Java load client (replaces the legacy Python load client).
 *
 * <p>Replays trace JSONL files against a running FlexLB master via gRPC Schedule RPC.
 * Supports multi-shard replay, configurable speed, semaphore-based concurrency control,
 * and optional engine stream reading for TTFT/total latency. The client records raw
 * data only — client_events.jsonl rows (renamed from per_request.jsonl: the
 * client-side half of the multi-component JSONL event streams, rid-joined
 * offline against the mock engine's engine_events.jsonl), the terminal
 * server_latency.json snapshot, and pushgateway metrics; every derived
 * statistic is computed by the run-level aggregator (aggregate_canvas_run.py),
 * never here.
 *
 * <p>FETCH_OUTPUT_STREAM (default true) controls ONLY the client-side read of engine
 * output streams (phase-2 FetchResponse/GenerateStreamCall). With FETCH_OUTPUT_STREAM=0
 * the client stops after a successful Schedule RPC; the engine still executes the
 * request in full (BATCH dispatcher: master enqueued it via EnqueueBatch during the
 * Schedule RPC). This trims the client's stream-reading network cost from load tests
 * while keeping engine-side load identical to the read-stream mode.
 *
 * <p>Configuration is read exclusively from environment variables at startup (no
 * multi-layer override). Run as:
 * <pre>{@code
 *   java -cp <jar> org.flexlb.mockengine.JavaLoadClient
 * }</pre>
 */
public final class JavaLoadClient {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BLOCK_SIZE = 1024;
    /**
     * Sweep cadence for the outstanding-result collector: completed futures are
     * harvested at this granularity while slow RPCs are still in flight. All
     * latency timestamps are stamped inside handleRequest when the event
     * happens, so sweep latency only delays row collection, never per-row data
     * fidelity.
     */
    private static final long COLLECTION_SWEEP_INTERVAL_NANOS =
            TimeUnit.MILLISECONDS.toNanos(100);

    private final Config config;
    private final EventLoopGroup eventLoopGroup;
    private final ManagedChannel[] scheduleChannels;
    private final FlexlbServiceGrpc.FlexlbServiceBlockingStub[] scheduleStubs;
    private final AtomicInteger scheduleStubRR = new AtomicInteger();
    private final Map<String, ManagedChannel[]> engineChannelPools = new ConcurrentHashMap<>();
    private final Map<String, AtomicInteger> engineChannelRR = new ConcurrentHashMap<>();
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5)).build();
    private final List<RequestResult> results = new ArrayList<>();
    final AtomicInteger actualSentCount = new AtomicInteger();
    private final AtomicInteger responseCount = new AtomicInteger();
    private volatile long replayStartedEpochMs;
    private volatile long replayStartedNanos;
    private volatile long sendStartNanos;
    private volatile long sendEndNanos;
    final AtomicInteger successCount = new AtomicInteger();
    final AtomicInteger errorCount = new AtomicInteger();
    final AtomicInteger inflightCount = new AtomicInteger();
    final AtomicInteger sentTotal = new AtomicInteger();
    final List<RequestResult> completedResults = Collections.synchronizedList(new ArrayList<>());
    private volatile ScheduledExecutorService pushgatewayExecutor;
    private volatile double lastGradientLogS = -10.0;
    final List<String> fallbackPrefillAddrs = new ArrayList<>();
    final List<String> fallbackDecodeAddrs = new ArrayList<>();
    private final AtomicInteger fallbackPrefillRR = new AtomicInteger();
    private final AtomicInteger fallbackDecodeRR = new AtomicInteger();

    JavaLoadClient(Config config) {
        this.config = config;
        if (config.dryRun) {
            // Test-only mode: no gRPC channels are created.
            this.eventLoopGroup = null;
            this.scheduleChannels = new ManagedChannel[0];
            this.scheduleStubs = new FlexlbServiceGrpc.FlexlbServiceBlockingStub[0];
            return;
        }
        this.eventLoopGroup = new NioEventLoopGroup(config.eventLoopThreads);
        this.scheduleChannels = new ManagedChannel[config.nChannels];
        this.scheduleStubs = new FlexlbServiceGrpc.FlexlbServiceBlockingStub[config.nChannels];
        for (int i = 0; i < config.nChannels; i++) {
            ManagedChannel channel = NettyChannelBuilder.forTarget(config.grpcTarget)
                    .eventLoopGroup(eventLoopGroup)
                    .channelType(NioSocketChannel.class)
                    .maxInboundMessageSize(16 * 1024 * 1024)
                    .flowControlWindow(1024 * 1024)
                    .keepAliveTime(30, TimeUnit.SECONDS)
                    .keepAliveTimeout(10, TimeUnit.SECONDS)
                    .usePlaintext()
                    .build();
            scheduleChannels[i] = channel;
            scheduleStubs[i] = FlexlbServiceGrpc.newBlockingStub(channel);
        }
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.fromEnv();
        config.print();
        if (config.traceFile.isEmpty()) {
            throw new IllegalArgumentException("TRACE_FILE environment variable is required");
        }
        JavaLoadClient client = new JavaLoadClient(config);
        try {
            client.run();
        } finally {
            client.close();
        }
    }

    void run() throws Exception {
        List<TraceRecord> records = loadTrace(config.traceFile);
        if (records.isEmpty()) {
            throw new RuntimeException("no replayable requests loaded from " + config.traceFile);
        }

        // Parity with the legacy Python load client: load_replay_requests applies
        // duration/limit filters FIRST, then num_shards slicing — so LIMIT applies
        // to the whole trace, not per shard. Loop mode skips both filters (duration
        // becomes a wall-clock timeout, limit a total sent cap) but still slices
        // the trace across shards so N workers do not each replay the full trace.
        // Uniform send mode reuses the loop-mode record semantics (shard slice
        // only, wall-clock duration, total sent cap): request bodies still come
        // from cycling the trace shard, only the arrival process changes.
        boolean cyclic = config.loop || config.isUniform();
        records = filterAndShard(records,
                cyclic ? 0 : config.durationS,
                cyclic ? 0 : config.limit,
                config.numShards, config.shardIndex);
        if (records.isEmpty()) {
            throw new RuntimeException("trace shard has no replayable requests");
        }

        // Parity with Python: length truncation applied after sharding, before replay.
        if (config.maxInputLen > 0 || config.maxOutputLen > 0) {
            records = truncateRecords(records, config.maxInputLen, config.maxOutputLen);
        }

        System.out.println("loaded " + records.size() + " requests from " + config.traceFile
                + " (shard=" + config.shardIndex + "/" + config.numShards + ")");

        if (config.isUniform()) {
            double perShardQps = config.sendModeQps / config.numShards;
            System.out.println(String.format(
                    "send mode: uniform — target_qps=%.3f, per_shard_qps=%.3f "
                            + "(interval %.3fms), trace timestamps ignored",
                    config.sendModeQps, perShardQps, 1000.0 / perShardQps));
            if (config.rampUpSeconds > 0) {
                System.out.println(String.format(
                        "ramp-up: per-shard QPS climbs linearly 0 -> %.3f over %.1fs, "
                                + "then constant (ramp sends ≈ %.1f per shard)",
                        perShardQps, config.rampUpSeconds,
                        perShardQps * config.rampUpSeconds / 2.0));
            }
        }

        if (config.enableFallback && !config.endpointsFile.isEmpty()) {
            loadFallbackEndpoints(config.endpointsFile);
        }

        Files.createDirectories(Path.of(config.outputDir));
        if (!config.skipServerLatency) {
            resetServerLatency();
        }

        long firstTsMs = records.get(0).tsMs;
        long traceSpanMs = Math.max(records.get(records.size() - 1).tsMs - firstTsMs, 1);

        if (config.gradient && config.isUniform()) {
            System.out.println("WARNING: GRADIENT is ignored in uniform send mode");
        }
        if (!config.isUniform() && config.rampUpSeconds > 0) {
            System.out.println("WARNING: RAMP_UP_SECONDS is ignored outside "
                    + "uniform send mode");
        }
        if (config.gradient && !config.isUniform() && config.durationS <= 0) {
            System.out.println("WARNING: GRADIENT requires DURATION_S > 0, "
                    + "falling back to fixed replay_speed");
        }
        if (config.gradient && !config.isUniform() && config.durationS > 0) {
            int startSpeed = Math.max(1, config.gradientStartSpeed);
            System.out.println("gradient mode: speed will increase from " + startSpeed
                    + "x to " + config.gradientMaxSpeed + "x over "
                    + config.durationS + "s");
        }

        if (config.startAtEpochMs > 0) {
            long waitMs = config.startAtEpochMs - System.currentTimeMillis();
            if (waitMs > 0) {
                System.out.println("waiting " + waitMs + "ms for start barrier...");
                Thread.sleep(waitMs);
            }
        }

        replayStartedEpochMs = System.currentTimeMillis();
        replayStartedNanos = System.nanoTime();
        sendStartNanos = replayStartedNanos;

        startPushgatewayLoop();

        Semaphore semaphore = new Semaphore(config.maxConcurrency);
        ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
        List<Future<RequestResult>> futures = new ArrayList<>();
        int sentCount = 0;
        int loopIdx = 0;
        // Per-shard uniform interval: total target rate SEND_MODE_QPS is split
        // evenly across NUM_SHARDS instances.
        double uniformIntervalS = config.isUniform()
                ? config.numShards / config.sendModeQps : 0.0;

        while (true) {
            for (TraceRecord record : records) {
                if (cyclic && config.durationS > 0) {
                    if ((System.nanoTime() - replayStartedNanos) / 1_000_000_000L >= config.durationS) {
                        break;
                    }
                }
                if (config.limit > 0 && sentCount >= config.limit) {
                    break;
                }

                // Parity with Python: gradient mode ramps speed linearly from
                // start to max over the duration window.
                double currentSpeed;
                if (config.gradient && !config.isUniform() && config.durationS > 0) {
                    double elapsedS = (System.nanoTime() - replayStartedNanos) / 1_000_000_000.0;
                    currentSpeed = gradientSpeed(elapsedS, config.durationS,
                            config.gradientStartSpeed, config.gradientMaxSpeed);
                    double progress = Math.min(elapsedS / config.durationS, 1.0);
                    if (elapsedS - lastGradientLogS >= 10) {
                        lastGradientLogS = elapsedS;
                        System.out.println(String.format(
                                "gradient speed: %.1fx (progress %.1f%%, elapsed %.1fs/%ds)",
                                currentSpeed, progress * 100, elapsedS, config.durationS));
                    }
                } else {
                    currentSpeed = config.replaySpeed;
                }

                double dueSeconds = 0;
                if (config.isUniform()) {
                    // Uniform arrival process: the ideal send schedule is
                    // t0 + i*interval, paced with the same sleep-until-due
                    // mechanism as replay so pacing_lag_ms stays meaningful.
                    // RAMP_UP_SECONDS > 0 replaces the fixed interval with a
                    // linear-QPS-climb schedule (see uniformDueSeconds):
                    // pacing lag keeps measuring send_start against the
                    // ramped ideal schedule, so it is not polluted by ramp.
                    dueSeconds = config.rampUpSeconds > 0
                            ? uniformDueSeconds(sentCount,
                                    config.sendModeQps / config.numShards,
                                    config.rampUpSeconds)
                            : sentCount * uniformIntervalS;
                    long dueNanos = replayStartedNanos + (long) (dueSeconds * 1_000_000_000L);
                    long sleepNanos = dueNanos - System.nanoTime();
                    if (sleepNanos > 0) {
                        Thread.sleep(sleepNanos / 1_000_000, (int) (sleepNanos % 1_000_000));
                    }
                } else if (currentSpeed > 0 && record.tsMs > 0) {
                    long loopOffsetMs = (long) loopIdx * traceSpanMs;
                    dueSeconds = (record.tsMs - firstTsMs + loopOffsetMs) / 1000.0 / currentSpeed;
                    long dueNanos = replayStartedNanos + (long) (dueSeconds * 1_000_000_000L);
                    long sleepNanos = dueNanos - System.nanoTime();
                    if (sleepNanos > 0) {
                        Thread.sleep(sleepNanos / 1_000_000, (int) (sleepNanos % 1_000_000));
                    }
                }

                if (cyclic && config.durationS > 0) {
                    if ((System.nanoTime() - replayStartedNanos) / 1_000_000_000L >= config.durationS) {
                        break;
                    }
                }

                final TraceRecord loopRecord;
                if (loopIdx > 0) {
                    loopRecord = makeLoopRequest(record, loopIdx, sentCount);
                } else {
                    loopRecord = record;
                }
                final double dueS = dueSeconds;
                futures.add(executor.submit(() -> handleRequest(loopRecord, semaphore, dueS)));
                sentCount++;
                sentTotal.set(sentCount);
            }

            if (!cyclic) {
                break;
            }
            if (config.durationS > 0
                    && (System.nanoTime() - replayStartedNanos) / 1_000_000_000L >= config.durationS) {
                break;
            }
            if (config.limit > 0 && sentCount >= config.limit) {
                break;
            }
            loopIdx++;
            if (futures.size() >= 100_000) {
                // Collect results from completed futures before removing them
                // to avoid losing latency/error statistics in loop mode.
                futures.removeIf(future -> {
                    if (future.isDone()) {
                        try {
                            RequestResult collected = future.get();
                            if (collected != null) {
                                results.add(collected);
                            }
                        } catch (Exception ignored) {
                            // Result could not be retrieved; will not be counted.
                        }
                        return true;
                    }
                    return false;
                });
            }
            System.out.println("loop replay: iteration " + loopIdx + " starting, sent " + sentCount
                    + " requests, elapsed " + (System.nanoTime() - replayStartedNanos) / 1_000_000_000L + "s");
        }

        sendEndNanos = System.nanoTime();
        double sendDurationS = (sendEndNanos - sendStartNanos) / 1_000_000_000.0;
        System.out.println("sending complete: sent=" + sentCount + " requests dispatched in "
                + String.format("%.1f", sendDurationS) + "s, waiting for responses...");

        ScheduledExecutorService progressMonitor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "progress-monitor");
            t.setDaemon(true);
            return t;
        });
        final int totalSent = sentCount;
        progressMonitor.scheduleAtFixedRate(() -> {
            System.out.println("  progress: " + responseCount.get() + "/" + totalSent
                    + " responses received, elapsed "
                    + String.format("%.1f", (System.nanoTime() - replayStartedNanos) / 1_000_000_000.0) + "s");
        }, 10, 10, TimeUnit.SECONDS);

        long deadlineNanos = System.nanoTime()
                + TimeUnit.SECONDS.toNanos(config.responseTimeoutSeconds);
        results.addAll(collectOutstandingResults(futures, deadlineNanos));
        progressMonitor.shutdownNow();
        executor.shutdownNow();

        long elapsedNanos = System.nanoTime() - replayStartedNanos;
        double elapsedS = elapsedNanos / 1_000_000_000.0;
        System.out.println("responses collected: " + results.size() + "/" + sentCount
                + " in " + String.format("%.1f", elapsedS) + "s");

        JsonNode serverLatency = config.skipServerLatency ? MAPPER.createObjectNode() : fetchServerLatency();
        writePerRequestResults();
        writeServerLatencySnapshot(serverLatency);
        stopPushgateway();
    }

    /**
     * Collects one result row per outstanding request future without letting a
     * single slow RPC block the collection cursor.
     *
     * <p>The legacy finalization loop blocked on {@code future.get(remaining)}
     * for each future in submission order, so one slow RPC parked the cursor
     * until the global deadline; every later future — including requests that
     * had long completed — was then synthesized as an empty timeout row with
     * no send_start. On run 20260829_094522 (A档) that truncated the real
     * per_request.jsonl rows to the first ~58% of the send window (cursor only
     * reached send_start≈70s of the 0-120s window; 206,628 synthesized rows).
     *
     * <p>This collector never blocks on an individual future. Completed futures
     * are harvested by periodic sweeps while slow ones are still in flight, so
     * every request that reaches a terminal state before the deadline
     * contributes a REAL row (send_start + terminal status) even when earlier
     * requests are still stuck. Only futures still incomplete at the deadline
     * are cancelled and synthesized. Guarantees:
     * <ul>
     *   <li>exactly one row per future — a real row when terminal before the
     *       deadline, a synthetic timeout row otherwise;</li>
     *   <li>row count conservation: real + synthetic == futures.size();</li>
     *   <li>total collection time bounded by the deadline (at most one sweep
     *       interval beyond it);</li>
     *   <li>rows returned in future submission order (legacy ordering).</li>
     * </ul>
     *
     * <p>Package-visible for the collection-semantics unit tests.
     *
     * @param futures outstanding futures as submitted by the send loop (may
     *         already contain completed entries; loop mode may have removed
     *         already-collected ones)
     * @param deadlineNanos absolute System.nanoTime() after which incomplete
     *         futures are cancelled and synthesized
     * @return one result row per future, in submission order
     */
    static List<RequestResult> collectOutstandingResults(List<Future<RequestResult>> futures,
            long deadlineNanos) {
        if (futures.isEmpty()) {
            return new ArrayList<>();
        }
        RequestResult[] collected = new RequestResult[futures.size()];
        int[] pending = new int[futures.size()];
        for (int i = 0; i < pending.length; i++) {
            pending[i] = i;
        }
        int pendingCount = pending.length;
        while (pendingCount > 0) {
            pendingCount = sweepCompletedFutures(futures, collected, pending, pendingCount);
            if (pendingCount == 0) {
                break;
            }
            long remainingNanos = deadlineNanos - System.nanoTime();
            if (remainingNanos <= 0) {
                break;
            }
            long sleepNanos = Math.min(remainingNanos, COLLECTION_SWEEP_INTERVAL_NANOS);
            try {
                // Sleep is capped by the remaining deadline budget so the final
                // sweep cannot overshoot the deadline by a full interval.
                Thread.sleep(sleepNanos / 1_000_000, (int) (sleepNanos % 1_000_000));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        for (int i = 0; i < pendingCount; i++) {
            int idx = pending[i];
            Future<RequestResult> future = futures.get(idx);
            // A future may race to completion between the last sweep and this
            // cancellation — prefer its real row when already available.
            if (future.isDone()) {
                collected[idx] = harvestFutureResult(future);
                continue;
            }
            future.cancel(true);
            RequestResult timeoutResult = new RequestResult();
            timeoutResult.status = "timeout";
            timeoutResult.error = "response deadline exceeded";
            timeoutResult.synthetic = true;
            collected[idx] = timeoutResult;
        }
        List<RequestResult> rows = new ArrayList<>(collected.length);
        for (RequestResult row : collected) {
            rows.add(row);
        }
        return rows;
    }

    /**
     * One sweep over the pending index list: harvests futures that reached a
     * terminal state since the last sweep and compacts the list in place.
     * Returns the new pending count.
     */
    private static int sweepCompletedFutures(List<Future<RequestResult>> futures,
            RequestResult[] collected, int[] pending, int pendingCount) {
        int kept = 0;
        for (int i = 0; i < pendingCount; i++) {
            int idx = pending[i];
            if (futures.get(idx).isDone()) {
                collected[idx] = harvestFutureResult(futures.get(idx));
            } else {
                pending[kept++] = idx;
            }
        }
        return kept;
    }

    /**
     * Retrieves the terminal result of a finished future. handleRequest always
     * returns a RequestResult (it converts its own exceptions into error rows),
     * so the catch only fires when the task itself died or the future was
     * cancelled externally — a synthetic exception row keeps the
     * one-row-per-future invariant (legacy catch-Exception parity).
     */
    private static RequestResult harvestFutureResult(Future<RequestResult> future) {
        try {
            return future.get();
        } catch (Exception e) {
            RequestResult errorResult = new RequestResult();
            errorResult.status = "exception";
            errorResult.error = e.toString();
            errorResult.synthetic = true;
            return errorResult;
        }
    }

    // Package-visible for loop-mode rid disjointness assertions in tests.
    TraceRecord makeLoopRequest(TraceRecord req, int loopIdx, int sentCount) {
        // Suffix carries the shard index defensively: even though loop mode shards
        // the trace, this keeps rid namespaces disjoint across shards if shard
        // counts change or a trace contains duplicated sourceRids.
        String loopSuffix = "_S" + config.shardIndex + "_L" + loopIdx;
        String newSourceRid = req.sourceRid + loopSuffix;
        String newTraceId = req.traceId.isEmpty() ? "" : req.traceId + loopSuffix;
        long newRequestId = stableRequestId(newSourceRid);
        // REPLAY_UNIQUE_PREFIX (default on): without re-salting, every loop
        // round presents byte-identical block_cache_keys, so cache affinity
        // routes each rid to the SAME prefill engine round after round and
        // the P-side load collapses onto a handful of engines (Gini ~0.56).
        // Re-salting only keys[0] keeps the shared suffix blocks (cross-
        // request prefix reuse) while giving every round a unique routing
        // prefix. The source list is shared across rounds, so it is copied
        // here and never mutated in place.
        List<Long> blockKeys = req.blockKeys;
        if (config.replayUniquePrefix && !blockKeys.isEmpty()) {
            List<Long> salted = new ArrayList<>(blockKeys);
            salted.set(0, roundSaltedKey(blockKeys.get(0), loopIdx));
            blockKeys = salted;
        }
        return new TraceRecord(newRequestId, newSourceRid, newTraceId, req.tsMs,
                req.inputLen, req.outputLen, blockKeys, req.tokenIds, req.priority);
    }

    private RequestResult handleRequest(TraceRecord record, Semaphore semaphore, double dueS) {
        long startedNanos = System.nanoTime();
        double sendDueEpochMs = replayStartedEpochMs + dueS * 1000.0;

        RequestResult result = new RequestResult();
        result.rid = record.sourceRid;
        result.traceId = record.traceId;
        result.requestId = record.requestId;
        result.ts = record.tsMs;
        result.inputLen = record.inputLen;
        result.outputLen = record.outputLen;
        result.status = "unknown";
        result.routePath = "master";
        result.sendDueEpochMs = sendDueEpochMs;
        result.priority = record.priority;

        EngineRpcService.GenerateInputPB inputPb = null;
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB scheduleResponse = null;

        try {
            semaphore.acquire();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            result.status = "exception";
            result.error = "interrupted";
            result.totalMs = (System.nanoTime() - startedNanos) / 1_000_000.0;
            errorCount.incrementAndGet();
            responseCount.incrementAndGet();
            return result;
        }

        Exception scheduleExc = null;
        inflightCount.incrementAndGet();
        try {
            double sendStartEpochMs = replayStartedEpochMs
                    + (System.nanoTime() - replayStartedNanos) / 1_000_000.0;
            result.sendStartEpochMs = sendStartEpochMs;
            result.pacingLagMs = Math.max(0.0, sendStartEpochMs - sendDueEpochMs);
            actualSentCount.incrementAndGet();

            inputPb = buildGenerateInput(record);
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB scheduleReq = buildScheduleRequest(record, inputPb);

            long scheduleStartNanos = System.nanoTime();
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub = nextScheduleStub()
                    .withDeadlineAfter(config.timeoutMs, TimeUnit.MILLISECONDS);
            scheduleResponse = stub.schedule(scheduleReq);

            result.scheduleMs = (System.nanoTime() - scheduleStartNanos) / 1_000_000.0;
            // sched_done epoch-ms (client_events.jsonl): the absolute moment the
            // schedule RPC returned — send_start_epoch_ms + scheduleMs keeps the
            // same wall clock as the engine-side engine_arrival_ms stamps.
            result.schedDoneEpochMs = sendStartEpochMs + result.scheduleMs;
            result.enqueuedByMaster = scheduleResponse.getEnqueuedByMaster();

            if (scheduleResponse.getCode() != 200 || !scheduleResponse.getSuccess()) {
                result.status = "schedule_error";
                result.error = scheduleResponse.getErrorMessage().isEmpty()
                        ? "code=" + scheduleResponse.getCode()
                        : scheduleResponse.getErrorMessage();
                result.totalMs = (System.nanoTime() - startedNanos) / 1_000_000.0;
            } else {
                result.prefill = roleAddr(scheduleResponse, "PREFILL");
                result.decode = roleAddr(scheduleResponse, "DECODE");

                if (!config.fetchOutputStream) {
                    // Under a NON_BATCH dispatcher the engine only receives the
                    // request through the client's own GenerateStreamCall
                    // (submission and stream reading are the same streaming
                    // call), so skipping the fetch would mean the request
                    // never reaches any engine. Fail fast instead of silently
                    // producing a run with zero engine load.
                    if (!scheduleResponse.getEnqueuedByMaster()) {
                        System.err.println(
                                "FATAL: FETCH_OUTPUT_STREAM=0 requires dispatcher.type=BATCH: "
                                + "schedule response reports enqueued_by_master=false "
                                + "(request_id=" + record.requestId + "). Under a NON_BATCH "
                                + "dispatcher the engine only receives requests through the "
                                + "client's GenerateStreamCall stream, so skipping the fetch "
                                + "would leave the engine idle. Re-enable stream reading or "
                                + "switch the master to a BATCH dispatcher.");
                        System.exit(86);
                    }
                    result.status = "scheduled";
                    result.totalMs = (System.nanoTime() - startedNanos) / 1_000_000.0;
                    // Route through tallyResult so the result lands in
                    // completedResults (drives flexlb_client_completed_total and
                    // the schedule-latency pushgateway series) instead of
                    // bumping counters by hand and skipping the record.
                    tallyResult(result);
                    responseCount.incrementAndGet();
                    return result;
                }
            }
        } catch (Exception e) {
            scheduleExc = e;
            result.status = "exception";
            result.error = e.toString();
            result.totalMs = (System.nanoTime() - startedNanos) / 1_000_000.0;
        } finally {
            inflightCount.decrementAndGet();
            semaphore.release();
        }

        // Escape hatch (default OFF — see enableFallback javadoc): on schedule
        // failure (exception or schedule_error), try fallback direct to
        // engines, outside the semaphore. Only reachable when the operator
        // explicitly opted in; standard load tests keep it off so every
        // failure surfaces as an error row instead of bypassing the master.
        if (scheduleResponse == null
                || "schedule_error".equals(result.status)
                || "exception".equals(result.status)) {
            if (config.enableFallback && !fallbackPrefillAddrs.isEmpty()) {
                String prefix = scheduleExc != null
                        ? "master=" + scheduleExc
                        : "master=" + result.error;
                attemptFallback(record, result, startedNanos, prefix);
            }
            tallyResult(result);
            responseCount.incrementAndGet();
            return result;
        }

        // Phase 2: engine stream reading (outside semaphore)
        if (scheduleResponse != null && config.fetchOutputStream) {
            try {
                Double firstFrameNanos = null;
                Double terminalNanos = null;

                String prefillAddr = roleAddr(scheduleResponse, "PREFILL");
                if (prefillAddr.isEmpty()) {
                    prefillAddr = roleAddr(scheduleResponse, "PDFUSION");
                }
                if (prefillAddr.isEmpty()) {
                    throw new RuntimeException("schedule response has no PREFILL/PDFUSION address");
                }

                EngineRpcService.GenerateInputPB modifiedInput = copyRoleAddrs(inputPb, scheduleResponse);
                ManagedChannel engineChannel = getEngineChannel(prefillAddr);
                RpcServiceGrpc.RpcServiceBlockingStub engineStub = RpcServiceGrpc.newBlockingStub(engineChannel)
                        .withDeadlineAfter(config.timeoutMs, TimeUnit.MILLISECONDS);

                Iterator<EngineRpcService.GenerateOutputsPB> stream;
                if (scheduleResponse.getEnqueuedByMaster()) {
                    stream = engineStub.fetchResponse(EngineRpcService.FetchRequestPB.newBuilder()
                            .setRequestId(inputPb.getRequestId())
                            .build());
                } else {
                    stream = engineStub.generateStreamCall(modifiedInput);
                }

                while (stream.hasNext()) {
                    EngineRpcService.GenerateOutputsPB output = stream.next();
                    long now = System.nanoTime();
                    if (firstFrameNanos == null) {
                        firstFrameNanos = (double) now;
                    }
                    EngineRpcService.FlattenOutputPB flatten = output.getFlattenOutput();
                    for (int j = 0; j < flatten.getFinishedCount(); j++) {
                        if (flatten.getFinished(j)) {
                            terminalNanos = (double) now;
                        }
                    }
                }

                if (firstFrameNanos == null) {
                    // Stream completed with zero outputs — mark as error to avoid
                    // masking underlying engine issues as successful requests.
                    result.status = "empty_response";
                    result.error = "stream completed with zero outputs";
                    result.totalMs = (System.nanoTime() - startedNanos) / 1_000_000.0;
                } else {
                    long endNanos = terminalNanos != null ? terminalNanos.longValue() : System.nanoTime();
                    result.ttftMs = (firstFrameNanos - startedNanos) / 1_000_000.0;
                    result.totalMs = (endNanos - startedNanos) / 1_000_000.0;
                    result.status = "ok";
                }
                result.wallClockTs = System.currentTimeMillis() / 1000.0;
            } catch (Exception e) {
                // Escape hatch (default OFF): on fetch/stream failure, try
                // fallback — only when the operator explicitly opted in;
                // otherwise the failure is recorded as an error row.
                if (config.enableFallback && !fallbackPrefillAddrs.isEmpty()) {
                    attemptFallback(record, result, startedNanos, "fetch=" + e);
                } else {
                    result.status = "exception";
                    result.error = e.toString();
                    result.totalMs = (System.nanoTime() - startedNanos) / 1_000_000.0;
                }
            }
        }

        tallyResult(result);
        responseCount.incrementAndGet();
        return result;
    }

    private void tallyResult(RequestResult result) {
        completedResults.add(result);
        if ("ok".equals(result.status) || "scheduled".equals(result.status)) {
            successCount.incrementAndGet();
        } else {
            errorCount.incrementAndGet();
        }
    }

    /**
     * Fallback direct-to-engine send (parity with Python _try_fallback): round-robin
     * picks prefill/decode addresses from endpoints.json, attaches role_addrs and
     * streams GenerateStreamCall directly against the prefill engine.
     */
    private void attemptFallback(TraceRecord record, RequestResult result, long startedNanos,
                                String masterErrorPrefix) {
        try {
            runFallbackStream(record, result, startedNanos);
        } catch (Exception fbExc) {
            result.status = "exception";
            result.error = masterErrorPrefix + "; fallback=" + fbExc;
            result.routePath = "fallback";
            result.totalMs = (System.nanoTime() - startedNanos) / 1_000_000.0;
            result.wallClockTs = System.currentTimeMillis() / 1000.0;
        }
    }

    private void runFallbackStream(TraceRecord record, RequestResult result, long startedNanos)
            throws IOException {
        String prefillAddr = roundRobinAddr(fallbackPrefillAddrs, fallbackPrefillRR,
                "no fallback prefill addresses available");
        String decodeAddr = fallbackDecodeAddrs.isEmpty()
                ? "" : roundRobinAddr(fallbackDecodeAddrs, fallbackDecodeRR, "");

        EngineRpcService.GenerateConfigPB.Builder fbConfig =
                buildGenerateInput(record).getGenerateConfig().toBuilder().clearRoleAddrs();
        fbConfig.addRoleAddrs(toRoleAddrPb(
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL, prefillAddr));
        if (!decodeAddr.isEmpty()) {
            fbConfig.addRoleAddrs(toRoleAddrPb(
                    EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE, decodeAddr));
        }
        EngineRpcService.GenerateInputPB fbInput = buildGenerateInput(record).toBuilder()
                .setGenerateConfig(fbConfig)
                .build();

        ManagedChannel channel = getEngineChannel(prefillAddr);
        RpcServiceGrpc.RpcServiceBlockingStub stub = RpcServiceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(config.timeoutMs, TimeUnit.MILLISECONDS);
        Iterator<EngineRpcService.GenerateOutputsPB> stream = stub.generateStreamCall(fbInput);

        Double firstFrameNanos = null;
        Double terminalNanos = null;
        while (stream.hasNext()) {
            EngineRpcService.GenerateOutputsPB output = stream.next();
            long now = System.nanoTime();
            if (firstFrameNanos == null) {
                firstFrameNanos = (double) now;
            }
            EngineRpcService.FlattenOutputPB flatten = output.getFlattenOutput();
            for (int j = 0; j < flatten.getFinishedCount(); j++) {
                if (flatten.getFinished(j)) {
                    terminalNanos = (double) now;
                }
            }
        }

        result.scheduleMs = 0.0;
        result.prefill = prefillAddr;
        result.decode = decodeAddr;
        result.routePath = "fallback";
        result.wallClockTs = System.currentTimeMillis() / 1000.0;
        if (firstFrameNanos == null) {
            // Parity with the main stream path: a stream that completes with
            // zero outputs is an error, not a 0-ms TTFT success.
            result.status = "empty_response";
            result.error = "stream completed with zero outputs";
            result.totalMs = (System.nanoTime() - startedNanos) / 1_000_000.0;
            return;
        }
        long endNanos = terminalNanos != null ? terminalNanos.longValue() : System.nanoTime();
        result.ttftMs = (firstFrameNanos - startedNanos) / 1_000_000.0;
        result.totalMs = (endNanos - startedNanos) / 1_000_000.0;
        result.status = "ok";
    }

    private static EngineRpcService.RoleAddrPB toRoleAddrPb(
            EngineRpcService.RoleTypePB roleType, String addr) {
        int colon = addr.lastIndexOf(':');
        if (colon <= 0 || colon == addr.length() - 1) {
            throw new IllegalArgumentException(
                    "invalid engine address '" + addr + "' (expected host:port)");
        }
        int grpcPort;
        try {
            grpcPort = Integer.parseInt(addr.substring(colon + 1));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "invalid engine address '" + addr + "' (non-numeric port)");
        }
        RoleType domainRole = RoleTypeProtoConverter.fromProto(roleType);
        return EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(RoleTypeProtoConverter.toLegacyProto(domainRole))
                .setRoleStr(domainRole.getCode())
                .setIp(addr.substring(0, colon))
                .setHttpPort(0)
                .setGrpcPort(grpcPort)
                .build();
    }

    private static String roundRobinAddr(List<String> addrs, AtomicInteger rr, String emptyError) {
        if (addrs.isEmpty()) {
            if (emptyError.isEmpty()) {
                return "";
            }
            throw new IllegalStateException(emptyError);
        }
        int idx = Math.floorMod(rr.getAndIncrement(), addrs.size());
        return addrs.get(idx);
    }

    /**
     * Loads fallback engine addresses from endpoints.json (parity with Python
     * _load_fallback_endpoints): prefers DOMAIN_ADDRESS:{domain} env entries
     * (HTTP port + 1 = gRPC port), falls back to the "engines" array.
     */
    void loadFallbackEndpoints(String path) throws IOException {
        JsonNode data = MAPPER.readTree(Path.of(path).toFile());
        String prefillDomain = data.path("prefill_domain").asText("");
        String decodeDomain = data.path("decode_domain").asText("");
        JsonNode env = data.path("env");

        parseDomainAddrs(env, "DOMAIN_ADDRESS:" + prefillDomain, fallbackPrefillAddrs);
        parseDomainAddrs(env, "DOMAIN_ADDRESS:" + decodeDomain, fallbackDecodeAddrs);

        if (fallbackPrefillAddrs.isEmpty()) {
            for (JsonNode e : data.path("engines")) {
                if ("prefill".equals(e.path("role").asText())
                        && !e.path("grpc_addr").asText("").isEmpty()) {
                    fallbackPrefillAddrs.add(e.get("grpc_addr").asText());
                }
            }
        }
        if (fallbackDecodeAddrs.isEmpty()) {
            for (JsonNode e : data.path("engines")) {
                if ("decode".equals(e.path("role").asText())
                        && !e.path("grpc_addr").asText("").isEmpty()) {
                    fallbackDecodeAddrs.add(e.get("grpc_addr").asText());
                }
            }
        }

        if (!fallbackPrefillAddrs.isEmpty()) {
            System.out.println("fallback prefill addrs: " + fallbackPrefillAddrs);
        }
        if (!fallbackDecodeAddrs.isEmpty()) {
            System.out.println("fallback decode addrs: " + fallbackDecodeAddrs);
        }
    }

    private static void parseDomainAddrs(JsonNode env, String key, List<String> out) {
        JsonNode node = env.path(key);
        if (node.isMissingNode() || node.asText("").isEmpty()) {
            return;
        }
        for (String part : node.asText().split(",")) {
            String addr = part.trim();
            if (addr.isEmpty()) {
                continue;
            }
            try {
                int colon = addr.lastIndexOf(':');
                if (colon <= 0 || colon == addr.length() - 1) {
                    throw new IllegalArgumentException("expected host:port");
                }
                // DOMAIN_ADDRESS holds the HTTP port; gRPC port = HTTP port + 1.
                out.add(addr.substring(0, colon) + ":"
                        + (Integer.parseInt(addr.substring(colon + 1)) + 1));
            } catch (RuntimeException e) {
                // One malformed entry must not abort the whole replay.
                System.err.println("WARNING: skipping malformed DOMAIN_ADDRESS entry '"
                        + addr + "' for " + key + ": " + e);
            }
        }
    }

    private EngineRpcService.GenerateInputPB buildGenerateInput(TraceRecord record) throws IOException {
        ObjectNode meta = MAPPER.createObjectNode();
        meta.put("rid", record.sourceRid);
        meta.put("trace_id", record.traceId);
        meta.put("input_len", record.inputLen);
        meta.put("output_len", record.outputLen);
        ArrayNode keysArray = meta.putArray("block_cache_keys");
        for (long key : record.blockKeys) {
            keysArray.add(key);
        }
        String uniqueKey = "flexlb_eval:" + MAPPER.writeValueAsString(meta);

        EngineRpcService.GenerateConfigPB.Builder genConfig = EngineRpcService.GenerateConfigPB.newBuilder()
                .setMaxNewTokens(Math.max(1, record.outputLen))
                .setNumReturnSequences(1)
                .setTopP(1.0f)
                .setTopK(0)
                .setTemperature(1.0f)
                .setReturnIncremental(true)
                .setIsStreaming(true)
                .setTimeoutMs((int) Math.min(config.timeoutMs, Integer.MAX_VALUE))
                .setUniqueKey(uniqueKey);

        EngineRpcService.RequestInfoPB.Builder info = EngineRpcService.RequestInfoPB.newBuilder()
                .setRequestId(record.sourceRid)
                .setTraceId(record.traceId)
                .setSourceRole("flexlb_eval");

        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(record.requestId)
                .addAllTokenIds(record.tokenIds)
                .setGenerateConfig(genConfig)
                .setClientId("flexlb_eval_client")
                .setStartTime(System.currentTimeMillis())
                .setRequestInfo(info);

        return input.build();
    }

    // Package-visible for priority wire-format assertions in tests.
    FlexlbScheduleProtocol.FlexlbScheduleRequestPB buildScheduleRequest(
            TraceRecord record, EngineRpcService.GenerateInputPB inputPb) {
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB.Builder builder =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(record.requestId)
                .setGenerateInput(inputPb.toByteString())
                .addAllBlockCacheKeys(record.blockKeys)
                .setSeqLen(record.inputLen)
                .setGenerateTimeout(config.timeoutMs)
                .setRequestTimeMs(System.currentTimeMillis())
                .setMaxNewTokens(Math.max(1, record.outputLen))
                .setNumBeams(1)
                .setForceDisableSpRun(false)
                .setModel(config.model)
                .setApiKey(config.apiKey)
                .setCacheKeyBlockSize(BLOCK_SIZE);
        if (record.priority > 0) {
            builder.setPriority(record.priority);
        }
        return builder.build();
    }

    private EngineRpcService.GenerateInputPB copyRoleAddrs(
            EngineRpcService.GenerateInputPB inputPb,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
        EngineRpcService.GenerateInputPB.Builder modified = inputPb.toBuilder();
        modified.getGenerateConfigBuilder().clearRoleAddrs();
        for (FlexlbScheduleProtocol.FlexlbServerStatusPB status : response.getServerStatusList()) {
            EngineRpcService.RoleTypePB roleType = switch (status.getRole()) {
                case "PREFILL" -> EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL;
                case "DECODE" -> EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE;
                default -> EngineRpcService.RoleTypePB.ROLE_TYPE_PDFUSION;
            };
            modified.getGenerateConfigBuilder().addRoleAddrs(
                    EngineRpcService.RoleAddrPB.newBuilder()
                            .setRole(RoleTypeProtoConverter.toLegacyProto(
                                    RoleTypeProtoConverter.fromProto(roleType)))
                            .setRoleStr(RoleTypeProtoConverter.fromProto(roleType).getCode())
                            .setIp(status.getServerIp())
                            .setHttpPort(status.getHttpPort())
                            .setGrpcPort(status.getGrpcPort())
                            .build());
        }
        return modified.build();
    }

    private String roleAddr(FlexlbScheduleProtocol.FlexlbScheduleResponsePB response, String role) {
        for (FlexlbScheduleProtocol.FlexlbServerStatusPB status : response.getServerStatusList()) {
            if (status.getRole().equals(role) && !status.getServerIp().isEmpty()) {
                return status.getServerIp() + ":" + status.getGrpcPort();
            }
        }
        return "";
    }

    private FlexlbServiceGrpc.FlexlbServiceBlockingStub nextScheduleStub() {
        int idx = scheduleStubRR.getAndIncrement() % config.nChannels;
        return scheduleStubs[Math.floorMod(idx, config.nChannels)];
    }

    private ManagedChannel getEngineChannel(String target) {
        ManagedChannel[] pool = engineChannelPools.computeIfAbsent(target, t -> {
            ManagedChannel[] channels = new ManagedChannel[config.nChannels];
            for (int i = 0; i < config.nChannels; i++) {
                channels[i] = NettyChannelBuilder.forTarget(t)
                        .eventLoopGroup(eventLoopGroup)
                        .channelType(NioSocketChannel.class)
                        .maxInboundMessageSize(16 * 1024 * 1024)
                        .flowControlWindow(1024 * 1024)
                        .keepAliveTime(30, TimeUnit.SECONDS)
                        .keepAliveTimeout(10, TimeUnit.SECONDS)
                        .usePlaintext()
                        .build();
            }
            engineChannelRR.put(t, new AtomicInteger());
            return channels;
        });
        AtomicInteger rr = engineChannelRR.get(target);
        int idx = Math.floorMod(rr.getAndIncrement(), config.nChannels);
        return pool[idx];
    }

    // ---- Trace Loading ----

    private List<TraceRecord> loadTrace(String path) throws IOException {
        List<TraceRecord> records = new ArrayList<>();
        for (String line : Files.readAllLines(Path.of(path))) {
            if (line.isBlank()) {
                continue;
            }
            try {
                JsonNode raw = MAPPER.readTree(line);
                TraceRecord record = parseTraceRecord(raw);
                if (record != null) {
                    records.add(record);
                }
            } catch (Exception e) {
                System.err.println("skipping malformed trace line: " + e.getMessage());
            }
        }
        records.sort(Comparator.comparingLong(r -> r.tsMs));
        return records;
    }

    // Package-visible for per-record priority parsing assertions in tests.
    TraceRecord parseTraceRecord(JsonNode raw) {
        int inputLen = raw.path("il").asInt(raw.path("input_token_len")
                .asInt(raw.path("backend_input_token_len").asInt(0)));
        if (inputLen <= 0) {
            return null;
        }

        int outputLen = raw.path("ol").asInt(raw.path("output_token_len").asInt(0));
        if (outputLen <= 0) {
            if ("skip".equals(config.zeroOutputPolicy)) {
                return null;
            } else if ("one".equals(config.zeroOutputPolicy)) {
                outputLen = 1;
            } else if ("default100".equals(config.zeroOutputPolicy)) {
                outputLen = 100;
            }
        }

        String sourceRid = raw.has("request_id") ? raw.get("request_id").asText()
                : raw.has("rid") ? raw.get("rid").asText()
                : Long.toString(stableRequestId(raw.toString()));
        String traceId = extractTraceId(raw);
        if (traceId.isEmpty()) {
            traceId = sourceRid;
        }

        long requestId;
        JsonNode ridIntNode = raw.get("request_id_int");
        if (ridIntNode != null) {
            requestId = toSignedInt64(ridIntNode.bigIntegerValue());
        } else {
            requestId = stableRequestId(sourceRid);
        }

        long tsMs = raw.path("ts").asLong(raw.path("request_enter_ts_epoch_ms")
                .asLong(raw.path("ts_epoch_ms").asLong(0)));

        List<Integer> tokenIds = null;
        JsonNode inputIdsNode = raw.get("input_ids");
        if (inputIdsNode != null && inputIdsNode.isArray()) {
            tokenIds = new ArrayList<>(inputIdsNode.size());
            for (JsonNode token : inputIdsNode) {
                tokenIds.add(token.asInt());
            }
        } else {
            tokenIds = Collections.nCopies(inputLen, 0);
        }

        List<Long> blockKeys = new ArrayList<>();
        JsonNode bhNode = raw.get("bh");
        if (bhNode == null) {
            bhNode = raw.get("block_cache_keys");
        }
        if (bhNode != null && bhNode.isArray()) {
            for (JsonNode key : bhNode) {
                blockKeys.add(toSignedInt64(key.bigIntegerValue()));
            }
        } else if (tokenIds != null) {
            blockKeys = computeBlockKeys(tokenIds, BLOCK_SIZE);
        }

        // Auto-TPM QoS priority: FORCE_PRIORITY > 0 pins every replayed
        // request to that single level (single-QoS runs); otherwise the
        // per-record "priority" field wins, else the client-wide PRIORITY env
        // default (50, the neutral QoS level — p0 traffic is rejected by
        // master admission); an explicit PRIORITY=0 keeps the field unset
        // on the wire (legacy behavior). A trace value outside 1-100 is not
        // clamped into the QoS domain: warn and fall back to the config
        // default so one messy line cannot skew a whole run (robustness
        // over strictness for a load-test tool).
        int priority;
        if (config.forcePriority > 0) {
            priority = config.forcePriority;
        } else {
            priority = raw.path("priority").asInt(config.priority);
            if (priority != 0 && !PriorityNormalizer.isValid(priority)) {
                System.err.println("ignoring invalid trace priority " + priority
                        + " (must be 1-100) for rid=" + sourceRid
                        + "; falling back to PRIORITY=" + config.priority);
                priority = config.priority;
            }
        }

        return new TraceRecord(requestId, sourceRid, traceId, tsMs,
                inputLen, outputLen, blockKeys, tokenIds, priority);
    }

    private String extractTraceId(JsonNode raw) {
        JsonNode traceIdNode = raw.get("trace_id");
        if (traceIdNode != null && !traceIdNode.asText().isEmpty()) {
            return traceIdNode.asText();
        }
        JsonNode controls = raw.get("request_controls");
        if (controls == null || !controls.isObject()) {
            return "";
        }
        JsonNode params = controls.get("parameters");
        if (params != null && params.isObject()) {
            for (String key : List.of("trace_id", "traceparent")) {
                JsonNode val = params.get(key);
                if (val != null && !val.asText().isEmpty()) {
                    return val.asText();
                }
            }
        }
        JsonNode metadata = controls.get("metadata");
        if (metadata != null && metadata.isArray()) {
            for (JsonNode item : metadata) {
                if (!item.isObject()) {
                    continue;
                }
                String key = item.path("key").asText().toLowerCase();
                if (key.equals("eagleeye-traceid") || key.equals("trace-id") || key.equals("x-trace-id")) {
                    JsonNode val = item.get("value");
                    if (val != null && !val.asText().isEmpty()) {
                        return val.asText();
                    }
                }
            }
        }
        return "";
    }

    private static List<Long> computeBlockKeys(List<Integer> tokenIds, int blockSize) {
        List<Long> keys = new ArrayList<>();
        int numBlocks = tokenIds.size() / blockSize;
        for (int b = 0; b < numBlocks; b++) {
            Hasher hasher = Hashing.murmur3_128().newHasher();
            for (int i = b * blockSize; i < (b + 1) * blockSize; i++) {
                hasher.putInt(tokenIds.get(i));
            }
            keys.add(hasher.hash().asLong());
        }
        return keys;
    }

    static long stableRequestId(String value) {
        return Hashing.murmur3_128()
                .hashString(value, StandardCharsets.UTF_8)
                .asLong() & 0x7FFF_FFFF_FFFF_FFFFL;
    }

    // Package-visible for loop-mode unique-prefix assertions in tests.
    // Deterministic per-round salt: the same (key, loop) pair always maps to
    // the same value, different loops map to different values.
    static long roundSaltedKey(long blockKey, int loopIdx) {
        return Hashing.murmur3_128().newHasher()
                .putLong(blockKey)
                .putInt(loopIdx)
                .hash()
                .asLong();
    }

    private static long toSignedInt64(BigInteger value) {
        BigInteger mod = value.mod(BigInteger.ONE.shiftLeft(64));
        if (mod.compareTo(BigInteger.valueOf(Long.MAX_VALUE)) > 0) {
            mod = mod.subtract(BigInteger.ONE.shiftLeft(64));
        }
        return mod.longValue();
    }

    // ---- Server Latency HTTP ----

    private void resetServerLatency() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + config.targetAddr + "/rtp_llm/server_latency/reset"))
                    .POST(HttpRequest.BodyPublishers.noBody())
                    .timeout(Duration.ofSeconds(5))
                    .build();
            httpClient.send(request, HttpResponse.BodyHandlers.discarding());
        } catch (Exception e) {
            System.out.println("server latency reset unavailable: " + e.getMessage());
        }
    }

    private JsonNode fetchServerLatency() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + config.targetAddr + "/rtp_llm/server_latency"))
                    .GET()
                    .timeout(Duration.ofSeconds(5))
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return MAPPER.readTree(response.body());
        } catch (Exception e) {
            System.out.println("server latency snapshot unavailable, using client RTT: " + e.getMessage());
            return MAPPER.createObjectNode();
        }
    }

    // ---- Output Writing ----

    private void writePerRequestResults() throws IOException {
        // client_events.jsonl (renamed from per_request.jsonl): the client-side
        // half of the multi-component JSONL event streams — one row per
        // request, rid-joined offline by aggregate_canvas_run.py against the
        // mock engine's engine_events.jsonl.
        Path perRequestPath = Path.of(config.outputDir, "client_events.jsonl");
        try (BufferedWriter writer = Files.newBufferedWriter(perRequestPath)) {
            for (RequestResult result : results) {
                writer.write(perRequestNode(result).toString());
                writer.newLine();
            }
        }
    }

    /**
     * Serializes one per-request row. Synthesized rows (collector timeout /
     * dead-future exception fallbacks) omit the "priority" key entirely
     * instead of writing a misleading 0: they never carried a request, so
     * downstream aggregation distinguishes them by key absence rather than
     * counting them as unset p0 traffic.
     */
    // Package-visible for per-request row serialization assertions in tests.
    static ObjectNode perRequestNode(RequestResult result) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("rid", result.rid);
        node.put("trace_id", result.traceId);
        node.put("request_id", result.requestId);
        node.put("ts", result.ts);
        node.put("input_len", result.inputLen);
        node.put("output_len", result.outputLen);
        node.put("status", result.status);
        node.put("schedule_ms", result.scheduleMs);
        node.put("sched_done_epoch_ms", result.schedDoneEpochMs);
        node.put("ttft_ms", result.ttftMs);
        node.put("total_ms", result.totalMs);
        node.put("enqueued_by_master", result.enqueuedByMaster);
        node.put("prefill", result.prefill);
        node.put("decode", result.decode);
        node.put("error", result.error);
        node.put("route_path", result.routePath);
        node.put("wall_clock_ts", result.wallClockTs);
        node.put("send_due_epoch_ms", result.sendDueEpochMs);
        node.put("send_start_epoch_ms", result.sendStartEpochMs);
        node.put("pacing_lag_ms", result.pacingLagMs);
        if (!result.synthetic) {
            node.put("priority", result.priority);
        }
        return node;
    }

    /**
     * Persists the terminal server-side latency snapshot next to the raw
     * per-request rows. This is raw aggregation input (master arrival and
     * completion counters feed the run-level validity checks in
     * aggregate_canvas_run.py), not client-derived statistics — the client
     * computes no summaries anymore. Multi-worker shards skip this
     * (run_online_eval.sh takes one unified final fetch after all clients
     * exit); an empty or failed snapshot writes nothing.
     */
    private void writeServerLatencySnapshot(JsonNode serverLatency) throws IOException {
        if (!serverLatency.isMissingNode() && !serverLatency.isEmpty()) {
            Path serverLatencyPath = Path.of(config.outputDir, "server_latency.json");
            MAPPER.writerWithDefaultPrettyPrinter().writeValue(serverLatencyPath.toFile(), serverLatency);
        }
    }

    // ---- Trace Filtering / Truncation (Python parity) ----

    /**
     * Applies duration/limit filters first, then {@code i % numShards} slicing —
     * identical to Python load_replay_requests + shard slice order, so LIMIT is a
     * total cap across all shards rather than a per-shard cap.
     */
    static List<TraceRecord> filterAndShard(List<TraceRecord> records, int durationS, int limit,
                                            int numShards, int shardIndex) {
        List<TraceRecord> out = records;
        if (durationS > 0 && !out.isEmpty()) {
            long firstTs = out.get(0).tsMs;
            long endTs = firstTs + durationS * 1000L;
            List<TraceRecord> filtered = new ArrayList<>();
            for (TraceRecord r : out) {
                if (r.tsMs <= endTs) {
                    filtered.add(r);
                }
            }
            out = filtered;
        }
        if (limit > 0) {
            out = out.subList(0, Math.min(limit, out.size()));
        }
        return shardSlice(out, numShards, shardIndex);
    }

    /**
     * Pure {@code i % numShards == shardIndex} slice with no duration/limit
     * filtering. Used directly in loop mode, where duration is a wall-clock
     * timeout and limit a total sent cap rather than trace filters.
     */
    static List<TraceRecord> shardSlice(List<TraceRecord> records, int numShards, int shardIndex) {
        if (numShards <= 1) {
            return new ArrayList<>(records);
        }
        if (shardIndex < 0 || shardIndex >= numShards) {
            throw new IllegalArgumentException("SHARD_INDEX must be in [0, NUM_SHARDS)");
        }
        List<TraceRecord> sharded = new ArrayList<>();
        for (int i = 0; i < records.size(); i++) {
            if (i % numShards == shardIndex) {
                sharded.add(records.get(i));
            }
        }
        return sharded;
    }

    /**
     * Caps input_len/output_len (and truncates token_ids accordingly), matching
     * Legacy Python client truncation: block_keys are left untouched.
     * Returns new records; counts each field truncation like Python does.
     */
    static List<TraceRecord> truncateRecords(List<TraceRecord> records, int maxInputLen, int maxOutputLen) {
        int truncated = 0;
        List<TraceRecord> out = new ArrayList<>(records.size());
        for (TraceRecord r : records) {
            TraceRecord cur = r;
            if (maxInputLen > 0 && cur.inputLen > maxInputLen) {
                List<Integer> tokens = cur.tokenIds;
                if (tokens != null && !tokens.isEmpty()) {
                    tokens = new ArrayList<>(tokens.subList(0, Math.min(maxInputLen, tokens.size())));
                }
                cur = new TraceRecord(cur.requestId, cur.sourceRid, cur.traceId, cur.tsMs,
                        maxInputLen, cur.outputLen, cur.blockKeys, tokens, cur.priority);
                truncated++;
            }
            if (maxOutputLen > 0 && cur.outputLen > maxOutputLen) {
                cur = new TraceRecord(cur.requestId, cur.sourceRid, cur.traceId, cur.tsMs,
                        cur.inputLen, maxOutputLen, cur.blockKeys, cur.tokenIds, cur.priority);
                truncated++;
            }
            out.add(cur);
        }
        System.out.println("truncated " + truncated + " request length(s): "
                + "max_input_len=" + maxInputLen + ", max_output_len=" + maxOutputLen);
        return out;
    }

    // ---- Pushgateway ----

    private void startPushgatewayLoop() {
        if (config.pushgatewayUrl.isEmpty()) {
            return;
        }
        pushgatewayExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "pushgateway-push");
            t.setDaemon(true);
            return t;
        });
        pushgatewayExecutor.scheduleAtFixedRate(() -> {
            try {
                pushMetrics();
            } catch (Exception e) {
                System.out.println("pushgateway push error: " + e);
            }
        }, 5, 5, TimeUnit.SECONDS);
        System.out.println("pushgateway metrics push enabled: " + config.pushgatewayUrl);
    }

    private void stopPushgateway() {
        if (pushgatewayExecutor != null) {
            pushgatewayExecutor.shutdownNow();
        }
        if (config.pushgatewayUrl.isEmpty()) {
            return;
        }
        try {
            pushMetrics();
            System.out.println("pushgateway final push done");
        } catch (Exception e) {
            System.out.println("pushgateway final push error: " + e);
        }
    }

    /** Builds the Prometheus text-format payload (parity with Python _push_metrics). */
    String buildPushMetricsBody() {
        List<String> lines = new ArrayList<>();
        int inflight = inflightCount.get();
        lines.add("flexlb_client_send_total{route_path=\"master\"} " + sentTotal.get());
        lines.add("flexlb_client_actual_send_total{route_path=\"master\"} " + actualSentCount.get());
        lines.add("flexlb_client_completed_total{route_path=\"master\"} " + completedResults.size());
        lines.add("flexlb_client_success_total{route_path=\"master\"} " + successCount.get());
        lines.add("flexlb_client_error_total{route_path=\"master\"} " + errorCount.get());
        lines.add("flexlb_client_inflight_count{route_path=\"master\"} " + inflight);
        lines.add("flexlb_client_max_concurrency{route_path=\"master\"} " + config.maxConcurrency);
        if (config.maxConcurrency > 0) {
            double util = inflight / (double) config.maxConcurrency;
            lines.add(String.format("flexlb_client_semaphore_utilization{route_path=\"master\"} %.4f", util));
        }

        List<RequestResult> snapshot;
        synchronized (completedResults) {
            snapshot = new ArrayList<>(completedResults);
        }
        if (!snapshot.isEmpty()) {
            Map<String, Map<String, List<Double>>> groups = new LinkedHashMap<>();
            for (RequestResult r : snapshot) {
                String rp = r.routePath == null ? "unknown" : r.routePath;
                Map<String, List<Double>> vals = groups.computeIfAbsent(rp, k -> {
                    Map<String, List<Double>> m = new LinkedHashMap<>();
                    m.put("schedule_ms", new ArrayList<>());
                    m.put("total_ms", new ArrayList<>());
                    m.put("ttft_ms", new ArrayList<>());
                    return m;
                });
                if (r.scheduleMs > 0) {
                    vals.get("schedule_ms").add(r.scheduleMs);
                }
                if (r.totalMs > 0) {
                    vals.get("total_ms").add(r.totalMs);
                }
                if (r.ttftMs > 0) {
                    vals.get("ttft_ms").add(r.ttftMs);
                }
            }
            for (Map.Entry<String, Map<String, List<Double>>> entry : groups.entrySet()) {
                String label = "route_path=\"" + entry.getKey() + "\"";
                for (String metricName : List.of("schedule_ms", "total_ms", "ttft_ms")) {
                    List<Double> values = entry.getValue().get(metricName);
                    if (values.isEmpty()) {
                        continue;
                    }
                    double sum = 0;
                    double mx = Double.NEGATIVE_INFINITY;
                    for (double v : values) {
                        sum += v;
                        mx = Math.max(mx, v);
                    }
                    double avg = sum / values.size();
                    double p50 = percentileNearestRank(values, 50);
                    double p99 = percentileNearestRank(values, 99);
                    lines.add(String.format("flexlb_client_%s_avg{%s} %.3f", metricName, label, avg));
                    lines.add(String.format("flexlb_client_%s_p50{%s} %.3f", metricName, label, p50));
                    lines.add(String.format("flexlb_client_%s_p99{%s} %.3f", metricName, label, p99));
                    lines.add(String.format("flexlb_client_%s_max{%s} %.3f", metricName, label, mx));
                    lines.add("flexlb_client_" + metricName + "_count{" + label + "} " + values.size());
                }
            }
        }
        return String.join("\n", lines) + "\n";
    }

    void pushMetrics() throws IOException, InterruptedException {
        String body = buildPushMetricsBody();
        String hostname;
        try {
            hostname = InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            hostname = "unknown";
        }
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(config.pushgatewayUrl
                        + "/metrics/job/flexlb_client/instance/" + hostname))
                .header("Content-Type", "text/plain; version=0.0.4")
                .PUT(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .timeout(Duration.ofSeconds(10))
                .build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200 && response.statusCode() != 202) {
            System.out.println("pushgateway push failed: " + response.statusCode() + " " + response.body());
        }
    }

    /**
     * Linear gradient replay speed (parity with Python gradient pacing): speed ramps
     * from max(1, startSpeed) to maxSpeed as elapsed time progresses over duration.
     */
    static double gradientSpeed(double elapsedS, int durationS, int gradientStartSpeed, int gradientMaxSpeed) {
        double progress = Math.min(Math.max(elapsedS, 0.0) / durationS, 1.0);
        int startSpeed = Math.max(1, gradientStartSpeed);
        return startSpeed + (gradientMaxSpeed - startSpeed) * progress;
    }

    /**
     * Ideal send time of the {@code index}-th request (0-based, per shard) in
     * uniform mode with traffic ramp-up.
     *
     * <p>During the ramp the per-shard instantaneous rate climbs linearly,
     * {@code q(t) = perShardQps * t / rampUpSeconds} for
     * {@code t <= rampUpSeconds}, then holds at {@code perShardQps}.
     * Integrating gives the cumulative send count
     * {@code N(t) = perShardQps * t^2 / (2 * rampUpSeconds)}; inverting
     * {@code N(t) = index} yields the ideal send time:
     * <ul>
     *   <li>ramp phase ({@code index < perShardQps * rampUpSeconds / 2}):
     *       {@code t = sqrt(2 * rampUpSeconds * index / perShardQps)};</li>
     *   <li>steady phase: {@code t = rampUpSeconds
     *       + (index - rampCount) / perShardQps}.</li>
     * </ul>
     *
     * <p>Properties: both pieces meet at {@code t = rampUpSeconds} with
     * matching slope {@code 1 / perShardQps} (the rate reaches exactly
     * {@code perShardQps} when the ramp ends), the send count during the
     * ramp equals the triangle integral {@code perShardQps * rampUpSeconds / 2}
     * (no lost or extra sends — pacing quality is conserved), and
     * {@code rampUpSeconds <= 0} degenerates to the fixed-interval schedule
     * {@code index / perShardQps} (legacy behavior).
     *
     * <p>Package-visible for the ramp-up schedule unit tests.
     */
    static double uniformDueSeconds(int index, double perShardQps, double rampUpSeconds) {
        if (rampUpSeconds <= 0) {
            return index / perShardQps;
        }
        double rampCount = perShardQps * rampUpSeconds / 2.0;
        if (index < rampCount) {
            return Math.sqrt(2.0 * rampUpSeconds * index / perShardQps);
        }
        return rampUpSeconds + (index - rampCount) / perShardQps;
    }

    /** Nearest-rank percentile (parity with Python LoadClient._percentile). */
    static double percentileNearestRank(List<Double> values, double p) {
        if (values.isEmpty()) {
            return 0.0;
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int idx = (int) (sorted.size() * p / 100.0);
        if (idx >= sorted.size()) {
            idx = sorted.size() - 1;
        }
        return sorted.get(idx);
    }

    // ---- Cleanup ----

    private void close() {
        for (ManagedChannel channel : scheduleChannels) {
            channel.shutdown();
        }
        for (ManagedChannel[] pool : engineChannelPools.values()) {
            for (ManagedChannel channel : pool) {
                channel.shutdown();
            }
        }
        if (eventLoopGroup != null) {
            eventLoopGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        }
    }

    // ---- Inner Classes ----

    static final class Config {
        final String traceFile;
        final String targetAddr;
        final String grpcTarget;
        final int durationS;
        final int maxConcurrency;
        final double replaySpeed;
        final int loadClientWorkers;
        final String outputDir;
        final int numShards;
        final int shardIndex;
        final int limit;
        final long timeoutMs;
        final double slaTtftMs;
        final String zeroOutputPolicy;
        /** When false the client skips reading engine output streams (FetchResponse/
         *  GenerateStreamCall phase 2) after a successful Schedule RPC. The engine
         *  still executes prefill + decode in full — only the client-side read is
         *  trimmed. Requires a BATCH dispatcher (see the enqueued_by_master
         *  fail-fast in handleRequest). */
        final boolean fetchOutputStream;
        final boolean loop;
        final int nChannels;
        final int eventLoopThreads;
        final long startAtEpochMs;
        final int responseTimeoutSeconds;
        final boolean skipServerLatency;
        final String model;
        final String apiKey;
        final boolean gradient;
        final int gradientStartSpeed;
        final int gradientMaxSpeed;
        final int maxInputLen;
        final int maxOutputLen;
        final String pushgatewayUrl;
        /**
         * Direct-to-engine fallback escape hatch (bypasses the master when
         * the Schedule RPC fails or the engine stream read fails). DEFAULT
         * OFF and it must stay off for load tests: a fallback send skips
         * master admission, routing and the schedule-latency leg entirely
         * (route_path="fallback", schedule_ms=0), so any fallback traffic
         * pollutes load-test calibers. Opt in explicitly with
         * ENABLE_FALLBACK=1 plus ENDPOINTS_FILE=&lt;a valid endpoints.json&gt;
         * only for case-test-style direct-connect scenarios that
         * deliberately want the escape hatch.
         */
        final boolean enableFallback;
        final String endpointsFile;
        final boolean dryRun;
        /** Default Auto-TPM QoS priority for all replayed requests. Defaults to
         *  the neutral level 50 (same as scheduler.ordering.defaultPriority):
         *  priority 0 is not a valid QoS level — load-test traffic at 0 gets
         *  100% rejected by master admission. Pass PRIORITY=0 explicitly to
         *  restore the old "leave the field unset on the wire" behavior. */
        final int priority;
        /** Single-priority override: > 0 pins every request to that level. */
        final int forcePriority;
        /** Arrival process: "replay" (trace ts pacing) or "uniform" (fixed interval). */
        final String sendMode;
        /** Total target QPS across all shards; required > 0 in uniform mode. */
        final double sendModeQps;
        /** Uniform-mode traffic ramp-up: per-shard QPS climbs linearly from 0
         *  to sendModeQps/numShards over RAMP_UP_SECONDS, then stays constant.
         *  0 (default) disables it — the arrival process stays a fixed
         *  interval, byte-identical to the pre-ramp behavior. Ignored in
         *  replay mode. Distinct from the orchestrator-level
         *  FLEXLB_WARMUP_SECONDS (a no-traffic prepare sleep before load
         *  starts); this knob shapes the arrival process once traffic begins. */
        final double rampUpSeconds;
        /**
         * REPLAY_UNIQUE_PREFIX: re-salt blockKeys[0] per loop round so every
         * replay round presents a fresh cache-affinity prefix (default on).
         */
        final boolean replayUniquePrefix;

        Config(String traceFile, String targetAddr, String grpcTarget,
               int durationS, int maxConcurrency, double replaySpeed,
               int loadClientWorkers, String outputDir, int numShards,
               int shardIndex, int limit, long timeoutMs, double slaTtftMs,
               String zeroOutputPolicy, boolean fetchOutputStream, boolean loop,
               int nChannels, int eventLoopThreads, long startAtEpochMs,
               int responseTimeoutSeconds, boolean skipServerLatency,
               String model, String apiKey, boolean gradient,
               int gradientStartSpeed, int gradientMaxSpeed,
               int maxInputLen, int maxOutputLen, String pushgatewayUrl,
               boolean enableFallback, String endpointsFile, boolean dryRun) {
            this(traceFile, targetAddr, grpcTarget, durationS, maxConcurrency, replaySpeed,
                    loadClientWorkers, outputDir, numShards, shardIndex, limit, timeoutMs,
                    slaTtftMs, zeroOutputPolicy, fetchOutputStream, loop, nChannels,
                    eventLoopThreads, startAtEpochMs, responseTimeoutSeconds,
                    skipServerLatency, model, apiKey, gradient,
                    gradientStartSpeed, gradientMaxSpeed, maxInputLen, maxOutputLen,
                    pushgatewayUrl, enableFallback, endpointsFile, dryRun, 0, 0, "replay", 0.0,
                    true);
        }

        Config(String traceFile, String targetAddr, String grpcTarget,
               int durationS, int maxConcurrency, double replaySpeed,
               int loadClientWorkers, String outputDir, int numShards,
               int shardIndex, int limit, long timeoutMs, double slaTtftMs,
               String zeroOutputPolicy, boolean fetchOutputStream, boolean loop,
               int nChannels, int eventLoopThreads, long startAtEpochMs,
               int responseTimeoutSeconds, boolean skipServerLatency,
               String model, String apiKey, boolean gradient,
               int gradientStartSpeed, int gradientMaxSpeed,
               int maxInputLen, int maxOutputLen, String pushgatewayUrl,
               boolean enableFallback, String endpointsFile, boolean dryRun,
               int priority) {
            this(traceFile, targetAddr, grpcTarget, durationS, maxConcurrency, replaySpeed,
                    loadClientWorkers, outputDir, numShards, shardIndex, limit, timeoutMs,
                    slaTtftMs, zeroOutputPolicy, fetchOutputStream, loop, nChannels,
                    eventLoopThreads, startAtEpochMs, responseTimeoutSeconds,
                    skipServerLatency, model, apiKey, gradient,
                    gradientStartSpeed, gradientMaxSpeed, maxInputLen, maxOutputLen,
                    pushgatewayUrl, enableFallback, endpointsFile, dryRun, priority,
                    0, "replay", 0.0, true);
        }

        Config(String traceFile, String targetAddr, String grpcTarget,
               int durationS, int maxConcurrency, double replaySpeed,
               int loadClientWorkers, String outputDir, int numShards,
               int shardIndex, int limit, long timeoutMs, double slaTtftMs,
               String zeroOutputPolicy, boolean fetchOutputStream, boolean loop,
               int nChannels, int eventLoopThreads, long startAtEpochMs,
               int responseTimeoutSeconds, boolean skipServerLatency,
               String model, String apiKey, boolean gradient,
               int gradientStartSpeed, int gradientMaxSpeed,
               int maxInputLen, int maxOutputLen, String pushgatewayUrl,
               boolean enableFallback, String endpointsFile, boolean dryRun,
               int priority, int forcePriority, String sendMode, double sendModeQps,
               boolean replayUniquePrefix) {
            this(traceFile, targetAddr, grpcTarget, durationS, maxConcurrency, replaySpeed,
                    loadClientWorkers, outputDir, numShards, shardIndex, limit, timeoutMs,
                    slaTtftMs, zeroOutputPolicy, fetchOutputStream, loop, nChannels,
                    eventLoopThreads, startAtEpochMs, responseTimeoutSeconds,
                    skipServerLatency, model, apiKey, gradient,
                    gradientStartSpeed, gradientMaxSpeed, maxInputLen, maxOutputLen,
                    pushgatewayUrl, enableFallback, endpointsFile, dryRun, priority,
                    forcePriority, sendMode, sendModeQps, 0.0, replayUniquePrefix);
        }

        Config(String traceFile, String targetAddr, String grpcTarget,
               int durationS, int maxConcurrency, double replaySpeed,
               int loadClientWorkers, String outputDir, int numShards,
               int shardIndex, int limit, long timeoutMs, double slaTtftMs,
               String zeroOutputPolicy, boolean fetchOutputStream, boolean loop,
               int nChannels, int eventLoopThreads, long startAtEpochMs,
               int responseTimeoutSeconds, boolean skipServerLatency,
               String model, String apiKey, boolean gradient,
               int gradientStartSpeed, int gradientMaxSpeed,
               int maxInputLen, int maxOutputLen, String pushgatewayUrl,
               boolean enableFallback, String endpointsFile, boolean dryRun,
               int priority, int forcePriority, String sendMode, double sendModeQps,
               double rampUpSeconds, boolean replayUniquePrefix) {
            this.traceFile = traceFile;
            this.targetAddr = targetAddr;
            this.grpcTarget = grpcTarget;
            this.durationS = durationS;
            this.maxConcurrency = maxConcurrency;
            this.replaySpeed = replaySpeed;
            this.loadClientWorkers = loadClientWorkers;
            this.outputDir = outputDir;
            this.numShards = numShards;
            this.shardIndex = shardIndex;
            this.limit = limit;
            this.timeoutMs = timeoutMs;
            this.slaTtftMs = slaTtftMs;
            this.zeroOutputPolicy = zeroOutputPolicy;
            this.fetchOutputStream = fetchOutputStream;
            this.loop = loop;
            this.nChannels = nChannels;
            this.eventLoopThreads = eventLoopThreads;
            this.startAtEpochMs = startAtEpochMs;
            this.responseTimeoutSeconds = responseTimeoutSeconds;
            this.skipServerLatency = skipServerLatency;
            this.model = model;
            this.apiKey = apiKey;
            this.gradient = gradient;
            this.gradientStartSpeed = gradientStartSpeed;
            this.gradientMaxSpeed = gradientMaxSpeed;
            this.maxInputLen = maxInputLen;
            this.maxOutputLen = maxOutputLen;
            this.pushgatewayUrl = pushgatewayUrl;
            this.enableFallback = enableFallback;
            this.endpointsFile = endpointsFile;
            this.dryRun = dryRun;
            this.priority = priority;
            this.forcePriority = forcePriority;
            this.sendMode = sendMode;
            this.sendModeQps = sendModeQps;
            this.rampUpSeconds = rampUpSeconds;
            this.replayUniquePrefix = replayUniquePrefix;
            if (!"replay".equals(sendMode) && !"uniform".equals(sendMode)) {
                throw new IllegalArgumentException(
                        "SEND_MODE must be 'replay' or 'uniform', got '" + sendMode + "'");
            }
            if ("uniform".equals(sendMode) && sendModeQps <= 0) {
                throw new IllegalArgumentException(
                        "SEND_MODE=uniform requires SEND_MODE_QPS > 0 (total target QPS)");
            }
            if (rampUpSeconds < 0) {
                throw new IllegalArgumentException(
                        "RAMP_UP_SECONDS must be >= 0, got " + rampUpSeconds);
            }
        }

        boolean isUniform() {
            return "uniform".equals(sendMode);
        }

        static Config fromEnv() {
            String targetAddr = env("TARGET_ADDR", "127.0.0.1:7001");
            String grpcTarget = env("GRPC_TARGET", "");
            if (grpcTarget.isEmpty()) {
                int colon = targetAddr.lastIndexOf(':');
                String host = targetAddr.substring(0, colon);
                int httpPort = Integer.parseInt(targetAddr.substring(colon + 1));
                grpcTarget = host + ":" + (httpPort + 2);
            }
            boolean fetchOutputStream = envBool("FETCH_OUTPUT_STREAM", true);
            return new Config(
                    env("TRACE_FILE", ""),
                    targetAddr,
                    grpcTarget,
                    envInt("DURATION_S", 0),
                    envInt("MAX_CONCURRENCY", 999_999_999),
                    envDouble("REPLAY_SPEED", 10.0),
                    envInt("LOAD_CLIENT_WORKERS", 1),
                    env("OUTPUT_DIR", "load_client_output"),
                    envInt("NUM_SHARDS", 1),
                    envInt("SHARD_INDEX", 0),
                    envInt("LIMIT", 0),
                    envLong("TIMEOUT_MS", 3_600_000L),
                    envDouble("SLA_TTFT_MS", 500.0),
                    env("ZERO_OUTPUT_POLICY", "skip"),
                    fetchOutputStream,
                    envBool("LOOP", false),
                    envInt("N_CHANNELS", 8),
                    envInt("EVENT_LOOP_THREADS", 32),
                    envLong("START_AT_EPOCH_MS", 0L),
                    envInt("RESPONSE_TIMEOUT", 120),
                    envBool("SKIP_SERVER_LATENCY", false),
                    env("MODEL", "engine_service"),
                    env("API_KEY", ""),
                    envBool("GRADIENT", false),
                    envInt("GRADIENT_START_SPEED", 10),
                    envInt("GRADIENT_MAX_SPEED", 1000),
                    envInt("MAX_INPUT_LEN", 0),
                    envInt("MAX_OUTPUT_LEN", 0),
                    env("PUSHGATEWAY_URL", ""),
                    // Fallback default OFF (load-test accuracy): fallback
                    // traffic bypasses the master and pollutes calibers;
                    // case-test direct-connect scenarios opt in explicitly
                    // via ENABLE_FALLBACK=1 + ENDPOINTS_FILE. See the
                    // enableFallback field javadoc.
                    envBool("ENABLE_FALLBACK", false),
                    env("ENDPOINTS_FILE", ""),
                    envBool("DRY_RUN", false),
                    // Default 50, not 0: priority 0 is rejected by master
                    // admission (all-p0 load-test traffic never enters the
                    // scheduling pool), and 50 is the scheduler's neutral
                    // defaultPriority — see docs/priority-scheduler-delivery-modes.md.
                    sanitizePriority(envInt("PRIORITY", 50)),
                    sanitizeForcePriority(envInt("FORCE_PRIORITY", 0)),
                    env("SEND_MODE", "replay"),
                    envDouble("SEND_MODE_QPS", 0.0),
                    envDouble("RAMP_UP_SECONDS", 0.0),
                    envBool("REPLAY_UNIQUE_PREFIX", true)
            );
        }

        void print() {
            System.out.println("=== JavaLoadClient Configuration ===");
            System.out.println("  TRACE_FILE=" + traceFile);
            System.out.println("  TARGET_ADDR=" + targetAddr);
            System.out.println("  GRPC_TARGET=" + grpcTarget);
            System.out.println("  DURATION_S=" + durationS);
            System.out.println("  MAX_CONCURRENCY=" + maxConcurrency);
            System.out.println("  REPLAY_SPEED=" + replaySpeed);
            System.out.println("  LOAD_CLIENT_WORKERS=" + loadClientWorkers);
            System.out.println("  OUTPUT_DIR=" + outputDir);
            System.out.println("  NUM_SHARDS=" + numShards);
            System.out.println("  SHARD_INDEX=" + shardIndex);
            System.out.println("  LIMIT=" + limit);
            System.out.println("  TIMEOUT_MS=" + timeoutMs);
            System.out.println("  SLA_TTFT_MS=" + slaTtftMs);
            System.out.println("  ZERO_OUTPUT_POLICY=" + zeroOutputPolicy);
            System.out.println("  LOOP=" + loop);
            System.out.println("  N_CHANNELS=" + nChannels);
            System.out.println("  EVENT_LOOP_THREADS=" + eventLoopThreads);
            System.out.println("  START_AT_EPOCH_MS=" + startAtEpochMs);
            System.out.println("  RESPONSE_TIMEOUT=" + responseTimeoutSeconds);
            System.out.println("  SKIP_SERVER_LATENCY=" + skipServerLatency);
            System.out.println("  MODEL=" + model);
            System.out.println("  API_KEY=" + (apiKey.isEmpty() ? "<empty>" : "<set>"));
            System.out.println("  FETCH_OUTPUT_STREAM=" + fetchOutputStream);
            System.out.println("  GRADIENT=" + gradient);
            System.out.println("  GRADIENT_START_SPEED=" + gradientStartSpeed);
            System.out.println("  GRADIENT_MAX_SPEED=" + gradientMaxSpeed);
            System.out.println("  MAX_INPUT_LEN=" + maxInputLen);
            System.out.println("  MAX_OUTPUT_LEN=" + maxOutputLen);
            System.out.println("  PUSHGATEWAY_URL=" + pushgatewayUrl);
            System.out.println("  ENABLE_FALLBACK=" + enableFallback);
            System.out.println("  ENDPOINTS_FILE=" + endpointsFile);
            System.out.println("  PRIORITY=" + priority);
            System.out.println("  FORCE_PRIORITY=" + forcePriority);
            System.out.println("  SEND_MODE=" + sendMode);
            System.out.println("  SEND_MODE_QPS=" + sendModeQps);
            System.out.println("  RAMP_UP_SECONDS=" + rampUpSeconds);
            System.out.println("  REPLAY_UNIQUE_PREFIX=" + replayUniquePrefix);
            System.out.println("=====================================");
        }

        private static String env(String key, String def) {
            String val = System.getenv(key);
            return val != null && !val.isEmpty() ? val : def;
        }

        private static int envInt(String key, int def) {
            String val = System.getenv(key);
            return val != null && !val.isEmpty() ? Integer.parseInt(val) : def;
        }

        private static long envLong(String key, long def) {
            String val = System.getenv(key);
            return val != null && !val.isEmpty() ? Long.parseLong(val) : def;
        }

        private static double envDouble(String key, double def) {
            String val = System.getenv(key);
            return val != null && !val.isEmpty() ? Double.parseDouble(val) : def;
        }

        private static boolean envBool(String key, boolean def) {
            String val = System.getenv(key);
            if (val == null || val.isEmpty()) {
                return def;
            }
            return val.equals("1") || val.equalsIgnoreCase("true") || val.equalsIgnoreCase("yes");
        }

        /**
         * PRIORITY must be a valid QoS level (1-100) or the explicit 0
         * "leave unset on the wire" legacy value; anything else warns and
         * falls back to the neutral default 50 — a load-test tool stays
         * robust on config typos instead of failing the run.
         */
        // Package-visible for env-validation assertions in tests.
        static int sanitizePriority(int value) {
            if (value == PriorityNormalizer.NO_PRIORITY || PriorityNormalizer.isValid(value)) {
                return value;
            }
            System.err.println("invalid PRIORITY=" + value + " (must be 0-100); falling back to 50");
            return PriorityNormalizer.DEFAULT_PRIORITY;
        }

        /**
         * FORCE_PRIORITY is either disabled (0) or a valid QoS level (1-100);
         * an invalid pin warns and disables instead of failing the run.
         */
        // Package-visible for env-validation assertions in tests.
        static int sanitizeForcePriority(int value) {
            if (value == PriorityNormalizer.NO_PRIORITY || PriorityNormalizer.isValid(value)) {
                return value;
            }
            System.err.println("invalid FORCE_PRIORITY=" + value
                    + " (must be 0-100); disabling the priority pin");
            return PriorityNormalizer.NO_PRIORITY;
        }
    }

    static final class TraceRecord {
        final long requestId;
        final String sourceRid;
        final String traceId;
        final long tsMs;
        final int inputLen;
        final int outputLen;
        final List<Long> blockKeys;
        final List<Integer> tokenIds;
        /** Auto-TPM QoS priority (30/40/50/60/70); 0 means unset. */
        final int priority;

        TraceRecord(long requestId, String sourceRid, String traceId, long tsMs,
                    int inputLen, int outputLen, List<Long> blockKeys, List<Integer> tokenIds) {
            this(requestId, sourceRid, traceId, tsMs, inputLen, outputLen, blockKeys, tokenIds, 0);
        }

        TraceRecord(long requestId, String sourceRid, String traceId, long tsMs,
                    int inputLen, int outputLen, List<Long> blockKeys, List<Integer> tokenIds,
                    int priority) {
            this.requestId = requestId;
            this.sourceRid = sourceRid;
            this.traceId = traceId;
            this.tsMs = tsMs;
            this.inputLen = inputLen;
            this.outputLen = outputLen;
            this.blockKeys = blockKeys;
            this.tokenIds = tokenIds;
            this.priority = priority;
        }
    }

    static final class RequestResult {
        String rid = "";
        String traceId = "";
        long requestId;
        long ts;
        int inputLen;
        int outputLen;
        String status = "unknown";
        double scheduleMs;
        /** Absolute epoch-ms when the schedule RPC returned (send_start + schedule_ms); 0 on the direct fallback path (no master schedule hop). */
        double schedDoneEpochMs;
        double ttftMs;
        double totalMs;
        boolean enqueuedByMaster;
        String prefill = "";
        String decode = "";
        String error = "";
        String routePath = "master";
        double wallClockTs;
        double sendDueEpochMs;
        double sendStartEpochMs;
        double pacingLagMs;
        /** Auto-TPM QoS priority carried by the schedule request; 0 means unset. */
        int priority;
        /** True for rows synthesized by the collector (deadline timeout /
         *  dead-future exception) rather than produced by a real request:
         *  they carry no priority and stay out of priority-scoped output
         *  and aggregates. */
        boolean synthetic;
    }

}
