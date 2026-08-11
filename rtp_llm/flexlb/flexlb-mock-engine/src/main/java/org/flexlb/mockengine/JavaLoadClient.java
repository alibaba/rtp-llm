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
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;

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
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Standalone Java load client (replaces the legacy Python load client).
 *
 * <p>Replays trace JSONL files against a running FlexLB master via gRPC Schedule RPC.
 * Supports multi-shard replay, configurable speed, semaphore-based concurrency control,
 * optional engine stream reading for TTFT/total latency, and generates summary.json +
 * per_request.jsonl matching the Python client format.
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

    private final Config config;
    // Parsed PRIORITY_MIX (null when unset: every request keeps priority 0,
    // which proto3 does not serialize — wire-identical to the legacy client).
    private final PriorityMix priorityMix;
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
        this.priorityMix = PriorityMix.parse(config.priorityMix);
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
        // becomes a wall-clock timeout, limit a total sent cap) but MUST still
        // shard: request_ids are deterministic hashes of sourceRid, so without
        // slicing every shard replays the identical trace and the master rejects
        // all but the first arrival of each rid as "duplicate request_id".
        // Uniform send mode reuses the loop-mode record semantics (shard slice
        // only, wall-clock duration, total sent cap): request bodies still come
        // from cycling the trace shard, only the arrival process changes.
        boolean cyclic = config.loop || config.isUniform();
        if (!cyclic) {
            records = filterAndShard(records, config.durationS, config.limit,
                    config.numShards, config.shardIndex);
        } else {
            records = shardSlice(records, config.numShards, config.shardIndex);
        }
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
                    dueSeconds = sentCount * uniformIntervalS;
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
        for (int i = 0; i < futures.size(); i++) {
            long remaining = deadlineNanos - System.nanoTime();
            if (remaining <= 0) {
                futures.get(i).cancel(true);
                // Count timed-out requests as errors so they are reflected in error_count
                RequestResult timeoutResult = new RequestResult();
                timeoutResult.status = "timeout";
                timeoutResult.error = "response deadline exceeded";
                results.add(timeoutResult);
                continue;
            }
            try {
                RequestResult result = futures.get(i).get(remaining, TimeUnit.NANOSECONDS);
                results.add(result);
            } catch (java.util.concurrent.TimeoutException e) {
                futures.get(i).cancel(true);
                RequestResult timeoutResult = new RequestResult();
                timeoutResult.status = "timeout";
                timeoutResult.error = "response timeout";
                results.add(timeoutResult);
            } catch (Exception e) {
                futures.get(i).cancel(true);
                RequestResult errorResult = new RequestResult();
                errorResult.status = "exception";
                errorResult.error = e.toString();
                results.add(errorResult);
            }
        }
        progressMonitor.shutdownNow();
        executor.shutdownNow();

        long elapsedNanos = System.nanoTime() - replayStartedNanos;
        double elapsedS = elapsedNanos / 1_000_000_000.0;
        System.out.println("responses collected: " + results.size() + "/" + sentCount
                + " in " + String.format("%.1f", elapsedS) + "s");

        JsonNode serverLatency = config.skipServerLatency ? MAPPER.createObjectNode() : fetchServerLatency();
        writePerRequestResults();
        ObjectNode summary = writeSummary(serverLatency, elapsedS, sendDurationS, sentCount);
        writeMarkdownReport(summary);
        stopPushgateway();
    }

    TraceRecord makeLoopRequest(TraceRecord req, int loopIdx, int sentCount) {
        // Suffix carries the shard index defensively: even though loop mode now
        // shards the trace, this keeps rid namespaces disjoint across shards if
        // shard counts change or a trace contains duplicated sourceRids.
        String loopSuffix = "_S" + config.shardIndex + "_L" + loopIdx;
        String newSourceRid = req.sourceRid + loopSuffix;
        String newTraceId = req.traceId.isEmpty() ? "" : req.traceId + loopSuffix;
        long newRequestId = stableRequestId(newSourceRid);
        return new TraceRecord(newRequestId, newSourceRid, newTraceId, req.tsMs,
                req.inputLen, req.outputLen, req.blockKeys, req.tokenIds);
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
            result.priority = priorityMix == null ? 0 : priorityMix.sample(ThreadLocalRandom.current());
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB scheduleReq =
                    buildScheduleRequest(record, inputPb, result.priority);

            long scheduleStartNanos = System.nanoTime();
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub = nextScheduleStub()
                    .withDeadlineAfter(config.timeoutMs, TimeUnit.MILLISECONDS);
            scheduleResponse = stub.schedule(scheduleReq);

            result.scheduleMs = (System.nanoTime() - scheduleStartNanos) / 1_000_000.0;
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

                if (!config.fetchResponseEnabled) {
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

        // Parity with Python: on schedule failure (exception or schedule_error),
        // try fallback direct to engines — outside the semaphore.
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
        if (scheduleResponse != null && config.fetchResponseEnabled) {
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
                // Parity with Python: on fetch/stream failure, try fallback.
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
        fbConfig.addRoleAddrs(toRoleAddrPb("PREFILL",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL, prefillAddr));
        if (!decodeAddr.isEmpty()) {
            fbConfig.addRoleAddrs(toRoleAddrPb("DECODE",
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
            String role, EngineRpcService.RoleTypePB roleType, String addr) {
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
        return EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(role)
                .setRoleType(roleType)
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

    private FlexlbScheduleProtocol.FlexlbScheduleRequestPB buildScheduleRequest(
            TraceRecord record, EngineRpcService.GenerateInputPB inputPb, int priority) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
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
                .setCacheKeyBlockSize(BLOCK_SIZE)
                // 0 (legacy) is proto3-default and never serialized on the wire.
                .setPriority(priority)
                .build();
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
                            .setRole(status.getRole())
                            .setRoleType(roleType)
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

    private TraceRecord parseTraceRecord(JsonNode raw) {
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

        return new TraceRecord(requestId, sourceRid, traceId, tsMs,
                inputLen, outputLen, blockKeys, tokenIds);
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
        Path perRequestPath = Path.of(config.outputDir, "per_request.jsonl");
        try (BufferedWriter writer = Files.newBufferedWriter(perRequestPath)) {
            for (RequestResult result : results) {
                ObjectNode node = MAPPER.createObjectNode();
                node.put("rid", result.rid);
                node.put("trace_id", result.traceId);
                node.put("request_id", result.requestId);
                node.put("ts", result.ts);
                node.put("input_len", result.inputLen);
                node.put("output_len", result.outputLen);
                node.put("status", result.status);
                node.put("schedule_ms", result.scheduleMs);
                node.put("ttft_ms", result.ttftMs);
                node.put("total_ms", result.totalMs);
                node.put("enqueued_by_master", result.enqueuedByMaster);
                node.put("prefill", result.prefill);
                node.put("decode", result.decode);
                node.put("error", result.error);
                node.put("route_path", result.routePath);
                node.put("priority", result.priority);
                node.put("wall_clock_ts", result.wallClockTs);
                node.put("send_due_epoch_ms", result.sendDueEpochMs);
                node.put("send_start_epoch_ms", result.sendStartEpochMs);
                node.put("pacing_lag_ms", result.pacingLagMs);
                writer.write(node.toString());
                writer.newLine();
            }
        }
    }

    private ObjectNode writeSummary(JsonNode serverLatency, double elapsedS, double sendDurationS, int sentCount)
            throws IOException {
        List<RequestResult> ok = results.stream().filter(r -> "ok".equals(r.status)).toList();
        List<RequestResult> scheduled = results.stream()
                .filter(r -> "ok".equals(r.status) || "scheduled".equals(r.status)).toList();
        int errorCount = results.size() - scheduled.size();
        int successCount = scheduled.size();

        List<Double> ttft = ok.stream().filter(r -> r.ttftMs > 0).map(r -> r.ttftMs).toList();
        List<Double> total = ok.stream().filter(r -> r.totalMs > 0).map(r -> r.totalMs).toList();
        List<Double> schedule = results.stream().filter(r -> r.scheduleMs > 0).map(r -> r.scheduleMs).toList();
        LatencySummary clientScheduleSummary = summarizeLatencies(schedule);

        JsonNode serverScheduleSummary = serverLatency.path("server_total_ms");
        boolean hasServerLatency = serverScheduleSummary.has("count") && serverScheduleSummary.get("count").asInt() > 0;

        ObjectNode serverStageLatency = MAPPER.createObjectNode();
        for (String stage : List.of("grpc_queue_ms", "route_submit_ms", "batch_wait_ms",
                "dispatch_ack_ms", "ack_response_ms")) {
            JsonNode stageNode = serverLatency.path(stage);
            serverStageLatency.set(stage, stageNode.isMissingNode() ? MAPPER.createObjectNode() : stageNode);
        }

        int slaViolations = (int) ok.stream().filter(r -> r.ttftMs > config.slaTtftMs).count();

        List<Double> sendStartTimes = new ArrayList<>();
        List<Double> pacingLags = new ArrayList<>();
        for (RequestResult r : results) {
            if (r.sendStartEpochMs > 0) {
                sendStartTimes.add(r.sendStartEpochMs);
                pacingLags.add(r.pacingLagMs);
            }
        }
        Collections.sort(sendStartTimes);
        double actualRpcQps = 0.0;
        if (sendStartTimes.size() > 1
                && sendStartTimes.get(sendStartTimes.size() - 1) > sendStartTimes.get(0)) {
            actualRpcQps = Math.round((sendStartTimes.size() - 1) * 1000.0
                    / (sendStartTimes.get(sendStartTimes.size() - 1) - sendStartTimes.get(0)) * 1000) / 1000.0;
        }

        ObjectNode summary = MAPPER.createObjectNode();
        summary.put("trace", config.traceFile);
        summary.put("max_concurrency", config.maxConcurrency);
        summary.put("elapsed_s", Math.round(elapsedS * 1000) / 1000.0);
        summary.put("total_requests", results.size());
        summary.put("scheduled", scheduled.size());
        summary.put("completed", ok.size());
        summary.put("errors", errorCount);
        summary.put("success_count", successCount);
        summary.put("error_count", errorCount);
        summary.put("offered_qps", elapsedS > 0 ? Math.round(results.size() / elapsedS * 1000) / 1000.0 : 0.0);
        summary.put("completed_qps", elapsedS > 0 ? Math.round(ok.size() / elapsedS * 1000) / 1000.0 : 0.0);
        summary.put("success_qps", elapsedS > 0 ? Math.round(successCount / elapsedS * 1000) / 1000.0 : 0.0);
        summary.put("error_qps", elapsedS > 0 ? Math.round(errorCount / elapsedS * 1000) / 1000.0 : 0.0);
        summary.put("send_duration_s", Math.round(sendDurationS * 1000) / 1000.0);
        summary.put("sent_count", sentCount);
        summary.put("actual_sent_count", actualSentCount.get());
        summary.put("recorded_result_count", results.size());
        summary.put("send_qps", sendDurationS > 0
                ? Math.round(results.size() / sendDurationS * 1000) / 1000.0 : 0.0);
        summary.put("actual_send_qps", actualRpcQps);
        summary.set("pacing_lag_ms", summarizeLatencies(pacingLags).toJson());
        if (config.isUniform()) {
            // Only emitted in uniform mode so the replay summary stays
            // byte-identical to the pre-uniform format.
            summary.put("send_mode", "uniform");
            summary.put("target_qps", config.sendModeQps);
            summary.put("per_shard_qps", config.sendModeQps / config.numShards);
            summary.put("uniform_interval_ms",
                    Math.round(1000.0 * config.numShards / config.sendModeQps * 1000) / 1000.0);
        }

        ObjectNode peakQps = MAPPER.createObjectNode();
        for (int windowMs : List.of(1, 10, 100, 1000)) {
            peakQps.put(windowMs + "ms", peakBucketQps(sendStartTimes, windowMs));
        }
        summary.set("send_peak_qps", peakQps);
        summary.put("server_arrival_qps", serverLatency.path("arrival_qps").asDouble(0.0));
        summary.put("server_completion_qps", serverLatency.path("completion_qps").asDouble(0.0));
        summary.put("n_channels", config.nChannels);
        summary.put("sla_ttft_ms", config.slaTtftMs);
        summary.put("sla_violations", slaViolations);
        summary.put("sla_violation_rate", ok.isEmpty() ? 0.0
                : Math.round(slaViolations / (double) ok.size() * 1_000_000) / 1_000_000.0);
        summary.put("schedule_latency_source", hasServerLatency ? "server" : "client");
        summary.set("schedule_latency_ms", hasServerLatency ? serverScheduleSummary : clientScheduleSummary.toJson());
        summary.set("server_schedule_latency_ms", serverScheduleSummary.isMissingNode()
                ? MAPPER.createObjectNode() : serverScheduleSummary);
        summary.set("server_stage_latency_ms", serverStageLatency);
        summary.set("client_schedule_latency_ms", clientScheduleSummary.toJson());
        summary.set("ttft_ms", summarizeLatencies(ttft).toJson());
        summary.set("total_ms", summarizeLatencies(total).toJson());
        summary.set("prefill_balance", loadBalanceSummary(ok.stream().map(r -> r.prefill).toList()));
        summary.set("decode_balance", loadBalanceSummary(ok.stream().map(r -> r.decode).toList()));
        summary.set("status_counts", countBy(results, r -> r.status));
        summary.set("route_path_counts", countBy(results, r -> r.routePath));
        if (priorityMix != null) {
            // Per-priority success/fail accounting — the data base for comparing
            // success rates across priorities in PRIORITY_MIX runs. Absent in
            // legacy runs so replay summaries stay byte-identical.
            summary.set("priority_stats", buildPriorityStats());
        }

        Path summaryPath = Path.of(config.outputDir, "summary.json");
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(summaryPath.toFile(), summary);
        System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(summary));

        if (!serverLatency.isMissingNode() && !serverLatency.isEmpty()) {
            Path serverLatencyPath = Path.of(config.outputDir, "server_latency.json");
            MAPPER.writerWithDefaultPrettyPrinter().writeValue(serverLatencyPath.toFile(), serverLatency);
        }
        return summary;
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
                        maxInputLen, cur.outputLen, cur.blockKeys, tokens);
                truncated++;
            }
            if (maxOutputLen > 0 && cur.outputLen > maxOutputLen) {
                cur = new TraceRecord(cur.requestId, cur.sourceRid, cur.traceId, cur.tsMs,
                        cur.inputLen, maxOutputLen, cur.blockKeys, cur.tokenIds);
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

    // ---- Markdown Report ----

    /**
     * Writes a minimal report.md aligned with Python report.write_markdown_report:
     * Overview + Latency table + Status counts + Top errors.
     */
    private void writeMarkdownReport(ObjectNode summary) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("# FlexLB Online Evaluation Report");
        lines.add("");
        lines.add("## Overview");
        lines.add("");
        lines.add("- Trace: `" + summary.path("trace").asText("") + "`");
        lines.add("- Total requests: " + summary.path("total_requests").asInt());
        lines.add("- Scheduled: " + summary.path("scheduled").asInt());
        lines.add("- Completed: " + summary.path("completed").asInt());
        lines.add("- Errors: " + summary.path("errors").asInt());
        lines.add("- Offered QPS: " + summary.path("offered_qps").asDouble());
        lines.add("- Completed QPS: " + summary.path("completed_qps").asDouble());
        lines.add("- Server arrival QPS: " + summary.path("server_arrival_qps").asDouble());
        lines.add("- Server completion QPS: " + summary.path("server_completion_qps").asDouble());
        lines.add("- Schedule latency source: " + summary.path("schedule_latency_source").asText("client"));
        lines.add("- SLA TTFT: " + summary.path("sla_ttft_ms").asDouble() + " ms");
        lines.add("- SLA violations: " + summary.path("sla_violations").asInt()
                + " (" + summary.path("sla_violation_rate").asDouble() + ")");
        lines.add("");
        lines.add("## Latency");
        lines.add("");
        lines.add("| Metric | Count | P50 | P90 | P95 | P99 | Max | Mean |");
        lines.add("|---|---:|---:|---:|---:|---:|---:|---:|");
        String source = summary.path("schedule_latency_source").asText("client");
        String scheduleLabel = "server".equals(source) ? "Schedule (server)" : "Schedule (client RTT)";
        appendLatencyRow(lines, scheduleLabel, summary.path("schedule_latency_ms"));
        if ("server".equals(source)) {
            appendLatencyRow(lines, "Schedule (client RTT)", summary.path("client_schedule_latency_ms"));
        }
        appendLatencyRow(lines, "TTFT", summary.path("ttft_ms"));
        appendLatencyRow(lines, "Total", summary.path("total_ms"));
        lines.add("");
        lines.add("## Status Counts");
        lines.add("");
        JsonNode statusCounts = summary.path("status_counts");
        if (statusCounts.isMissingNode() || statusCounts.isEmpty()) {
            lines.add("_empty_");
        } else {
            lines.add("| Key | Value |");
            lines.add("|---|---:|");
            List<String> keys = new ArrayList<>();
            statusCounts.fieldNames().forEachRemaining(keys::add);
            Collections.sort(keys);
            for (String key : keys) {
                lines.add("| `" + key + "` | " + statusCounts.get(key).asInt() + " |");
            }
        }
        lines.add("");
        lines.add("## Top Errors");
        lines.add("");
        Map<String, Integer> errors = new LinkedHashMap<>();
        for (RequestResult r : results) {
            if (!"ok".equals(r.status) && !"scheduled".equals(r.status)) {
                String key = !r.error.isEmpty() ? r.error : (!r.status.isEmpty() ? r.status : "unknown");
                errors.merge(key, 1, Integer::sum);
            }
        }
        if (errors.isEmpty()) {
            lines.add("_none_");
        } else {
            lines.add("| Error | Count |");
            lines.add("|---|---:|");
            errors.entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .limit(10)
                    .forEach(e -> lines.add("| `"
                            + e.getKey().replace("\n", " ").substring(0, Math.min(240, e.getKey().length()))
                            + "` | " + e.getValue() + " |"));
        }
        Path reportPath = Path.of(config.outputDir, "report.md");
        Files.writeString(reportPath, String.join("\n", lines) + "\n");
        System.out.println("report: " + reportPath);
    }

    private static void appendLatencyRow(List<String> lines, String name, JsonNode row) {
        lines.add("| " + name
                + " | " + row.path("count").asInt()
                + " | " + row.path("p50").asDouble()
                + " | " + row.path("p90").asDouble()
                + " | " + row.path("p95").asDouble()
                + " | " + row.path("p99").asDouble()
                + " | " + row.path("max").asDouble()
                + " | " + row.path("mean").asDouble() + " |");
    }

    // ---- Statistics Helpers ----

    private static LatencySummary summarizeLatencies(List<Double> values) {
        if (values.isEmpty()) {
            return new LatencySummary(0, 0, 0, 0, 0, 0, 0);
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        double sum = 0;
        for (double v : sorted) {
            sum += v;
        }
        double mean = sum / sorted.size();
        return new LatencySummary(
                sorted.size(),
                percentile(sorted, 50),
                percentile(sorted, 90),
                percentile(sorted, 95),
                percentile(sorted, 99),
                sorted.get(sorted.size() - 1),
                Math.round(mean * 1000) / 1000.0
        );
    }

    private static double percentile(List<Double> sorted, double p) {
        if (sorted.isEmpty()) {
            return 0.0;
        }
        if (sorted.size() == 1) {
            return sorted.get(0);
        }
        double rank = (sorted.size() - 1) * p / 100.0;
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        if (lo == hi) {
            return sorted.get(lo);
        }
        double weight = rank - lo;
        return Math.round((sorted.get(lo) * (1.0 - weight) + sorted.get(hi) * weight) * 1000) / 1000.0;
    }

    private static double peakBucketQps(List<Double> epochMsValues, int windowMs) {
        if (epochMsValues.isEmpty() || windowMs <= 0) {
            return 0.0;
        }
        Map<Long, Integer> buckets = new HashMap<>();
        for (double value : epochMsValues) {
            long bucket = (long) (value / windowMs);
            buckets.merge(bucket, 1, Integer::sum);
        }
        int max = buckets.values().stream().max(Integer::compare).orElse(0);
        return Math.round(max * 1000.0 / windowMs * 1000) / 1000.0;
    }

    private static ObjectNode loadBalanceSummary(List<String> assignments) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String addr : assignments) {
            if (addr != null && !addr.isEmpty()) {
                counts.merge(addr, 1, Integer::sum);
            }
        }
        ObjectNode node = MAPPER.createObjectNode();
        if (counts.isEmpty()) {
            node.set("counts", MAPPER.createObjectNode());
            node.put("stddev", 0.0);
            node.put("max_over_avg", 0.0);
            return node;
        }
        ObjectNode countsNode = MAPPER.createObjectNode();
        counts.forEach(countsNode::put);
        node.set("counts", countsNode);
        double avg = counts.values().stream().mapToInt(Integer::intValue).sum() / (double) counts.size();
        double variance = 0;
        for (int c : counts.values()) {
            variance += (c - avg) * (c - avg);
        }
        double stddev = Math.sqrt(variance / counts.size());
        int maxVal = counts.values().stream().max(Integer::compare).orElse(0);
        node.put("stddev", Math.round(stddev * 1000) / 1000.0);
        node.put("max_over_avg", avg > 0 ? Math.round(maxVal / avg * 1000) / 1000.0 : 0.0);
        return node;
    }

    private static ObjectNode countBy(List<RequestResult> rows, java.util.function.Function<RequestResult, String> extractor) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (RequestResult row : rows) {
            String value = extractor.apply(row);
            counts.merge(value != null ? value : "", 1, Integer::sum);
        }
        ObjectNode node = MAPPER.createObjectNode();
        counts.forEach(node::put);
        return node;
    }

    // Groups results by injected priority (descending) and reports per-priority
    // total/success/fail plus a fail breakdown keyed by result status.
    private ObjectNode buildPriorityStats() {
        Map<Integer, List<RequestResult>> byPriority = new TreeMap<>(Comparator.reverseOrder());
        for (RequestResult result : results) {
            byPriority.computeIfAbsent(result.priority, key -> new ArrayList<>()).add(result);
        }
        ObjectNode stats = MAPPER.createObjectNode();
        for (Map.Entry<Integer, List<RequestResult>> entry : byPriority.entrySet()) {
            List<RequestResult> rows = entry.getValue();
            List<RequestResult> failures = rows.stream()
                    .filter(r -> !"ok".equals(r.status) && !"scheduled".equals(r.status)).toList();
            ObjectNode node = MAPPER.createObjectNode();
            node.put("total", rows.size());
            node.put("success", rows.size() - failures.size());
            node.put("fail", failures.size());
            node.set("error_status_counts", countBy(failures, r -> r.status));
            stats.set(String.valueOf(entry.getKey()), node);
        }
        return stats;
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

    /**
     * Parsed PRIORITY_MIX spec: "70:10,60:15,50:50,40:15,30:10"
     * (priority:percent). Each request samples a priority according to the
     * weights. An empty/unset spec parses to null and every request keeps
     * priority 0 — the proto3 default, never serialized, i.e. the legacy path.
     */
    static final class PriorityMix {
        final int[] priorities;
        private final int[] cumulativeWeights;
        final int totalWeight;

        private PriorityMix(int[] priorities, int[] cumulativeWeights, int totalWeight) {
            this.priorities = priorities;
            this.cumulativeWeights = cumulativeWeights;
            this.totalWeight = totalWeight;
        }

        /** Returns null for an empty spec (legacy: priority stays 0). */
        static PriorityMix parse(String spec) {
            if (spec == null || spec.isBlank()) {
                return null;
            }
            String[] parts = spec.split(",");
            int[] priorities = new int[parts.length];
            int[] cumulative = new int[parts.length];
            int total = 0;
            for (int i = 0; i < parts.length; i++) {
                String[] kv = parts[i].trim().split(":");
                if (kv.length != 2) {
                    throw new IllegalArgumentException(
                            "PRIORITY_MIX entry must be 'priority:percent', got '" + parts[i].trim() + "'");
                }
                int priority = Integer.parseInt(kv[0].trim());
                int weight = Integer.parseInt(kv[1].trim());
                if (priority <= 0 || weight <= 0) {
                    throw new IllegalArgumentException(
                            "PRIORITY_MIX requires priority > 0 and percent > 0, got '" + parts[i].trim() + "'");
                }
                priorities[i] = priority;
                total += weight;
                cumulative[i] = total;
            }
            return new PriorityMix(priorities, cumulative, total);
        }

        int sample(Random random) {
            return priorityFor(random.nextInt(totalWeight));
        }

        /** Maps a roll in [0, totalWeight) onto a priority (visible for tests). */
        int priorityFor(int roll) {
            for (int i = 0; i < cumulativeWeights.length; i++) {
                if (roll < cumulativeWeights[i]) {
                    return priorities[i];
                }
            }
            return priorities[priorities.length - 1];
        }
    }

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
        final boolean scheduleOnly;
        final boolean loop;
        final int nChannels;
        final int eventLoopThreads;
        final long startAtEpochMs;
        final int responseTimeoutSeconds;
        final boolean skipServerLatency;
        final String model;
        final String apiKey;
        final boolean fetchResponseEnabled;
        final boolean gradient;
        final int gradientStartSpeed;
        final int gradientMaxSpeed;
        final int maxInputLen;
        final int maxOutputLen;
        final String pushgatewayUrl;
        final boolean enableFallback;
        final String endpointsFile;
        final boolean dryRun;
        final String sendMode;
        final double sendModeQps;
        final String priorityMix;

        Config(String traceFile, String targetAddr, String grpcTarget,
               int durationS, int maxConcurrency, double replaySpeed,
               int loadClientWorkers, String outputDir, int numShards,
               int shardIndex, int limit, long timeoutMs, double slaTtftMs,
               String zeroOutputPolicy, boolean scheduleOnly, boolean loop,
               int nChannels, int eventLoopThreads, long startAtEpochMs,
               int responseTimeoutSeconds, boolean skipServerLatency,
               String model, String apiKey, boolean fetchResponseEnabled,
               boolean gradient, int gradientStartSpeed, int gradientMaxSpeed,
               int maxInputLen, int maxOutputLen, String pushgatewayUrl,
               boolean enableFallback, String endpointsFile, boolean dryRun,
               String sendMode, double sendModeQps, String priorityMix) {
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
            this.scheduleOnly = scheduleOnly;
            this.loop = loop;
            this.nChannels = nChannels;
            this.eventLoopThreads = eventLoopThreads;
            this.startAtEpochMs = startAtEpochMs;
            this.responseTimeoutSeconds = responseTimeoutSeconds;
            this.skipServerLatency = skipServerLatency;
            this.model = model;
            this.apiKey = apiKey;
            this.fetchResponseEnabled = fetchResponseEnabled;
            this.gradient = gradient;
            this.gradientStartSpeed = gradientStartSpeed;
            this.gradientMaxSpeed = gradientMaxSpeed;
            this.maxInputLen = maxInputLen;
            this.maxOutputLen = maxOutputLen;
            this.pushgatewayUrl = pushgatewayUrl;
            this.enableFallback = enableFallback;
            this.endpointsFile = endpointsFile;
            this.dryRun = dryRun;
            this.sendMode = sendMode;
            this.sendModeQps = sendModeQps;
            this.priorityMix = priorityMix;
            if (!"replay".equals(sendMode) && !"uniform".equals(sendMode)) {
                throw new IllegalArgumentException(
                        "SEND_MODE must be 'replay' or 'uniform', got '" + sendMode + "'");
            }
            if ("uniform".equals(sendMode) && sendModeQps <= 0) {
                throw new IllegalArgumentException(
                        "SEND_MODE=uniform requires SEND_MODE_QPS > 0 (total target QPS)");
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
            boolean scheduleOnly = envBool("SCHEDULE_ONLY", false);
            String expectFetchResponse = env("FLEXLB_EXPECT_FETCH_RESPONSE", "");
            boolean fetchResponseEnabled = !scheduleOnly
                    && !expectFetchResponse.equalsIgnoreCase("0")
                    && !expectFetchResponse.equalsIgnoreCase("false")
                    && !expectFetchResponse.equalsIgnoreCase("no");
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
                    scheduleOnly,
                    envBool("LOOP", false),
                    envInt("N_CHANNELS", 8),
                    envInt("EVENT_LOOP_THREADS", 32),
                    envLong("START_AT_EPOCH_MS", 0L),
                    envInt("RESPONSE_TIMEOUT", 120),
                    envBool("SKIP_SERVER_LATENCY", false),
                    env("MODEL", "engine_service"),
                    env("API_KEY", ""),
                    fetchResponseEnabled,
                    envBool("GRADIENT", false),
                    envInt("GRADIENT_START_SPEED", 10),
                    envInt("GRADIENT_MAX_SPEED", 1000),
                    envInt("MAX_INPUT_LEN", 0),
                    envInt("MAX_OUTPUT_LEN", 0),
                    env("PUSHGATEWAY_URL", ""),
                    envBool("ENABLE_FALLBACK", false),
                    env("ENDPOINTS_FILE", ""),
                    envBool("DRY_RUN", false),
                    env("SEND_MODE", "replay"),
                    envDouble("SEND_MODE_QPS", 0.0),
                    env("PRIORITY_MIX", "")
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
            System.out.println("  SCHEDULE_ONLY=" + scheduleOnly);
            System.out.println("  LOOP=" + loop);
            System.out.println("  N_CHANNELS=" + nChannels);
            System.out.println("  EVENT_LOOP_THREADS=" + eventLoopThreads);
            System.out.println("  START_AT_EPOCH_MS=" + startAtEpochMs);
            System.out.println("  RESPONSE_TIMEOUT=" + responseTimeoutSeconds);
            System.out.println("  SKIP_SERVER_LATENCY=" + skipServerLatency);
            System.out.println("  MODEL=" + model);
            System.out.println("  API_KEY=" + (apiKey.isEmpty() ? "<empty>" : "<set>"));
            System.out.println("  FETCH_RESPONSE_ENABLED=" + fetchResponseEnabled);
            System.out.println("  GRADIENT=" + gradient);
            System.out.println("  GRADIENT_START_SPEED=" + gradientStartSpeed);
            System.out.println("  GRADIENT_MAX_SPEED=" + gradientMaxSpeed);
            System.out.println("  MAX_INPUT_LEN=" + maxInputLen);
            System.out.println("  MAX_OUTPUT_LEN=" + maxOutputLen);
            System.out.println("  PUSHGATEWAY_URL=" + pushgatewayUrl);
            System.out.println("  ENABLE_FALLBACK=" + enableFallback);
            System.out.println("  ENDPOINTS_FILE=" + endpointsFile);
            System.out.println("  SEND_MODE=" + sendMode);
            System.out.println("  SEND_MODE_QPS=" + sendModeQps);
            System.out.println("  PRIORITY_MIX=" + (priorityMix.isEmpty() ? "<unset: legacy priority 0>" : priorityMix));
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

        TraceRecord(long requestId, String sourceRid, String traceId, long tsMs,
                    int inputLen, int outputLen, List<Long> blockKeys, List<Integer> tokenIds) {
            this.requestId = requestId;
            this.sourceRid = sourceRid;
            this.traceId = traceId;
            this.tsMs = tsMs;
            this.inputLen = inputLen;
            this.outputLen = outputLen;
            this.blockKeys = blockKeys;
            this.tokenIds = tokenIds;
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
        double ttftMs;
        double totalMs;
        boolean enqueuedByMaster;
        String prefill = "";
        String decode = "";
        String error = "";
        String routePath = "master";
        int priority;
        double wallClockTs;
        double sendDueEpochMs;
        double sendStartEpochMs;
        double pacingLagMs;
    }

    record LatencySummary(int count, double p50, double p90, double p95, double p99,
                          double max, double mean) {
        ObjectNode toJson() {
            ObjectNode node = MAPPER.createObjectNode();
            node.put("count", count);
            node.put("p50", p50);
            node.put("p90", p90);
            node.put("p95", p95);
            node.put("p99", p99);
            node.put("max", max);
            node.put("mean", mean);
            return node;
        }
    }
}
