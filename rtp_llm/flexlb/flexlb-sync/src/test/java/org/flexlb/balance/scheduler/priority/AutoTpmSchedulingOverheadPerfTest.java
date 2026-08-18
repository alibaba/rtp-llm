package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * A/B performance comparison: Auto-TPM enabled vs disabled scheduling path.
 *
 * <p>Measures the per-request overhead of the PriorityAdmissionScheduler
 * plan/commit path vs the legacy direct-enqueue path. Both paths share the
 * same underlying PriorityScheduler batch/dispatch pipeline — only the
 * admission layer differs.
 *
 * <p>Topology: 1 prefill + 2 decode (mirroring MasterBatchEndToEndPerformanceTest
 * baseline). Requests use log-like varying priority (uniform 1-100) and seqLen.
 */
@Tag("performance-regression")
class AutoTpmSchedulingOverheadPerfTest {

    private static final int WARMUP_REQUESTS = 256;
    private static final int REQUEST_COUNT =
            Integer.getInteger("flexlb.perf.autotpm.requests", 8_192);
    private static final int ROUNDS = 2;

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void autoTpmOverheadVsBaselineBurst() throws Exception {
        // ================ Run baseline (autoTpmEnabled=false) ================
        List<RoundResult> baselineRounds = new ArrayList<>();
        for (int round = 0; round < ROUNDS; round++) {
            try (PerfHarness h = new PerfHarness(cfg -> { /* baseline: all defaults */ })) {
                baselineRounds.add(runBurst(h, "baseline-" + round));
            }
        }

        // ================ Run Auto-TPM enabled ================
        List<RoundResult> autoTpmRounds = new ArrayList<>();
        for (int round = 0; round < ROUNDS; round++) {
            try (PerfHarness h = new PerfHarness(cfg -> {
                cfg.setAutoTpmEnabled(true);
                cfg.setAutoTpmPrefillQueueEvictEnabled(true);
                cfg.setAutoTpmDecodeReservedEvictEnabled(true);
            })) {
                autoTpmRounds.add(runBurst(h, "autotpm-" + round));
            }
        }

        // ================ Report ================
        RoundResult baseline = best(baselineRounds);
        RoundResult autoTpm = best(autoTpmRounds);
        double overheadPct = (baseline.qps - autoTpm.qps) / baseline.qps * 100.0;
        double p50DeltaUs = (autoTpm.p50Ns - baseline.p50Ns) / 1_000.0;
        double p99DeltaUs = (autoTpm.p99Ns - baseline.p99Ns) / 1_000.0;

        System.out.printf(
                "AutoTpm scheduling overhead A/B comparison (burst, %d requests):%n"
                        + "  BASELINE (off): qps=%.1f p50=%.3fms p90=%.3fms p99=%.3fms avg=%.3fms%n"
                        + "  AUTO-TPM (on):  qps=%.1f p50=%.3fms p90=%.3fms p99=%.3fms avg=%.3fms%n"
                        + "  DELTA: throughput=%.1f%% p50_overhead=%.1fus p99_overhead=%.1fus%n",
                REQUEST_COUNT,
                baseline.qps, baseline.p50Ns / 1e6, baseline.p90Ns / 1e6,
                baseline.p99Ns / 1e6, baseline.avgNs / 1e6,
                autoTpm.qps, autoTpm.p50Ns / 1e6, autoTpm.p90Ns / 1e6,
                autoTpm.p99Ns / 1e6, autoTpm.avgNs / 1e6,
                overheadPct, p50DeltaUs, p99DeltaUs);

        // SLO: Auto-TPM path must deliver at least 5000 QPS (same floor)
        long minimumQps = Long.getLong("flexlb.perf.min-autotpm-qps", 5_000L);
        assertTrue(autoTpm.qps >= minimumQps,
                () -> String.format("Auto-TPM throughput %.1f QPS below floor %d",
                        autoTpm.qps, minimumQps));
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void autoTpmOverheadVsBaselineRateLimited() throws Exception {
        int[] targetQpsValues = {2_000, 5_000, 10_000};

        System.out.println("AutoTpm rate-limited A/B comparison:");
        System.out.printf("%-10s %-12s %-12s %-10s %-10s %-10s %-10s %-12s %-12s%n",
                "target", "mode", "actual_qps", "p50_ms", "p90_ms", "p99_ms", "avg_ms",
                "p50_delta_us", "p99_delta_us");

        for (int targetQps : targetQpsValues) {
            int requestCount = Math.max(1_024, targetQps / 2);

            RoundResult baseline;
            try (PerfHarness h = new PerfHarness(cfg -> { })) {
                runWarmup(h);
                baseline = runRateLimited(h, requestCount, targetQps);
            }

            RoundResult autoTpm;
            try (PerfHarness h = new PerfHarness(cfg -> {
                cfg.setAutoTpmEnabled(true);
                cfg.setAutoTpmPrefillQueueEvictEnabled(true);
                cfg.setAutoTpmDecodeReservedEvictEnabled(true);
            })) {
                runWarmup(h);
                autoTpm = runRateLimited(h, requestCount, targetQps);
            }

            double p50DeltaUs = (autoTpm.p50Ns - baseline.p50Ns) / 1_000.0;
            double p99DeltaUs = (autoTpm.p99Ns - baseline.p99Ns) / 1_000.0;

            System.out.printf("%-10d %-12s %-12.1f %-10.3f %-10.3f %-10.3f %-10.3f %-12s %-12s%n",
                    targetQps, "baseline", baseline.qps,
                    baseline.p50Ns / 1e6, baseline.p90Ns / 1e6,
                    baseline.p99Ns / 1e6, baseline.avgNs / 1e6, "-", "-");
            System.out.printf("%-10d %-12s %-12.1f %-10.3f %-10.3f %-10.3f %-10.3f %-12.1f %-12.1f%n",
                    targetQps, "autotpm", autoTpm.qps,
                    autoTpm.p50Ns / 1e6, autoTpm.p90Ns / 1e6,
                    autoTpm.p99Ns / 1e6, autoTpm.avgNs / 1e6,
                    p50DeltaUs, p99DeltaUs);
        }
    }

    // ==================== Measurement logic ====================

    private RoundResult runBurst(PerfHarness h, String label) throws Exception {
        // Warmup
        for (int i = 0; i < WARMUP_REQUESTS; i++) {
            h.submit(i, syntheticPriority(i), syntheticSeqLen(i));
        }
        awaitAllDispatched(h, WARMUP_REQUESTS);

        // Measurement
        long[] latencies = new long[REQUEST_COUNT];
        long startNanos = System.nanoTime();
        for (int i = 0; i < REQUEST_COUNT; i++) {
            long t0 = System.nanoTime();
            h.submit(1_000_000L + i, syntheticPriority(i), syntheticSeqLen(i));
            latencies[i] = System.nanoTime() - t0;
        }
        awaitAllDispatched(h, WARMUP_REQUESTS + REQUEST_COUNT);
        long elapsedNanos = System.nanoTime() - startNanos;

        Arrays.sort(latencies);
        double qps = REQUEST_COUNT * 1_000_000_000.0 / elapsedNanos;
        return new RoundResult(qps, latencies, label);
    }

    private void runWarmup(PerfHarness h) throws Exception {
        for (int i = 0; i < WARMUP_REQUESTS; i++) {
            h.submit(i, syntheticPriority(i), syntheticSeqLen(i));
        }
        awaitAllDispatched(h, WARMUP_REQUESTS);
    }

    private RoundResult runRateLimited(PerfHarness h, int requestCount, int targetQps)
            throws Exception {
        long[] latencies = new long[requestCount];
        long startNanos = System.nanoTime();
        for (int i = 0; i < requestCount; i++) {
            long targetNano = startNanos + (long) i * 1_000_000_000L / targetQps;
            while (System.nanoTime() < targetNano) {
                Thread.onSpinWait();
            }
            long t0 = System.nanoTime();
            h.submit(2_000_000L + i, syntheticPriority(i), syntheticSeqLen(i));
            latencies[i] = System.nanoTime() - t0;
        }
        awaitAllDispatched(h, WARMUP_REQUESTS + requestCount);
        long elapsedNanos = System.nanoTime() - startNanos;

        Arrays.sort(latencies);
        double qps = requestCount * 1_000_000_000.0 / elapsedNanos;
        return new RoundResult(qps, latencies, "rate-" + targetQps);
    }

    private static void awaitAllDispatched(PerfHarness h, int expectedTotal) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (h.dispatchCount.get() < expectedTotal && System.nanoTime() < deadline) {
            Thread.sleep(5);
        }
    }

    private static int syntheticPriority(int index) {
        // Spread priorities across 1-100 range
        return 1 + (index * 7 + 13) % 100;
    }

    private static long syntheticSeqLen(int index) {
        // Vary input length: 64 to 8192 tokens
        return 64L + (index * 127) % 8128;
    }

    private static RoundResult best(List<RoundResult> rounds) {
        return rounds.stream().max((a, b) -> Double.compare(a.qps, b.qps)).orElseThrow();
    }

    // ==================== Records ====================

    private record RoundResult(double qps, long p50Ns, long p90Ns, long p99Ns, long avgNs, String label) {
        RoundResult(double qps, long[] sortedLatencies, String label) {
            this(qps,
                    percentile(sortedLatencies, 0.50),
                    percentile(sortedLatencies, 0.90),
                    percentile(sortedLatencies, 0.99),
                    average(sortedLatencies),
                    label);
        }

        private static long percentile(long[] sorted, double pct) {
            int idx = Math.max(0, (int) Math.ceil(sorted.length * pct) - 1);
            return sorted[idx];
        }

        private static long average(long[] values) {
            long sum = 0;
            for (long v : values) sum += v;
            return sum / values.length;
        }
    }

    // ==================== Per-test harness ====================

    private static final class PerfHarness implements AutoCloseable {
        final FlexlbConfig config = new FlexlbConfig();
        final PriorityScheduler scheduler;
        final PriorityAdmissionScheduler priorityScheduler;
        final EndpointRegistry endpointRegistry;
        final AtomicLong dispatchCount = new AtomicLong();
        final WorkerStatus decodeStatus = decodeWs();
        final DefaultBatchDispatcher dispatcher;

        private static final String PREFILL_IP = "127.0.0.1";
        private static final int PREFILL_HTTP = 9000;
        private static final int PREFILL_GRPC = 9001;
        private static final String PREFILL_IP_PORT = PREFILL_IP + ":" + PREFILL_HTTP;
        private static final String DECODE_IP = "127.0.0.2";
        private static final int DECODE_HTTP = 9100;
        private static final int DECODE_GRPC = 9101;
        private static final String DECODE_IP_PORT = DECODE_IP + ":" + DECODE_HTTP;

        PerfHarness(Consumer<FlexlbConfig> customize) {
            ConfigService configService = mock(ConfigService.class);
            EngineGrpcClient grpcClient = mock(EngineGrpcClient.class);
            BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
            PrioritySchedulerReporter priorityReporter = mock(PrioritySchedulerReporter.class);

            // Perf config matching MasterBatchEndToEndPerformanceTest
            config.setFlexlbBatchAlgorithm("fixed_window");
            config.setFlexlbBatchFixedWaitMs(10L);
            config.setFlexlbBatchPredictThresholdMs(0L);
            config.setFlexlbBatchSizeMax(16);
            config.setFlexlbBatchQueueMaxSize(4_096);
            config.setFlexlbBatchMaxInflight(20_000);
            config.setFlexlbBatchDispatchPoolSize(32);
            config.setFlexlbBatchDispatchQueueSize(2_048);
            config.setPrefillQueueSizeThreshold(1_000_000L);
            config.setScheduleWorkerSize(1);
            config.setCostSloMs(50_000L);
            config.setCostSloRiskMarginMs(50L);
            config.setDecodeConcurrencyLimit(100_000);
            customize.accept(config);
            when(configService.loadBalanceConfig()).thenReturn(config);

            AtomicReference<PriorityScheduler> schedulerRef = new AtomicReference<>();
            AtomicReference<EndpointRegistry> endpointRegistryRef = new AtomicReference<>();

            // Mock gRPC: immediate success followed by the same Decode
            // KV_ALLOCATED WorkerStatus signal that closes production leases.
            when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                    any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                    .thenAnswer(inv -> {
                        EngineRpcService.EnqueueBatchRequestPB req = inv.getArgument(2);
                        int count = 0;
                        for (EngineRpcService.EnqueueBatchDpSlotPB slot : req.getDpSlotsList()) {
                            count += slot.getRequestsCount();
                        }
                        dispatchCount.addAndGet(count);
                        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                                        .setBatchId(req.getBatchId());
                        List<Long> acceptedRequestIds = new ArrayList<>();
                        for (EngineRpcService.EnqueueBatchDpSlotPB slot : req.getDpSlotsList()) {
                            for (EngineRpcService.EnqueueBatchExternalInputPB input
                                    : slot.getRequestsList()) {
                                long requestId = input.getInput().getRequestId();
                                acceptedRequestIds.add(requestId);
                                response.addSuccesses(
                                        EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                                .setRequestId(requestId));
                            }
                        }
                        reportDecodeAccepted(schedulerRef.get(), endpointRegistryRef.get(),
                                decodeStatus, acceptedRequestIds);
                        return CompletableFuture.completedFuture(response.build());
                    });

            // Router: always succeeds, reserves on decode
            Router router = mock(Router.class);
            endpointRegistry = new EndpointRegistry(configService, schedulerRef::get, reporter);
            endpointRegistryRef.set(endpointRegistry);

            when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
                BalanceContext ctx = inv.getArgument(0);
                DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
                decodeEp.reserve(ctx.getRequestId(), (int) ctx.getRequest().getSeqLen(),
                        (int) ctx.getRequest().getSeqLen() + 8,
                        ctx.getPriority(), ctx.getDeadlineMs());
                return successRoute(ctx.getRequestId());
            });

            dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);

            // Build priority scheduler (always created; only active when autoTpmEnabled=true)
            priorityScheduler = new PriorityAdmissionScheduler(
                    configService, router, endpointRegistry, new PlanCommitter(),
                    new PrioritySloPolicy(PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS,
                            PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS),
                    priorityReporter, reporter, new UnsupportedEngineCancelChannel());

            scheduler = new PriorityScheduler(configService, router,
                    endpointRegistry, dispatcher, reporter, priorityScheduler, null);
            schedulerRef.set(scheduler);

            // Register endpoints
            WorkerStatus prefillWs = new WorkerStatus();
            prefillWs.setIp(PREFILL_IP);
            prefillWs.setPort(PREFILL_HTTP);
            prefillWs.setGrpcPort(PREFILL_GRPC);
            endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefillWs);

            endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeStatus);
            endpointRegistry.getDecode(DECODE_IP_PORT)
                    .onWorkerStatusUpdate(decodeStatus, new WorkerStatusResponse());
        }

        private static void reportDecodeAccepted(PriorityScheduler scheduler,
                                                 EndpointRegistry registry,
                                                 WorkerStatus decodeStatus,
                                                 List<Long> requestIds) {
            Map<String, TaskInfo> running = new java.util.HashMap<>();
            for (long requestId : requestIds) {
                TaskInfo task = new TaskInfo();
                task.setRequestId(requestId);
                task.setPhase(TaskPhase.KV_ALLOCATED);
                running.put(String.valueOf(requestId), task);
            }
            WorkerStatusResponse response = new WorkerStatusResponse();
            response.setRole(RoleType.DECODE);
            response.setRunningTaskInfo(running);
            registry.getDecode(DECODE_IP_PORT).onWorkerStatusUpdate(decodeStatus, response);
            scheduler.onWorkerStatusUpdate(response);
        }

        private static WorkerStatus decodeWs() {
            WorkerStatus status = new WorkerStatus();
            status.setIp(DECODE_IP);
            status.setPort(DECODE_HTTP);
            status.setGrpcPort(DECODE_GRPC);
            status.setAvailableKvCacheTokens(new AtomicLong(1_000_000_000L));
            status.setTotalKvCacheTokens(new AtomicLong(2_000_000_000L));
            return status;
        }

        CompletableFuture<Response> submit(long requestId, int priority, long seqLen) {
            BalanceContext ctx = context(requestId, priority, seqLen);
            return scheduler.submit(ctx);
        }

        @Override
        public void close() {
            priorityScheduler.shutdown();
            scheduler.shutdown();
            dispatcher.shutdown();
        }

        private BalanceContext context(long requestId, int priority, long seqLen) {
            Request request = new Request();
            request.setRequestId(requestId);
            request.setSeqLen(seqLen);
            request.setMaxNewTokens(8);
            request.setNumBeams(1);
            request.setModel("perf-model");
            request.setPriority(priority);

            BalanceContext ctx = new BalanceContext();
            ctx.setRequest(request);
            ctx.setConfig(config);
            ctx.setGenerateInputPbBytes(generateInputBytes(requestId, (int) seqLen));
            ctx.setBudget(ScheduleBudget.forDeadline(priority,
                    ctx.getStartTime(), ctx.getStartTime() + 30_000));
            return ctx;
        }

        private static byte[] generateInputBytes(long requestId, int tokenCount) {
            EngineRpcService.GenerateInputPB.Builder input =
                    EngineRpcService.GenerateInputPB.newBuilder()
                            .setRequestId(requestId)
                            .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                                    .setMaxNewTokens(8)
                                    .build());
            for (int i = 0; i < Math.min(tokenCount, 128); i++) {
                input.addTokenIds(100 + i);
            }
            return input.build().toByteArray();
        }

        private static Response successRoute(long requestId) {
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(List.of(
                    server(RoleType.PREFILL, PREFILL_IP, PREFILL_HTTP, PREFILL_GRPC, requestId),
                    server(RoleType.DECODE, DECODE_IP, DECODE_HTTP, DECODE_GRPC, requestId)));
            return response;
        }

        private static ServerStatus server(RoleType role, String ip, int httpPort,
                                           int grpcPort, long requestId) {
            ServerStatus status = new ServerStatus();
            status.setSuccess(true);
            status.setRole(role);
            status.setServerIp(ip);
            status.setHttpPort(httpPort);
            status.setGrpcPort(grpcPort);
            status.setDpRank(0);
            status.setGroup("perf-group");
            status.setRequestId(requestId);
            return status;
        }
    }
}
