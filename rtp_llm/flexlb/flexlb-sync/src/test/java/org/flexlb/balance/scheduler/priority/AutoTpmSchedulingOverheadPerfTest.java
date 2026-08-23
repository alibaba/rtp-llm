package org.flexlb.balance.scheduler.priority;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
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
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.RouteService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Rate-limited performance matrix for every legal scheduler, ordering,
 * decision, and dispatcher combination.
 *
 * <p>The test deliberately enters through {@link RouteService}, including DIRECT,
 * so it measures the public configuration switch rather than calling a scheduler
 * implementation directly. Topology and load are controlled with:
 *
 * <ul>
 *   <li>{@code flexlb.perf.prefill-workers} (default 1)</li>
 *   <li>{@code flexlb.perf.decode-workers} (default 1)</li>
 *   <li>{@code flexlb.perf.target-qps} (default 3000)</li>
 *   <li>{@code flexlb.perf.autotpm.requests} (default max(1024, target QPS))</li>
 * </ul>
 *
 * <p>DIRECT + BATCH is rejected by the public config validator and is therefore
 * intentionally absent. Every submitted future is retained and must complete
 * successfully; dispatch and completion counts are exact hard gates.
 */
@Tag("performance-regression")
class AutoTpmSchedulingOverheadPerfTest {

    private static final int WARMUP_REQUESTS = 256;
    private static final int PREFILL_WORKERS = integerProperty(
            "flexlb.perf.prefill-workers", "flexlb.perf.autotpm.prefills", 1);
    private static final int DECODE_WORKERS = integerProperty(
            "flexlb.perf.decode-workers", "flexlb.perf.autotpm.decodes", 1);
    private static final int TARGET_QPS =
            Integer.getInteger("flexlb.perf.target-qps", 3_000);
    private static final int REQUEST_COUNT = Integer.getInteger(
            "flexlb.perf.autotpm.requests", Math.max(1_024, TARGET_QPS));
    private static final long PHASE_TIMEOUT_SECONDS =
            Long.getLong("flexlb.perf.phase-timeout-seconds", 20L);
    private static Logger flexlbLogger;
    private static Logger syncLogger;
    private static Level previousFlexlbLogLevel;
    private static Level previousSyncLogLevel;

    @BeforeAll
    static void suppressHotPathLogging() {
        flexlbLogger = (Logger) LoggerFactory.getLogger("flexlbLogger");
        syncLogger = (Logger) LoggerFactory.getLogger("syncLogger");
        previousFlexlbLogLevel = flexlbLogger.getLevel();
        previousSyncLogLevel = syncLogger.getLevel();
        flexlbLogger.setLevel(Level.ERROR);
        syncLogger.setLevel(Level.WARN);
    }

    @AfterAll
    static void restoreLogging() {
        flexlbLogger.setLevel(previousFlexlbLogLevel);
        syncLogger.setLevel(previousSyncLogLevel);
    }

    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void legalSchedulingConfigMatrixRateLimited() throws Exception {
        assertTrue(PREFILL_WORKERS > 0, "prefill worker count must be positive");
        assertTrue(DECODE_WORKERS > 0, "decode worker count must be positive");
        assertTrue(TARGET_QPS > 0, "target QPS must be positive");
        assertTrue(REQUEST_COUNT > 0, "request count must be positive");

        System.out.printf(
                "Scheduling config matrix: target_qps=%d requests=%d topology=%dP/%dD%n",
                TARGET_QPS, REQUEST_COUNT, PREFILL_WORKERS, DECODE_WORKERS);
        System.out.printf("%-42s %-12s %-12s %-12s %-12s %-12s%n",
                "mode", "actual_qps", "e2e_p50_ms", "e2e_p90_ms",
                "e2e_p99_ms", "e2e_avg_ms");

        for (Mode mode : Mode.values()) {
            RoundResult result;
            try (PerfHarness harness = new PerfHarness(mode)) {
                runWarmup(harness);
                result = runRateLimited(harness);
            }
            System.out.printf("%-42s %-12.1f %-12.3f %-12.3f %-12.3f %-12.3f%n",
                    mode.label, result.qps,
                    result.p50Ns / 1e6, result.p90Ns / 1e6,
                    result.p99Ns / 1e6, result.avgNs / 1e6);
        }
    }

    private void runWarmup(PerfHarness harness) throws Exception {
        List<CompletableFuture<Response>> futures = new ArrayList<>(WARMUP_REQUESTS);
        for (int i = 0; i < WARMUP_REQUESTS; i++) {
            futures.add(harness.submit(i, syntheticPriority(i), syntheticSeqLen(i)));
        }
        harness.awaitSuccessful(futures, WARMUP_REQUESTS);
    }

    private RoundResult runRateLimited(PerfHarness harness) throws Exception {
        List<CompletableFuture<Response>> futures = new ArrayList<>(REQUEST_COUNT);
        List<CompletableFuture<Response>> latencyObservers =
                new ArrayList<>(REQUEST_COUNT);
        long[] completionLatencies = new long[REQUEST_COUNT];
        long startNanos = System.nanoTime();
        for (int i = 0; i < REQUEST_COUNT; i++) {
            long targetNanos = startNanos + (long) i * 1_000_000_000L / TARGET_QPS;
            while (System.nanoTime() < targetNanos) {
                Thread.onSpinWait();
            }
            int latencyIndex = i;
            long requestStarted = System.nanoTime();
            CompletableFuture<Response> future = harness.submit(
                    1_000_000L + i, syntheticPriority(i), syntheticSeqLen(i));
            futures.add(future);
            latencyObservers.add(future.whenComplete((ignored, error) ->
                    completionLatencies[latencyIndex] =
                            System.nanoTime() - requestStarted));
        }
        harness.awaitSuccessful(futures, WARMUP_REQUESTS + REQUEST_COUNT);
        CompletableFuture.allOf(latencyObservers.toArray(CompletableFuture[]::new))
                .get(PHASE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        long elapsedNanos = System.nanoTime() - startNanos;

        Arrays.sort(completionLatencies);
        double qps = REQUEST_COUNT * 1_000_000_000.0 / elapsedNanos;
        return new RoundResult(qps, completionLatencies);
    }

    private static int integerProperty(String preferred, String compatibility,
                                       int defaultValue) {
        String preferredValue = System.getProperty(preferred);
        return preferredValue != null
                ? Integer.parseInt(preferredValue)
                : Integer.getInteger(compatibility, defaultValue);
    }

    private static int syntheticPriority(int index) {
        return 1 + (index * 7 + 13) % 100;
    }

    private static long syntheticSeqLen(int index) {
        return 64L + (index * 127) % 8128;
    }

    private enum Mode {
        QUEUE_PRIORITY_SINGLE_BATCH(
                "QUEUE+PRIORITY+SINGLE+BATCH", true, true, true, false),
        QUEUE_FIFO_SINGLE_BATCH(
                "QUEUE+FIFO+SINGLE+BATCH", true, true, false, false),
        QUEUE_PRIORITY_FIXED_WINDOW_BATCH(
                "QUEUE+PRIORITY+FIXED_WINDOW+BATCH", true, true, true, true),
        QUEUE_FIFO_FIXED_WINDOW_BATCH(
                "QUEUE+FIFO+FIXED_WINDOW+BATCH", true, true, false, true),
        QUEUE_PRIORITY_SINGLE_NON_BATCH(
                "QUEUE+PRIORITY+SINGLE+NON_BATCH", true, false, true, false),
        QUEUE_FIFO_SINGLE_NON_BATCH(
                "QUEUE+FIFO+SINGLE+NON_BATCH", true, false, false, false),
        QUEUE_PRIORITY_FIXED_WINDOW_NON_BATCH(
                "QUEUE+PRIORITY+FIXED_WINDOW+NON_BATCH", true, false, true, true),
        QUEUE_FIFO_FIXED_WINDOW_NON_BATCH(
                "QUEUE+FIFO+FIXED_WINDOW+NON_BATCH", true, false, false, true),
        DIRECT_NON_BATCH("DIRECT+NON_BATCH", false, false, false, false);

        private final String label;
        private final boolean queue;
        private final boolean batch;
        private final boolean priority;
        private final boolean fixedWindow;

        Mode(String label, boolean queue, boolean batch,
             boolean priority, boolean fixedWindow) {
            this.label = label;
            this.queue = queue;
            this.batch = batch;
            this.priority = priority;
            this.fixedWindow = fixedWindow;
        }

        void configure(FlexlbConfig config) {
            if (!queue) {
                config.setScheduler(new DirectSchedulerConfig());
                SchedulingTestConfig.useNonBatchDispatcher(config);
                return;
            }
            if (priority) {
                SchedulingTestConfig.usePriorityQueue(config);
            } else {
                SchedulingTestConfig.useFifoQueue(config);
            }
            if (fixedWindow) {
                SchedulingTestConfig.useFixedWindowDecision(config);
            } else {
                SchedulingTestConfig.useSingleDecision(config);
            }
            if (batch) {
                SchedulingTestConfig.useBatchDispatcher(config);
            } else {
                SchedulingTestConfig.useNonBatchDispatcher(config);
            }
        }
    }

    private record RoundResult(double qps, long p50Ns, long p90Ns, long p99Ns, long avgNs) {
        RoundResult(double qps, long[] sortedLatencies) {
            this(qps,
                    percentile(sortedLatencies, 0.50),
                    percentile(sortedLatencies, 0.90),
                    percentile(sortedLatencies, 0.99),
                    average(sortedLatencies));
        }

        private static long percentile(long[] sorted, double percentile) {
            int index = Math.max(0, (int) Math.ceil(sorted.length * percentile) - 1);
            return sorted[index];
        }

        private static long average(long[] values) {
            long sum = 0;
            for (long value : values) {
                sum += value;
            }
            return sum / values.length;
        }
    }

    private record WorkerTarget(String ip, int httpPort, int grpcPort,
                                String ipPort, WorkerStatus status) {
    }

    private static final class PerfHarness implements AutoCloseable {
        private static final int PREFILL_HTTP_BASE = 9_000;
        private static final int PREFILL_GRPC_BASE = 19_000;
        private static final int DECODE_HTTP_BASE = 29_000;
        private static final int DECODE_GRPC_BASE = 39_000;

        private final Mode mode;
        private final FlexlbConfig config = new FlexlbConfig();
        private final PriorityScheduler scheduler;
        private final PriorityAdmissionScheduler priorityScheduler;
        private final EndpointRegistry endpointRegistry;
        private final DefaultBatchDispatcher dispatcher;
        private final RouteService routeService;
        private final List<WorkerTarget> prefills = new ArrayList<>();
        private final List<WorkerTarget> decodes = new ArrayList<>();
        private final Map<Long, Integer> decodeAssignment = new ConcurrentHashMap<>();
        private final List<Map<Long, TaskInfo>> activeDecodeTasks = new ArrayList<>();
        private final List<Object> decodeStatusLocks = new ArrayList<>();
        private final AtomicLong dispatchCount = new AtomicLong();
        private final AtomicLong completionCount = new AtomicLong();
        private final AtomicInteger maxDecisionGroupSize = new AtomicInteger();
        private final AtomicReference<String> firstFailure = new AtomicReference<>();

        PerfHarness(Mode mode) {
            this.mode = mode;
            mode.configure(config);
            configureCapacity(config, mode);

            ConfigService configService = mock(ConfigService.class);
            EngineGrpcClient grpcClient = mock(EngineGrpcClient.class);
            BatchSchedulerReporter reporter =
                    new BatchSchedulerReporter(NoOpFlexMonitor.getInstance());
            PrioritySchedulerReporter priorityReporter =
                    new PrioritySchedulerReporter(NoOpFlexMonitor.getInstance());
            RecentCacheKeyTraceReporter traceReporter = mock(RecentCacheKeyTraceReporter.class);
            DefaultRouter router = mock(DefaultRouter.class);
            when(configService.loadBalanceConfig()).thenReturn(config);

            AtomicReference<PriorityScheduler> schedulerRef = new AtomicReference<>();
            endpointRegistry = new EndpointRegistry(configService, schedulerRef::get, reporter);

            when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
                BalanceContext context = invocation.getArgument(0);
                long requestId = context.getRequestId();
                int prefillIndex = Math.floorMod(requestId, prefills.size());
                int decodeIndex = Math.floorMod(requestId, decodes.size());
                WorkerTarget prefill = prefills.get(prefillIndex);
                WorkerTarget decode = decodes.get(decodeIndex);
                if (mode.queue) {
                    DecodeEndpoint decodeEndpoint = endpointRegistry.getDecode(decode.ipPort);
                    decodeEndpoint.reserve(requestId,
                            (int) context.getRequest().getSeqLen(),
                            (int) context.getRequest().getSeqLen() + 8,
                            context.getPriority());
                    decodeAssignment.put(requestId, decodeIndex);
                }
                return successRoute(requestId, prefill, decode);
            });

            when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                    any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                    .thenAnswer(invocation -> {
                        EngineRpcService.EnqueueBatchRequestPB request = invocation.getArgument(2);
                        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                                        .setBatchId(request.getBatchId());
                        List<Long> accepted = new ArrayList<>();
                        for (EngineRpcService.EnqueueBatchDpSlotPB slot
                                : request.getDpSlotsList()) {
                            for (EngineRpcService.EnqueueBatchExternalInputPB input
                                    : slot.getRequestsList()) {
                                long requestId = input.getInput().getRequestId();
                                accepted.add(requestId);
                                response.addSuccesses(
                                        EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                                .setRequestId(requestId));
                            }
                        }
                        try {
                            // Acceptance is part of this synthetic RPC's success
                            // contract. Publish dispatch only after every endpoint
                            // status reducer completed, matching the phase gate used
                            // by NON_BATCH delivery.
                            reportDecodeAccepted(accepted);
                            dispatchCount.addAndGet(accepted.size());
                            return CompletableFuture.completedFuture(response.build());
                        } catch (Throwable acceptanceFailure) {
                            firstFailure.compareAndSet(null,
                                    "batch_id=" + request.getBatchId()
                                            + " decode_acceptance=" + acceptanceFailure);
                            // Preserve the real dispatcher's asynchronous RPC
                            // failure path instead of throwing only from Mockito.
                            return CompletableFuture.failedFuture(acceptanceFailure);
                        }
                    });

            dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
            priorityScheduler = new PriorityAdmissionScheduler(
                    configService, router, endpointRegistry, new PlanCommitter(),
                    priorityReporter, reporter, new UnsupportedEngineCancelChannel());
            scheduler = new PriorityScheduler(
                    configService, router, endpointRegistry, dispatcher, reporter,
                    priorityScheduler, null, new UnsupportedEngineCancelChannel()) {
                @Override
                public void onDecisionGroupReady(
                        List<BatchItem> items, DecisionGroupMetadata metadata) {
                    maxDecisionGroupSize.accumulateAndGet(items.size(), Math::max);
                    super.onDecisionGroupReady(items, metadata);
                }
            };
            schedulerRef.set(scheduler);
            routeService = new RouteService(
                    configService, router, scheduler, traceReporter);

            registerTopology();
        }

        private static void configureCapacity(FlexlbConfig config, Mode mode) {
            int requestedCapacity = Math.max(20_000,
                    WARMUP_REQUESTS + REQUEST_COUNT + 1_024);
            config.getRouter().getRoles().getPrefill().getAvailability()
                    .setMaxPendingRequests(1_000_000L);
            config.getRouter().getRoles().getDecode().getAvailability()
                    .setMaxEngineRequests(100_000L);
            if (!mode.queue) {
                return;
            }
            config.queueScheduler().getCapacity()
                    .setMaxOutstandingRequestsGlobal(requestedCapacity);
            config.queueScheduler().getCapacity()
                    .setMaxWaitingRequestsPerPrefillWorker(requestedCapacity);
            config.queueScheduler().getLifecycle()
                    .setMaxDeliveredNotAcceptedRequestsGlobal(requestedCapacity);
            if (mode.fixedWindow) {
                SchedulingTestConfig.useFixedWindowDecision(config)
                        .setMaxCollectionWaitMs(10L);
                SchedulingTestConfig.useFixedWindowDecision(config)
                        .setMaxRequests(16);
            }
        }

        private void registerTopology() {
            for (int i = 0; i < PREFILL_WORKERS; i++) {
                WorkerStatus status = workerStatus(
                        "127.1." + (i / 250) + "." + (1 + i % 250),
                        PREFILL_HTTP_BASE + i, PREFILL_GRPC_BASE + i, false);
                WorkerTarget target = new WorkerTarget(
                        status.getIp(), status.getPort(), status.getGrpcPort(),
                        status.getIpPort(), status);
                prefills.add(target);
                endpointRegistry.ensureEndpoint(RoleType.PREFILL, target.ipPort, status);
            }
            for (int i = 0; i < DECODE_WORKERS; i++) {
                WorkerStatus status = workerStatus(
                        "127.2." + (i / 250) + "." + (1 + i % 250),
                        DECODE_HTTP_BASE + i, DECODE_GRPC_BASE + i, true);
                WorkerTarget target = new WorkerTarget(
                        status.getIp(), status.getPort(), status.getGrpcPort(),
                        status.getIpPort(), status);
                decodes.add(target);
                activeDecodeTasks.add(new HashMap<>());
                decodeStatusLocks.add(new Object());
                endpointRegistry.ensureEndpoint(RoleType.DECODE, target.ipPort, status);
                endpointRegistry.getDecode(target.ipPort)
                        .onWorkerStatusUpdate(status, new WorkerStatusResponse());
            }
        }

        CompletableFuture<Response> submit(long requestId, int priority, long seqLen) {
            CompletableFuture<Response> future = routeService.route(
                    context(requestId, priority, seqLen));
            future.whenComplete((response, error) -> {
                try {
                    if (error != null) {
                        firstFailure.compareAndSet(null,
                                "request_id=" + requestId + " exceptional=" + error);
                        return;
                    }
                    if (response == null || !response.isSuccess()) {
                        firstFailure.compareAndSet(null,
                                "request_id=" + requestId + " response=" + response);
                        return;
                    }
                    if (!mode.batch) {
                        if (mode.queue) {
                            reportDecodeAccepted(List.of(requestId));
                        } else {
                            decodeAssignment.remove(requestId);
                        }
                        dispatchCount.incrementAndGet();
                    }
                } catch (Throwable callbackFailure) {
                    // whenComplete returns a dependent stage. Do not leave a
                    // callback failure observable only on that otherwise-unused
                    // stage: publish it into the phase gate so the test fails
                    // immediately while the source future is still retained and
                    // validated independently below.
                    firstFailure.compareAndSet(null,
                            "request_id=" + requestId
                                    + " completion_callback=" + callbackFailure);
                } finally {
                    // Publish completion only after the synthetic Decode status has
                    // finished reducing, so the phase gate cannot race harness close.
                    completionCount.incrementAndGet();
                }
            });
            return future;
        }

        void awaitSuccessful(List<CompletableFuture<Response>> futures,
                             long expectedTotal) throws Exception {
            long deadline = System.nanoTime()
                    + TimeUnit.SECONDS.toNanos(PHASE_TIMEOUT_SECONDS);
            while ((dispatchCount.get() < expectedTotal
                    || completionCount.get() < expectedTotal)
                    && System.nanoTime() < deadline) {
                String failure = firstFailure.get();
                if (failure != null) {
                    fail(mode.label + " failed before delivery: " + failure);
                }
                Thread.sleep(2L);
            }

            assertEquals(expectedTotal, dispatchCount.get(),
                    () -> timeoutMessage("dispatch", expectedTotal));
            assertEquals(expectedTotal, completionCount.get(),
                    () -> timeoutMessage("completion", expectedTotal));
            String failure = firstFailure.get();
            assertTrue(failure == null,
                    () -> mode.label + " returned a failed response: " + failure);
            for (CompletableFuture<Response> future : futures) {
                assertTrue(future.isDone(),
                        () -> mode.label + " retained a non-terminal future");
                Response response = future.getNow(null);
                assertNotNull(response, mode.label + " completed without a response");
                assertTrue(response.isSuccess(),
                        () -> mode.label + " response failed: code=" + response.getCode()
                                + " message=" + response.getErrorMessage());
                assertEquals(mode.batch, response.isEnqueuedByMaster(),
                        () -> mode.label + " returned the wrong delivery protocol");
            }
            // A sparse, user-configured topology can legitimately time out
            // every FIXED_WINDOW as a singleton. Require a multi-request group
            // only when the unrated warmup provides at least two requests per
            // Prefill worker; deterministic algorithm tests cover grouping for
            // every topology independently of this load generator.
            if (mode.queue && mode.fixedWindow
                    && WARMUP_REQUESTS >= 2 * PREFILL_WORKERS) {
                assertTrue(maxDecisionGroupSize.get() > 1,
                        () -> mode.label + " never formed a multi-request decision group");
            } else if (mode.queue && !mode.fixedWindow) {
                assertEquals(1, maxDecisionGroupSize.get(),
                        () -> mode.label + " did not preserve SINGLE decision groups");
            }
        }

        private String timeoutMessage(String counter, long expected) {
            return mode.label + " " + counter + " count mismatch: expected=" + expected
                    + " dispatched=" + dispatchCount.get()
                    + " completed=" + completionCount.get()
                    + " queued=" + (mode.queue ? scheduler.getQueuedRequestCount() : 0)
                    + " first_failure=" + firstFailure.get();
        }

        private void reportDecodeAccepted(List<Long> requestIds) {
            Map<Integer, List<Long>> byDecode = new HashMap<>();
            for (long requestId : requestIds) {
                Integer decodeIndex = decodeAssignment.remove(requestId);
                if (decodeIndex == null) {
                    firstFailure.compareAndSet(null,
                            "missing decode assignment for request_id=" + requestId);
                    continue;
                }
                byDecode.computeIfAbsent(decodeIndex, ignored -> new ArrayList<>())
                        .add(requestId);
            }

            for (Map.Entry<Integer, List<Long>> entry : byDecode.entrySet()) {
                int decodeIndex = entry.getKey();
                WorkerTarget decode = decodes.get(decodeIndex);
                synchronized (decodeStatusLocks.get(decodeIndex)) {
                    Map<Long, TaskInfo> active = activeDecodeTasks.get(decodeIndex);
                    for (long requestId : entry.getValue()) {
                        TaskInfo task = new TaskInfo();
                        task.setRequestId(requestId);
                        task.setPhase(TaskPhase.KV_ALLOCATED);
                        active.put(requestId, task);
                    }
                    Map<String, TaskInfo> snapshot = new HashMap<>(active.size());
                    for (Map.Entry<Long, TaskInfo> task : active.entrySet()) {
                        snapshot.put(String.valueOf(task.getKey()), task.getValue());
                    }
                    WorkerStatusResponse response = new WorkerStatusResponse();
                    response.setRole(RoleType.DECODE);
                    response.setRunningTaskInfo(snapshot);
                    endpointRegistry.getDecode(decode.ipPort)
                            .onWorkerStatusUpdate(decode.status, response);
                    scheduler.onWorkerStatusUpdate(response);
                }
            }
        }

        private BalanceContext context(long requestId, int priority, long seqLen) {
            Request request = new Request();
            request.setRequestId(requestId);
            request.setSeqLen(seqLen);
            request.setMaxNewTokens(8);
            request.setNumBeams(1);
            request.setModel("perf-model");
            request.setPriority(priority);

            BalanceContext context = new BalanceContext();
            context.setRequest(request);
            context.setGenerateInputPbBytes(generateInputBytes(requestId, (int) seqLen));
            context.setSchedulingMetadata(SchedulingMetadata.explicit(
                    priority, context.getStartTime() + 30_000));
            return context;
        }

        @Override
        public void close() {
            try {
                priorityScheduler.shutdown();
                scheduler.shutdown();
            } finally {
                // PriorityScheduler owns the registry in production; the explicit
                // close also protects the harness if scheduler shutdown aborts.
                endpointRegistry.close();
                dispatcher.shutdown();
            }
        }

        private static WorkerStatus workerStatus(String ip, int httpPort,
                                                 int grpcPort, boolean decode) {
            WorkerStatus status = new WorkerStatus();
            status.setIp(ip);
            status.setPort(httpPort);
            status.setGrpcPort(grpcPort);
            status.setGroup("perf-group");
            status.setAlive(true);
            if (decode) {
                status.setAvailableKvCacheTokens(new AtomicLong(1_000_000_000L));
                status.setTotalKvCacheTokens(new AtomicLong(2_000_000_000L));
            }
            return status;
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

        private static Response successRoute(long requestId,
                                             WorkerTarget prefill,
                                             WorkerTarget decode) {
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(List.of(
                    server(RoleType.PREFILL, prefill, requestId),
                    server(RoleType.DECODE, decode, requestId)));
            return response;
        }

        private static ServerStatus server(RoleType role, WorkerTarget worker,
                                           long requestId) {
            ServerStatus status = new ServerStatus();
            status.setSuccess(true);
            status.setRole(role);
            status.setServerIp(worker.ip);
            status.setHttpPort(worker.httpPort);
            status.setGrpcPort(worker.grpcPort);
            status.setDpRank(0);
            status.setGroup("perf-group");
            status.setRequestId(requestId);
            return status;
        }
    }
}
