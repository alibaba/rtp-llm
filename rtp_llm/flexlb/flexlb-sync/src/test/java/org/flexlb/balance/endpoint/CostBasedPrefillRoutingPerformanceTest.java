package org.flexlb.balance.endpoint;

import ch.qos.logback.classic.Level;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.mockito.Mockito;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.withSettings;

@Tag("performance-regression")
class CostBasedPrefillRoutingPerformanceTest {

    private static final int ENGINE_COUNT = 750;
    private static final int MEASUREMENT_ROUNDS = 3;
    private static final int WARMUP_OPERATIONS_PER_THREAD = 100;
    private static final int OPERATIONS_PER_THREAD =
            Integer.getInteger("flexlb.perf.routing.operations-per-thread", 500);

    private EndpointRegistry endpointRegistry;
    private ch.qos.logback.classic.Logger syncLogger;
    private Level previousSyncLogLevel;

    @BeforeEach
    void suppressFixtureLogs() {
        syncLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("syncLogger");
        previousSyncLogLevel = syncLogger.getLevel();
        syncLogger.setLevel(Level.WARN);
    }

    @AfterEach
    void tearDown() {
        if (endpointRegistry != null) {
            endpointRegistry.close();
        }
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        syncLogger.setLevel(previousSyncLogLevel);
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void routesAcross750EnginesWithoutThroughputRegression() throws Exception {
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int threadCount = Math.min(16, Math.max(4, availableProcessors));
        long defaultMinimumQps = Math.min(20_000L, Math.max(2_000L, availableProcessors * 1_000L));
        long minimumQps = Long.getLong("flexlb.perf.min-routing-qps", defaultMinimumQps);
        double maximumP99Ms = Double.parseDouble(
                System.getProperty("flexlb.perf.max-routing-p99-ms", "20"));

        CostBasedPrefillStrategy strategy = createStrategy();
        assertEquals(ENGINE_COUNT, endpointRegistry.getEndpointCount(RoleType.PREFILL));

        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        try {
            List<RoundResult> rounds = new ArrayList<>(MEASUREMENT_ROUNDS);
            for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
                rounds.add(runRound(strategy, executor, threadCount));
            }
            int totalFailures = rounds.stream().mapToInt(RoundResult::failures).sum();
            rounds.sort((left, right) -> Double.compare(left.qps(), right.qps()));
            RoundResult median = rounds.get(MEASUREMENT_ROUNDS / 2);

            System.out.printf(
                    "FlexLB 750-engine routing performance: threads=%d operations=%d qps=%.1f p50=%.3fms p99=%.3fms%n",
                    threadCount, median.operations(), median.qps(), median.p50Ms(), median.p99Ms());

            assertEquals(0, totalFailures, "all routing decisions must succeed");
            assertTrue(median.qps() >= minimumQps,
                    () -> String.format("routing throughput %.1f QPS is below regression floor %d QPS",
                            median.qps(), minimumQps));
            assertTrue(median.p99Ms() <= maximumP99Ms,
                    () -> String.format("routing P99 %.3f ms exceeds regression ceiling %.3f ms",
                            median.p99Ms(), maximumP99Ms));
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    private CostBasedPrefillStrategy createStrategy() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchEnabled(true);
        config.setCostSloMs(50_000L);
        config.setPrefillQueueSizeThreshold(1_000_000L);

        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config);
        BatchSchedulerReporter reporter = Mockito.mock(BatchSchedulerReporter.class, withSettings().stubOnly());
        FlexlbBatchScheduler scheduler = Mockito.mock(
                FlexlbBatchScheduler.class, withSettings().stubOnly());
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);

        for (int index = 0; index < ENGINE_COUNT; index++) {
            int port = 61_000 + index;
            WorkerStatus status = workerStatus(port);
            PrefillEndpoint endpoint = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                    RoleType.PREFILL, status.getIpPort(), status);
            endpoint.getBatcher().shutdown();
        }

        EngineWorkerStatus engineWorkerStatus =
                new EngineWorkerStatus(endpointRegistry);
        ResourceMeasureFactory resourceMeasureFactory =
                new ResourceMeasureFactory(List.of(new PrefillResourceMeasure(configService)));
        EngineHealthReporter healthReporter =
                Mockito.mock(EngineHealthReporter.class, withSettings().stubOnly());

        return new CostBasedPrefillStrategy(
                engineWorkerStatus, new EmptyCacheAwareService(), resourceMeasureFactory, healthReporter);
    }

    private RoundResult runRound(CostBasedPrefillStrategy strategy,
                                 ExecutorService executor,
                                 int threadCount) throws Exception {
        CountDownLatch ready = new CountDownLatch(threadCount);
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger failures = new AtomicInteger();
        List<Future<long[]>> futures = new ArrayList<>(threadCount);

        for (int threadIndex = 0; threadIndex < threadCount; threadIndex++) {
            final long requestId = threadIndex + 1L;
            futures.add(executor.submit(() -> {
                BalanceContext context = context(requestId);
                for (int operation = 0; operation < WARMUP_OPERATIONS_PER_THREAD; operation++) {
                    strategy.select(context, RoleType.PREFILL, null);
                }
                ready.countDown();
                start.await();

                long[] latencies = new long[OPERATIONS_PER_THREAD];
                for (int operation = 0; operation < OPERATIONS_PER_THREAD; operation++) {
                    long operationStart = System.nanoTime();
                    ServerStatus result = strategy.select(context, RoleType.PREFILL, null);
                    latencies[operation] = System.nanoTime() - operationStart;
                    if (!result.isSuccess()) {
                        failures.incrementAndGet();
                    }
                }
                return latencies;
            }));
        }

        assertTrue(ready.await(10, TimeUnit.SECONDS), "routing workers did not finish warmup");
        long startNanos = System.nanoTime();
        start.countDown();

        long[] allLatencies = new long[threadCount * OPERATIONS_PER_THREAD];
        int offset = 0;
        for (Future<long[]> future : futures) {
            long[] latencies = future.get(20, TimeUnit.SECONDS);
            System.arraycopy(latencies, 0, allLatencies, offset, latencies.length);
            offset += latencies.length;
        }
        long elapsedNanos = System.nanoTime() - startNanos;
        Arrays.sort(allLatencies);

        int operations = allLatencies.length;
        double qps = operations * 1_000_000_000.0 / elapsedNanos;
        double p50Ms = percentileNanos(allLatencies, 0.50) / 1_000_000.0;
        double p99Ms = percentileNanos(allLatencies, 0.99) / 1_000_000.0;
        return new RoundResult(operations, failures.get(), qps, p50Ms, p99Ms);
    }

    private static long percentileNanos(long[] sortedValues, double percentile) {
        int index = Math.max(0, (int) Math.ceil(sortedValues.length * percentile) - 1);
        return sortedValues[index];
    }

    private static WorkerStatus workerStatus(int port) {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        status.setRole(RoleType.PREFILL);
        status.setGroup("performance-regression");
        status.setAlive(true);
        status.setRunningTaskList(new HashMap<>());
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(256);
        cacheStatus.setAvailableKvCache(1_000_000L);
        status.setCacheStatus(cacheStatus);
        return status;
    }

    private static BalanceContext context(long requestId) {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchEnabled(true);
        config.setCostSloMs(50_000L);
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1_024L);
        request.setBlockCacheKeys(List.of(1L, 2L, 3L, 4L));
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        return context;
    }

    private record RoundResult(int operations, int failures, double qps,
                               double p50Ms, double p99Ms) {
    }

    private static final class EmptyCacheAwareService implements CacheAwareService {
        @Override
        public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
                                                        RoleType roleType,
                                                        String group) {
            return Map.of();
        }

        @Override
        public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
            return null;
        }
    }
}
