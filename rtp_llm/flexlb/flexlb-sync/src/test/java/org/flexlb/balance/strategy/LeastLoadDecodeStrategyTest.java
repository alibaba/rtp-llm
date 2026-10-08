package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class LeastLoadDecodeStrategyTest {
    private ConfigService configService;
    private FlexlbConfig config;
    private EngineWorkerStatus engineWorkerStatus;
    private Map<String, WorkerStatus> workers;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        config = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);
        engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        workers = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();
        workers.clear();
    }

    @AfterEach
    void tearDown() {
        workers.clear();
    }

    @Test
    void empty_workers_should_return_failure() {
        ServerStatus status = strategy().select(context(1000L, 10), RoleType.DECODE, null);
        assertFalse(status.isSuccess());
        assertNotNull(status.getMessage());
    }

    @Test
    void concurrency_should_take_priority_when_cache_usage_prefers_another_worker() {
        WorkerStatus lessBusy = worker("127.0.0.1", 800, 200);
        lessBusy.setWaitingTaskList(tasks(1L));
        WorkerStatus lessCache = worker("127.0.0.2", 10, 990);
        lessCache.setRunningTaskList(tasks(2L, 3L));

        assertSelected(lessBusy, strategy().select(context(1000L, 10), RoleType.DECODE, null));
    }

    @Test
    void equal_concurrency_should_compare_cache_usage_ratio_instead_of_absolute_tokens() {
        WorkerStatus smallerCapacity = worker("127.0.0.1", 100, 100);
        WorkerStatus lowerUsageRatio = worker("127.0.0.2", 200, 1800);
        smallerCapacity.setRunningTaskList(tasks(1L));
        lowerUsageRatio.setWaitingTaskList(tasks(2L));

        assertSelected(lowerUsageRatio, strategy().select(context(1000L, 10), RoleType.DECODE, null));
    }

    @Test
    void concurrency_should_deduplicate_ids_across_waiting_running_and_local_tasks() {
        config.setDecodeConcurrencyLimit(3);
        WorkerStatus duplicated = worker("127.0.0.1", 800, 200);
        duplicated.setWaitingTaskList(tasks(1L));
        duplicated.setRunningTaskList(tasks(1L));
        duplicated.getLocalTaskMap().put(1L, task(1L));
        WorkerStatus distinct = worker("127.0.0.2", 10, 990);
        distinct.setRunningTaskList(tasks(2L, 3L));

        assertSelected(duplicated, strategy().select(context(1000L, 10), RoleType.DECODE, null));
    }

    @Test
    void newly_selected_in_transit_task_should_count_before_engine_reports_it() {
        WorkerStatus lowerCache = worker("127.0.0.1", 100, 9900);
        WorkerStatus other = worker("127.0.0.2", 200, 9800);
        LeastLoadDecodeStrategy strategy = strategy();

        assertSelected(lowerCache, strategy.select(context(1000L, 10), RoleType.DECODE, null));
        assertEquals(TaskStateEnum.IN_TRANSIT, lowerCache.getLocalTaskMap().get(1000L).getTaskState());
        assertSelected(other, strategy.select(context(1001L, 10), RoleType.DECODE, null));
        assertEquals(1, lowerCache.getLocalTaskMap().size());
        assertEquals(1, other.getLocalTaskMap().size());
    }

    @Test
    void group_selection_should_exclude_lower_load_workers_from_other_groups() {
        WorkerStatus requested = worker("127.0.0.1", 100, 900);
        requested.setGroup("group-a");
        requested.setRunningTaskList(tasks(1L));
        WorkerStatus outside = worker("127.0.0.2", 0, 1000);
        outside.setGroup("group-b");
        LeastLoadDecodeStrategy strategy = strategy();

        ServerStatus selected = strategy.select(context(1000L, 10), RoleType.DECODE, "group-a");
        assertSelected(requested, selected);
        assertEquals("group-a", selected.getGroup());
        assertTrue(outside.getLocalTaskMap().isEmpty());
        assertFalse(strategy.select(context(1001L, 10), RoleType.DECODE, "missing").isSuccess());
    }

    @Test
    void resource_filters_should_exclude_dead_kv_full_and_concurrency_full_workers() {
        config.setDecodeConcurrencyLimit(2);
        WorkerStatus dead = worker("127.0.0.1", 0, 1000);
        dead.setAlive(false);
        WorkerStatus kvFull = worker("127.0.0.2", 950, 50);
        WorkerStatus concurrencyFull = worker("127.0.0.3", 0, 1000);
        concurrencyFull.setWaitingTaskList(tasks(1L));
        concurrencyFull.getLocalTaskMap().put(2L, task(2L));
        WorkerStatus available = worker("127.0.0.4", 100, 900);
        available.setRunningTaskList(tasks(3L));
        LeastLoadDecodeStrategy strategy = strategy();

        assertSelected(available, strategy.select(context(1000L, 10), RoleType.DECODE, null));
        assertTrue(dead.getLocalTaskMap().isEmpty());
        assertTrue(kvFull.getLocalTaskMap().isEmpty());
        assertEquals(1, concurrencyFull.getLocalTaskMap().size());
        assertFalse(strategy.select(context(1001L, 10), RoleType.DECODE, null).isSuccess());
        assertEquals(1, available.getLocalTaskMap().size());
    }

    @Test
    void identical_scores_should_round_robin_across_requests_even_after_rollback() {
        worker("127.0.0.1", 100, 9900);
        worker("127.0.0.2", 100, 9900);
        worker("127.0.0.3", 100, 9900);
        LeastLoadDecodeStrategy strategy = strategy();
        Map<String, Integer> counts = new HashMap<>();

        for (long requestId = 1000; requestId < 1060; requestId++) {
            ServerStatus status = strategy.select(context(requestId, 10), RoleType.DECODE, null);
            assertTrue(status.isSuccess());
            counts.merge(status.getServerIp(), 1, Integer::sum);
            strategy.rollBack(status.getServerIp() + ":8080", requestId);
        }

        assertEquals(Map.of("127.0.0.1", 20, "127.0.0.2", 20, "127.0.0.3", 20), counts);
        workers.values().forEach(worker -> assertTrue(worker.getLocalTaskMap().isEmpty()));
    }

    @Test
    void concurrent_selection_with_static_reports_should_balance_reservations_and_respect_limit() throws Exception {
        config.setDecodeConcurrencyLimit(100);
        worker("127.0.0.1", 0, 100000);
        worker("127.0.0.2", 0, 100000);
        worker("127.0.0.3", 0, 100000);
        LeastLoadDecodeStrategy strategy = strategy();

        List<ServerStatus> initial = selectConcurrently(strategy, 1000L, 180);
        assertEquals(180, initial.stream().filter(ServerStatus::isSuccess).count());
        workers.values().forEach(worker -> assertEquals(60, worker.getLocalTaskMap().size()));

        List<ServerStatus> overflow = selectConcurrently(strategy, 2000L, 180);
        assertEquals(120, overflow.stream().filter(ServerStatus::isSuccess).count());
        assertEquals(60, overflow.stream().filter(status -> !status.isSuccess()).count());
        workers.values().forEach(worker -> assertEquals(100, worker.getLocalTaskMap().size()));
        assertEquals(300, workers.values().stream()
                .flatMap(worker -> worker.getLocalTaskMap().keySet().stream()).distinct().count());
    }

    @Test
    void rollback_should_restore_reservation_cache_and_admission_capacity() {
        config.setDecodeConcurrencyLimit(1);
        WorkerStatus worker = worker("127.0.0.1", 100, 900);
        LeastLoadDecodeStrategy strategy = strategy();

        ServerStatus selected = strategy.select(context(1000L, 25), RoleType.DECODE, null);
        assertSelected(worker, selected);
        assertEquals(1000L, selected.getRequestId());
        assertEquals(25, worker.getLocalTaskMap().get(1000L).getInputLength());
        assertEquals(125, worker.getUsedKvCacheTokens().get());
        assertEquals(875, worker.getAvailableKvCacheTokens().get());
        assertFalse(strategy.select(context(1001L, 25), RoleType.DECODE, null).isSuccess());

        strategy.rollBack(worker.getIpPort(), 1000L);
        strategy.rollBack(worker.getIpPort(), 1000L);
        assertTrue(worker.getLocalTaskMap().isEmpty());
        assertEquals(100, worker.getUsedKvCacheTokens().get());
        assertEquals(900, worker.getAvailableKvCacheTokens().get());
        assertSelected(worker, strategy.select(context(1002L, 25), RoleType.DECODE, null));
    }

    @Test
    void routing_cost_with_many_endpoints_and_reported_requests() throws Exception {
        for (int[] shape : List.of(new int[]{8, 64}, new int[]{8, 256}, new int[]{64, 256})) {
            workers.clear();
            config.setDecodeConcurrencyLimit(shape[1] + 100);
            for (int dp = 0; dp < shape[0]; dp++) {
                WorkerStatus worker = worker("10.0.0." + (dp + 1), 1000, 999000);
                Map<String, TaskInfo> running = new HashMap<>();
                for (int row = 0; row < shape[1]; row++) {
                    long id = dp * 10000L + row;
                    running.put(String.valueOf(id), task(id));
                    TaskInfo local = task(id);
                    local.updateTaskState(TaskStateEnum.RUNNING);
                    worker.getLocalTaskMap().put(id, local);
                }
                worker.setRunningTaskList(running);
            }
            ResourceMeasureFactory factory = mock(ResourceMeasureFactory.class);
            DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
            when(factory.getMeasure(any())).thenReturn(measure);
            for (LoadBalancer strategy : List.of(
                    new WeightedCacheLoadBalancer(configService, engineWorkerStatus, factory),
                    new LeastLoadDecodeStrategy(configService, engineWorkerStatus, factory))) {
                for (long id = 1000000; id < 1000200; id++) {
                    ServerStatus result = strategy.select(context(id, 0), RoleType.DECODE, null);
                    assertTrue(result.isSuccess());
                    strategy.rollBack(result.getServerIp() + ":8080", id);
                }
                int requests = 1200;
                long[] latencies = new long[requests];
                ExecutorService executor = Executors.newFixedThreadPool(12);
                long start = System.nanoTime();
                try {
                    List<Future<?>> futures = new ArrayList<>();
                    for (int index = 0; index < requests; index++) {
                        final int sample = index;
                        futures.add(executor.submit(() -> {
                            long begin = System.nanoTime();
                            ServerStatus result = strategy.select(context(2000000L + sample, 0), RoleType.DECODE, null);
                            latencies[sample] = System.nanoTime() - begin;
                            assertTrue(result.isSuccess());
                            strategy.rollBack(result.getServerIp() + ":8080", result.getRequestId());
                        }));
                    }
                    for (Future<?> future : futures) {
                        future.get(30, TimeUnit.SECONDS);
                    }
                } finally {
                    executor.shutdownNow();
                    assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
                }
                double elapsed = (System.nanoTime() - start) / 1e9;
                Arrays.sort(latencies);
                System.out.printf("LOCAL_ROUTING_COST strategy=%s dp=%d reported_per_dp=%d threads=12 "
                                + "requests=%d qps=%.1f p50_us=%.1f p99_us=%.1f%n",
                        strategy.getClass().getSimpleName(), shape[0], shape[1], requests,
                        requests / elapsed, latencies[requests / 2] / 1000.0,
                        latencies[(int) (requests * .99)] / 1000.0);
                workers.values().forEach(worker -> {
                    assertEquals(shape[1], worker.getLocalTaskMap().size());
                    assertEquals(shape[1], worker.getDecodeConcurrency());
                });
            }
        }
    }

    @Test
    void concurrent_reports_completions_and_rollbacks_should_drain_local_reservations() throws Exception {
        config.setDecodeConcurrencyLimit(1000);
        WorkerStatus worker = worker("127.0.0.1", 100, 999900);
        LeastLoadDecodeStrategy strategy = strategy();
        AtomicBoolean done = new AtomicBoolean();
        CountDownLatch reportsStarted = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(3);
        try {
            Future<?> reports = executor.submit(() -> {
                int round = 0;
                while (!done.get()) {
                    Map<String, TaskInfo> reported = new HashMap<>();
                    worker.getLocalTaskMap().forEach((id, task) -> reported.put(String.valueOf(id), task(id)));
                    Map<String, TaskInfo> waiting = round % 3 == 0 ? reported : Map.of();
                    Map<String, TaskInfo> running = round % 3 == 1 ? reported : Map.of();
                    Map<String, TaskInfo> finished = round % 3 == 2 ? reported : Map.of();
                    worker.setWaitingTaskList(waiting);
                    worker.setRunningTaskList(running);
                    worker.updateTaskStates(waiting, running, finished);
                    worker.updateKvCacheTokens(100, 999900);
                    reportsStarted.countDown();
                    round++;
                    Thread.yield();
                }
            });
            List<Future<?>> schedulers = new ArrayList<>();
            for (int scheduler = 0; scheduler < 2; scheduler++) {
                final long firstId = 100000L + scheduler * 1000L;
                schedulers.add(executor.submit(() -> {
                    assertTrue(reportsStarted.await(10, TimeUnit.SECONDS));
                    for (long id = firstId; id < firstId + 500; id++) {
                        ServerStatus selected = strategy.select(context(id, 10), RoleType.DECODE, null);
                        assertTrue(selected.isSuccess());
                        strategy.rollBack(worker.getIpPort(), id);
                        strategy.rollBack(worker.getIpPort(), id); // repeated cancellation/rollback
                    }
                    return null;
                }));
            }
            try {
                for (Future<?> scheduler : schedulers) {
                    scheduler.get(20, TimeUnit.SECONDS);
                }
            } finally {
                done.set(true);
            }
            reports.get(20, TimeUnit.SECONDS);
            assertTrue(worker.getLocalTaskMap().isEmpty());
            // A final authoritative report settles stale snapshots and KV corrections.
            // This tests drain/recovery, not a strict live-global concurrency guarantee.
            worker.setWaitingTaskList(Map.of());
            worker.setRunningTaskList(Map.of());
            worker.updateKvCacheTokens(100, 999900);
            assertEquals(0, worker.getDecodeConcurrency());
            assertEquals(100, worker.getUsedKvCacheTokens().get());
            assertEquals(999900, worker.getAvailableKvCacheTokens().get());
            assertSelected(worker, strategy.select(context(999999L, 10), RoleType.DECODE, null));
            strategy.rollBack(worker.getIpPort(), 999999L);
            assertEquals(0, worker.getDecodeConcurrency());
        } finally {
            done.set(true);
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
        }
    }

    private LeastLoadDecodeStrategy strategy() {
        ResourceMeasureFactory factory = mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        when(factory.getMeasure(any())).thenReturn(measure);
        return new LeastLoadDecodeStrategy(configService, engineWorkerStatus, factory);
    }

    private WorkerStatus worker(String ip, long used, long available) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(8080);
        worker.setAlive(true);
        worker.setRole(RoleType.DECODE.getCode());
        worker.getUsedKvCacheTokens().set(used);
        worker.getAvailableKvCacheTokens().set(available);
        workers.put(worker.getIpPort(), worker);
        return worker;
    }

    private BalanceContext context(long requestId, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        return context;
    }

    private Map<String, TaskInfo> tasks(Long... requestIds) {
        return Arrays.stream(requestIds).collect(Collectors.toMap(String::valueOf, this::task));
    }

    private TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        return task;
    }

    private void assertSelected(WorkerStatus worker, ServerStatus status) {
        assertTrue(status.isSuccess());
        assertEquals(worker.getIp(), status.getServerIp());
        assertEquals(worker.getPort(), status.getHttpPort());
        assertEquals(RoleType.DECODE, status.getRole());
    }

    private List<ServerStatus> selectConcurrently(LeastLoadDecodeStrategy strategy, long firstId,
                                                int requestCount) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(12);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<ServerStatus>> futures = new ArrayList<>();
        try {
            for (int index = 0; index < requestCount; index++) {
                long requestId = firstId + index;
                futures.add(executor.submit(() -> {
                    assertTrue(start.await(10, TimeUnit.SECONDS));
                    return strategy.select(context(requestId, 10), RoleType.DECODE, null);
                }));
            }
            start.countDown();
            List<ServerStatus> results = new ArrayList<>();
            for (Future<ServerStatus> future : futures) {
                results.add(future.get(20, TimeUnit.SECONDS));
            }
            return results;
        } finally {
            start.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
        }
    }
}
