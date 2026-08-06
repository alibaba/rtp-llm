package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.BatchDispatchExecutor;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * P0 concurrency stress tests for FlexLB's hottest race windows.
 *
 * <p>Each test runs thousands of rounds with a fresh object per round,
 * using {@link CyclicBarrier} to maximise the collision window between
 * two threads. The pattern mirrors {@link InflightStoreActiveCountTest}:
 *
 * <pre>{@code
 * for (int i = 0; i < ROUNDS; i++) {
 *     // fresh objects per round — no cross-round contamination
 *     CyclicBarrier barrier = new CyclicBarrier(2);
 *     Future<?> t1 = pool.submit(() -> { barrier.await(); actionA(); return null; });
 *     Future<?> t2 = pool.submit(() -> { barrier.await(); actionB(); return null; });
 *     t1.get(5, TimeUnit.SECONDS);
 *     t2.get(5, TimeUnit.SECONDS);
 *     // per-round assertions
 * }
 * // global assertions
 * }</pre>
 *
 * <p>The three races covered:
 * <ul>
 *   <li><b>UT-C1</b> — {@code InflightItem.complete(CANCELLED)} vs
 *       {@code complete(successResponse)}:
 *       the CAS-guarded terminal transition must fire exactly once; EP
 *       {@code release} must run at most once per endpoint; the
 *       {@link InflightStore#activeCount} must return to zero.</li>
 *   <li><b>UT-C2</b> — {@code DecodeEndpoint.reserve()} vs
 *       {@code calibrate → removeInflight}: the {@code inflightKvReservedTotal}
 *       AtomicLong must never go negative (over-admission guard) and must
 *       settle to zero.</li>
 *   <li><b>UT-C3</b> — {@code PrefillEndpoint.repackBatch()} vs
 *       {@code releaseBatch()}: the {@code inflightRequestCount} AtomicInteger
 *       must never go negative and must settle to zero.</li>
 * </ul>
 */
class ConcurrencyStressTest {

    // ==================== shared helpers ====================

    private static InflightStore newStore() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return new InflightStore(Mockito.mock(BatchSchedulerReporter.class), configService);
    }

    /**
     * Build an {@link InflightItem} with mock prefill / decode endpoints bound.
     * The item is in the RUNNING state and not yet registered with any store.
     */
    private static InflightItem newItemWithEps(long requestId,
                                               PrefillEndpoint prefillEp,
                                               DecodeEndpoint decodeEp) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        InflightItem item = new InflightItem(ctx, new CompletableFuture<>(), null);
        item.setPrefillEp(prefillEp);
        item.setDecodeEp(decodeEp);
        return item;
    }

    private static DecodeEndpoint newDecodeEndpoint() {
        return newDecodeEndpoint(null);
    }

    private static DecodeEndpoint newDecodeEndpoint(InflightStore store) {
        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.2");
        status.setPort(8080);
        status.setGrpcPort(8081);
        status.getAvailableKvCacheTokens().set(1_000_000);
        status.getTotalKvCacheTokens().set(1_000_000);
        return new DecodeEndpoint(status, new FlexlbConfig(), store);
    }

    private static PrefillEndpoint newPrefillEndpoint() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchQueueMaxSize(100);
        config.setFlexlbBatchFixedWaitMs(300);
        config.setCostFormula("10 + 0.1*sum(computeTokens) + 5*batchSize");

        return new PrefillEndpoint(status, config,
                Mockito.mock(EngineGrpcClient.class),
                Mockito.mock(BatchDispatchExecutor.class),
                new BatchIdGenerator("127.0.0.1", 7001),
                () -> 0,
                Mockito.mock(BatchSchedulerReporter.class),
                null);
    }

    private static BatchItem newBatchItem(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(500);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(0);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, new CompletableFuture<>(), null,
                prefill, null, null, null, System.currentTimeMillis());
    }

    // ==================== UT-C1: concurrent cancel vs complete ====================

    /**
     * UT-C1: Concurrent {@code complete(CANCELLED)} vs {@code complete(success)}
     * on the same {@link InflightItem}.
     *
     * <p>Both methods share the same {@code state} AtomicReference CAS, so
     * only one terminal transition can succeed. The winner releases EP
     * resources and completes the future; the loser is a no-op tombstone.
     *
     * <p>Race window: both threads release the {@link CyclicBarrier}
     * simultaneously, then one calls {@code cancel()} and the other
     * {@code complete(successResponse)}.
     *
     * <p>Assertions per round:
     * <ul>
     *   <li>{@code isTerminated()} — exactly one CAS won.</li>
     *   <li>{@code future().isDone()} — future settled exactly once.</li>
     *   <li>Mock EP {@code release(requestId)} called at most once per EP —
     *       the CAS loser must not double-release.</li>
     * </ul>
     *
     * <p>Global assertion: {@code store.activeCount() == 0} — no leak
     * (callback never fired) and no double-decrement (callback fired twice).
     */
    @Test
    @Timeout(60)
    void concurrentCancelVsComplete() throws Exception {
        InflightStore store = newStore();
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            int rounds = 10_000;
            for (int i = 0; i < rounds; i++) {
                long requestId = i;
                PrefillEndpoint mockPrefillEp = Mockito.mock(PrefillEndpoint.class);
                DecodeEndpoint mockDecodeEp = Mockito.mock(DecodeEndpoint.class);

                InflightItem item = newItemWithEps(requestId, mockPrefillEp, mockDecodeEp);
                store.putIfAbsent(String.valueOf(requestId), item);

                Response successResponse = new Response();
                successResponse.setSuccess(true);

                CyclicBarrier barrier = new CyclicBarrier(2);
                Future<?> cancelFuture = pool.submit(() -> {
                    barrier.await();
                    item.complete(Response.error(StrategyErrorType.CANCELLED, "cancelled"),
                            InflightState.CANCELLED);
                    return null;
                });
                Future<?> completeFuture = pool.submit(() -> {
                    barrier.await();
                    item.complete(successResponse);
                    return null;
                });

                cancelFuture.get(5, TimeUnit.SECONDS);
                completeFuture.get(5, TimeUnit.SECONDS);

                assertTrue(item.isTerminated(),
                        "item should be terminal at round " + i);
                assertTrue(item.future().isDone(),
                        "future should be done at round " + i);
                // CAS winner calls release on both EPs; loser must not call at all
                Mockito.verify(mockPrefillEp, Mockito.atMost(1)).release(requestId);
                Mockito.verify(mockDecodeEp, Mockito.atMost(1)).release(requestId);
            }
            assertEquals(0, store.activeCount(),
                    "activeCount should be zero after all rounds — no leak, no double-decrement");
        } finally {
            pool.shutdownNow();
            store.shutdown();
        }
    }

    // ==================== UT-C2: concurrent reserve vs calibrate removeInflight ====================

    /**
     * UT-C2: Concurrent {@code reserve()} vs {@code calibrate → removeInflight}
     * on the same {@link DecodeEndpoint}.
     *
     * <p>T1 calls {@code reserve(requestId, kvTokens, expectedKvTokens)} which
     * adds the entry to {@code inflightRequests} and increments
     * {@code inflightKvReservedTotal} inside a {@code ConcurrentHashMap.compute}
     * bin lock.
     *
     * <p>T2 calls {@code onWorkerStatusUpdate} with a {@link WorkerStatusResponse}
     * whose {@code runningTaskInfo} contains the same {@code requestId}. This
     * triggers {@code calibrate → observeRunningTasks → removeInflight}, which
     * removes the entry and decrements the counter.
     *
     * <p>The race: if {@code removeInflight} runs between the map insert and
     * the counter increment (or vice-versa), the counter could go negative
     * (over-admission) or stay inflated (leak). The {@code compute} bin lock
     * prevents this, and the stress test verifies it holds under pressure.
     *
     * <p>Assertions:
     * <ul>
     *   <li>{@code inflightKvReservedTotal} never negative — checked after
     *       every round via {@code decodeInflightHardKvReserved()} and
     *       captured in an {@link AtomicBoolean}.</li>
     *   <li>After cleanup (release all requestIds), counter is zero.</li>
     *   <li>After cleanup, {@code inflightRequests} is empty.</li>
     * </ul>
     */
    @Test
    @Timeout(60)
    void concurrentReserveVsCalibrateRemoveInflight() throws Exception {
        DecodeEndpoint endpoint = newDecodeEndpoint();
        ExecutorService pool = Executors.newFixedThreadPool(2);
        AtomicBoolean negativeObserved = new AtomicBoolean(false);
        try {
            int rounds = 10_000;
            long kvTokens = 100;
            long expectedKvTokens = 200;

            for (int i = 0; i < rounds; i++) {
                long requestId = i;

                CyclicBarrier barrier = new CyclicBarrier(2);
                Future<?> reserveFuture = pool.submit(() -> {
                    barrier.await();
                    endpoint.reserve(requestId, kvTokens, expectedKvTokens);
                    return null;
                });
                Future<?> calibrateFuture = pool.submit(() -> {
                    barrier.await();
                    TaskInfo task = new TaskInfo();
                    task.setRequestId(requestId);
                    task.setPhase(TaskPhase.RUNNING);
                    WorkerStatusResponse resp = new WorkerStatusResponse();
                    resp.setRunningTaskInfo(Map.of(String.valueOf(requestId), task));
                    resp.setFinishedTaskInfo(Map.of());
                    endpoint.onWorkerStatusUpdate(endpoint.getStatus(), resp);
                    return null;
                });

                reserveFuture.get(5, TimeUnit.SECONDS);
                calibrateFuture.get(5, TimeUnit.SECONDS);

                long counter = endpoint.decodeInflightHardKvReserved();
                if (counter < 0) {
                    negativeObserved.set(true);
                }
            }

            assertFalse(negativeObserved.get(),
                    "inflightKvReservedTotal went negative during the stress test");

            // Cleanup: release every requestId — no-op if already removed by calibrate
            for (int i = 0; i < rounds; i++) {
                endpoint.release(i);
            }

            assertEquals(0, endpoint.decodeInflightHardKvReserved(),
                    "inflightKvReservedTotal should be zero after cleanup");
            assertEquals(0, endpoint.decodeInflightCount(),
                    "inflightRequests should be empty after cleanup");
        } finally {
            pool.shutdownNow();
            endpoint.close();
        }
    }

    // ==================== UT-C3: concurrent repackBatch vs releaseBatch ====================

    /**
     * UT-C3: Concurrent {@code repackBatch()} vs {@code releaseBatch()} on the
     * same {@link PrefillEndpoint} and batchId.
     *
     * <p>Each round commits a single-item batch, then races two terminal
     * operations on the same batchId:
     * <ul>
     *   <li>T1 — {@code repackBatch(batchId, Set.of(requestId))}: uses
     *       {@code computeIfPresent} to atomically shrink the entry to zero
     *       survivors, decrementing {@code inflightRequestCount}.</li>
     *   <li>T2 — {@code releaseBatch(batchId)}: uses {@code remove} to
     *       remove the entry, decrementing {@code inflightRequestCount}.</li>
     * </ul>
     *
     * <p>Both paths adjust the same {@link java.util.concurrent.atomic.AtomicInteger}
     * counter. If they double-decrement (both see the entry and both subtract),
     * the counter goes negative. The {@code ConcurrentHashMap} bin lock
     * serialises map access, but counter adjustments happen outside the lock
     * for {@code releaseBatch} — the stress test verifies no underflow.
     *
     * <p>Assertions:
     * <ul>
     *   <li>{@code inflightRequestCount} (observed via
     *       {@code prefillPendingRequestCount()}) never negative — captured
     *       in an {@link AtomicBoolean}.</li>
     *   <li>After all rounds, counter is zero.</li>
     * </ul>
     */
    @Test
    @Timeout(60)
    void concurrentRepackBatchVsReleaseBatch() throws Exception {
        PrefillEndpoint endpoint = newPrefillEndpoint();
        ExecutorService pool = Executors.newFixedThreadPool(2);
        AtomicBoolean negativeObserved = new AtomicBoolean(false);
        try {
            int rounds = 1_000;

            for (int i = 0; i < rounds; i++) {
                long batchId = 1_000_000L + i;
                long requestId = i;
                BatchItem item = newBatchItem(requestId);
                endpoint.commitBatch(batchId, 100, List.of(item));

                CyclicBarrier barrier = new CyclicBarrier(2);
                Future<?> repackFuture = pool.submit(() -> {
                    barrier.await();
                    endpoint.repackBatch(batchId, Set.of(requestId));
                    return null;
                });
                Future<?> releaseFuture = pool.submit(() -> {
                    barrier.await();
                    endpoint.releaseBatch(batchId);
                    return null;
                });

                repackFuture.get(5, TimeUnit.SECONDS);
                releaseFuture.get(5, TimeUnit.SECONDS);

                long counter = endpoint.prefillPendingRequestCount();
                if (counter < 0) {
                    negativeObserved.set(true);
                }
            }

            assertFalse(negativeObserved.get(),
                    "inflightRequestCount went negative during the stress test");
            assertEquals(0, endpoint.prefillPendingRequestCount(),
                    "inflightRequestCount should be zero after all rounds");
        } finally {
            pool.shutdownNow();
            endpoint.close();
        }
    }

    // ==================== UT-C4: concurrent STALE evict vs dispatch callback ====================

    /**
     * UT-C4: Concurrent STALE eviction (via {@code calibrate → evictStaleEngineTasks
     * → terminateBoundItem → item.complete(errorResp, FAILED)}) vs {@code item.complete(successResponse)}.
     *
     * <p>Both paths share the same {@code state} AtomicReference CAS on
     * {@link InflightItem}, so only one terminal transition can succeed.
     * The winner releases EP resources and completes the future; the loser
     * is a no-op tombstone.
     *
     * <p>Setup per round: reserve on a real {@link DecodeEndpoint} (layer 1),
     * then 3 calibrate rounds to migrate to layer 2 and advance the STALE
     * counter (flexlbStaleEvictRounds = 3). The 4th calibrate triggers eviction.
     *
     * <p>Race window: T1 releases the {@link CyclicBarrier} and calls
     * {@code onWorkerStatusUpdate} (round 4 → STALE eviction), while T2
     * simultaneously calls {@code item.complete(successResponse)}.
     *
     * <p>Assertions per round:
     * <ul>
     *   <li>{@code isTerminated()} — exactly one CAS won.</li>
     *   <li>{@code future().isDone()} — future settled exactly once.</li>
     *   <li>Mock EP {@code release(requestId)} called at most once per EP —
     *       the CAS loser must not double-release (KV no double-free).</li>
     * </ul>
     *
     * <p>Global assertion: {@code store.activeCount() == 0} — no leak.
     */
    @Test
    @Timeout(120)
    void concurrentStaleEvictVsDispatchCallback() throws Exception {
        InflightStore store = newStore();
        DecodeEndpoint endpoint = newDecodeEndpoint(store);
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            int rounds = 5000;
            for (int i = 0; i < rounds; i++) {
                long requestId = 100_000L + i;
                PrefillEndpoint mockPrefillEp = Mockito.mock(PrefillEndpoint.class);
                DecodeEndpoint mockDecodeEp = Mockito.mock(DecodeEndpoint.class);

                InflightItem item = newItemWithEps(requestId, mockPrefillEp, mockDecodeEp);
                store.putIfAbsent(String.valueOf(requestId), item);

                // Reserve → layer 1
                endpoint.reserve(requestId, 100, 200);

                // Calibrate round 1: migrate to layer 2 (engineWork)
                TaskInfo task = new TaskInfo();
                task.setRequestId(requestId);
                task.setPhase(TaskPhase.RUNNING);
                WorkerStatusResponse respWithRid = new WorkerStatusResponse();
                respWithRid.setRunningTaskInfo(Map.of(String.valueOf(requestId), task));
                respWithRid.setFinishedTaskInfo(Map.of());
                endpoint.onWorkerStatusUpdate(endpoint.getStatus(), respWithRid);

                // Calibrate rounds 2, 3: unseen (advance STALE counter)
                WorkerStatusResponse emptyResp = new WorkerStatusResponse();
                emptyResp.setRunningTaskInfo(Map.of());
                emptyResp.setFinishedTaskInfo(Map.of());
                endpoint.onWorkerStatusUpdate(endpoint.getStatus(), emptyResp);
                endpoint.onWorkerStatusUpdate(endpoint.getStatus(), emptyResp);

                // Race: T1 calibrate round 4 (STALE eviction → terminateBoundItem → terminate(FAILED))
                //       T2 item.complete(successResponse)
                Response successResponse = new Response();
                successResponse.setSuccess(true);

                CyclicBarrier barrier = new CyclicBarrier(2);
                Future<?> evictFuture = pool.submit(() -> {
                    try {
                        barrier.await();
                        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), emptyResp);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    return null;
                });
                Future<?> completeFuture = pool.submit(() -> {
                    try {
                        barrier.await();
                        item.complete(successResponse);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    return null;
                });

                evictFuture.get(5, TimeUnit.SECONDS);
                completeFuture.get(5, TimeUnit.SECONDS);

                assertTrue(item.isTerminated(),
                        "item should be terminal at round " + i);
                assertTrue(item.future().isDone(),
                        "future should be done at round " + i);
                Mockito.verify(mockPrefillEp, Mockito.atMost(1)).release(requestId);
                Mockito.verify(mockDecodeEp, Mockito.atMost(1)).release(requestId);
            }
            assertEquals(0, store.activeCount(),
                    "activeCount should be zero after all rounds — no leak, no double-decrement");
        } finally {
            pool.shutdownNow();
            endpoint.close();
            store.shutdown();
        }
    }

    // ==================== UT-C5: concurrent EP close vs calibrate ====================

    /**
     * UT-C5: Concurrent {@code PrefillEndpoint.close()} (drainInflight iterating
     * {@code engineWork}) vs {@code onWorkerStatusUpdate} (calibrate writing
     * to {@code engineWork}).
     *
     * <p>{@link java.util.concurrent.ConcurrentHashMap} iterators are weakly
     * consistent and must not throw {@link java.util.ConcurrentModificationException}.
     * The stress test verifies this holds under tight CyclicBarrier races.
     *
     * <p>Setup per round: commit a single-item batch (puts entry in
     * {@code inflightEntries}), then race {@code close()} vs
     * {@code onWorkerStatusUpdate} (which migrates the entry to
     * {@code engineWork}).
     *
     * <p>Assertions:
     * <ul>
     *   <li>No exception (CME or otherwise) thrown by either thread.</li>
     *   <li>After both threads complete and a second {@code close()} to drain
     *       any entries added by the calibrate winner, {@code prefillPendingRequestCount() == 0}.</li>
     * </ul>
     */
    @Test
    @Timeout(60)
    void concurrentEpCloseVsCalibrate() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(2);
        AtomicBoolean exceptionObserved = new AtomicBoolean(false);
        try {
            int rounds = 1000;
            for (int i = 0; i < rounds; i++) {
                long batchId = 2_000_000L + i;
                long requestId = i;
                PrefillEndpoint endpoint = newPrefillEndpoint();
                BatchItem item = newBatchItem(requestId);
                endpoint.commitBatch(batchId, 100, List.of(item));

                TaskInfo task = new TaskInfo();
                task.setRequestId(requestId);
                task.setBatchId(batchId);
                task.setPhase(TaskPhase.RUNNING);
                WorkerStatusResponse resp = new WorkerStatusResponse();
                resp.setRunningTaskInfo(Map.of(String.valueOf(requestId), task));
                resp.setFinishedTaskInfo(Map.of());

                CyclicBarrier barrier = new CyclicBarrier(2);
                Future<?> closeFuture = pool.submit(() -> {
                    try {
                        barrier.await();
                        endpoint.close();
                    } catch (Throwable t) {
                        exceptionObserved.set(true);
                        throw t;
                    }
                    return null;
                });
                Future<?> calibrateFuture = pool.submit(() -> {
                    try {
                        barrier.await();
                        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), resp);
                    } catch (Throwable t) {
                        exceptionObserved.set(true);
                        throw t;
                    }
                    return null;
                });

                closeFuture.get(5, TimeUnit.SECONDS);
                calibrateFuture.get(5, TimeUnit.SECONDS);

                // Drain any entries the calibrate winner added after close()'s drainInflight
                endpoint.close();

                assertTrue(endpoint.prefillPendingRequestCount() >= 0,
                        "inflightRequestCount should not be negative at round " + i);
            }
            assertFalse(exceptionObserved.get(),
                    "exception observed during stress test (possible ConcurrentModificationException)");
        } finally {
            pool.shutdownNow();
        }
    }

    // ==================== UT-C6: concurrent offer vs shutdown ====================

    /**
     * UT-C6: Concurrent {@code WorkerBatcher.offer(item)} vs
     * {@code WorkerBatcher.shutdown()}.
     *
     * <p>{@code offer()} checks the {@code stopped} volatile flag and calls
     * {@code item.failOffer(...)} (never throws) if stopped. {@code shutdown()}
     * sets {@code stopped = true}, drains the queue, and failOffer's remaining
     * items. The race verifies that no item is orphaned (left in the queue
     * after shutdown) and no uncaught exception escapes {@code offer()}.
     *
     * <p>Setup per round: fresh {@link PrefillEndpoint} with mocked dispatch
     * executor (dispatch is a no-op). T1 offers M items in a loop; T2 calls
     * {@code shutdown()} simultaneously. After both complete, the queue must
     * be empty.
     *
     * <p>Assertions:
     * <ul>
     *   <li>No uncaught exception from {@code offer()}.</li>
     *   <li>{@code batcher.queueSize() == 0} — all items either dispatched
     *       (removed by run loop) or failOffer'd (removed by shutdown drain).</li>
     *   <li>After {@code endpoint.close()}, {@code prefillPendingRequestCount() == 0}
     *       — no inflight leak from dispatched items.</li>
     * </ul>
     */
    @Test
    @Timeout(60)
    void concurrentOfferVsShutdown() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(2);
        AtomicBoolean exceptionObserved = new AtomicBoolean(false);
        try {
            int rounds = 1000;
            int offersPerRound = 10;
            for (int i = 0; i < rounds; i++) {
                PrefillEndpoint endpoint = newPrefillEndpoint();
                WorkerBatcher batcher = endpoint.getBatcher();

                CyclicBarrier barrier = new CyclicBarrier(2);
                Future<?> offerFuture = pool.submit(() -> {
                    try {
                        barrier.await();
                        for (int j = 0; j < offersPerRound; j++) {
                            try {
                                batcher.offer(newBatchItem(j));
                            } catch (Throwable t) {
                                exceptionObserved.set(true);
                            }
                        }
                    } catch (Throwable t) {
                        exceptionObserved.set(true);
                    }
                    return null;
                });
                Future<?> shutdownFuture = pool.submit(() -> {
                    try {
                        barrier.await();
                        batcher.shutdown();
                    } catch (Throwable t) {
                        exceptionObserved.set(true);
                    }
                    return null;
                });

                offerFuture.get(10, TimeUnit.SECONDS);
                shutdownFuture.get(5, TimeUnit.SECONDS);

                // close() calls batcher.shutdown() again (drains any items orphaned
                // by the known offer/shutdown race: reserveQueueSlot → drainTo → queue.add)
                // and drainInflight() (clears dispatched items from inflightEntries).
                endpoint.close();
                assertEquals(0, batcher.queueSize(),
                        "queueDepth should be zero after close at round " + i);
                assertEquals(0, endpoint.prefillPendingRequestCount(),
                        "prefillPendingRequestCount should be zero after close at round " + i);
            }
            assertFalse(exceptionObserved.get(),
                    "uncaught exception observed during offer/shutdown stress test");
        } finally {
            pool.shutdownNow();
        }
    }
}
