package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Thread-safe store for inflight {@link InflightItem}s, keyed by string-form
 * request ID.
 *
 * <p>Provides O(1) put / get / remove backed by {@link ConcurrentHashMap}.
 * A background evictor thread periodically sweeps the store: RUNNING items
 * older than the configured inflight TTL are timed out with an error
 * response ({@link InflightItem#timeoutWithError()}), and tombstones (items
 * that have reached a terminal state) are removed after a safety TTL,
 * preventing unbounded growth from leaked entries that were never
 * explicitly removed.
 *
 * <p>The TTL is a safety net — normal operation does NOT remove items on
 * terminal transition. Instead, terminated items remain as tombstones
 * (terminal state in {@link InflightItem#state()}) so that late cancel lookups return
 * false (already terminal) rather than null (not found). The evictor
 * only catches items whose TTL has expired since termination.
 */
@Component
public class InflightStore {

    private final ConcurrentMap<String, InflightItem> store = new ConcurrentHashMap<>();
    private final ScheduledExecutorService evictor;
    private final BatchSchedulerReporter reporter;
    private final ConfigService configService;

    /**
     * Number of non-terminal items registered via {@link #putIfAbsent}.
     * Decremented by the item's terminal callback (CAS-guarded, exactly once).
     * Items registered via plain {@link #put} are not counted.
     */
    private final AtomicInteger activeCount = new AtomicInteger(0);

    /**
     * Per-scheduler active counters, bucketed by {@link InflightItem#scheduler()}.
     * Incremented alongside {@link #activeCount} on registration and decremented
     * inside the same exactly-once terminal callback, so the bucket can never
     * drift from the global counter. Lets each scheduler gate admission on its
     * own inflight population (e.g. {@code flexlbBatchMaxInflight} is
     * BATCH-only and must not be consumed by DIRECT/QUEUE traffic).
     */
    private final ConcurrentMap<AbstractScheduler, AtomicInteger> schedulerActiveCounts =
            new ConcurrentHashMap<>();

    /** Tombstone retention period: items stay in the store this long after termination. */
    private static final long TTL_MS = 60_000;

    /** Evictor runs at this interval. */
    private static final long EVICT_INTERVAL_MS = 10_000;

    public InflightStore(BatchSchedulerReporter reporter, ConfigService configService) {
        this.reporter = reporter;
        this.configService = configService;
        this.evictor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "inflight-evictor");
            t.setDaemon(true);
            return t;
        });
        this.evictor.scheduleAtFixedRate(this::evict,
                EVICT_INTERVAL_MS, EVICT_INTERVAL_MS, TimeUnit.MILLISECONDS);
    }

    /**
     * Atomically register an item. Returns {@code null} if inserted, or the
     * existing item if the key is already present (including terminal
     * tombstones within TTL — duplicate request IDs are rejected until the
     * evictor removes the tombstone).
     *
     * <p>On successful insert the active counter is incremented and the item's
     * terminal callback is wired to decrement it (exactly once).
     *
     * <p>Race compensation: another thread (cancel / TTL cleanup / worker
     * callback) may drive the item terminal between the map insert and the
     * callback wiring — in that window the terminal transition sees a null
     * callback and never decrements. We therefore re-check the state after
     * wiring and, if already terminal, atomically claim the callback via
     * {@link InflightItem#takeOnTerminal()} and run it ourselves. Both this
     * path and the terminal transition claim via getAndSet(null), so the
     * decrement runs exactly once regardless of interleaving.
     */
    public InflightItem putIfAbsent(String requestId, InflightItem item) {
        InflightItem existing = store.putIfAbsent(requestId, item);
        if (existing == null) {
            activeCount.incrementAndGet();
            AtomicInteger schedulerCount = item.scheduler() == null ? null
                    : schedulerActiveCounts.computeIfAbsent(item.scheduler(), s -> new AtomicInteger());
            if (schedulerCount != null) {
                schedulerCount.incrementAndGet();
            }
            item.setOnTerminal(() -> {
                activeCount.decrementAndGet();
                if (schedulerCount != null) {
                    schedulerCount.decrementAndGet();
                }
            });
            if (item.state().isTerminal()) {
                Runnable cb = item.takeOnTerminal();
                if (cb != null) {
                    cb.run();
                }
            }
        }
        return existing;
    }

    public InflightItem get(String requestId) {
        return store.get(requestId);
    }

    /** Number of non-terminal items registered via {@link #putIfAbsent}. */
    public int activeCount() {
        return activeCount.get();
    }

    /**
     * Number of non-terminal items registered via {@link #putIfAbsent} by the
     * given scheduler ({@link InflightItem#scheduler()} bucket). Used by
     * per-path admission gates (e.g. {@code flexlbBatchMaxInflight}).
     */
    public int activeCount(AbstractScheduler scheduler) {
        AtomicInteger count = schedulerActiveCounts.get(scheduler);
        return count == null ? 0 : count.get();
    }

    /**
     * Periodically report the number of active (non-terminal) requests via
     * {@code flexlb.scheduler.inflight.size}. Tombstones within TTL are
     * excluded — external monitors treat this metric as the live inflight
     * count, and the tombstone tail would systematically inflate it.
     */
    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    public void reportInflightSize() {
        reporter.reportSchedulerInflightSize(activeCount.get());
    }

    /**
     * Single-pass TTL sweep, handling both inflight expiry and tombstone eviction:
     * <ul>
     *   <li>RUNNING items whose age exceeds the configured inflight TTL
     *       ({@code flexlbInflightTtlMs}) are timed out via
     *       {@link InflightItem#timeoutWithError()} — the TTL safety net for
     *       requests that never reached a terminal state (e.g. a lost ACK).
     *       Error semantics and resource rollback are owned by the item; the
     *       resulting tombstone is evicted on a later sweep.</li>
     *   <li>Terminated items (tombstones) older than {@link #TTL_MS} are
     *       removed from the store.</li>
     * </ul>
     *
     * <p>Non-terminated items within TTL are never evicted; they represent
     * genuinely inflight requests that must be tracked until a response or
     * timeout is delivered. Public so tests can trigger the sweep manually
     * (production runs it on the evictor thread).
     */
    public void evict() {
        try {
            long inflightTtlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
            long now = System.currentTimeMillis();
            store.forEach((reqId, item) -> {
                if (item.state() == InflightState.RUNNING
                        && (now - item.createdAtMs()) > inflightTtlMs) {
                    if (item.timeoutWithError()) {
                        Logger.warn("FlexLB inflight TTL expired: request_id={}", reqId);
                    }
                } else if (item.state().isTerminal()
                        && item.getTerminalTime() > 0
                        && (now - item.getTerminalTime()) > TTL_MS) {
                    store.remove(reqId);
                }
            });
        } catch (Exception e) {
            // Guard: prevent evict exceptions from silently cancelling subsequent scheduled runs
            Thread.currentThread().getUncaughtExceptionHandler().uncaughtException(Thread.currentThread(), e);
        }
    }

    public void shutdown() {
        evictor.shutdown();
    }
}
