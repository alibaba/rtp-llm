package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.config.PrefillQueueSheddingConfig;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/** Periodically evaluates blocked placement candidates independently of placement planning. */
final class PrefillSheddingMonitor implements AutoCloseable {
    private final PrefillQueueSheddingConfig sheddingConfig;
    private final DefaultRouter router;

    private final boolean priorityOrdering;

    private final ApplyVictim applyVictim;
    private final Map<GlobalQueueEntry, Blocked> candidates = new ConcurrentHashMap<>();
    private final ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor(task -> {
        var thread = new Thread(task, "flexlb-prefill-shedding");
        thread.setDaemon(true);
        return thread;
    });
    private final Map<String, PrefillQueueShedding.Controller> sheddingControllers =
            new LinkedHashMap<>();
    private final Map<String, PrefillTrafficTrend> trafficTrends = new LinkedHashMap<>();
    private final Map<String, PrefillDeadlineBudget> deadlineBudgets = new LinkedHashMap<>();
    /** Arrival classification funnel from ingress threads to the shedding thread. */
    private final ConcurrentLinkedQueue<GlobalQueueEntry> arrivals = new ConcurrentLinkedQueue<>();
    /** Hard-timeout evidence funnel from request lifecycle to the shedding thread. */
    private final ConcurrentLinkedQueue<HardTimeout> hardTimeouts = new ConcurrentLinkedQueue<>();

    private volatile Map<String, PrefillTrafficTrend.Snapshot> publishedTrafficTrends = Map.of();

    PrefillSheddingMonitor(PrefillQueueSheddingConfig config, DefaultRouter router,
            boolean priorityOrdering, ApplyVictim applyVictim) {
        this.sheddingConfig = config;
        this.router = router;
        this.priorityOrdering = priorityOrdering;
        this.applyVictim = applyVictim;
    }

    @Override
    public void close() {
        executor.shutdownNow();
        candidates.clear();
        arrivals.clear();
        hardTimeouts.clear();
    }

    void start() {
        long interval = sheddingConfig.getDeadlineBudget() == null ? 100L : PrefillQueueShedding.SOFT_INTERVAL_MS;
        executor.scheduleWithFixedDelay(() -> {
            try {
                runRound();
            } catch (RuntimeException failure) {
                Logger.error("Prefill shedding round failed", failure);
            }
        }, interval, interval, TimeUnit.MILLISECONDS);
    }

    void arrive(GlobalQueueEntry entry) { arrivals.add(entry); }

    void blocked(GlobalQueueEntry entry) { candidates.put(entry, new Blocked(entry, entry.sheddingEstimate)); }

    boolean isCurrent(Blocked blocked) { return candidates.get(blocked.entry()) == blocked; }

    void remove(GlobalQueueEntry entry) { candidates.remove(entry); }

    void hardTimeout(GlobalQueueEntry entry) {
        hardTimeouts.add(new HardTimeout(entry.routingGroup, System.currentTimeMillis()));
    }

    Map<String, PrefillTrafficTrend.Snapshot> snapshot() { return publishedTrafficTrends; }

    private void runRound() {
        drainArrivals();
        drainHardTimeouts();
        long nowMs = System.currentTimeMillis();
        long nowNanos = System.nanoTime();
        Map<String, List<PrefillQueueShedding.Candidate<Blocked>>> pools = new LinkedHashMap<>();
        for (var blocked : candidates.values()) {
            var entry = blocked.entry();
            var estimate = blocked.estimate();
            if (estimate == null || entry.removed || entry.future.isDone() || entry.context.requestExpired(nowMs)) {
                continue;
            }
            long waitedMs = TimeUnit.NANOSECONDS.toMillis(nowNanos - entry.queuedAtNanos);
            pools.computeIfAbsent(entry.routingGroup, ignored -> new ArrayList<>())
                    .add(new PrefillQueueShedding.Candidate<>(blocked, entry.priority, entry.sequence, estimate, waitedMs));
        }
        pools.forEach((pool, entries) -> shedPool(pool, entries, nowMs));
        publishTrafficTrends();
    }

    private void drainArrivals() {
        GlobalQueueEntry entry;
        while ((entry = arrivals.poll()) != null) {
            if (entry.removed || entry.future.isDone()) {
                continue;
            }
            PrefillQueueShedding.Controller controller = controllerFor(entry.routingGroup);
            PrefillTrafficTrend trend = trendFor(entry.routingGroup);
            if (trend != null) {
                trend.arrive(Math.max(0L, entry.context.getRequest().getSeqLen()));
            }
            PrefillQueueShedding.Estimate estimate = entry.sheddingEstimate;
            if (estimate != null && estimate.hitTokens() == 0L) {
                controller.coldArrival();
            } else {
                controller.arrive();
            }
        }
    }

    private PrefillQueueShedding.Controller controllerFor(String pool) {
        return sheddingControllers.computeIfAbsent(
                pool, ignored -> new PrefillQueueShedding.Controller());
    }

    private PrefillTrafficTrend trendFor(String pool) {
        PrefillQueueSheddingConfig.RecoveryConfig recovery = sheddingConfig.getRecovery();
        if (recovery == null) {
            return null;
        }
        return trafficTrends.computeIfAbsent(
                pool, ignored -> new PrefillTrafficTrend(recovery));
    }

    private void drainHardTimeouts() {
        HardTimeout timeout;
        while ((timeout = hardTimeouts.poll()) != null) {
            controllerFor(timeout.pool()).hardTimeout(timeout.atMs());
        }
    }

    private void shedPool(
            String pool,
            List<PrefillQueueShedding.Candidate<Blocked>> candidates,
            long nowMs) {
        CostBasedPrefillStrategy.PrefillPoolSnapshot snapshot = router.prefillPoolSnapshot(pool);
        if (snapshot == null || snapshot.workers() <= 0) {
            return;
        }
        PrefillQueueShedding.Controller controller = controllerFor(pool);
        PrefillTrafficTrend trend = trendFor(pool);
        double queuedWorkMs = 0.0;
        for (PrefillQueueShedding.Candidate<Blocked> candidate : candidates) {
            queuedWorkMs += candidate.estimate().fleetWorkMs();
        }
        PrefillTrafficTrend.Snapshot observed = trend == null ? null
                : trend.observe(nowMs, queuedWorkMs,
                        sheddingConfig.getExpectedWaitThresholdMs(),
                        sheddingConfig.getMaxWaitMs());
        List<Blocked> victims;
        if (sheddingConfig.getDeadlineBudget() != null) {
            victims = deadlineBudgetVictims(pool, candidates, snapshot, nowMs);
        } else {
            // Soft eviction requires sustained pool-wide Engine queue pressure;
            // recovery headroom only reduces, never triggers, eviction.
            victims = PrefillQueueShedding.adaptiveVictims(
                    candidates, sheddingConfig, observed, controller, nowMs,
                    router.isPrefillClusterOverloaded(pool, nowMs));
        }
        double acceptingPercent = snapshot.accepting() * 100.0 / snapshot.workers();
        int selected = victims.size();
        int accepted = 0;
        for (Blocked victim : victims) {
            boolean cold = victim.entry().knownColdRequest || victim.estimate().hitTokens() == 0L;
            boolean retry = cold && acceptingPercent >= sheddingConfig.getMinAcceptingWorkerPercent();
            if (applyVictim.apply(victim, retry)) {
                if (trend != null) {
                    trend.discard(victim.estimate().fleetWorkMs());
                }
                accepted++;
            }
        }
        controller.recordEvictions(nowMs, selected, accepted);
    }

    private List<Blocked> deadlineBudgetVictims(
            String pool,
            List<PrefillQueueShedding.Candidate<Blocked>> candidates,
            CostBasedPrefillStrategy.PrefillPoolSnapshot snapshot,
            long nowMs) {
        PrefillDeadlineBudget budget = deadlineBudgets.computeIfAbsent(
                pool, ignored -> new PrefillDeadlineBudget());
        PrefillDeadlineBudget.Fleet fleet = new PrefillDeadlineBudget.Fleet(
                snapshot.workers(),
                snapshot.workers() - snapshot.accepting(),
                snapshot.pendingTokens(),
                true);
        List<PrefillDeadlineBudget.Candidate<Blocked>> budgeted =
                new ArrayList<>(candidates.size());
        for (PrefillQueueShedding.Candidate<Blocked> candidate : candidates) {
            GlobalQueueEntry entry = candidate.value().entry();
            PrefillQueueShedding.Estimate estimate = candidate.estimate();
            long computeTokens = Math.max(
                    0L, estimate.inputTokens() - estimate.hitTokens());
            boolean cold = entry.knownColdRequest || estimate.hitTokens() == 0L;
            long remainingMs = Math.max(
                    0L, entry.context.getRequestExpiresAtMs() - nowMs);
            budgeted.add(new PrefillDeadlineBudget.Candidate<>(
                    candidate.value(), candidate.priority(), candidate.sequence(),
                    computeTokens, cold, remainingMs));
        }
        return budget.victims(budgeted, fleet,
                sheddingConfig.getDeadlineBudget(), priorityOrdering, nowMs);
    }

    private void publishTrafficTrends() {
        if (trafficTrends.isEmpty()) {
            return;
        }
        Map<String, PrefillTrafficTrend.Snapshot> snapshot = new LinkedHashMap<>();
        for (Map.Entry<String, PrefillTrafficTrend> entry : trafficTrends.entrySet()) {
            PrefillTrafficTrend.Snapshot latest = entry.getValue().latestSnapshot();
            if (latest != null) {
                snapshot.put(entry.getKey(), latest);
            }
        }
        publishedTrafficTrends = Collections.unmodifiableMap(snapshot);
    }

    @FunctionalInterface
    interface ApplyVictim {
        boolean apply(Blocked blocked, boolean retry);
    }

    record Blocked(GlobalQueueEntry entry, PrefillQueueShedding.Estimate estimate) { }

    private record HardTimeout(String pool, long atMs) { }
}
