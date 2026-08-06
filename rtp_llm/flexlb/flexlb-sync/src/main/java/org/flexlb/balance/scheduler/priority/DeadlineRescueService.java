package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Phase 6 deadline rescue: a background scanner that migrates danger-zone
 * requests (deadline within {@code autoTpmDangerThresholdMs}) out of their
 * prefill queue onto a better P/D placement (design doc 4.4, 14).
 *
 * <p>Each tick takes per-endpoint {@code PrefillQueueSnapshot}s, picks
 * candidates (priority above the lowest level, transferCount below
 * {@code autoTpmMaxTransferCount}), CAS-removes them from the source queue
 * (reason {@code RESCUE}) and re-enters {@link PriorityAdmissionScheduler}
 * with the original arrival/deadline preserved — a rescue never resets the
 * SLO. A failed re-entry completes the original future with an explicit
 * error; a rescued request is never retried.
 *
 * <p>Storm control (design doc 14.4, 22.5): per-request transfer cap, per-tick
 * global cap ({@code autoTpmMaxRescuePerTick}) and per-source-endpoint cap
 * ({@code autoTpmMaxRescuePerEndpointPerTick}). Concurrency safety relies
 * solely on the existing queueVersion/admissionVersion CAS protocol — no
 * cross-endpoint lock is taken.
 *
 * <p>The scanner thread is created only when both {@code autoTpmEnabled} and
 * {@code autoTpmDeadlineRescueEnabled} are set (default off).
 */
@Component
public class DeadlineRescueService {

    /** Queue-removal reason recorded by the batcher for a rescued request. */
    static final String REMOVE_REASON_RESCUE = "RESCUE";

    private static final String RESULT_SUCCESS = "success";
    private static final String RESULT_REQUEUE_FAILED = "requeue_failed";
    private static final String RESULT_CAS_SKIPPED = "cas_skipped";
    private static final String RESULT_LIMITED = "limited";

    private final ConfigService configService;
    private final EndpointRegistry endpointRegistry;
    private final PriorityAdmissionScheduler admissionScheduler;
    private final InflightRegistrar registrar;
    private final PrioritySchedulerReporter priorityReporter;

    private volatile boolean running;
    private Thread scannerThread;

    public DeadlineRescueService(ConfigService configService,
                                 EndpointRegistry endpointRegistry,
                                 PriorityAdmissionScheduler admissionScheduler,
                                 InflightRegistrar registrar,
                                 PrioritySchedulerReporter priorityReporter) {
        this.configService = configService;
        this.endpointRegistry = endpointRegistry;
        this.admissionScheduler = admissionScheduler;
        this.registrar = registrar;
        this.priorityReporter = priorityReporter;
    }

    /** Start the scanner thread iff both Auto-TPM and the rescue switch are on. */
    @PostConstruct
    public synchronized void start() {
        FlexlbConfig config = configService.loadBalanceConfig();
        if (!config.isAutoTpmEnabled() || !config.isAutoTpmDeadlineRescueEnabled()) {
            Logger.info("[auto-tpm] deadline rescue disabled (autoTpmEnabled={}, autoTpmDeadlineRescueEnabled={})",
                    config.isAutoTpmEnabled(), config.isAutoTpmDeadlineRescueEnabled());
            return;
        }
        if (scannerThread != null) {
            return;
        }
        running = true;
        scannerThread = new Thread(this::runLoop, "flexlb-deadline-rescue");
        scannerThread.setDaemon(true);
        scannerThread.start();
        Logger.info("[auto-tpm] deadline rescue scanner started, interval_ms={} danger_threshold_ms={}",
                config.getAutoTpmRescueScanIntervalMs(), config.getAutoTpmDangerThresholdMs());
    }

    @PreDestroy
    public synchronized void stop() {
        running = false;
        if (scannerThread != null) {
            scannerThread.interrupt();
            try {
                scannerThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            scannerThread = null;
        }
    }

    /** Whether the scanner thread is active (observability / tests). */
    public boolean isRunning() {
        return running && scannerThread != null && scannerThread.isAlive();
    }

    private void runLoop() {
        long intervalMs = Math.max(1, configService.loadBalanceConfig().getAutoTpmRescueScanIntervalMs());
        while (running) {
            try {
                rescueTick(System.currentTimeMillis());
            } catch (Throwable t) {
                Logger.error("[auto-tpm] deadline rescue tick failed", t);
            }
            try {
                Thread.sleep(intervalMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    /** A danger-zone request spotted in one endpoint's queue snapshot. */
    private record Candidate(String endpointId, PrefillQueueManager manager, QueuedRequestSnapshot snap) {
    }

    /**
     * One scan pass: collect danger-zone candidates across all prefill queues,
     * order them (priority desc → deadline asc → requestId asc) and migrate
     * within the per-tick / per-endpoint budgets.
     *
     * @return migrations count = requests that actually entered
     *         {@link #rescueOne} (i.e. CAS-removed from their source queue,
     *         limit-consuming); candidates dropped as {@code limited} or
     *         {@code cas_skipped} are NOT counted
     */
    int rescueTick(long nowMs) {
        FlexlbConfig config = configService.loadBalanceConfig();
        long dangerThresholdMs = config.getAutoTpmDangerThresholdMs();
        int maxTransferCount = config.getAutoTpmMaxTransferCount();
        int maxPerTick = config.getAutoTpmMaxRescuePerTick();
        int maxPerEndpoint = config.getAutoTpmMaxRescuePerEndpointPerTick();
        int lowestPriority = lowestPriority(config);

        // Per-endpoint snapshot + expected version for the CAS removals below.
        List<Candidate> candidates = new ArrayList<>();
        Map<String, Long> versionByEndpoint = new HashMap<>();
        endpointRegistry.getPrefillEndpoints().forEach((key, ep) -> {
            PrefillQueueManager manager = ep.getBatcher().queueManager();
            PrefillQueueSnapshot snapshot = manager.snapshot();
            versionByEndpoint.put(key, snapshot.queueVersion());
            for (QueuedRequestSnapshot snap : snapshot.items()) {
                // Task40: rescue only handles the danger zone with remaining
                // time left (deadline > now); already-expired requests are
                // rejected by the scheduler's SLO check instead, and
                // no-priority items (priority 0, legacy path) never migrate.
                if (snap.deadlineMs() > 0
                        && snap.deadlineMs() > nowMs
                        && snap.deadlineMs() - nowMs <= dangerThresholdMs
                        && snap.priority() > 0
                        && snap.priority() > lowestPriority
                        && snap.transferCount() < maxTransferCount) {
                    candidates.add(new Candidate(key, manager, snap));
                }
            }
        });
        if (candidates.isEmpty()) {
            return 0;
        }
        candidates.sort(Comparator
                .comparingInt((Candidate c) -> c.snap().priority()).reversed()
                .thenComparingLong(c -> c.snap().deadlineMs())
                .thenComparingLong(c -> c.snap().requestId()));

        int migrations = 0;
        Map<String, Integer> perEndpoint = new HashMap<>();
        for (Candidate candidate : candidates) {
            int priority = candidate.snap().priority();
            if (migrations >= maxPerTick
                    || perEndpoint.getOrDefault(candidate.endpointId(), 0) >= maxPerEndpoint) {
                priorityReporter.reportRescue(priority, RESULT_LIMITED);
                continue;
            }
            onCandidateSelected(candidate.snap().requestId());

            long expectedVersion = versionByEndpoint.get(candidate.endpointId());
            List<BatchItem> removed = candidate.manager().tryRemove(
                    List.of(candidate.snap().requestId()), expectedVersion, REMOVE_REASON_RESCUE);
            if (removed == null) {
                // Queue mutated since the snapshot — refresh the expected
                // version so later candidates on this endpoint can still CAS.
                versionByEndpoint.put(candidate.endpointId(), candidate.manager().queueVersion());
                priorityReporter.reportRescue(priority, RESULT_CAS_SKIPPED);
                continue;
            }
            versionByEndpoint.put(candidate.endpointId(), candidate.manager().queueVersion());
            if (removed.isEmpty()) {
                // Request left the queue between snapshot and removal.
                priorityReporter.reportRescue(priority, RESULT_CAS_SKIPPED);
                continue;
            }
            BatchItem item = removed.get(0);
            if (item.future().isDone()) {
                // Terminal race (TTL / preemption already completed it): the
                // terminal path owns rollback, nothing left to migrate.
                priorityReporter.reportRescue(priority, RESULT_CAS_SKIPPED);
                continue;
            }
            migrations++;
            perEndpoint.merge(candidate.endpointId(), 1, Integer::sum);
            rescueOne(item, candidate.endpointId(), nowMs);
        }
        return migrations;
    }

    /**
     * Migrate one request already removed from its source queue: undo the old
     * inflight registration and decode reservation (rescue is NOT a terminal
     * state — the future stays pending), bump transferCount and re-enter the
     * admission scheduler with the same ctx/future. On failure the scheduler
     * itself completes the future with an explicit error; no retry here.
     */
    private void rescueOne(BatchItem item, String fromEndpoint, long nowMs) {
        long startNanos = System.nanoTime();
        long remainingMs = item.deadlineMs() - nowMs;

        registrar.unregisterInflight(item);
        if (item.decodeEp() != null && item.decode() != null) {
            item.decodeEp().release(item.decode().getRequestId());
        }

        BalanceContext ctx = item.ctx();
        ctx.setTransferCount(item.transferCount() + 1);
        admissionScheduler.schedule(ctx, item.future(), registrar);

        // schedule() completes the future synchronously on every failure path;
        // a still-pending (or successful) future means the request was placed.
        Response outcome = item.future().getNow(null);
        boolean success = outcome == null || outcome.isSuccess();
        String result = success ? RESULT_SUCCESS : RESULT_REQUEUE_FAILED;
        String toEndpoint = success ? ctx.getScheduledPrefillEndpoint() : "-";
        long tookMs = (System.nanoTime() - startNanos) / 1_000_000;

        priorityReporter.reportRescue(item.priority(), result);
        priorityReporter.reportTransfer(item.priority(), result);
        priorityReporter.reportRescueLatency(item.priority(), result, fromEndpoint, toEndpoint, tookMs);
        Logger.info("[auto-tpm] rescue request_id={} priority={} from={} to={} transfer_count={} "
                        + "remaining_ms={} took_ms={} result={}",
                item.requestId(), item.priority(), fromEndpoint, toEndpoint,
                ctx.getTransferCount(), remainingMs, tookMs, result);
    }

    /**
     * Test seam invoked after a candidate passed the budget checks and before
     * its CAS removal — lets tests inject concurrent queue mutations.
     */
    protected void onCandidateSelected(long requestId) {
        // no-op in production
    }

    /** Lowest configured priority level — never rescued (design doc 14.2). */
    private static int lowestPriority(FlexlbConfig config) {
        String spec = config.getAutoTpmPriorityLevels();
        int min = Integer.MAX_VALUE;
        if (spec != null) {
            for (String part : spec.split(",")) {
                try {
                    min = Math.min(min, Integer.parseInt(part.trim()));
                } catch (NumberFormatException ignored) {
                    // malformed entry — fall back below if nothing parses
                }
            }
        }
        return min == Integer.MAX_VALUE ? 30 : min;
    }
}
