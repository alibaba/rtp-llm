package org.flexlb.balance.scheduler.priority;

import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * AutoTPM Cancel: WorkerStatus-driven release confirmation.
 *
 * <p>A victim's capacity may only be reclaimed when the resource-owning worker
 * itself reports {@code resource_released=true} for the matching
 * {@code request_id}. This tracker turns that condition
 * into a {@link CompletableFuture} completed from the normal periodic
 * WorkerStatus path — it never sends any RPC of its own and never blocks the
 * caller thread.
 *
 * <p>Concurrency contract:
 * <ul>
 *   <li>{@link #awaitReleased} synchronously consults the latest cached
 *       observation UNDER THE SAME LOCK used by {@link #onWorkerStatus} before
 *       registering a waiter — a status that arrived earlier can never be
 *       missed (missed-wakeup fix);</li>
 *   <li>futures complete exactly once; the deadline timer is cancelled on
 *       completion (no timer leak);</li>
 *   <li>a worker-epoch change or an explicitly reported unhealthy worker fails
 *       all its waiters — capacity is NOT optimistically reclaimed;</li>
 *   <li>task absence from the running/finished lists keeps the waiter waiting
 *       (absence is never release proof).</li>
 * </ul>
 */
public class ReleaseTracker implements AutoCloseable {

    /**
     * Process-wide instance: fed by endpoint calibrate paths (which are not
     * Spring-managed) and consumed by the admission scheduler. Tests may still
     * construct isolated instances.
     */
    private static final ReleaseTracker GLOBAL = new ReleaseTracker();

    public static ReleaseTracker global() {
        return GLOBAL;
    }

    public static final long DEFAULT_DEADLINE_MS = 5_000L;
    public static final long MAX_DEADLINE_MS = 30_000L;

    /** One release observation extracted from a WorkerStatus report. */
    public record ReleaseObservation(String workerKey,
                                     long workerEpoch,
                                     long statusVersion,
                                     String requestId,
                                     boolean resourceReleased,
                                     long lifecycleRevision,
                                     int terminalErrorCode) {
    }

    private record WaiterKey(String workerKey, String requestId) {
    }

    private static final class Waiter {
        final CompletableFuture<ReleaseObservation> future = new CompletableFuture<>();
        ScheduledFuture<?> deadlineTask;
    }

    /** Last confirmed-released observation per (worker, request). */
    private final Map<WaiterKey, ReleaseObservation> releasedCache = new HashMap<>();
    private final Map<WaiterKey, List<Waiter>> waiters = new HashMap<>();
    private final Map<String, Long> workerEpochs = new HashMap<>();
    private final Object lock = new Object();
    private final ScheduledExecutorService deadlineScheduler;
    private static final int RELEASED_CACHE_CAP = 100_000;

    public ReleaseTracker() {
        this.deadlineScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "auto-tpm-release-tracker-deadline");
            t.setDaemon(true);
            return t;
        });
    }

    /**
     * Register interest in a release. Never blocks: returns a future completed
     * by {@link #onWorkerStatus}, failed on deadline / worker-epoch change.
     */
    public CompletableFuture<ReleaseObservation> awaitReleased(String workerKey,
                                                               String requestId,
                                                               long deadlineMs) {
        long boundedDeadline = Math.min(Math.max(1, deadlineMs <= 0 ? DEFAULT_DEADLINE_MS : deadlineMs),
                MAX_DEADLINE_MS);
        WaiterKey key = new WaiterKey(workerKey, requestId);
        Waiter waiter = new Waiter();
        synchronized (lock) {
            // Sync-check-then-register under ONE lock: status that already
            // arrived completes immediately (missed-wakeup fix).
            ReleaseObservation cached = releasedCache.get(key);
            if (cached != null) {
                waiter.future.complete(cached);
                return waiter.future;
            }
            waiters.computeIfAbsent(key, k -> new ArrayList<>(2)).add(waiter);
        }
        waiter.deadlineTask = deadlineScheduler.schedule(() -> {
            boolean removed;
            synchronized (lock) {
                List<Waiter> list = waiters.get(key);
                removed = list != null && list.remove(waiter);
                if (list != null && list.isEmpty()) {
                    waiters.remove(key);
                }
            }
            if (removed) {
                waiter.future.completeExceptionally(new TimeoutException(
                        "release not confirmed within " + boundedDeadline + "ms for request "
                                + requestId + " on " + workerKey));
            }
        }, boundedDeadline, TimeUnit.MILLISECONDS);
        // If the future completed concurrently before the timer was assigned,
        // cancel the timer now (exactly-once + no timer leak).
        waiter.future.whenComplete((r, e) -> {
            ScheduledFuture<?> task = waiter.deadlineTask;
            if (task != null) {
                task.cancel(false);
            }
        });
        return waiter.future;
    }

    /**
     * Feed one observation from the periodic WorkerStatus path. Also used to
     * pre-warm the released cache so late {@code awaitReleased} calls complete
     * immediately.
     */
    public void onWorkerStatus(ReleaseObservation observation) {
        List<Waiter> completed = null;
        synchronized (lock) {
            Long knownEpoch = workerEpochs.get(observation.workerKey());
            if (knownEpoch == null || knownEpoch != observation.workerEpoch()) {
                if (knownEpoch != null) {
                    // Epoch change: restart detected — fail every waiter of
                    // this worker, do NOT reclaim capacity.
                    failWorkerWaitersLocked(observation.workerKey(),
                            "worker epoch changed " + knownEpoch + " -> " + observation.workerEpoch());
                }
                if (observation.workerEpoch() != 0) {
                    workerEpochs.put(observation.workerKey(), observation.workerEpoch());
                }
            }
            if (!observation.resourceReleased()) {
                return;  // absence / not-yet-released keeps waiters waiting
            }
            WaiterKey key = new WaiterKey(observation.workerKey(), observation.requestId());
            if (releasedCache.size() < RELEASED_CACHE_CAP) {
                releasedCache.put(key, observation);
            }
            List<Waiter> list = waiters.remove(key);
            if (list != null) {
                completed = list;
            }
        }
        if (completed != null) {
            for (Waiter waiter : completed) {
                waiter.future.complete(observation);
            }
        }
    }

    /**
     * Test hygiene for the process-wide {@link #global()} instance: drops the
     * released cache, worker epochs and every pending waiter (failed, never
     * silently released — iron rule 4 holds even in tests). Production code
     * never calls this.
     */
    public void reset() {
        List<Waiter> orphaned = new ArrayList<>();
        synchronized (lock) {
            waiters.values().forEach(orphaned::addAll);
            waiters.clear();
            releasedCache.clear();
            workerEpochs.clear();
        }
        for (Waiter waiter : orphaned) {
            waiter.future.completeExceptionally(new IllegalStateException("release tracker reset"));
        }
    }

    /** Fail all waiters of an unhealthy worker (alive=false path). */
    public void onWorkerUnhealthy(String workerKey) {
        synchronized (lock) {
            failWorkerWaitersLocked(workerKey, "worker unhealthy");
        }
    }

    /** Drop cached releases of a terminal request (bounded-cache hygiene). */
    public void forget(String workerKey, String requestId) {
        synchronized (lock) {
            releasedCache.remove(new WaiterKey(workerKey, requestId));
        }
    }

    private void failWorkerWaitersLocked(String workerKey, String reason) {
        List<Waiter> failed = new ArrayList<>();
        waiters.entrySet().removeIf(entry -> {
            if (entry.getKey().workerKey().equals(workerKey)) {
                failed.addAll(entry.getValue());
                return true;
            }
            return false;
        });
        releasedCache.keySet().removeIf(key -> key.workerKey().equals(workerKey));
        if (!failed.isEmpty()) {
            Logger.warn("[auto-tpm] failing {} release waiters on {}: {}", failed.size(), workerKey, reason);
            for (Waiter waiter : failed) {
                waiter.future.completeExceptionally(new IllegalStateException(reason));
            }
        }
    }

    @Override
    public void close() {
        deadlineScheduler.shutdownNow();
    }
}
