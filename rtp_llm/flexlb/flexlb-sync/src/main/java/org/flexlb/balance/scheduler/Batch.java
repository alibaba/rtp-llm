package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.loadbalance.Response;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A group of {@link InflightItem}s dispatched together to one or more decode
 * endpoints (DPs).
 *
 * <p><b>DP-aware:</b> {@link #itemsByDp} is a two-dimensional structure —
 * the outer list is indexed by DP position within the batch, and each inner
 * list holds the items routed to that DP. All traversal (terminate, complete,
 * removeItem) respects this grouping.
 *
 * <p>Terminal transition: {@code Batch} uses a single {@link AtomicBoolean}
 * {@code terminated} for CAS-guarded idempotent terminal transition, whereas
 * {@link InflightItem} uses an {@code AtomicReference<InflightState>} that
 * additionally distinguishes the terminal kind (COMPLETED/FAILED/TIMED_OUT).
 *
 * <p>Thread-safety: {@link CopyOnWriteArrayList} for read-heavy access
 * (status callbacks, metric reporting) with infrequent mutation
 * (item removal during repack).
 */
public final class Batch implements InflightEntry {

    private final long batchId;

    /** DP-aware: outer list indexed by DP position, inner list = items for that DP. */
    private final List<List<InflightItem>> itemsByDp;

    /** The prefill endpoint this batch is dispatched to. */
    volatile PrefillEndpoint prefillEp;

    /** The scheduler that manages this batch. */
    volatile AbstractScheduler scheduler;

    /** CAS flag — {@code true} once the batch reaches a terminal state. */
    final AtomicBoolean terminated = new AtomicBoolean(false);

    public Batch(long batchId, List<List<InflightItem>> itemsByDp) {
        this.batchId = batchId;
        this.itemsByDp = new CopyOnWriteArrayList<>();
        for (List<InflightItem> dpItems : itemsByDp) {
            this.itemsByDp.add(new CopyOnWriteArrayList<>(dpItems));
        }
    }

    // ---- accessors ----

    public long batchId() {
        return batchId;
    }

    public List<List<InflightItem>> itemsByDp() {
        return itemsByDp;
    }

    public PrefillEndpoint prefillEp() {
        return prefillEp;
    }

    public void setPrefillEp(PrefillEndpoint ep) {
        this.prefillEp = ep;
    }

    public AbstractScheduler scheduler() {
        return scheduler;
    }

    public void setScheduler(AbstractScheduler scheduler) {
        this.scheduler = scheduler;
    }

    public boolean isTerminated() {
        return terminated.get();
    }

    // ---- error paths (CAS-guarded, double-layer traversal) ----

    /**
     * Terminate the entire batch — CAS-guarded, then propagate to all items.
     *
     * <p>Each item's {@link InflightItem#terminate(TerminalReason)} is called,
     * which internally checks {@code batch.terminated} to skip batch-level
     * callbacks (prevent recursion).
     */
    public void terminate(TerminalReason reason) {
        if (!terminated.compareAndSet(false, true)) return;
        for (List<InflightItem> dpItems : itemsByDp) {
            for (InflightItem item : dpItems) {
                item.terminate(reason);
            }
        }
        cleanupBatchLevel();
    }

    /**
     * Fail all items with the given cause — CAS-guarded, preserves cause in
     * each item's exceptional completion.
     */
    public void failAll(Throwable cause) {
        if (!terminated.compareAndSet(false, true)) return;
        for (List<InflightItem> dpItems : itemsByDp) {
            for (InflightItem item : dpItems) {
                item.fail(cause);
            }
        }
        cleanupBatchLevel();
    }

    /** Convenience: terminate all items as TIMED_OUT. */
    public void timeout() {
        terminate(TerminalReason.TIMED_OUT);
    }

    /** Convenience: terminate all items as CANCELLED. */
    public void cancel() {
        terminate(TerminalReason.CANCELLED);
    }

    // ---- success path (CAS-guarded, double-layer traversal) ----

    /**
     * Complete all items with their respective responses.
     *
     * <p>Each inner list in {@code responsesByDp} corresponds to the items
     * in {@link #itemsByDp} at the same index. Items are completed in order.
     *
     * @param responsesByDp per-DP response lists, aligned with {@link #itemsByDp}
     */
    public void complete(List<List<Response>> responsesByDp) {
        if (!terminated.compareAndSet(false, true)) return;
        for (int dpIdx = 0; dpIdx < itemsByDp.size(); dpIdx++) {
            List<InflightItem> items = itemsByDp.get(dpIdx);
            if (dpIdx >= responsesByDp.size()) {
                break;
            }
            List<Response> responses = responsesByDp.get(dpIdx);
            for (int i = 0; i < items.size() && i < responses.size(); i++) {
                items.get(i).complete(responses.get(i));
            }
        }
        cleanupBatchLevel();
    }

    // ---- item management ----

    /**
     * Remove a single item from all DP lists (used during repack or
     * when an item completes/fails before batch dispatch).
     *
     * <p>Checks {@code terminated} to prevent recursion when called from
     * {@link InflightItem#terminate(TerminalReason)} during batch-level
     * termination.
     *
     * @param item the item to remove
     */
    void removeItem(InflightItem item) {
        if (terminated.get()) return; // prevent recursion during batch terminate
        for (List<InflightItem> dpItems : itemsByDp) {
            dpItems.remove(item);
        }
    }

    // ---- private helpers ----

    /**
     * Batch-level cleanup: release EP resources and notify scheduler.
     * Called exactly once (inside CAS-guarded terminate/complete/failAll).
     */
    private void cleanupBatchLevel() {
        if (prefillEp != null) {
            prefillEp.releaseBatch(batchId);
        }
        if (scheduler != null) {
            scheduler.removeBatchInflight(String.valueOf(batchId));
        }
    }
}
