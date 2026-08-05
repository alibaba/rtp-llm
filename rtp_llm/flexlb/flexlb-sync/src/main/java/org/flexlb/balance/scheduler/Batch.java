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
 * (status callbacks, metric reporting). Items are never physically
 * removed — {@link #removeItem} is a no-op to preserve index alignment
 * for {@link #complete} (see F7).
 */
public final class Batch implements InflightEntry {

    private final long batchId;

    /** DP-aware: outer list indexed by DP position, inner list = items for that DP. */
    private final List<List<InflightItem>> itemsByDp;

    /** The prefill endpoint this batch is dispatched to. */
    volatile PrefillEndpoint prefillEp;

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
     * in {@link #itemsByDp} at the same index, matched by <b>original
     * position</b>. Items that were already driven terminal by repack,
     * cancel, or early failure are naturally skipped — their
     * {@link InflightItem#complete(Response)} CAS returns false (no-op),
     * so the response is harmlessly discarded while surviving items at
     * their original positions receive the correct response.
     *
     * <p>This position-stable alignment is preserved because
     * {@link #removeItem} is a no-op: items are never physically removed
     * from {@link #itemsByDp}, preventing index drift when a middle item
     * is repacked out.
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
     * No-op: items remain at their original positions so that
     * {@link #complete} can match responses by original index without
     * position drift from repack removals.
     *
     * <p>Previously this method physically removed the item from
     * {@link #itemsByDp} (CopyOnWriteArrayList). When a middle item was
     * repacked out, the surviving items shifted down, breaking the
     * index alignment between {@code itemsByDp} and the caller-built
     * {@code responsesByDp} in {@link #complete} — some items received
     * the wrong response or were never marked complete.
     *
     * <p>With this no-op, terminal items stay in place and are
     * harmlessly skipped by the CAS guard in
     * {@link InflightItem#complete(Response)}. The batch is released
     * as a unit in {@link #cleanupBatchLevel()}.
     *
     * <p>The {@code terminated} check is retained to document that
     * batch-level {@link #terminate(TerminalReason)} iterates items and
     * calls {@link InflightItem#terminate(TerminalReason)}, which
     * recurses back into this method.
     */
    void removeItem(InflightItem item) {
        if (terminated.get()) return; // prevent recursion during batch terminate
        // intentionally no-op — see javadoc above
    }

    // ---- private helpers ----

    /**
     * Batch-level cleanup: release EP resources.
     * Called exactly once (inside CAS-guarded terminate/complete/failAll).
     */
    private void cleanupBatchLevel() {
        if (prefillEp != null) {
            prefillEp.releaseBatch(batchId);
        }
    }
}
