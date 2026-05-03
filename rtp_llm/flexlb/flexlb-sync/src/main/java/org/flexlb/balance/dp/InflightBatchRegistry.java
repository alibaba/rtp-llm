package org.flexlb.balance.dp;

import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Tracks in-flight batches dispatched via Master.Enqueue so that {@code Cancel}
 * (frontend → Master → P/D) can locate which Prefill / Decode workers actually
 * hold the request.
 *
 * <p>Three indices kept in sync:
 * <ul>
 *   <li>{@code byBatch}: batchId → all requestIds + prefill target</li>
 *   <li>{@code byRequest}: requestId → its batch + decode target</li>
 *   <li>{@code requestStates}: requestId → state machine (PENDING_ACK / ACTIVE / CANCELLED)
 *       used to handle the cancel-before-ack race (see below).</li>
 * </ul>
 *
 * <h3>Per-request state machine</h3>
 * <pre>
 *                     register()
 *                   PENDING_ACK
 *                  /          \
 *  cancel()       /            \  ack accepted
 *                v              v
 *           CANCELLED          ACTIVE
 *           (tombstone)            \
 *                                   \  cancel()
 *                                    v
 *                                CANCELLED
 * </pre>
 * The PENDING_ACK → CANCELLED transition is the tombstone path: a Cancel arriving
 * before {@code Master.Enqueue} has been acknowledged would otherwise be silently
 * lost (Engine doesn't yet know about the request, so an immediate Cancel RPC is
 * a no-op; once Engine processes Enqueue, no further Cancel arrives and the
 * request runs to completion). The state machine lets {@code DpBatchScheduler.handleAck}
 * detect the tombstone and re-issue Cancel after the Engine has the request.
 *
 * <h3>Stale eviction</h3>
 * Normal cleanup happens through {@link #removeRequest} / {@link #remove} on
 * Cancel and on completion notification. {@link #evictStale()} is the safety net
 * for missed notifications. The threshold is set well above any plausible single
 * request lifetime ({@value #STALE_THRESHOLD_MS} ms = 10 minutes) so that the
 * eviction window is NOT mistaken for an "expected completion time" — entries
 * disappearing from the registry should normally come from explicit cleanup,
 * never from this safety net. Eviction count is exposed via
 * {@link #getEvictedCount()} for monitoring; non-zero values during normal
 * operation indicate a leak in the cleanup path.
 */
@Component
public class InflightBatchRegistry {

    /**
     * Drop entries older than this. Sized to comfortably exceed any plausible
     * single-request lifetime (single request max decode time + KV transfer +
     * margin), so this is purely a memory-leak safety net rather than a normal
     * cleanup mechanism.
     *
     * <p>10 minutes is generous for typical chat/RAG; bump per deployment if your
     * workload has long-running generations.
     */
    static final long STALE_THRESHOLD_MS = 10L * 60_000L;

    public enum RequestState {
        /** register() has been called; Enqueue ack not yet received. */
        PENDING_ACK,
        /** Enqueue accepted by Engine; request is being processed. */
        ACTIVE,
        /** Cancel received (either before or after ack). */
        CANCELLED
    }

    private final Map<Long /*batchId*/, BatchEntry> byBatch = new ConcurrentHashMap<>();
    private final Map<Long /*requestId*/, RequestEntry> byRequest = new ConcurrentHashMap<>();
    private final Map<Long /*requestId*/, AtomicReference<RequestState>> requestStates = new ConcurrentHashMap<>();

    /** Counter for evicted entries — observable via {@link #getEvictedCount()}. */
    private final AtomicLong evictedCount = new AtomicLong();

    public record BatchEntry(long batchId,
                             ServerStatus prefill,
                             List<Long> requestIds,
                             long createdAtMs) {
    }

    public record RequestEntry(long requestId,
                               long batchId,
                               ServerStatus prefill,
                               ServerStatus decode,
                               long createdAtMs) {
    }

    /** Register a freshly-Enqueued batch. All requests start in PENDING_ACK. */
    public void register(long batchId, PrefillBatch batch) {
        long now = System.currentTimeMillis();
        List<Long> requestIds = new ArrayList<>(batch.size());
        for (PendingRequest req : batch.requests()) {
            requestIds.add(req.requestId());
        }
        byBatch.put(batchId, new BatchEntry(batchId, batch.prefillTarget(), requestIds, now));
        for (PendingRequest req : batch.requests()) {
            byRequest.put(req.requestId(),
                    new RequestEntry(req.requestId(), batchId, batch.prefillTarget(), req.decode(), now));
            requestStates.put(req.requestId(), new AtomicReference<>(RequestState.PENDING_ACK));
        }
    }

    public RequestEntry lookupByRequest(long requestId) {
        return byRequest.get(requestId);
    }

    public BatchEntry lookupByBatch(long batchId) {
        return byBatch.get(batchId);
    }

    /** Current state, or {@code null} if the request is unknown / already evicted. */
    public RequestState getState(long requestId) {
        AtomicReference<RequestState> ref = requestStates.get(requestId);
        return ref == null ? null : ref.get();
    }

    /**
     * Atomically transition {@code PENDING_ACK → ACTIVE}. Returns true on success.
     * If the request is already CANCELLED (cancel-before-ack tombstone), returns
     * false — caller MUST treat this as a tombstone and skip success completion.
     */
    public boolean markActive(long requestId) {
        AtomicReference<RequestState> ref = requestStates.get(requestId);
        return ref != null && ref.compareAndSet(RequestState.PENDING_ACK, RequestState.ACTIVE);
    }

    /**
     * Atomically transition any state to CANCELLED. Returns the previous state
     * (or {@code null} if request is unknown).
     */
    public RequestState markCancelled(long requestId) {
        AtomicReference<RequestState> ref = requestStates.get(requestId);
        if (ref == null) return null;
        return ref.getAndSet(RequestState.CANCELLED);
    }

    /** Remove a single request from the registry (e.g., after successful Cancel ack). */
    public void removeRequest(long requestId) {
        RequestEntry entry = byRequest.remove(requestId);
        requestStates.remove(requestId);
        if (entry != null) {
            BatchEntry batchEntry = byBatch.get(entry.batchId());
            if (batchEntry != null) {
                // Whole batch only goes when all members are gone — cheap to keep until
                // evictStale() reaps. We don't mutate the immutable record's list.
                boolean anyLeft = batchEntry.requestIds().stream().anyMatch(byRequest::containsKey);
                if (!anyLeft) {
                    byBatch.remove(entry.batchId());
                }
            }
        }
    }

    /** Remove an entire batch (e.g., Enqueue rejected). */
    public void remove(long batchId) {
        BatchEntry entry = byBatch.remove(batchId);
        if (entry != null) {
            for (Long requestId : entry.requestIds()) {
                byRequest.remove(requestId);
                requestStates.remove(requestId);
            }
        }
    }

    public int sizeBatches() {
        return byBatch.size();
    }

    public int sizeRequests() {
        return byRequest.size();
    }

    /** Total entries evicted by the stale safety net (cumulative). For monitoring. */
    public long getEvictedCount() {
        return evictedCount.get();
    }

    /**
     * Periodic safety net for missed cleanup notifications. Should not normally
     * fire for any entry that ran a healthy request lifecycle — non-zero eviction
     * count under normal operation indicates a leak in the explicit cleanup path
     * (Cancel cascade or completion notification not arriving).
     */
    @Scheduled(fixedDelay = 30_000L)
    public void evictStale() {
        long now = System.currentTimeMillis();
        int evicted = 0;
        Iterator<Map.Entry<Long, BatchEntry>> it = byBatch.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<Long, BatchEntry> e = it.next();
            if (now - e.getValue().createdAtMs() > STALE_THRESHOLD_MS) {
                it.remove();
                for (Long requestId : e.getValue().requestIds()) {
                    byRequest.remove(requestId);
                    requestStates.remove(requestId);
                }
                evicted++;
            }
        }
        // Defensive sweep over byRequest: register() double-writes both maps, so
        // residue should be impossible, but keep this as a safety net under extreme
        // concurrency.
        Iterator<Map.Entry<Long, RequestEntry>> rit = byRequest.entrySet().iterator();
        while (rit.hasNext()) {
            Map.Entry<Long, RequestEntry> e = rit.next();
            if (now - e.getValue().createdAtMs() > STALE_THRESHOLD_MS) {
                rit.remove();
                requestStates.remove(e.getKey());
                evicted++;
            }
        }
        if (evicted > 0) {
            evictedCount.addAndGet(evicted);
            Logger.warn("InflightBatchRegistry safety-net evicted {} stale entries (>{}ms); "
                    + "this should be ZERO in steady state — investigate the explicit cleanup path",
                    evicted, STALE_THRESHOLD_MS);
        }
    }
}
