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

/**
 * Tracks in-flight batches dispatched via Master.Enqueue so that {@code Cancel}
 * (frontend → Master → P/D) can locate which Prefill / Decode workers actually
 * hold the request.
 *
 * <p>Two indices kept in sync:
 * <ul>
 *   <li>{@code byBatch}: batchId → all requestIds + the prefill target (cancel-by-batch)</li>
 *   <li>{@code byRequest}: requestId → its batch + decode target (cancel-by-request)</li>
 * </ul>
 *
 * <h3>Eviction</h3>
 * Decode FetchResponse normally finishes a stream and could notify Master to clean up,
 * but cross-process notifications are unreliable. The {@link #evictStale()} scheduled
 * task removes entries older than {@code STALE_THRESHOLD_MS} (default 60s) to prevent
 * unbounded memory growth in the face of dropped notifications. 60s is generous —
 * even a long generation typically completes within that window.
 */
@Component
public class InflightBatchRegistry {

    /** Drop entries older than 60s. Must exceed typical longest generate time. */
    static final long STALE_THRESHOLD_MS = 60_000L;

    private final Map<Long /*batchId*/, BatchEntry> byBatch = new ConcurrentHashMap<>();
    private final Map<Long /*requestId*/, RequestEntry> byRequest = new ConcurrentHashMap<>();

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

    /** Register a freshly-Enqueued batch. */
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
        }
    }

    public RequestEntry lookupByRequest(long requestId) {
        return byRequest.get(requestId);
    }

    public BatchEntry lookupByBatch(long batchId) {
        return byBatch.get(batchId);
    }

    /** Remove a single request from the registry (e.g., after successful Cancel ack). */
    public void removeRequest(long requestId) {
        RequestEntry entry = byRequest.remove(requestId);
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
            }
        }
    }

    public int sizeBatches() {
        return byBatch.size();
    }

    public int sizeRequests() {
        return byRequest.size();
    }

    /** Periodic safety net for missed cleanup notifications. */
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
                evicted++;
            }
        }
        if (evicted > 0) {
            Logger.warn("InflightBatchRegistry evicted {} stale entries (>{}ms)",
                    evicted, STALE_THRESHOLD_MS);
        }
    }
}
