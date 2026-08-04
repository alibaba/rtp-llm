package org.flexlb.balance.endpoint;

/**
 * Non-batch-path inflight entry: a single request dispatched directly by a
 * strategy (CostBased / ShortestTTFT). Keyed by requestId — the engine
 * reports these with {@code batch_id=-1}.
 *
 * <p>Committed via {@code commitRequest(requestId, predictMs)} — a typed
 * single-request entry, so {@code inflightRequestCount} tracks the request
 * exactly (the legacy untyped batch shape used to miss it).
 *
 * @param requestId   request identifier (also the inflight map key)
 * @param predictMs   predicted execution time in milliseconds
 * @param createdAtMs epoch millis when the request was committed
 */
public record PrefillInflightRequest(
        long requestId, long predictMs,
        long createdAtMs) implements PrefillInflightEntry {

    @Override
    public int requestCount() {
        return 1;
    }
}
