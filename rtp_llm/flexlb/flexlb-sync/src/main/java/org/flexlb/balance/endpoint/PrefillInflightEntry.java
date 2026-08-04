package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;

/**
 * First-layer prefill inflight entry: dispatched to the engine but not yet
 * acknowledged in any engine status report (strict inflight semantics).
 *
 * <p>Sealed so consumers can exhaustively switch over the two dispatch
 * shapes at compile time:
 * <ul>
 *   <li>{@link PrefillInflightBatch} — batch path, keyed by a real batchId</li>
 *   <li>{@link PrefillInflightRequest} — non-batch path (CostBased /
 *       ShortestTTFT direct dispatch), keyed by requestId</li>
 * </ul>
 *
 * <p>Extends {@link InflightEvictor.TtlTracked} so the first layer keeps the
 * existing TTL-eviction backstop for lost requests (network failure, engine
 * never received the dispatch).
 */
public sealed interface PrefillInflightEntry extends InflightEvictor.TtlTracked
        permits PrefillInflightBatch, PrefillInflightRequest {

    /** Predicted execution time in milliseconds for this entry. */
    long predictMs();

    /** Number of requests this entry accounts for in inflight counters. */
    int requestCount();
}
