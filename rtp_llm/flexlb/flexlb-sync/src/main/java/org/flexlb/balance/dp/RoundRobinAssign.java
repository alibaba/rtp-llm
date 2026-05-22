package org.flexlb.balance.dp;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Round-robin dp_rank assignment with a persistent cursor that survives across
 * batches. Within a batch the i-th request goes to slot
 * {@code Math.floorMod(start + i, dpSize)}, where {@code start} is the cursor
 * value taken atomically before assignment.
 *
 * <h3>Why a persistent cursor</h3>
 * Under low traffic, most flushes drain on the {@code dpBatchWindowMs} timer
 * with batch size 1. A purely positional ({@code i % dpSize}) scheme would then
 * fill slot 0 every time and leave higher ranks idle, producing the empirical
 * 18× rank=0 / rank=1 imbalance observed in 5x soak. The cursor advances by
 * {@code requests.size()} per batch, so successive small batches rotate through
 * all dp ranks.
 *
 * <h3>Partial batches (window/timeout flush)</h3>
 * When {@code batch.size() < dpSize}, slots not covered by this batch are
 * picked up by the next batch starting where this one left off. Engine-side
 * {@code is_fake_query} placeholder injection (V1-α plan) covers within-batch
 * gaps when the engine requires every dp rank populated for a single
 * all-to-all step.
 */
@Component("rrDpAssign")
public class RoundRobinAssign implements DpAssignStrategy {

    public static final String NAME = "RR";

    private final AtomicInteger cursor = new AtomicInteger(0);

    @Override
    public List<RankAssignment> assign(PrefillBatch batch) {
        int dpSize = batch.dpSize();
        if (dpSize <= 0) {
            throw new IllegalArgumentException("dpSize must be > 0, got " + dpSize);
        }
        List<PendingRequest> requests = batch.requests();
        int start = cursor.getAndAdd(requests.size());
        List<RankAssignment> out = new ArrayList<>(requests.size());
        for (int i = 0; i < requests.size(); i++) {
            out.add(new RankAssignment(requests.get(i), Math.floorMod(start + i, dpSize)));
        }
        return out;
    }

    @Override
    public String name() {
        return NAME;
    }
}
