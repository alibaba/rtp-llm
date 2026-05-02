package org.flexlb.balance.dp;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * V1-α: simple Round-Robin assignment.
 *
 * <h3>Do not over-credit RR's balancing power</h3>
 * RR is length-agnostic — it cannot compress per-batch length variance. It does
 * <b>NOT</b> improve single-step GPU util, does <b>NOT</b> improve TPOT, and does
 * <b>NOT</b> shorten the latency of long requests themselves.
 *
 * <p>Its real value: it guarantees the batch boundary + DP barrier so the engine
 * stops degenerating into "pathological steady state" (long requests accumulating
 * on one rank, short requests starving the others, permanent skew). This is the
 * required scaffolding for V2 LPT / cache-aware strategies.
 *
 * <h3>Why the cursor advances across batches</h3>
 * Without it, every batch starts at rank 0, and since concurrent arrivals are
 * frequently sorted with the longest first, rank 0 would systematically take the
 * longest request in each batch. {@code getAndAdd(n)} together with {@code floorMod}
 * also handles cursor wrap-around (negative values).
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
        int n = batch.size();
        // Single getAndAdd(n) so concurrent batches receive disjoint contiguous
        // cursor ranges.
        int start = cursor.getAndAdd(n);
        List<PendingRequest> requests = batch.requests();
        List<RankAssignment> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            // floorMod handles negative values produced by cursor wrap-around.
            int rank = Math.floorMod(start + i, dpSize);
            out.add(new RankAssignment(requests.get(i), rank));
        }
        return out;
    }

    @Override
    public String name() {
        return NAME;
    }
}
