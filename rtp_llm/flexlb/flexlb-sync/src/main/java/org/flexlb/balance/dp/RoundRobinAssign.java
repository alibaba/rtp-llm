package org.flexlb.balance.dp;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * Positional dp_rank assignment within a single batch: the i-th request goes
 * to slot {@code i % dpSize}.
 *
 * <h3>Why positional, not cursor-based</h3>
 * Cross-pod fairness is enforced one level up by {@link GroupSelector} (the
 * RR/cache-affinity/etc. plug-point inside {@link DispatchPlanner}). A batch
 * arriving at this assigner is already destined for one specific pod, and the
 * batcher caps drain at {@code dpSize}, so positional assignment fills slots
 * 0..dpSize-1 exactly once per full batch. A persistent cursor here would
 * just add jitter.
 *
 * <h3>Partial batches (window/timeout flush)</h3>
 * When {@code batch.size() < dpSize}, slots {@code batch.size()..dpSize-1} stay
 * empty. Engine-side {@code is_fake_query} placeholder injection (V1-α plan)
 * fills those — Master does NOT generate fake queries today; that is tracked as
 * a TODO once the engine contract solidifies.
 */
@Component("rrDpAssign")
public class RoundRobinAssign implements DpAssignStrategy {

    public static final String NAME = "RR";

    @Override
    public List<RankAssignment> assign(PrefillBatch batch) {
        int dpSize = batch.dpSize();
        if (dpSize <= 0) {
            throw new IllegalArgumentException("dpSize must be > 0, got " + dpSize);
        }
        List<PendingRequest> requests = batch.requests();
        List<RankAssignment> out = new ArrayList<>(requests.size());
        for (int i = 0; i < requests.size(); i++) {
            out.add(new RankAssignment(requests.get(i), i % dpSize));
        }
        return out;
    }

    @Override
    public String name() {
        return NAME;
    }
}
