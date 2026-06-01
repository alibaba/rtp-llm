package org.flexlb.balance.dp;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;

import java.util.List;

/**
 * Plug-point for "given this drained chunk, decide how to dispatch it".
 *
 * <p>Responsibilities of a planner:
 * <ol>
 *   <li>Pick a DP group for each batch (delegates to {@link GroupSelector}).</li>
 *   <li>Pair each request with a decode worker in the same group.</li>
 *   <li>Slice the drain into one or more batches if needed (V1: one).</li>
 *   <li>Surface unplaceable requests as {@link FailedRequest} entries.</li>
 * </ol>
 *
 * <p>This is the seam where future strategies plug in:
 * <ul>
 *   <li>Cache-affinity: pick the group with the most prefix-hit blocks.</li>
 *   <li>Length-aware bin-pack: split a heterogeneous chunk into per-rank slots
 *       sized by token count.</li>
 *   <li>Multi-group fan-out: when drain &gt; dpSize, slice across groups.</li>
 * </ul>
 *
 * <p>Implementations MUST be thread-safe (called from the per-model batcher).
 */
public interface DispatchPlanner {

    DispatchPlan plan(List<QueuedRequest> drained, DispatchContext context);

    /**
     * Select a target group for request routing. Called at request arrival
     * time to determine which group (and its prefill worker) the request
     * should be batched with.
     *
     * @return the prefill WorkerStatus representing the selected group, or {@code null}
     */
    default WorkerStatus selectPrefillWorker(String model, FlexlbConfig cfg, int dpSize) {
        return null;
    }

    /**
     * Select a decode worker in the given group. Used by the planner for
     * per-request decode assignment within a batch.
     *
     * @return an alive decode worker as ServerStatus, or {@code null}
     */
    default org.flexlb.dao.loadbalance.ServerStatus selectDecodeWorker(String model, FlexlbConfig cfg, String group) {
        return null;
    }
}
