package org.flexlb.balance.dp;

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
}
