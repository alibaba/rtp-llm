package org.flexlb.balance.dp;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;

import java.util.function.ToIntFunction;

/**
 * Plug-point for worker selection in the DP batch pipeline.
 *
 * <p>Implementations MUST be thread-safe (called from per-worker batchers).
 */
public interface DispatchPlanner {

    /**
     * Select the best prefill worker for a request. Scores each candidate by
     * estimated prefill cost (accounting for cache hits) plus queue wait penalty,
     * with gating on per-worker queue size and decode availability.
     *
     * @param queueSizeByWorker returns the current batcher queue size for a worker key (ipPort)
     * @return the best prefill worker, or {@code null} if none passes gating
     */
    WorkerStatus selectPrefillWorker(String model, FlexlbConfig cfg,
                                     BalanceContext ctx,
                                     ToIntFunction<String> queueSizeByWorker);

    /**
     * Select a decode worker for a single request within a given group.
     *
     * @param ctx the request context (carries config, request details)
     * @param group the prefill worker's group — decode must be in the same group
     * @return a successful ServerStatus for the decode worker, or a failed/null
     *         ServerStatus if no suitable decode worker is available
     */
    ServerStatus selectDecodeWorker(BalanceContext ctx, String group);
}
