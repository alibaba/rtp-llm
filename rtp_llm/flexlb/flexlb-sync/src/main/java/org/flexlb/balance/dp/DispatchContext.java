package org.flexlb.balance.dp;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;

import java.util.List;

/**
 * Per-drain context shared by {@link DispatchPlanner#plan} and
 * {@link GroupSelector#select}. Carries everything either of them might want:
 * the model name (for affinity / logging), the dpSize (slot count for the
 * batch), the live config (so impls stay stateless), and the request chunk
 * (for selectors that score groups by request content — cache-affinity,
 * length-aware, etc.).
 *
 * <p>{@link #requests()} is the same list the planner is currently working
 * on; sharing it avoids the planner having to translate into a separate
 * "selector hint" record on every drain.
 *
 * <p>{@link #preSelectedPrefill()} is non-null when the caller (e.g.
 * {@link SloBudgetBatcher}) has already chosen the prefill target before
 * forming the batch. The planner skips its own group selection and uses
 * this worker directly.
 */
public record DispatchContext(String model,
                              int dpSize,
                              FlexlbConfig config,
                              List<QueuedRequest> requests,
                              WorkerStatus preSelectedPrefill,
                              ServerStatus preSelectedDecode) {

    public DispatchContext(String model, int dpSize, FlexlbConfig config,
                          List<QueuedRequest> requests) {
        this(model, dpSize, config, requests, null, null);
    }

    public DispatchContext(String model, int dpSize, FlexlbConfig config,
                          List<QueuedRequest> requests, WorkerStatus preSelectedPrefill) {
        this(model, dpSize, config, requests, preSelectedPrefill, null);
    }
}
