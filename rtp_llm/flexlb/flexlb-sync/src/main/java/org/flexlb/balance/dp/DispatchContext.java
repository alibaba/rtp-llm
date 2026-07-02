package org.flexlb.balance.dp;

import org.flexlb.config.FlexlbConfig;

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
 */
public record DispatchContext(String model,
                              int dpSize,
                              FlexlbConfig config,
                              List<QueuedRequest> requests) {
}
