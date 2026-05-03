package org.flexlb.balance.dp;

import org.flexlb.config.FlexlbConfig;

/**
 * Per-drain context handed to {@link DispatchPlanner#plan}. Bundles the
 * model + dpSize + live config so the planner can stay stateless.
 */
public record DispatchContext(String model, int dpSize, FlexlbConfig config) {
}
