package org.flexlb.balance.dp;

import org.flexlb.dao.master.WorkerStatus;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * No-cache group selector. Returns the first candidate — relies on the upstream
 * filter ({@code DefaultDispatchPlanner.dpEnabledPrefillCandidates}) to enforce
 * {@code isAlive} + {@code dpSize > 1}. Used as the kill-switch fallback when
 * {@code flexlb.cacheAwareSchedulingEnabled = false}, and exposable via
 * {@code dpGroupSelector = FIRST_ALIVE}.
 */
@Component("firstAliveGroupSelector")
public class FirstAliveGroupSelector implements GroupSelector {

    public static final String NAME = "FIRST_ALIVE";

    @Override
    public WorkerStatus select(List<WorkerStatus> candidates, DispatchContext ctx) {
        if (candidates == null || candidates.isEmpty()) {
            return null;
        }
        return candidates.get(0);
    }

    @Override
    public String name() {
        return NAME;
    }
}
