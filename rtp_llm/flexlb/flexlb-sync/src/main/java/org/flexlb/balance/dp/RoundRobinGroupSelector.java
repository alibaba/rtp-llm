package org.flexlb.balance.dp;

import org.flexlb.dao.master.WorkerStatus;
import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Round-robin across DP-enabled groups. Cursor advances ONCE PER BATCH, so the
 * fairness unit is "one full DP cycle" (≤ {@code dpSize} requests), not one
 * request — that way batching efficiency is independent of QPS.
 *
 * <p>Stability: candidates are sorted by {@code ipPort} so the cursor index
 * lands on the same physical pod across calls even if the upstream
 * {@code ConcurrentHashMap} iteration order shifts.
 */
@Component("rrGroupSelector")
public class RoundRobinGroupSelector implements GroupSelector {

    public static final String NAME = "RR";

    private final AtomicLong cursor = new AtomicLong(0);

    @Override
    public WorkerStatus select(List<WorkerStatus> candidates, BatchHint hint) {
        if (candidates == null || candidates.isEmpty()) {
            return null;
        }
        List<WorkerStatus> sorted = candidates.stream()
                .sorted(Comparator.comparing(WorkerStatus::getIpPort))
                .toList();
        int idx = (int) Math.floorMod(cursor.getAndIncrement(), (long) sorted.size());
        return sorted.get(idx);
    }

    @Override
    public String name() {
        return NAME;
    }
}
