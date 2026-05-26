package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Mutable implementation of {@link DpGroupTopologyView}. Built from DP0's
 * {@code CacheStatusPB.dp_cache[]} via {@code DefaultCacheAwareService} on
 * every cache-status sync; cleaned up by the same service-discovery hook
 * that prunes {@link EngineLocalView} / {@link GlobalCacheIndex}.
 *
 * <p>Two indices kept in lock-step:
 * <ul>
 *   <li>{@code groupRanks}: groupIpPort → ordered list of per-rank addresses
 *       (forward lookup, drives {@link #ranksOf}).</li>
 *   <li>{@code rankToGroup}: rankIpPort → groupIpPort (reverse lookup, drives
 *       {@link #groupOf}).</li>
 * </ul>
 * Updates are atomic per group: the new rank list replaces the old, and the
 * reverse map is patched (remove old keys, add new keys) under the same
 * critical section so that {@link #groupOf} never sees a stale rank pointing
 * at a vanished group.
 */
@Slf4j
@Component
public class DpGroupTopology implements DpGroupTopologyView {

    /** groupIpPort -> ordered per-rank addresses (index = dp_rank). */
    private final ConcurrentHashMap<String, List<DpRankAddress>> groupRanks = new ConcurrentHashMap<>();

    /** rankIpPort -> groupIpPort. */
    private final ConcurrentHashMap<String, String> rankToGroup = new ConcurrentHashMap<>();

    /**
     * Replace the rank list for a group. Empty / null inputs are treated as
     * "no DP" and skipped — they do NOT clear an existing entry, since an
     * occasional empty dp_cache[] (a probe race, an aggregation bug) should
     * not cause the topology to forget a known group.
     */
    public void update(String groupIpPort, List<DpRankAddress> ranks) {
        if (groupIpPort == null || ranks == null || ranks.isEmpty()) {
            return;
        }
        // Defensive copy so callers can't mutate the stored snapshot.
        List<DpRankAddress> snapshot = List.copyOf(ranks);
        synchronized (lockFor(groupIpPort)) {
            List<DpRankAddress> previous = groupRanks.put(groupIpPort, snapshot);
            if (previous != null) {
                for (DpRankAddress old : previous) {
                    // Only remove if it still points at this group — under a
                    // re-key (unlikely but defensive), the new owner has already
                    // been recorded and we must not undo that.
                    rankToGroup.remove(old.ipPort(), groupIpPort);
                }
            }
            for (DpRankAddress r : snapshot) {
                rankToGroup.put(r.ipPort(), groupIpPort);
            }
        }
    }

    /**
     * Drop a group from the topology. Called from the service-discovery
     * cleanup path when the group's worker disappears.
     */
    public void remove(String groupIpPort) {
        if (groupIpPort == null) {
            return;
        }
        synchronized (lockFor(groupIpPort)) {
            List<DpRankAddress> ranks = groupRanks.remove(groupIpPort);
            if (ranks != null) {
                for (DpRankAddress r : ranks) {
                    rankToGroup.remove(r.ipPort(), groupIpPort);
                }
            }
        }
    }

    @Override
    public String groupOf(String rankIpPort) {
        return rankIpPort == null ? null : rankToGroup.get(rankIpPort);
    }

    @Override
    public List<DpRankAddress> ranksOf(String groupIpPort) {
        if (groupIpPort == null) {
            return Collections.emptyList();
        }
        List<DpRankAddress> ranks = groupRanks.get(groupIpPort);
        return ranks == null ? Collections.emptyList() : ranks;
    }

    @Override
    public Set<String> groups() {
        return groupRanks.keySet();
    }

    /** Test/observability: total reverse-map size. */
    public int totalRankMappings() {
        return rankToGroup.size();
    }

    /** Drop everything. Test hook + lifecycle reset. */
    public void clear() {
        groupRanks.clear();
        rankToGroup.clear();
    }

    /**
     * String interning gives us a per-group monitor without keeping a
     * dedicated lock map. Group keys are bounded (one per pod) and live for
     * the deployment's lifetime, so interning is safe here.
     */
    private static Object lockFor(String groupIpPort) {
        return groupIpPort.intern();
    }
}
