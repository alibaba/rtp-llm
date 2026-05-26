package org.flexlb.cache.core;

import java.util.List;
import java.util.Set;

/**
 * Read-only view of the DP topology — "which per-rank endpoint belongs to
 * which DP group". Exists so that future group-level load-balancing
 * strategies can consume topology without depending on the mutable
 * {@link DpGroupTopology} implementation.
 *
 * <h3>Why a separate index from EngineLocalView</h3>
 * {@link EngineLocalView} maps (group, rank-position) ↔ block sets — it knows
 * the cache <em>content</em> per rank. This view answers a different question:
 * given an arbitrary per-rank address (which the caller may have observed in
 * a {@code WorkerStatusPB.dp_status[i].ip:grpc_port} field, a metric, or a
 * future per-rank routing decision), find the externally-visible group it
 * belongs to. Cache content lookups go through {@link KvCacheManager}; identity
 * lookups go through here.
 *
 * <h3>What it does NOT do</h3>
 * <ul>
 *   <li>Does NOT change ShortestTTFT / WeightedCache behaviour. Those
 *       continue to use the union {@code cachedKeys} on the outer
 *       {@link org.flexlb.dao.master.CacheStatus} and operate at group
 *       granularity (ip:port = DP0).</li>
 *   <li>Does NOT replace {@link GlobalCacheIndex}. The block index stays
 *       group-keyed for now; if a future strategy wants per-rank prefix
 *       hits, it cross-references this view with
 *       {@link EngineLocalView#getPerRankBlocks}.</li>
 * </ul>
 *
 * <h3>Snapshot / freshness</h3>
 * The topology is refreshed every time DP0 reports a non-empty
 * {@code dp_cache[]}, which is the normal cache-status sync cadence
 * (per {@code dpCacheStatusCheckInterval}). Stale entries are evicted by the
 * service-discovery cleanup path that already removes {@link EngineLocalView}
 * and {@link GlobalCacheIndex} entries.
 */
public interface DpGroupTopologyView {

    /**
     * Group identifier (= DP0's ip:port) that owns the given per-rank ip:port,
     * or {@code null} if the rank is unknown — the worker has not been synced
     * yet, or the engine has not reported {@code dp_caches[]} for this group.
     */
    String groupOf(String rankIpPort);

    /**
     * All ranks in a group, ordered by {@code dp_rank} ascending. Empty if the
     * group is not DP-enabled or hasn't been synced yet.
     */
    List<DpRankAddress> ranksOf(String groupIpPort);

    /** All currently tracked DP-enabled group identifiers. */
    Set<String> groups();

    /** Number of ranks tracked for a group, or 0 if unknown. */
    default int rankCount(String groupIpPort) {
        return ranksOf(groupIpPort).size();
    }
}
