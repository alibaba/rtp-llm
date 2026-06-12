package org.flexlb.cache.core;

/**
 * Endpoint of one DP rank inside a pod. Carried by {@link DpGroupTopologyView}
 * to bridge per-rank addresses (dp_rank &gt; 0 endpoints, never registered in
 * service discovery) and the externally-visible group address (= DP0's ip:port).
 *
 * <p>Intentionally minimal — topology only needs identity, not cache or
 * load metadata. The latter live in {@code DpRankCacheStatus} /
 * {@code DpRankStatus} respectively and are tracked separately.
 */
public record DpRankAddress(int dpRank, String ip, int grpcPort) {

    public String ipPort() {
        return ip + ":" + grpcPort;
    }
}
