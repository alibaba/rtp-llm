package org.flexlb.dao.cache;

import java.util.Map;
import java.util.stream.Collectors;

/**
 * Cache match details for one target worker.
 *
 * <p>{@code globalMatchBlocks} includes {@code localMatchBlocks} and additionally counts
 * remote-source hits. Both fields are the raw KVCM response; routing strategies apply
 * their own remote policy when scoring a worker.
 */
public record HostCacheMatch(
        long localMatchBlocks,
        long globalMatchBlocks) {

    public static HostCacheMatch local(long matchBlocks) {
        return new HostCacheMatch(matchBlocks, matchBlocks);
    }

    /**
     * Converts local match counts while preserving logical {@code ip:port@engineIndex} keys.
     */
    public static Map<String, HostCacheMatch> fromLocalMatches(Map<String, Integer> localMatches) {
        return localMatches.entrySet().stream().collect(Collectors.toMap(
                Map.Entry::getKey,
                entry -> local(entry.getValue())));
    }
}
