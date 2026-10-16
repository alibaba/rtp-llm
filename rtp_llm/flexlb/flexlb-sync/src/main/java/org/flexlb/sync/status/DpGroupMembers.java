package org.flexlb.sync.status;

import java.util.HashSet;
import java.util.Set;

/**
 * Tracks the routable Decode ranks reported for one DP master address.
 */
final class DpGroupMembers {
    private final Set<Long> ranks = new HashSet<>();
    private long dpSize;
    private boolean consistentRanks = true;

    void add(long size, long rank) {
        if (dpSize == 0) {
            dpSize = size;
        }
        if (size != dpSize || rank < 0 || rank >= size || !ranks.add(rank)) {
            consistentRanks = false;
        }
    }

    boolean hasAllRanks() {
        return consistentRanks && ranks.size() == dpSize;
    }
}
