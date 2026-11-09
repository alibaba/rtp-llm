package org.flexlb.consistency;

/**
 * Read-only leadership view used to decide local scheduling versus forwarding.
 */
public interface MasterStatusView {
    /** Standalone nodes and the elected master schedule locally. */
    default boolean shouldForwardToMaster() {
        return isNeedConsistency() && !isMaster();
    }

    boolean isNeedConsistency();

    boolean isMaster();
}
