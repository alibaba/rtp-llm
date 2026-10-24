package org.flexlb.consistency;

/** Read-only leadership view used to decide local scheduling versus forwarding. */
public interface MasterStatusView {

    boolean isNeedConsistency();

    boolean isMaster();

    /** Standalone nodes and the elected master schedule locally. */
    default boolean shouldForwardToMaster() {
        return isNeedConsistency() && !isMaster();
    }
}
