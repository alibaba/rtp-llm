package org.flexlb.consistency;

/**
 * Read-only leadership view used to decide local scheduling versus forwarding.
 */
public interface MasterElectService {

    default void start() {}

    default void offline() {}

    default void destroy() {}

    boolean isNeedConsistency();

    boolean isMaster();

    /** Returns the active ZooKeeper leadership term, or zero without cluster leadership. */
    default long getLeadershipEpoch() {
        return 0L;
    }

    default void refreshMasterHost(boolean forceSync) {}
}
