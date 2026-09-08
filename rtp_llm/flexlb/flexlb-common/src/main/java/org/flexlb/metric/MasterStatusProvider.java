package org.flexlb.metric;

/**
 * Provides the current master status for metric tagging.
 * Implemented by modules that have access to master election state.
 *
 * @author saichen.sm
 */
public interface MasterStatusProvider {

    /**
     * Check if the current node is the master.
     *
     * @return true if this node is the master, false otherwise
     */
    boolean isMaster();
}
