package org.flexlb.master;

/**
 * Receives master identity and election events without depending on Worker monitoring.
 */
public interface MasterMetrics {
    void reportMasterNode(String host);

    void reportPrefillBalanceMasterEvent(ZkMasterEvent event);
}
