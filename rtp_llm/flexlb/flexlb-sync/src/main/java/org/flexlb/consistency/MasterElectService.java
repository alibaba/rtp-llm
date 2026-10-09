package org.flexlb.consistency;

/** Read-only leadership view used to decide local scheduling versus forwarding. */
public interface MasterElectService {

    boolean isNeedConsistency();

    boolean isMaster();
}
