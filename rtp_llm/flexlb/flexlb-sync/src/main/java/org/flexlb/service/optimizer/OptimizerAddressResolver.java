package org.flexlb.service.optimizer;

import java.util.List;

public interface OptimizerAddressResolver {

    List<String> getAddresses();

    void shutdown();

    /**
     * Idempotent + retryable start. Returns true if started or already started;
     * returns false on transient failure (state rolled back, caller should retry).
     */
    default boolean start() {
        return true;
    }
}
