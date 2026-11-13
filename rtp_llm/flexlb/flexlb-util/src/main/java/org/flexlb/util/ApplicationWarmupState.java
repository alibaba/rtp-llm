package org.flexlb.util;

import org.springframework.stereotype.Component;

/**
 * Exposes whether application warm-up has completed.
 */
@Component
public class ApplicationWarmupState {

    private volatile boolean warmupFinished;

    public boolean isWarmupFinished() {
        return warmupFinished;
    }

    public void setWarmupFinished(boolean warmupFinished) {
        this.warmupFinished = warmupFinished;
    }
}
