package org.flexlb.listener;

/**
 * Exposes whether application warm-up has completed.
 */
@SuppressWarnings("BooleanMethodIsAlwaysInverted")
public interface ApplicationWarmupState {

    boolean isWarmupFinished();
}
