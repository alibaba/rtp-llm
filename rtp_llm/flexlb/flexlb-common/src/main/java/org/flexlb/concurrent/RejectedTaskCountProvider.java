package org.flexlb.concurrent;

/**
 * Provides the number of tasks rejected by an executor.
 */
public interface RejectedTaskCountProvider {

    long getRejectedTaskCount();
}
