package org.flexlb.balance.scheduler;

/** How the scheduler should finish work that was accepted before shutdown. */
public enum ShutdownMode {
    /** Stop intake and let already accepted queue work drain until the deadline. */
    DRAIN,
    /** Stop selecting new work immediately and settle the remaining work. */
    IMMEDIATE
}
