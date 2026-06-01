package org.flexlb.balance.dp;

public interface DispatchBatcher {

    void offer(QueuedRequest req);

    boolean cancelInQueue(long requestId);

    int queueSize();

    boolean isAlive();

    void shutdown();
}
