package org.flexlb.balance.scheduler;

import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Tracks requests in GlobalQueue and EngineSubQueue.
 * PREFILL and PDFUSION have separate EngineSubQueue limits.
 */
@Component
public final class QueueCapacity {
    private final AtomicLong globalQueueAndEngineSubQueueCount = new AtomicLong();
    private final AtomicInteger prefillEngineSubQueueCount = new AtomicInteger();
    private final AtomicInteger pdFusionEngineSubQueueCount = new AtomicInteger();
    private final Set<CompletableFuture<?>> queuedRequestFutures = ConcurrentHashMap.newKeySet();

    public static boolean isEngineSubQueueMode(FlexlbConfig config) {
        return config.isQueue() && !config.isBatchDispatch();
    }

    public long globalQueueAndEngineSubQueueCount() {
        return globalQueueAndEngineSubQueueCount.get();
    }

    public int engineSubQueueWaiting(RoleType role) {
        return engineSubQueueCount(role).get();
    }

    public boolean isGlobalQueueEntryLimitReached(FlexlbConfig config, int globalQueueSize) {
        long queuedRequests = isEngineSubQueueMode(config)
                ? globalQueueAndEngineSubQueueCount.get()
                : globalQueueSize;
        return queuedRequests >= config.queueScheduler().getMaxQueuedRequests();
    }

    /**
     * Count a new request once toward the GlobalQueue and EngineSubQueue total.
     * Only QUEUE + NON_BATCH uses this total to decide whether to admit a request.
     */
    public void increaseTotalQueueCount(CompletableFuture<?> future) {
        if (queuedRequestFutures.add(future)) {
            globalQueueAndEngineSubQueueCount.incrementAndGet();
        }
    }

    /**
     * Stop counting a request after delivery, preemption, cancellation or failure.
     */
    public void decreaseTotalQueueCount(CompletableFuture<?> future) {
        if (future == null) {
            return;
        }
        if (queuedRequestFutures.remove(future)) {
            globalQueueAndEngineSubQueueCount.decrementAndGet();
        }
    }

    public long engineSubQueueLimit(FlexlbConfig config, int logicalEndpointCount) {
        DispatcherConfig dispatcher = config.getDispatcher();
        long totalQueueLimit = config.queueScheduler().getMaxQueuedRequests();
        long engineSubQueueLimit = (long) logicalEndpointCount * dispatcher.getMaxInflightPerPrefillWorker();
        engineSubQueueLimit = Math.min(totalQueueLimit, engineSubQueueLimit);
        return Math.min(totalQueueLimit, engineSubQueueLimit * dispatcher.getPrefillSubqueueLimitMultiplier());
    }

    /**
     * Reserves one shared waiting slot for a NON_BATCH request. The caller
     * releases the slot if enqueue fails.
     */
    public boolean tryReserveEngineSubQueueSlot(FlexlbConfig config, RoleType role, int logicalEndpointCount) {
        AtomicInteger engineSubQueueCounter = engineSubQueueCount(role);
        long limit = engineSubQueueLimit(config, logicalEndpointCount);
        while (true) {
            int current = engineSubQueueCounter.get();
            if (current >= limit) {
                return false;
            }
            if (engineSubQueueCounter.compareAndSet(current, current + 1)) {
                return true;
            }
        }
    }

    /**
     * Counts a BATCH request entering an endpoint queue without a shared limit.
     */
    public void increaseEngineSubQueueCount(RoleType role) {
        engineSubQueueCount(role).incrementAndGet();
    }

    public void releaseEngineSubQueueSlot(RoleType role) {
        releaseEngineSubQueueSlot(role, true);
    }

    public void releaseEngineSubQueueSlot(RoleType role, boolean countWasAdded) {
        if (countWasAdded) {
            engineSubQueueCount(role).getAndUpdate(current -> Math.max(0, current - 1));
        }
    }

    private AtomicInteger engineSubQueueCount(RoleType role) {
        return switch (role) {
            case PREFILL -> prefillEngineSubQueueCount;
            case PDFUSION -> pdFusionEngineSubQueueCount;
            default -> throw new IllegalArgumentException("not a Prefill routing role: " + role);
        };
    }
}
