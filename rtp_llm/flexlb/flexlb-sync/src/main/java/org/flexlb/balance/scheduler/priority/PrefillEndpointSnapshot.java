package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.PrefillEndpoint;

/**
 * Read-only view of a prefill endpoint at snapshot time.
 *
 * <p>{@link #queueVersion} is re-checked at plan commit time; a mismatch means
 * the queue was mutated after the plan was built and the plan must be retried.
 *
 * @param endpoint        the live endpoint (used at commit time only)
 * @param queueVersion    batcher queue version at snapshot time
 * @param queueSize       pending items in the batcher queue
 * @param queueCapacity   configured max queue size (0 = unbounded)
 * @param estimatedWaitMs algorithm-estimated wait for a newly offered request
 */
public record PrefillEndpointSnapshot(
        PrefillEndpoint endpoint,
        long queueVersion,
        int queueSize,
        int queueCapacity,
        long estimatedWaitMs) {

    public static PrefillEndpointSnapshot capture(PrefillEndpoint endpoint, int queueCapacity) {
        return new PrefillEndpointSnapshot(
                endpoint,
                endpoint.getBatcher().queueVersion(),
                endpoint.getBatcher().queueSize(),
                queueCapacity,
                endpoint.getBatcher().queueWaitMs());
    }
}
