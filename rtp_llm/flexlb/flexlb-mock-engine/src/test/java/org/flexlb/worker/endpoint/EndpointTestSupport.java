package org.flexlb.worker.endpoint;

import org.flexlb.worker.status.WorkerStatus;

/**
 * Creates test Decode endpoints and reads endpoint request counts under their state lock.
 */
public final class EndpointTestSupport {
    private EndpointTestSupport() {}

    public static java.util.concurrent.locks.ReentrantLock stateLock(WorkerEndpoint endpoint) {
        return endpoint.stateLock();
    }

    public static DecodeEndpoint decode(WorkerStatus status) {
        return new DecodeEndpoint(status, new CapacityNotifier());
    }

    public static long occupiedRequestCount(PrefillEndpoint endpoint) {
        endpoint.stateLock().lock();
        try {
            var requests =
                    (java.util.Map<?, ?>)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    endpoint, "requests");
            long unknown =
                    (long)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    endpoint, "unknownEngineRequestCount");
            return unknown > Long.MAX_VALUE - requests.size()
                    ? Long.MAX_VALUE
                    : unknown + requests.size();
        } finally {
            endpoint.stateLock().unlock();
        }
    }
}
