package org.flexlb.mock;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Assertion utilities for verifying Prefill and Decode resource cleanup.
 */
public final class InflightAssertions {

    private InflightAssertions() {
    }

    /**
     * Wait for all inflight resources to be released, polling at the given interval.
     *
     * @param prefillEp  the prefill endpoint to check
     * @param decodeEp   the decode endpoint to check (may be null)
     * @param timeoutMs  maximum time to wait
     * @param pollMs     poll interval
     * @return true if all resources were released within the timeout
     */
    public static boolean waitForResourcesReleased(PrefillEndpoint prefillEp,
                                                    DecodeEndpoint decodeEp,
                                                    long timeoutMs, long pollMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            boolean prefillOk = prefillEp == null || prefillEp.ownershipStats().batchCount() == 0;
            boolean decodeOk = decodeEp == null || decodeEp.resourceSnapshot().reservedCount() == 0;
            if (prefillOk && decodeOk) {
                return true;
            }
            try {
                Thread.sleep(pollMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return false;
    }

    /**
     * Assert that all inflight resources are released within the given timeout.
     */
    public static void assertResourcesReleasedWithin(PrefillEndpoint prefillEp,
                                                     DecodeEndpoint decodeEp,
                                                     long timeoutMs) {
        assertTrue(waitForResourcesReleased(prefillEp, decodeEp, timeoutMs, 50),
                "Inflight resources not released within " + timeoutMs + "ms"
                        + " (prefill batches=" + (prefillEp != null ? prefillEp.ownershipStats().batchCount() : "null")
                        + ", decode inflight=" + (decodeEp != null ? decodeEp.resourceSnapshot().reservedCount() : "null") + ")");
    }
}
