package org.flexlb.mock;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Assertion utilities for verifying three-layer inflight resource cleanup.
 *
 * <p>FlexLB has three layers of inflight tracking:
 * <ol>
 *   <li>scheduler-level batch inflight tracking</li>
 *   <li>{@link PrefillEndpoint#prefillInflightCount()} + {@link PrefillEndpoint#prefillEngineWorkCount()}
 *       — per-worker batch tracking</li>
 *   <li>{@link DecodeEndpoint#decodeInflightCount()} — per-worker decode reservation</li>
 * </ol>
 */
public final class InflightAssertions {

    private InflightAssertions() {
    }

    private static int prefillTrackedCount(PrefillEndpoint prefillEp) {
        return prefillEp.prefillInflightCount() + prefillEp.prefillEngineWorkCount();
    }

    /**
     * Assert that the PrefillEndpoint for the given ip:port has no inflight batches.
     */
    public static void assertPrefillInflightEmpty(PrefillEndpoint prefillEp) {
        int batchCount = prefillTrackedCount(prefillEp);
        assertEquals(0, batchCount,
                "PrefillEndpoint inflightBatches should be empty but has " + batchCount + " batches");
    }

    /**
     * Assert that the DecodeEndpoint for the given ip:port has no inflight requests.
     */
    public static void assertDecodeInflightEmpty(DecodeEndpoint decodeEp) {
        int count = decodeEp.decodeInflightCount();
        assertEquals(0, count,
                "DecodeEndpoint inflightRequests should be empty but has " + count + " requests");
    }

    /**
     * Assert that both prefill and decode endpoints have released all resources.
     */
    public static void assertAllResourcesReleased(PrefillEndpoint prefillEp, DecodeEndpoint decodeEp) {
        assertPrefillInflightEmpty(prefillEp);
        assertDecodeInflightEmpty(decodeEp);
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
            boolean prefillOk = prefillEp == null || prefillTrackedCount(prefillEp) == 0;
            boolean decodeOk = decodeEp == null || decodeEp.decodeInflightCount() == 0;
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
                        + " (prefill batches=" + (prefillEp != null ? prefillTrackedCount(prefillEp) : "null")
                        + ", decode inflight=" + (decodeEp != null ? decodeEp.decodeInflightCount() : "null") + ")");
    }
}
