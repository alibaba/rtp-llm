package org.flexlb.api.server;

/**
 * Checks QUEUE Schedule RPC latency and process resources with the shared 750P/750D fixture.
 */
class QueuedRoutePerformanceTest extends AbstractRoutePerformanceTest {
    @Override
    protected String mode() {
        return "QUEUE";
    }
}
