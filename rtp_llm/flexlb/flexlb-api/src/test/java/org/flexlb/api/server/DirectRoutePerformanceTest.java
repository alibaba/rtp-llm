package org.flexlb.api.server;

/**
 * Checks DIRECT Schedule RPC latency and process resources with the shared 750P/750D fixture.
 */
class DirectRoutePerformanceTest extends AbstractRoutePerformanceTest {
    @Override
    protected String mode() {
        return "DIRECT";
    }
}
