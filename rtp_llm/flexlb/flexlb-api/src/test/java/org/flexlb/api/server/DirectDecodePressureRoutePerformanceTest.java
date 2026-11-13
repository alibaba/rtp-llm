package org.flexlb.api.server;

/**
 * Checks DIRECT routing under sustained Decode capacity shortage and recovery.
 */
class DirectDecodePressureRoutePerformanceTest extends AbstractRoutePerformanceTest {
    @Override
    protected String mode() {
        return "DIRECT";
    }

    @Override
    protected Scenario scenario() {
        return Scenario.DECODE_LIMITED;
    }
}
