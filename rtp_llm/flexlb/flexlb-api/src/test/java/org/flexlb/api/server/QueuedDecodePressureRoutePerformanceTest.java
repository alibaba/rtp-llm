package org.flexlb.api.server;

/**
 * Checks QUEUE routing under sustained Decode capacity shortage and recovery.
 */
class QueuedDecodePressureRoutePerformanceTest extends AbstractRoutePerformanceTest {
    @Override
    protected String mode() {
        return "QUEUE";
    }

    @Override
    protected Scenario scenario() {
        return Scenario.DECODE_LIMITED;
    }
}
