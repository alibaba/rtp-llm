package org.flexlb.api.server;

/**
 * Checks QUEUE routing under sustained Prefill capacity shortage and recovery.
 */
class QueuedPrefillPressureRoutePerformanceTest extends AbstractRoutePerformanceTest {
    @Override
    protected String mode() {
        return "QUEUE";
    }

    @Override
    protected Scenario scenario() {
        return Scenario.PREFILL_LIMITED;
    }
}
