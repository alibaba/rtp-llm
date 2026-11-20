package org.flexlb.api.server;

/**
 * Checks DIRECT routing under sustained Prefill capacity shortage and recovery.
 */
class DirectPrefillPressureRoutePerformanceTest extends AbstractRoutePerformanceTest {
    @Override
    protected String mode() {
        return "DIRECT";
    }

    @Override
    protected Scenario scenario() {
        return Scenario.PREFILL_LIMITED;
    }
}
