package org.flexlb.api.server;

/**
 * Checks priority-aware cost ordering and TOKEN_BALANCE under sustained Prefill shortage.
 * The shared fixture verifies queue expiry, capacity recovery and empty resource ledgers.
 */
class TokenBalancePrefillPressureRoutePerformanceTest
        extends TokenBalanceRequestOrderRoutePerformanceTest {
    @Override
    protected Scenario scenario() {
        return Scenario.PREFILL_LIMITED;
    }
}
