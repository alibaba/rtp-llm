package org.flexlb.api.server;

/**
 * Checks priority-aware cost ordering and TOKEN_BALANCE under sustained Decode shortage.
 * The shared fixture verifies queue expiry, capacity recovery and empty resource ledgers.
 */
class TokenBalanceDecodePressureRoutePerformanceTest
        extends TokenBalanceRequestOrderRoutePerformanceTest {
    @Override
    protected Scenario scenario() {
        return Scenario.DECODE_LIMITED;
    }
}
