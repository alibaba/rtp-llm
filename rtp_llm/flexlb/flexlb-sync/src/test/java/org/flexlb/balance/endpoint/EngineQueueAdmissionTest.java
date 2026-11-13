package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;
import static org.flexlb.balance.endpoint.EngineQueueAdmission.State.AVAILABLE;
import static org.flexlb.balance.endpoint.EngineQueueAdmission.State.CONGESTED;
import static org.flexlb.balance.endpoint.EngineQueueAdmission.State.UNKNOWN;
import static org.junit.jupiter.api.Assertions.assertEquals;

class EngineQueueAdmissionTest {
    @Test void unobservedAdmissionsReserveHeadroomWithoutClaimingSustainedEngineOverload() {
        var policy = new EngineQueueAdmission(30, 5000);
        policy.observe(0, 0);
        assertEquals(false, policy.dispatchBlocked(30));
        assertEquals(true, policy.dispatchBlocked(31));
        assertEquals(false, policy.dispatchBlocked());
        assertEquals(AVAILABLE, policy.state());
        policy.observe(29, 20);
        assertEquals(false, policy.dispatchBlocked(1));
        assertEquals(true, policy.dispatchBlocked(2));
        policy.observe(0, 40);
        assertEquals(false, policy.dispatchBlocked(0));
        policy.observe(-1, 60);
        assertEquals(false, policy.dispatchBlocked(100));
        assertEquals(UNKNOWN, policy.state());
    }

    @Test void dispatchWatermarkDoesNotWaitForOverloadTimer() {
        var policy = new EngineQueueAdmission(10, 5000);
        assertEquals(false, policy.dispatchBlocked());
        policy.observe(0, 0); assertEquals(false, policy.dispatchBlocked());
        policy.observe(10, 1); assertEquals(false, policy.dispatchBlocked());
        policy.observe(11, 2);
        assertEquals(AVAILABLE, policy.state());
        assertEquals(true, policy.dispatchBlocked());
        policy.observe(1000, 5002); assertEquals(CONGESTED, policy.state());
        assertEquals(true, policy.dispatchBlocked());
        policy.observe(9, 5003); assertEquals(false, policy.dispatchBlocked());
        policy.observe(-1, 5004); assertEquals(false, policy.dispatchBlocked());
    }

    @Test void requiresContinuousObservedPressureAndRecoversImmediately() {
        var policy = new EngineQueueAdmission(10, 5000);
        assertEquals(UNKNOWN, policy.state());
        policy.observe(10, 0); assertEquals(AVAILABLE, policy.state());
        policy.observe(11, 1000); assertEquals(AVAILABLE, policy.state());
        policy.observe(200, 5999); assertEquals(AVAILABLE, policy.state());
        policy.observe(11, 6000); assertEquals(CONGESTED, policy.state());
        policy.observe(10, 6001); assertEquals(AVAILABLE, policy.state());
        policy.observe(11, 6002); assertEquals(AVAILABLE, policy.state());
        policy.observe(11, 11002); assertEquals(CONGESTED, policy.state());
        policy.observe(-1, 11003); assertEquals(UNKNOWN, policy.state());
        policy.observe(100, 20000); assertEquals(AVAILABLE, policy.state());
        policy.observe(100, 25000); assertEquals(CONGESTED, policy.state());
    }

    @Test void shortBurstsNeverAccumulateIntoSustainedPressure() {
        var policy = new EngineQueueAdmission(10, 5000);
        for (int i = 0; i < 1000; i++) {
            long now = i * 5000L;
            policy.observe(1000, now);
            policy.observe(1000, now + 4998);
            assertEquals(AVAILABLE, policy.state());
            policy.observe(0, now + 4999);
        }
    }

    @Test void zeroDurationUsesTheExactCountBoundary() {
        var policy = new EngineQueueAdmission(10, 0);
        policy.observe(10, 0); assertEquals(AVAILABLE, policy.state());
        policy.observe(11, 1); assertEquals(CONGESTED, policy.state());
        policy.observe(0, 2); assertEquals(AVAILABLE, policy.state());
    }
}
