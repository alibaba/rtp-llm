package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.flexlb.balance.endpoint.EngineQueueAdmission.State.*;

class EngineQueueAdmissionTest {
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
