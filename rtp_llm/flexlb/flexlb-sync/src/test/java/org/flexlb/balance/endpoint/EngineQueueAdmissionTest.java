package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.flexlb.balance.endpoint.EngineQueueAdmission.State.*;

class EngineQueueAdmissionTest {
    @Test void alternatingSamplesAtTheLimitCannotReopenAFullWorkerQueue() {
        var policy = new EngineQueueAdmission(80, 5_000, 1_000, true);
        policy.observe(79, 0);
        assertEquals(false, policy.dispatchBlocked(1));
        for (int sample = 1; sample <= 60; sample++) {
            policy.observe(sample % 2 == 1 ? 80 : 79, sample * 100L);
            assertEquals(true, policy.dispatchBlocked());
        }
        assertEquals(AVAILABLE, policy.state());
    }

    @Test void reachingTheWatermarkKeepsWorkAtTheMasterUntilTheWorkerStaysEmpty() {
        var policy = new EngineQueueAdmission(80, 100, 1000, true);
        policy.observe(80, 0);
        policy.observe(79, 99);
        assertEquals(true, policy.dispatchBlocked(1));
        policy.observe(80, 100);
        policy.observe(80, 200);
        policy.observe(3, 201);
        assertEquals(true, policy.dispatchBlocked());
        policy.observe(0, 202);
        assertEquals(false, policy.dispatchBlocked(1));
        assertEquals(true, policy.dispatchBlocked(2));
        policy.observe(1, 290);
        policy.observe(0, 295);
        policy.observe(0, 394);
        assertEquals(true, policy.dispatchBlocked(2));
        policy.observe(0, 395);
        assertEquals(false, policy.dispatchBlocked(80));
    }

    @Test void ordinaryAdmissionKeepsItsConfiguredWatermarkAfterSustainedPressure() {
        var policy = new EngineQueueAdmission(80, 100, 1000, false);
        policy.observe(80, 0);
        policy.observe(80, 100);
        policy.observe(3, 200);
        assertEquals(false, policy.dispatchBlocked(77));
        assertEquals(true, policy.dispatchBlocked(78));
    }

    @Test void staleOrMissingQueueTelemetryCannotKeepRecoveryArmed() {
        var policy = new EngineQueueAdmission(80, 100, 1000, true);
        policy.observe(80, 0);
        policy.observe(80, 100);
        policy.observe(-1, 101);
        assertEquals(false, policy.dispatchBlocked(500));
        policy.observe(3, 102);
        assertEquals(false, policy.dispatchBlocked());
        policy.observe(80, 103);
        policy.observe(80, 203);
        policy.observe(3, 1204);
        assertEquals(false, policy.dispatchBlocked());
    }

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

    @Test void rescueRequiresContinuousFreshFullObservationsAndHasARecoveryBand() {
        var policy = new EngineQueueAdmission(80, 5000, 1000);
        policy.observe(80, 0);
        policy.observe(80, 5000);
        assertEquals(false, policy.freshPressure(true, 5000, Long.MIN_VALUE));
        for (long ms = 6000; ms <= 10000; ms += 1000) { policy.observe(80, ms); }
        assertEquals(true, policy.freshPressure(true, 10000, Long.MIN_VALUE));
        assertEquals(false, policy.freshPressure(true, 11001, Long.MIN_VALUE));
        assertEquals(false, policy.freshPressure(true, 10001, 10000));
        policy.observe(78, 10500);
        assertEquals(true, policy.freshPressure(true, 10500, Long.MIN_VALUE));
        policy.observe(60, 10600);
        assertEquals(false, policy.freshPressure(true, 10600, Long.MIN_VALUE));
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
