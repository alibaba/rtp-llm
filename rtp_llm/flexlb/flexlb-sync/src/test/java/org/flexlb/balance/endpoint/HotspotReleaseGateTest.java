package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Regression contracts for the integral hotspot release line. */
class HotspotReleaseGateTest {

    /** Climb requires sustained pressure AND pool spare capacity. */
    @Test
    void sustainedPressureWithSpareClimbsProportionallyToBacklog() {
        HotspotReleaseGate gate = new HotspotReleaseGate();
        long now = 0L;
        gate.govern(30L, true, now);
        // 10s of sustained pressure at waiting=30: 1.2*(30-10)*10 = 240pp, clamped.
        for (int round = 1; round <= 100; round++) {
            now += 100L;
            gate.govern(30L, true, now);
        }
        assertEquals(100.0, gate.line(), 1e-9);
    }

    /** Transient bumps below the sustained window must not open the line. */
    @Test
    void transientPressureDoesNotClimb() {
        HotspotReleaseGate gate = new HotspotReleaseGate();
        long now = 0L;
        gate.govern(30L, true, now);
        for (int round = 1; round <= 40; round++) {
            now += 100L;
            gate.govern(30L, true, now);
        }
        assertEquals(0.0, gate.line(), 1e-9, "4s of pressure is below the 5s window");
    }

    /** Sustained pressure without spare retreats instead of climbing. */
    @Test
    void saturatedPoolRetreatsInsteadOfClimbing() {
        HotspotReleaseGate gate = new HotspotReleaseGate();
        long now = 0L;
        gate.govern(30L, true, now);
        for (int round = 1; round <= 100; round++) {
            now += 100L;
            gate.govern(30L, false, now);
        }
        assertEquals(0.0, gate.line(), 1e-9, "saturation must never open the line");

        // Preload a line via a spare-capacity climb, then starve the pool:
        // the line must fall at the fixed saturation retreat rate.
        HotspotReleaseGate loaded = new HotspotReleaseGate();
        loaded.govern(30L, true, 0L);
        for (int round = 1; round <= 60; round++) {
            loaded.govern(30L, true, round * 100L);
        }
        double before = loaded.line();
        assertTrue(before > 0.0);
        now = 6_000L;
        for (int round = 1; round <= 10; round++) {
            now += 100L;
            loaded.govern(30L, false, now);
        }
        assertEquals(Math.max(0.0, before - 3.0 * 1.0), loaded.line(), 1e-9);
    }

    /** Relief decays the line toward closed, four times slower than climb. */
    @Test
    void reliefDecaysTheLine() {
        HotspotReleaseGate gate = new HotspotReleaseGate();
        long now = 0L;
        gate.govern(30L, true, now);
        for (int round = 1; round <= 60; round++) {
            now += 100L;
            gate.govern(30L, true, now);
        }
        double climbed = gate.line();
        // Relief at waiting=0 for 10s: 0.3*10*10 = 30pp decay.
        for (int round = 1; round <= 100; round++) {
            now += 100L;
            gate.govern(0L, true, now);
        }
        assertEquals(Math.max(0.0, climbed - 30.0), gate.line(), 1e-9);
    }

    /** A governor gap longer than the staleness bound restarts closed. */
    @Test
    void staleGapResetsTheLine() {
        HotspotReleaseGate gate = new HotspotReleaseGate();
        long now = 0L;
        gate.govern(30L, true, now);
        for (int round = 1; round <= 60; round++) {
            now += 100L;
            gate.govern(30L, true, now);
        }
        assertTrue(gate.line() > 0.0);
        gate.govern(0L, true, now + 5_000L);
        assertEquals(0.0, gate.line(), 1e-9);
    }

    /** The first sample only seeds the integrator and must not move the line. */
    @Test
    void firstRoundSeedsWithoutClimbing() {
        HotspotReleaseGate gate = new HotspotReleaseGate();
        gate.govern(1_000L, true, 123L);
        assertEquals(0.0, gate.line(), 1e-9);
    }

    /** A backwards clock jump is dropped instead of integrating in reverse. */
    @Test
    void backwardsClockRoundIsDropped() {
        HotspotReleaseGate gate = new HotspotReleaseGate();
        long now = 0L;
        gate.govern(30L, true, now);
        for (int round = 1; round <= 80; round++) {
            now += 100L;
            gate.govern(30L, true, now);
        }
        double climbed = gate.line();
        assertTrue(climbed > 50.0);

        // A clock step back with relief input must not run the decay in
        // reverse and raise the line; the round is simply dropped.
        gate.govern(0L, true, now - 500L);
        assertEquals(climbed, gate.line(), 1e-9);
    }
}
