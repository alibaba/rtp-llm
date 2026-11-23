package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Regression contracts for the integral hotspot release line, including the
 * fleet-relative setpoint the governor passes each round. */
class HotspotReleaseGateTest {

    private static final long SETPOINT = 10L;
    private static final long SUSTAINED_MS = 5_000L;

    /** Climb requires sustained pressure AND pool spare capacity. */
    @Test
    void sustainedPressureWithSpareClimbsProportionallyToBacklog() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        long now = 0L;
        gate.govern(30L, SETPOINT, true, now);
        // 10s of sustained pressure at waiting=30: 1.2*(30-10)*10 = 240pp, clamped.
        for (int round = 1; round <= 100; round++) {
            now += 100L;
            gate.govern(30L, SETPOINT, true, now);
        }
        assertEquals(100.0, gate.line(), 1e-9);
    }

    /** Transient bumps below the sustained window must not open the line. */
    @Test
    void transientPressureDoesNotClimb() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        long now = 0L;
        gate.govern(30L, SETPOINT, true, now);
        for (int round = 1; round <= 40; round++) {
            now += 100L;
            gate.govern(30L, SETPOINT, true, now);
        }
        assertEquals(0.0, gate.line(), 1e-9, "4s of pressure is below the 5s window");
    }

    /** Sustained pressure without spare retreats instead of climbing. */
    @Test
    void saturatedPoolRetreatsInsteadOfClimbing() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        long now = 0L;
        gate.govern(30L, SETPOINT, true, now);
        for (int round = 1; round <= 100; round++) {
            now += 100L;
            gate.govern(30L, SETPOINT, false, now);
        }
        assertEquals(0.0, gate.line(), 1e-9, "saturation must never open the line");

        // Preload a line via a spare-capacity climb, then starve the pool:
        // the line must fall at the fixed saturation retreat rate.
        HotspotReleaseGate loaded = new HotspotReleaseGate(SUSTAINED_MS);
        loaded.govern(30L, SETPOINT, true, 0L);
        for (int round = 1; round <= 60; round++) {
            loaded.govern(30L, SETPOINT, true, round * 100L);
        }
        double before = loaded.line();
        assertTrue(before > 0.0);
        now = 6_000L;
        for (int round = 1; round <= 10; round++) {
            now += 100L;
            loaded.govern(30L, SETPOINT, false, now);
        }
        assertEquals(Math.max(0.0, before - 3.0 * 1.0), loaded.line(), 1e-9);
    }

    /** Relief decays the line toward closed, four times slower than climb. */
    @Test
    void reliefDecaysTheLine() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        long now = 0L;
        gate.govern(30L, SETPOINT, true, now);
        for (int round = 1; round <= 60; round++) {
            now += 100L;
            gate.govern(30L, SETPOINT, true, now);
        }
        double climbed = gate.line();
        // Relief at waiting=0 for 10s: 0.3*10*10 = 30pp decay.
        for (int round = 1; round <= 100; round++) {
            now += 100L;
            gate.govern(0L, SETPOINT, true, now);
        }
        assertEquals(Math.max(0.0, climbed - 30.0), gate.line(), 1e-9);
    }

    /** A governor gap longer than the staleness bound restarts closed. */
    @Test
    void staleGapResetsTheLine() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        long now = 0L;
        gate.govern(30L, SETPOINT, true, now);
        for (int round = 1; round <= 60; round++) {
            now += 100L;
            gate.govern(30L, SETPOINT, true, now);
        }
        assertTrue(gate.line() > 0.0);
        gate.govern(0L, SETPOINT, true, now + 5_000L);
        assertEquals(0.0, gate.line(), 1e-9);
    }

    /** The first sample only seeds the integrator and must not move the line. */
    @Test
    void firstRoundSeedsWithoutClimbing() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        gate.govern(1_000L, SETPOINT, true, 123L);
        assertEquals(0.0, gate.line(), 1e-9);
    }

    /** A backwards clock jump is dropped instead of integrating in reverse. */
    @Test
    void backwardsClockRoundIsDropped() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        long now = 0L;
        gate.govern(30L, SETPOINT, true, now);
        for (int round = 1; round <= 80; round++) {
            now += 100L;
            gate.govern(30L, SETPOINT, true, now);
        }
        double climbed = gate.line();
        assertTrue(climbed > 50.0);

        // A clock step back with relief input must not run the decay in
        // reverse and raise the line; the round is simply dropped.
        gate.govern(0L, SETPOINT, true, now - 500L);
        assertEquals(climbed, gate.line(), 1e-9);
    }

    /**
     * Fleet-scaled setpoint: a large-batch engine whose healthy queue (40) is
     * below its own median-scaled setpoint (45) never opens the line even
     * under sustained pressure — the fleet-relative zero point, not the
     * absolute floor, decides congestion.
     */
    @Test
    void fleetScaledSetpointKeepsAHealthyBusyQueuePinned() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        long now = 0L;
        gate.govern(40L, 45L, true, now);
        for (int round = 1; round <= 100; round++) {
            now += 100L;
            gate.govern(40L, 45L, true, now);
        }
        assertEquals(0.0, gate.line(), 1e-9,
                "a busy-but-balanced fleet must stay pinned");
    }

    /** The setpoint floor keeps a shallow-queue hotspot (12 over floor 10) actionable. */
    @Test
    void setpointFloorStillOpensAShallowQueueHotspot() {
        HotspotReleaseGate gate = new HotspotReleaseGate(SUSTAINED_MS);
        long now = 0L;
        gate.govern(12L, SETPOINT, true, now);
        for (int round = 1; round <= 60; round++) {
            now += 100L;
            gate.govern(12L, SETPOINT, true, now);
        }
        // 1s of climbing past the 5s window: 1.2*(12-10)*1 = 2.4pp.
        assertEquals(1.2 * (12L - SETPOINT) * 1.0, gate.line(), 1e-9);
    }

    /** The sustained window follows the configured value, not a fixed 5s. */
    @Test
    void sustainedWindowIsConfigurable() {
        HotspotReleaseGate gate = new HotspotReleaseGate(2_000L);
        long now = 0L;
        gate.govern(30L, SETPOINT, true, now);
        for (int round = 1; round <= 19; round++) {
            now += 100L;
            gate.govern(30L, SETPOINT, true, now);
        }
        assertEquals(0.0, gate.line(), 1e-9, "1.9s is below the 2s window");
        for (int round = 20; round <= 30; round++) {
            now += 100L;
            gate.govern(30L, SETPOINT, true, now);
        }
        assertEquals(1.2 * (30L - SETPOINT) * 1.0, gate.line(), 1e-9,
                "one second of climbing after the 2s window");
    }
}
