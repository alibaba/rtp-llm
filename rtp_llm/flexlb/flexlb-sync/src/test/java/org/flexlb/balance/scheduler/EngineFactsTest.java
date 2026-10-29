package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineFacts;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineFactsTest {
    @ParameterizedTest
    @CsvSource({"false,0,3000", "false,1,3000", "true,0,3000", "true,1,15000"})
    void observationRequiresDecodeAndPredictionButLaterPredictionUsesCompletion(
            boolean hasDecode, long sequence, long observedVisibility) {
        var initial = new EngineFacts(false, 0L, false, 3000L, false, sequence);
        var completed = initial.observe(prefill(EngineKind.COMPLETED, 5000L), hasDecode);
        assertEquals(observedVisibility, completed.visibilityAtMs());
        assertEquals(5000L, completed.prefillCompletedAtMs());
        assertEquals(15000L, completed.predict(sequence + 1L, 4000L).visibilityAtMs());
        assertSame(completed, completed.observe(prefill(EngineKind.COMPLETED, 6000L), hasDecode));
        assertEquals(0L, initial.prefillCompletedAtMs());
    }

    @ParameterizedTest
    @CsvSource({
        "9223372036854765807,9223372036854775807",
        "9223372036854775806,9223372036854775807"
    })
    void handoffAdditionSaturatesForBothPredictionAndObservation(long completedAt, long expected) {
        var initial = new EngineFacts(false, 0L, false, 3000L, false, 1L);
        var completed = initial.observe(prefill(EngineKind.COMPLETED, completedAt), true);
        assertEquals(expected, completed.visibilityAtMs());
        assertEquals(expected, completed.predict(2L, 4000L).visibilityAtMs());
    }

    @Test
    void overdueSurvivesUnchangedObservationAndResetsOnlyOnNewFacts() {
        var expired = new EngineFacts(false, 0L, false, 3000L, false, 1L).expireVisibility();
        assertTrue(expired.visibilityOverdue());
        assertEquals(0L, expired.visibilityAtMs());
        assertSame(expired, expired.observe(prefill(EngineKind.RETIRED, 4000L), true));
        var active = expired.observe(prefill(EngineKind.ACTIVE, 4000L), true);
        assertFalse(active.visibilityOverdue());
        assertEquals(0L, active.visibilityAtMs());
        assertFalse(expired.predict(2L, 5000L).visibilityOverdue());
        assertThrows(IllegalStateException.class, () -> expired.predict(1L, 5000L));
        assertThrows(IllegalArgumentException.class, () -> expired.predict(2L, -1L));
        assertTrue(expired.visibilityOverdue());
    }

    @Test
    void decodeAcceptanceSuppressesVisibilityAcrossLaterPrefillFactsAndPrediction() {
        var initial = new EngineFacts(false, 5000L, true, 0L, false, 1L);
        assertEquals(0L, initial.predict(2L, 8000L).visibilityAtMs());
        var active = initial.observe(prefill(EngineKind.ACTIVE, 6000L), true);
        assertEquals(0L, active.visibilityAtMs());
        assertTrue(active.decodeAccepted());
    }

    private static EngineEvidence prefill(EngineKind kind, long nowMs) {
        return new EngineEvidence(null, null, RoleType.PREFILL, kind, 0L, null, nowMs);
    }
}
