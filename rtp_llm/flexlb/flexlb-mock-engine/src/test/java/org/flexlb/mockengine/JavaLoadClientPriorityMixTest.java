package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the PRIORITY_MIX weighted priority assignment in JavaLoadClient:
 * spec parsing, weight normalization, fixed-seed determinism and the single
 * override chain (trace explicit priority > PRIORITY_MIX > DEFAULT_PRIORITY).
 */
class JavaLoadClientPriorityMixTest {

    private static JavaLoadClient.TraceRecord rec(int idx, int priority) {
        return new JavaLoadClient.TraceRecord(idx, "rid-" + idx, "trace-" + idx, idx * 100L,
                128, 16, Collections.emptyList(), Collections.nCopies(128, 0), priority);
    }

    // ------------------------------------------------------------------
    // Parsing
    // ------------------------------------------------------------------

    @Test
    void parseReturnsNullForUnsetSpec() {
        assertNull(JavaLoadClient.PriorityMix.parse(null));
        assertNull(JavaLoadClient.PriorityMix.parse(""));
        assertNull(JavaLoadClient.PriorityMix.parse("   "));
    }

    @Test
    void parseReadsPriorityWeightPairs() {
        JavaLoadClient.PriorityMix mix =
                JavaLoadClient.PriorityMix.parse("70:10,60:15,50:50,40:15,30:10");
        assertEquals(5, mix.priorities.length);
        assertEquals(70, mix.priorities[0]);
        assertEquals(30, mix.priorities[4]);
        assertEquals(1.0, mix.cumulative[4], 1e-12);
    }

    @Test
    void parseNormalizesWeightsThatDoNotSumTo100() {
        // Weights 1:3 → 25% / 75% regardless of absolute scale.
        JavaLoadClient.PriorityMix mix = JavaLoadClient.PriorityMix.parse("70:1,30:3");
        assertEquals(0.25, mix.cumulative[0], 1e-12);
        assertEquals(1.0, mix.cumulative[1], 1e-12);
    }

    @Test
    void parseToleratesWhitespaceAroundEntries() {
        JavaLoadClient.PriorityMix mix = JavaLoadClient.PriorityMix.parse(" 70 : 10 , 50 : 90 ");
        assertEquals(70, mix.priorities[0]);
        assertEquals(50, mix.priorities[1]);
        assertEquals(0.1, mix.cumulative[0], 1e-12);
    }

    @Test
    void parseRejectsMalformedSpecs() {
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("70"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("70:10:5"));
        assertThrows(NumberFormatException.class,
                () -> JavaLoadClient.PriorityMix.parse("hi:10"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("70:-1,50:2"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("70:0,50:0"));
    }

    // ------------------------------------------------------------------
    // Assignment: determinism + proportions
    // ------------------------------------------------------------------

    @Test
    void assignmentIsDeterministicAcrossRuns() {
        List<JavaLoadClient.TraceRecord> records = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            records.add(rec(i, JavaLoadClient.PRIORITY_UNSET));
        }
        JavaLoadClient.PriorityMix mix =
                JavaLoadClient.PriorityMix.parse("70:10,60:15,50:50,40:15,30:10");

        List<JavaLoadClient.TraceRecord> first = JavaLoadClient.assignPriorities(records, mix);
        List<JavaLoadClient.TraceRecord> second = JavaLoadClient.assignPriorities(records, mix);

        assertEquals(first.size(), second.size());
        for (int i = 0; i < first.size(); i++) {
            assertEquals(first.get(i).priority, second.get(i).priority,
                    "assignment must be deterministic at index " + i);
        }
    }

    @Test
    void assignmentFollowsMixProportions() {
        List<JavaLoadClient.TraceRecord> records = new ArrayList<>();
        for (int i = 0; i < 10_000; i++) {
            records.add(rec(i, JavaLoadClient.PRIORITY_UNSET));
        }
        JavaLoadClient.PriorityMix mix = JavaLoadClient.PriorityMix.parse("70:10,50:80,30:10");

        Map<Integer, Integer> counts = new HashMap<>();
        for (JavaLoadClient.TraceRecord r : JavaLoadClient.assignPriorities(records, mix)) {
            counts.merge(r.priority, 1, Integer::sum);
        }

        assertEquals(3, counts.size());
        // 10% / 80% / 10% with generous tolerance for the fixed-seed draw.
        assertTrue(Math.abs(counts.get(70) - 1000) < 200, "p70 count=" + counts.get(70));
        assertTrue(Math.abs(counts.get(50) - 8000) < 400, "p50 count=" + counts.get(50));
        assertTrue(Math.abs(counts.get(30) - 1000) < 200, "p30 count=" + counts.get(30));
    }

    // ------------------------------------------------------------------
    // Override chain
    // ------------------------------------------------------------------

    @Test
    void explicitTracePriorityWinsOverMix() {
        List<JavaLoadClient.TraceRecord> records = List.of(
                rec(0, 99),                                // explicit trace value
                rec(1, JavaLoadClient.PRIORITY_UNSET));    // falls through to mix
        JavaLoadClient.PriorityMix mix = JavaLoadClient.PriorityMix.parse("40:1");

        List<JavaLoadClient.TraceRecord> out = JavaLoadClient.assignPriorities(records, mix);
        assertEquals(99, out.get(0).priority);
        assertEquals(40, out.get(1).priority);
    }

    @Test
    void unsetMixFallsBackToDefaultPriority() {
        List<JavaLoadClient.TraceRecord> records = List.of(
                rec(0, JavaLoadClient.PRIORITY_UNSET),
                rec(1, 70));
        List<JavaLoadClient.TraceRecord> out = JavaLoadClient.assignPriorities(records, null);
        assertEquals(JavaLoadClient.DEFAULT_PRIORITY, out.get(0).priority);
        assertEquals(70, out.get(1).priority);
    }
}
