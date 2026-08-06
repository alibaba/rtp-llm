package org.flexlb.autotpm;

import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Assertion matrix per D12 (task40 revision):
 * - proto=60 → 60  (legal proto value wins)
 * - proto=0, header="70" → 70  (header fallback)
 * - both absent → 0  (NO_PRIORITY sentinel, legacy path)
 * - header="0" → 0  (explicit 0 = "not carried" sentinel)
 * - proto=99 → 50  (carried illegal proto → default)
 * - header="25" / non-numeric / negative → 50  (carried illegal header → default)
 */
class PriorityNormalizerTest {

    private PriorityNormalizer normalizer;

    @BeforeEach
    void setUp() {
        FlexlbConfig config = new FlexlbConfig();
        // defaults: priorityLevels="30,40,50,60,70", defaultPriority=50
        normalizer = new PriorityNormalizer(config);
    }

    @Test
    void proto_legal_value_wins() {
        assertEquals(60, normalizer.normalize(60, null));
        assertEquals(60, normalizer.normalize(60, "70"));  // proto takes precedence over header
    }

    @Test
    void header_fallback_when_proto_unset() {
        assertEquals(70, normalizer.normalize(0, "70"));
    }

    @Test
    void no_priority_sentinel_when_both_absent() {
        assertEquals(PriorityNormalizer.NO_PRIORITY, normalizer.normalize(0, null));
        assertEquals(PriorityNormalizer.NO_PRIORITY, normalizer.normalize(0, ""));
        assertEquals(PriorityNormalizer.NO_PRIORITY, normalizer.normalize(0, "  "));
    }

    @Test
    void explicit_zero_header_is_no_priority_sentinel() {
        assertEquals(PriorityNormalizer.NO_PRIORITY, normalizer.normalize(0, "0"));
        assertEquals(PriorityNormalizer.NO_PRIORITY, normalizer.normalize(0, " 0 "));
    }

    @Test
    void illegal_proto_falls_to_default() {
        assertEquals(50, normalizer.normalize(99, null));
        assertEquals(50, normalizer.normalize(99, "70"));  // illegal proto still trumps header
    }

    @Test
    void non_numeric_header_falls_to_default() {
        assertEquals(50, normalizer.normalize(0, "high"));
        assertEquals(50, normalizer.normalize(0, "abc123"));
    }

    @Test
    void all_legal_levels_accepted_from_proto() {
        assertEquals(30, normalizer.normalize(30, null));
        assertEquals(40, normalizer.normalize(40, null));
        assertEquals(50, normalizer.normalize(50, null));
        assertEquals(60, normalizer.normalize(60, null));
        assertEquals(70, normalizer.normalize(70, null));
    }

    @Test
    void all_legal_levels_accepted_from_header() {
        assertEquals(30, normalizer.normalize(0, "30"));
        assertEquals(40, normalizer.normalize(0, "40"));
        assertEquals(50, normalizer.normalize(0, "50"));
        assertEquals(60, normalizer.normalize(0, "60"));
        assertEquals(70, normalizer.normalize(0, "70"));
    }

    @Test
    void header_illegal_value_falls_to_default() {
        // 99 / 25 are not in the legal set
        assertEquals(50, normalizer.normalize(0, "99"));
        assertEquals(50, normalizer.normalize(0, "25"));
        // negative is carried but illegal
        assertEquals(50, normalizer.normalize(0, "-1"));
    }
}
