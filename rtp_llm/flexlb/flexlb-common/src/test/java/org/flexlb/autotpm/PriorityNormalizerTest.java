package org.flexlb.autotpm;

import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Assertion matrix per blueprint §二:
 * - proto=60 → 60  (legal proto value wins)
 * - proto=0, header="70" → 70  (header fallback)
 * - both absent → 50  (default)
 * - proto=99 → 50  (illegal proto → default)
 * - header non-numeric → 50  (unparseable header → default)
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
    void default_when_both_absent() {
        assertEquals(50, normalizer.normalize(0, null));
        assertEquals(50, normalizer.normalize(0, ""));
        assertEquals(50, normalizer.normalize(0, "  "));
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
        // 99 is not in the legal set
        assertEquals(50, normalizer.normalize(0, "99"));
        // negative is not > 0
        assertEquals(50, normalizer.normalize(0, "-1"));
        // 0 is not > 0
        assertEquals(50, normalizer.normalize(0, "0"));
    }
}
