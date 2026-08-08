package org.flexlb.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PriorityNormalizerTest {

    // ==================== not carried (117790880: default passthrough) ====================

    @Test
    void unset_proto_and_missing_header_yield_the_builtin_default() {
        assertEquals(PriorityNormalizer.DEFAULT_PRIORITY, PriorityNormalizer.normalize(0, null));
        assertEquals(PriorityNormalizer.DEFAULT_PRIORITY, PriorityNormalizer.normalize(0, ""));
        assertEquals(PriorityNormalizer.DEFAULT_PRIORITY, PriorityNormalizer.normalize(0, "   "));
    }

    @Test
    void unset_channels_use_the_configured_default_normalized_when_invalid() {
        // 117790880: requests carrying no priority participate at the
        // configured default level instead of opting out (NO_PRIORITY).
        assertEquals(60, PriorityNormalizer.normalize(0, null, 60));
        assertEquals(50, PriorityNormalizer.normalize(0, null, 0));
        assertEquals(50, PriorityNormalizer.normalize(0, null, 45));
    }

    @Test
    void hasPriority_distinguishes_the_sentinel() {
        assertFalse(PriorityNormalizer.hasPriority(PriorityNormalizer.NO_PRIORITY));
        assertTrue(PriorityNormalizer.hasPriority(30));
        assertTrue(PriorityNormalizer.hasPriority(70));
    }

    // ==================== explicit invalid values ====================

    @ParameterizedTest
    @ValueSource(ints = {-1, 0, 101, 200})
    void invalid_proto_priority_falls_back_to_default_50(int protoPriority) {
        assertEquals(50, PriorityNormalizer.normalize(protoPriority, null));
    }

    @ParameterizedTest
    @ValueSource(strings = {"0", "-1", "101", "abc", "5o", "30.0"})
    void invalid_header_value_falls_back_to_default_50(String header) {
        assertEquals(50, PriorityNormalizer.normalize(0, header));
    }

    @Test
    void invalid_configured_default_is_normalized_to_50_for_explicit_invalid_values() {
        assertEquals(50, PriorityNormalizer.normalize(-1, null, 0));
        assertEquals(50, PriorityNormalizer.normalize(-1, null, -1));
        assertEquals(50, PriorityNormalizer.normalize(101, null, 101));
    }

    @Test
    void valid_configured_default_is_used_for_explicit_invalid_values() {
        assertEquals(60, PriorityNormalizer.normalize(-1, null, 60));
        assertEquals(60, PriorityNormalizer.normalize(0, "banana", 60));
    }

    // ==================== valid values kept ====================

    @ParameterizedTest
    @ValueSource(ints = {1, 30, 45, 50, 70, 100})
    void valid_proto_priority_is_kept(int priority) {
        assertEquals(priority, PriorityNormalizer.normalize(priority, null));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 30, 45, 50, 70, 100})
    void valid_header_priority_is_kept_when_proto_unset(int priority) {
        assertEquals(priority, PriorityNormalizer.normalize(0, String.valueOf(priority)));
    }

    @Test
    void header_value_is_trimmed_before_parsing() {
        assertEquals(70, PriorityNormalizer.normalize(0, " 70 "));
    }

    // ==================== resolution order: proto > header > default ====================

    @Test
    void valid_proto_wins_over_header_and_default() {
        assertEquals(30, PriorityNormalizer.normalize(30, "70", 60));
    }

    @Test
    void invalid_proto_falls_through_to_valid_header() {
        assertEquals(70, PriorityNormalizer.normalize(-1, "70", 60));
        assertEquals(70, PriorityNormalizer.normalize(0, "70", 60));
    }

    @Test
    void invalid_proto_and_invalid_header_fall_through_to_default() {
        assertEquals(60, PriorityNormalizer.normalize(-1, "banana", 60));
        assertEquals(50, PriorityNormalizer.normalize(-7, "abc", 0));
    }

    // ==================== isValid ====================

    @Test
    void isValid_accepts_range_1_to_100() {
        assertTrue(PriorityNormalizer.isValid(1));
        assertTrue(PriorityNormalizer.isValid(30));
        assertTrue(PriorityNormalizer.isValid(45));
        assertTrue(PriorityNormalizer.isValid(50));
        assertTrue(PriorityNormalizer.isValid(70));
        assertTrue(PriorityNormalizer.isValid(80));
        assertTrue(PriorityNormalizer.isValid(100));
        assertFalse(PriorityNormalizer.isValid(0));
        assertFalse(PriorityNormalizer.isValid(-1));
        assertFalse(PriorityNormalizer.isValid(101));
    }
}
