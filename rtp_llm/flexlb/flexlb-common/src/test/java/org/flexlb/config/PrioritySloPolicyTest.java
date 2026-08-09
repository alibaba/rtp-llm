package org.flexlb.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class PrioritySloPolicyTest {

    private static PrioritySloPolicy defaults() {
        return new PrioritySloPolicy(
                PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS,
                PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS);
    }

    // ==================== strict startup validators (F4/P0-4) ====================

    @Test
    void firstInvalidBucketEntry_reports_offending_fragment_and_accepts_valid_or_blank() {
        assertNull(PrioritySloPolicy.firstInvalidBucketEntry(
                PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS));
        assertNull(PrioritySloPolicy.firstInvalidBucketEntry(null));
        assertNull(PrioritySloPolicy.firstInvalidBucketEntry(""));
        assertNull(PrioritySloPolicy.firstInvalidBucketEntry("  "));
        assertEquals("256150", PrioritySloPolicy.firstInvalidBucketEntry("256150,1024:300"));
        assertEquals("abc:def", PrioritySloPolicy.firstInvalidBucketEntry("abc:def,1024:300"));
        assertEquals("0:100", PrioritySloPolicy.firstInvalidBucketEntry("0:100"));
    }

    @Test
    void firstInvalidMultiplierEntry_reports_offending_fragment_and_accepts_valid_or_blank() {
        assertNull(PrioritySloPolicy.firstInvalidMultiplierEntry(
                PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS));
        assertNull(PrioritySloPolicy.firstInvalidMultiplierEntry(null));
        assertNull(PrioritySloPolicy.firstInvalidMultiplierEntry(""));
        assertEquals("30:0", PrioritySloPolicy.firstInvalidMultiplierEntry("30:0,50:1.0"));
        assertEquals("x:1.0", PrioritySloPolicy.firstInvalidMultiplierEntry("x:1.0"));
    }

    // ==================== bucket boundaries (upper-bound inclusive) ====================

    @ParameterizedTest
    @CsvSource({
            "1, 150",
            "255, 150",
            "256, 150",      // boundary: inclusive upper bound
            "257, 300",
            "1024, 300",
            "1025, 600",
            "4096, 600",
            "4097, 1200",
            "16384, 1200",
            "16385, 2400",   // catch-all bucket
            "1000000, 2400",
    })
    void baseSloMs_uses_upper_bound_inclusive_buckets(long seqLen, long expectedSloMs) {
        assertEquals(expectedSloMs, defaults().baseSloMs(seqLen));
    }

    @Test
    void bucketLabel_maps_seqLen_to_bucket_tag() {
        PrioritySloPolicy policy = defaults();
        assertEquals("256", policy.bucketLabel(100));
        assertEquals("1024", policy.bucketLabel(1024));
        assertEquals("*", policy.bucketLabel(20000));
    }

    // ==================== priority multipliers (TreeMap interpolation) ====================

    @ParameterizedTest
    @CsvSource({
            "30, 2.0",
            "40, 1.5",
            "50, 1.0",
            "60, 0.75",
            "70, 0.5",
    })
    void multiplier_matches_default_table(int priority, double expected) {
        assertEquals(expected, defaults().multiplier(priority));
    }

    @ParameterizedTest
    @CsvSource({
            // Exact midpoints interpolate linearly
            "35, 1.75",  // 2.0 + 0.5*(1.5-2.0)
            "45, 1.25",  // 1.5 + 0.5*(1.0-1.5)
            "55, 0.875", // 1.0 + 0.5*(0.75-1.0)
            "65, 0.625", // 0.75 + 0.5*(0.5-0.75)
            // Below minimum anchor clamps to highest multiplier
            "0, 2.0",
            "10, 2.0",
            "29, 2.0",
            // Above maximum anchor clamps to lowest multiplier
            "71, 0.5",
            "80, 0.5",
            "100, 0.5",
    })
    void multiplier_interpolates_between_anchors_and_clamps_at_edges(int priority, double expected) {
        assertEquals(expected, defaults().multiplier(priority), 1e-9);
    }

    @ParameterizedTest
    @CsvSource({
            "256, 30, 300",    // 150 * 2.0
            "256, 70, 75",     // 150 * 0.5
            "1024, 60, 225",   // 300 * 0.75
            "4096, 40, 900",   // 600 * 1.5
            "20000, 50, 2400", // catch-all * 1.0
            "256, 60, 113",    // 150 * 0.75 = 112.5 -> rounds to 113
            // Interpolated multiplier (45 -> 1.25): 150 * 1.25 = 187.5 -> 188
            "256, 45, 188",
    })
    void requestSloMs_is_base_times_multiplier_rounded(long seqLen, int priority, long expected) {
        assertEquals(expected, defaults().requestSloMs(seqLen, priority));
    }

    // ==================== deadline ====================

    @Test
    void deadlineMs_is_arrival_plus_slo_minus_predicted_prefill() {
        assertEquals(1180L, PrioritySloPolicy.deadlineMs(1000L, 300L, 120L));
        assertEquals(950L, PrioritySloPolicy.deadlineMs(1000L, 150L, 200L));
    }

    // ==================== invalid spec fallback ====================

    @ParameterizedTest
    @ValueSource(strings = {
            "garbage",           // no key:value
            "256:abc",           // non-numeric slo
            "abc:150",           // non-numeric bound
            "256:150:extra",     // wrong arity
            "0:150",             // non-positive bound
            "256:0",             // non-positive slo
            "",                  // blank
    })
    void invalid_bucket_spec_falls_back_to_defaults(String spec) {
        PrioritySloPolicy policy = new PrioritySloPolicy(
                spec, PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS);
        assertEquals(150L, policy.baseSloMs(256));
        assertEquals(2400L, policy.baseSloMs(100000));
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "garbage",
            "30:abc",
            "abc:2.0",
            "30:0",              // non-positive multiplier
            "30:-1.5",
            "",
    })
    void invalid_multiplier_spec_falls_back_to_defaults(String spec) {
        PrioritySloPolicy policy = new PrioritySloPolicy(
                PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS, spec);
        assertEquals(2.0, policy.multiplier(30));
        assertEquals(0.5, policy.multiplier(70));
    }

    @Test
    void null_specs_fall_back_to_defaults() {
        PrioritySloPolicy policy = new PrioritySloPolicy(null, null);
        assertEquals(150L, policy.baseSloMs(1));
        assertEquals(1.0, policy.multiplier(50));
    }

    @Test
    void custom_valid_specs_are_honored() {
        PrioritySloPolicy policy = new PrioritySloPolicy("100:10,*:99", "50:3.0");
        assertEquals(10L, policy.baseSloMs(100));
        assertEquals(99L, policy.baseSloMs(101));
        assertEquals(30L, policy.requestSloMs(50, 50));
    }
}
