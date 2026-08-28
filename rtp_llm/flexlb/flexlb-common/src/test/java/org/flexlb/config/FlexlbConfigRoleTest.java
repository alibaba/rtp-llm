package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FlexlbConfigRoleTest {

    @Test
    void output_token_reservation_uses_absence_for_no_cap() {
        FlexlbConfig config = new FlexlbConfig();
        assertEquals(1000L, config.effectiveMaxOutputTokensForReservation(393216L));
        assertEquals(500L, config.effectiveMaxOutputTokensForReservation(500L));

        config.getRouter().getRoles().getDecode().getKvReservation()
                .setMaxOutputTokensForEstimate(8192L);
        assertEquals(8192L, config.effectiveMaxOutputTokensForReservation(393216L));

        config.getRouter().getRoles().getDecode().getKvReservation()
                .setMaxOutputTokensForEstimate(null);
        assertEquals(393216L,
                config.effectiveMaxOutputTokensForReservation(393216L));
    }

    @Test
    void decode_reservation_is_non_negative_saturating_and_capacity_capped() {
        FlexlbConfig config = new FlexlbConfig();

        assertEquals(1500L, config.decodeKvReservationTokens(500L, 10_000L, 0L));
        assertEquals(1200L, config.decodeKvReservationTokens(500L, 10_000L, 1200L));
        assertEquals(0L, config.decodeKvReservationTokens(-1L, -1L, 0L));

        config.getRouter().getRoles().getDecode().getKvReservation()
                .setMaxOutputTokensForEstimate(null);
        assertEquals(Long.MAX_VALUE, config.decodeKvReservationTokens(
                Long.MAX_VALUE - 10L, 100L, 0L));
        assertEquals(4096L, config.decodeKvReservationTokens(
                Long.MAX_VALUE - 10L, 100L, 4096L));
    }
}
