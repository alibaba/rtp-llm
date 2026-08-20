package org.flexlb.dao;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SchedulingMetadataTest {

    @Test
    void normalizesPriorityAndKeepsCallerExpirationUnchanged() {
        long expiresAtMs = 1_893_456_000_000L;

        SchedulingMetadata metadata = SchedulingMetadata.of(
                0, "70", expiresAtMs, 50);

        assertEquals(70, metadata.priority());
        assertEquals(SchedulingMetadata.PrioritySource.EXPLICIT, metadata.source());
        assertEquals(expiresAtMs, metadata.expiresAtMs());
    }

    @Test
    void missingPriorityUsesConfiguredDefault() {
        SchedulingMetadata metadata = SchedulingMetadata.of(
                0, null, 10_000L, 60);

        assertEquals(60, metadata.priority());
        assertEquals(SchedulingMetadata.PrioritySource.DEFAULT, metadata.source());
    }

    @Test
    void remainingLifetimeUsesTheSingleAbsoluteExpiration() {
        SchedulingMetadata metadata = SchedulingMetadata.explicit(50, 2_000L);

        assertEquals(500L, metadata.remainingMs(1_500L));
        assertFalse(metadata.expired(1_999L));
        assertTrue(metadata.expired(2_000L));
    }

    @Test
    void nonPositiveExpirationIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> SchedulingMetadata.explicit(50, 0));
    }
}
