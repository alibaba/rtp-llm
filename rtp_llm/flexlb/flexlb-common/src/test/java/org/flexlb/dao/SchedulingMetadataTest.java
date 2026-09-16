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
        SchedulingMetadata metadata = SchedulingMetadata.of(0, "70", expiresAtMs, 50);

        assertEquals(70, metadata.priority());
        assertEquals(SchedulingMetadata.PrioritySource.EXPLICIT, metadata.source());
        assertEquals(expiresAtMs, metadata.expiresAtMs());

        SchedulingMetadata fallback = SchedulingMetadata.of(0, "invalid", expiresAtMs, 60);
        assertEquals(60, fallback.priority());
        assertEquals(SchedulingMetadata.PrioritySource.EXPLICIT, fallback.source());
        assertEquals(expiresAtMs, fallback.expiresAtMs());
    }

    @Test
    void missingPriorityUsesConfiguredDefault() {
        for (String header : new String[]{null, "", "   "}) {
            SchedulingMetadata metadata = SchedulingMetadata.of(0, header, 10_000L, 60);

            assertEquals(60, metadata.priority());
            assertEquals(SchedulingMetadata.PrioritySource.DEFAULT, metadata.source());
        }
    }

    @Test
    void remainingLifetimeUsesTheSingleAbsoluteExpiration() {
        SchedulingMetadata metadata = SchedulingMetadata.explicit(50, 2_000L);

        assertEquals(500L, metadata.remainingMs(1_500L));
        assertFalse(metadata.expired(1_999L));
        assertEquals(0L, metadata.remainingMs(2_000L));
        assertTrue(metadata.expired(2_000L));
        assertEquals(-1L, metadata.remainingMs(2_001L));
        assertTrue(metadata.expired(2_001L));
        assertEquals(2_000L, metadata.expiresAtMs());
    }

    @Test
    void nonPositiveExpirationIsRejected() {
        for (long expiresAtMs : new long[]{0L, -1L}) {
            assertThrows(IllegalArgumentException.class,
                    () -> SchedulingMetadata.explicit(50, expiresAtMs));
            assertThrows(IllegalArgumentException.class,
                    () -> SchedulingMetadata.of(0, null, expiresAtMs, 60));
        }
    }
}
