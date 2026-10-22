package org.flexlb.dao;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SchedulingMetadataTest {

    @Test
    void invalidCallerQosDoesNotBecomeExplicitDefaultPriority() {
        for (int proto : new int[] {0, -1, 101}) {
            for (String header : new String[] {null, "", "abc", "-1", "0", "101", "49.5"}) {
                SchedulingMetadata metadata = SchedulingMetadata.of(proto, header, 10_000L, 50);
                assertEquals(50, metadata.priority());
                assertEquals(SchedulingMetadata.PrioritySource.DEFAULT, metadata.source());
            }
        }
        for (int priority : new int[] {1, 49, 50, 100}) {
            SchedulingMetadata header = SchedulingMetadata.of(0, String.valueOf(priority), 10_000L, 50);
            assertEquals(priority, header.priority());
            assertEquals(SchedulingMetadata.PrioritySource.EXPLICIT, header.source());
            SchedulingMetadata proto = SchedulingMetadata.of(priority, "invalid", 10_000L, 50);
            assertEquals(priority, proto.priority());
            assertEquals(SchedulingMetadata.PrioritySource.EXPLICIT, proto.source());
        }
    }

    @Test
    void normalizesPriorityAndKeepsCallerExpirationUnchanged() {
        long expiresAtMs = 1_893_456_000_000L;
        SchedulingMetadata metadata = SchedulingMetadata.of(0, "70", expiresAtMs, 50);

        assertEquals(70, metadata.priority());
        assertEquals(SchedulingMetadata.PrioritySource.EXPLICIT, metadata.source());
        assertEquals(expiresAtMs, metadata.expiresAtMs());

        SchedulingMetadata fallback = SchedulingMetadata.of(0, "invalid", expiresAtMs, 60);
        assertEquals(60, fallback.priority());
        assertEquals(SchedulingMetadata.PrioritySource.DEFAULT, fallback.source());
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
