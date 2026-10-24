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
            }
        }
        for (int priority : new int[] {1, 49, 50, 100}) {
            SchedulingMetadata header = SchedulingMetadata.of(0, String.valueOf(priority), 10_000L, 50);
            assertEquals(priority, header.priority());
            SchedulingMetadata proto = SchedulingMetadata.of(priority, "invalid", 10_000L, 50);
            assertEquals(priority, proto.priority());
        }
    }

    @Test
    void normalizesPriorityAndKeepsCallerExpirationUnchanged() {
        long expiresAtMs = 1_893_456_000_000L;
        SchedulingMetadata metadata = SchedulingMetadata.of(0, "70", expiresAtMs, 50);

        assertEquals(70, metadata.priority());
        assertEquals(expiresAtMs, metadata.expiresAtMs());

        SchedulingMetadata fallback = SchedulingMetadata.of(0, "invalid", expiresAtMs, 60);
        assertEquals(60, fallback.priority());
        assertEquals(expiresAtMs, fallback.expiresAtMs());
    }

    @Test
    void missingPriorityUsesConfiguredDefault() {
        for (String header : new String[]{null, "", "   "}) {
            SchedulingMetadata metadata = SchedulingMetadata.of(0, header, 10_000L, 60);

            assertEquals(60, metadata.priority());
        }
    }

    @Test
    void expirationUsesTheSingleAbsoluteDeadline() {
        SchedulingMetadata metadata = SchedulingMetadata.explicit(50, 2_000L);

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
