package org.flexlb.util;

import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.assertEquals;

class JsonUtilsTest {

    @Test
    void formattedWriterPreservesJsonPolicyWithoutChangingCompactOutput() {
        Payload payload = new Payload("test", null,
                LocalDateTime.of(2026, 9, 13, 12, 30), new Object());
        assertEquals("""
                {
                  "name" : "test",
                  "at" : "2026-09-13T12:30:00",
                  "empty" : { }
                }""", JsonUtils.toFormattedString(payload));
        assertEquals("{\"name\":\"test\",\"at\":\"2026-09-13T12:30:00\",\"empty\":{}}",
                JsonUtils.toString(payload));
    }

    private record Payload(String name, String absent, LocalDateTime at, Object empty) {
    }
}
