package org.flexlb.util;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import org.flexlb.exception.FlexLBException;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    @Test
    void nullInputReportsItsCause() {
        FlexLBException error = assertThrows(FlexLBException.class,
                () -> JsonUtils.toObject((Object) null, Payload.class));
        assertTrue(error.getMessage().contains("Input must not be null"));
    }

    @Test
    void deserializerErrorsAreNotWrappedAsJsonFailures() {
        assertThrows(AssertionError.class,
                () -> JsonUtils.toObject((Object) "{}", ErrorPayload.class));
        assertThrows(AssertionError.class,
                () -> JsonUtils.toObject("{}", ErrorPayload.class));
    }

    private record Payload(String name, String absent, LocalDateTime at, Object empty) {
    }

    @JsonDeserialize(using = FatalErrorDeserializer.class)
    private static class ErrorPayload {
    }

    public static class FatalErrorDeserializer extends JsonDeserializer<ErrorPayload> {
        @Override
        public ErrorPayload deserialize(JsonParser parser, DeserializationContext context) throws IOException {
            throw new AssertionError("fatal deserializer error");
        }
    }
}
