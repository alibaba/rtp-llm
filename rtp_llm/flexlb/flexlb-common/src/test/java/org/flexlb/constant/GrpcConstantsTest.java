package org.flexlb.constant;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GrpcConstantsTest {

    @Test
    void defaultAndConfiguredLimits() {
        assertEquals(512 * 1024 * 1024, GrpcConstants.parseMaxMessageSize(null));
        assertEquals(512 * 1024 * 1024, GrpcConstants.parseMaxMessageSize(""));
        assertEquals(512 * 1024 * 1024, GrpcConstants.parseMaxMessageSize("  "));
        assertEquals(256 * 1024 * 1024, GrpcConstants.parseMaxMessageSize("256"));
        assertEquals(768 * 1024 * 1024, GrpcConstants.parseMaxMessageSize(" 768 "));
        assertEquals(1024 * 1024, GrpcConstants.parseMaxMessageSize("1"));
        assertEquals(2047 * 1024 * 1024, GrpcConstants.parseMaxMessageSize("2047"));
    }

    @Test
    void invalidLimitsFailFast() {
        for (String value : new String[]{"0", "-1", "2048", "2147483648", "1.5", "512MiB", "1_024", "+512"}) {
            assertThrows(IllegalArgumentException.class, () -> GrpcConstants.parseMaxMessageSize(value));
        }
    }
}
