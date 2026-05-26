package org.flexlb.config;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EnvConfigOverridesTest {

    static class Sample {
        int intField = 10;
        long longField = 100L;
        double doubleField = 1.5;
        boolean booleanField = false;
        String stringField = "default";
        Mode enumField = Mode.A;

        enum Mode { A, B, C }
    }

    @Test
    void appliesIntOverrideViaCamelToUpperSnake() {
        Sample s = new Sample();
        EnvConfigOverrides.apply(s, "TEST_", Map.of("TEST_INT_FIELD", "42"));
        assertEquals(42, s.intField);
    }

    @Test
    void appliesAllSupportedPrimitivesAndString() {
        Sample s = new Sample();
        EnvConfigOverrides.apply(s, "TEST_", Map.of(
                "TEST_INT_FIELD", "7",
                "TEST_LONG_FIELD", "9999999999",
                "TEST_DOUBLE_FIELD", "3.14",
                "TEST_BOOLEAN_FIELD", "true",
                "TEST_STRING_FIELD", "hello",
                "TEST_ENUM_FIELD", "C"));
        assertEquals(7, s.intField);
        assertEquals(9999999999L, s.longField);
        assertEquals(3.14, s.doubleField, 1e-9);
        assertTrue(s.booleanField);
        assertEquals("hello", s.stringField);
        assertEquals(Sample.Mode.C, s.enumField);
    }

    @Test
    void prefixIsRequiredEnvVarsWithoutPrefixDoNotApply() {
        Sample s = new Sample();
        EnvConfigOverrides.apply(s, "DISPATCH_", Map.of(
                "INT_FIELD", "999",
                "OTHER_INT_FIELD", "555"));
        assertEquals(10, s.intField, "no DISPATCH_ prefix → not applied");
    }

    @Test
    void emptyEnvValueIsSkippedSoFieldKeepsDefault() {
        Sample s = new Sample();
        EnvConfigOverrides.apply(s, "TEST_", Map.of("TEST_STRING_FIELD", ""));
        assertEquals("default", s.stringField);
    }

    @Test
    void emptyPrefixWorksForBackwardsCompatibility() {
        // ConfigService historically used no prefix. Keep that contract working.
        Sample s = new Sample();
        EnvConfigOverrides.apply(s, "", Map.of("INT_FIELD", "42"));
        assertEquals(42, s.intField);
    }

    @Test
    void invalidNumberLeavesFieldAtDefaultAndDoesNotThrow() {
        Sample s = new Sample();
        EnvConfigOverrides.apply(s, "TEST_", Map.of("TEST_INT_FIELD", "not-a-number"));
        assertEquals(10, s.intField, "malformed env value must not crash startup");
    }

    @Test
    void unknownEnvVarsAreIgnored() {
        Sample s = new Sample();
        EnvConfigOverrides.apply(s, "TEST_", Map.of(
                "TEST_INT_FIELD", "42",
                "TEST_NO_SUCH_FIELD", "ignored"));
        assertEquals(42, s.intField);
    }

    @Test
    void booleanFalseSetsToFalseExplicitly() {
        Sample s = new Sample();
        s.booleanField = true;
        EnvConfigOverrides.apply(s, "TEST_", Map.of("TEST_BOOLEAN_FIELD", "false"));
        assertFalse(s.booleanField);
    }

    @Test
    void noMatchingEnvVarsLeavesAllDefaults() {
        Sample s = new Sample();
        EnvConfigOverrides.apply(s, "TEST_", Map.of());
        assertEquals(10, s.intField);
        assertEquals("default", s.stringField);
    }
}
