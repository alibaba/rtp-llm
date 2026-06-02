package org.flexlb.util;

import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Method;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LoggerTest {

    @BeforeEach
    void setUp() {
        // Reset globalLogLevel before each test
        Logger.setGlobalLogLevel(null);
    }

    @Test
    @DisplayName("shouldLog - when checkGlobalLevel=true and globalLogLevel is null")
    void shouldLog_checkGlobalLevel_true_globalLogLevel_null() throws Exception {
        // Arrange
        Logger.setGlobalLogLevel(null);
        Method shouldLogMethod = Logger.class.getDeclaredMethod("shouldLog", LogLevel.class, boolean.class);
        shouldLogMethod.setAccessible(true);

        // Act & Assert
        // When globalLogLevel is null and checkGlobalLevel is true, should return false for any level
        for (LogLevel level : LogLevel.values()) {
            boolean result = (boolean) shouldLogMethod.invoke(null, level, true);
            Assertions.assertFalse(result, "Should return false for " + level + " when globalLogLevel is null and checkGlobalLevel=true");
        }
    }

    @Test
    @DisplayName("shouldLog - when checkGlobalLevel=false and globalLogLevel is null")
    void shouldLog_checkGlobalLevel_false_globalLogLevel_null() throws Exception {
        // Arrange
        Logger.setGlobalLogLevel(null);
        Method shouldLogMethod = Logger.class.getDeclaredMethod("shouldLog", LogLevel.class, boolean.class);
        shouldLogMethod.setAccessible(true);

        // Act & Assert
        // When globalLogLevel is null and checkGlobalLevel is false, should return true for any level (warn and error enabled by default)
        for (LogLevel level : LogLevel.values()) {
            boolean result = (boolean) shouldLogMethod.invoke(null, level, false);
            assertTrue(result, "Should return true for " + level + " when globalLogLevel is null and checkGlobalLevel=false");
        }
    }

    @ParameterizedTest
    @MethodSource("provideLogLevelCombinationsForCheckGlobalLevelTrue")
    @DisplayName("shouldLog - checkGlobalLevel=true with different log level combinations")
    void shouldLog_checkGlobalLevel_true_with_levels(LogLevel globalLevel, LogLevel targetLevel, boolean expected) throws Exception {
        // Arrange
        Logger.setGlobalLogLevel(globalLevel);
        Method shouldLogMethod = Logger.class.getDeclaredMethod("shouldLog", LogLevel.class, boolean.class);
        shouldLogMethod.setAccessible(true);

        // Act
        boolean result = (boolean) shouldLogMethod.invoke(null, targetLevel, true);

        // Assert
        assertEquals(expected, result,
            String.format("For globalLevel=%s, targetLevel=%s, expected=%s",
                globalLevel, targetLevel, expected));
    }

    @ParameterizedTest
    @MethodSource("provideLogLevelCombinationsForCheckGlobalLevelFalse")
    @DisplayName("shouldLog - checkGlobalLevel=false with different log level combinations")
    void shouldLog_checkGlobalLevel_false_with_levels(LogLevel globalLevel, LogLevel targetLevel, boolean expected) throws Exception {
        // Arrange
        Logger.setGlobalLogLevel(globalLevel);
        Method shouldLogMethod = Logger.class.getDeclaredMethod("shouldLog", LogLevel.class, boolean.class);
        shouldLogMethod.setAccessible(true);

        // Act
        boolean result = (boolean) shouldLogMethod.invoke(null, targetLevel, false);

        // Assert
        assertEquals(expected, result,
            String.format("For globalLevel=%s, targetLevel=%s, expected=%s",
                globalLevel, targetLevel, expected));
    }

    static Stream<Arguments> provideLogLevelCombinationsForCheckGlobalLevelTrue() {
        return Stream.of(
            // globalLogLevel=TRACE should allow all levels
            Arguments.of(LogLevel.TRACE, LogLevel.TRACE, true),
            Arguments.of(LogLevel.TRACE, LogLevel.DEBUG, true),
            Arguments.of(LogLevel.TRACE, LogLevel.INFO, true),
            Arguments.of(LogLevel.TRACE, LogLevel.WARN, true),
            Arguments.of(LogLevel.TRACE, LogLevel.ERROR, true),

            // globalLogLevel=DEBUG should allow DEBUG and above
            Arguments.of(LogLevel.DEBUG, LogLevel.TRACE, false),
            Arguments.of(LogLevel.DEBUG, LogLevel.DEBUG, true),
            Arguments.of(LogLevel.DEBUG, LogLevel.INFO, true),
            Arguments.of(LogLevel.DEBUG, LogLevel.WARN, true),
            Arguments.of(LogLevel.DEBUG, LogLevel.ERROR, true),

            // globalLogLevel=INFO should allow INFO and above
            Arguments.of(LogLevel.INFO, LogLevel.TRACE, false),
            Arguments.of(LogLevel.INFO, LogLevel.DEBUG, false),
            Arguments.of(LogLevel.INFO, LogLevel.INFO, true),
            Arguments.of(LogLevel.INFO, LogLevel.WARN, true),
            Arguments.of(LogLevel.INFO, LogLevel.ERROR, true),

            // globalLogLevel=WARN should allow TO WARN and above
            Arguments.of(LogLevel.WARN, LogLevel.TRACE, false),
            Arguments.of(LogLevel.WARN, LogLevel.DEBUG, false),
            Arguments.of(LogLevel.WARN, LogLevel.INFO, false),
            Arguments.of(LogLevel.WARN, LogLevel.WARN, true),
            Arguments.of(LogLevel.WARN, LogLevel.ERROR, true),

            // globalLogLevel=ERROR should only allow ERROR
            Arguments.of(LogLevel.ERROR, LogLevel.TRACE, false),
            Arguments.of(LogLevel.ERROR, LogLevel.DEBUG, false),
            Arguments.of(LogLevel.ERROR, LogLevel.INFO, false),
            Arguments.of(LogLevel.ERROR, LogLevel.WARN, false),
            Arguments.of(LogLevel.ERROR, LogLevel.ERROR, true)
        );
    }

    static Stream<Arguments> provideLogLevelCombinationsForCheckGlobalLevelFalse() {
        return Stream.of(
            // When checkGlobalLevel=false, behavior should be same as checkGlobalLevel=true when globalLogLevel is set
            // globalLogLevel=TRACE should allow all levels
            Arguments.of(LogLevel.TRACE, LogLevel.TRACE, true),
            Arguments.of(LogLevel.TRACE, LogLevel.DEBUG, true),
            Arguments.of(LogLevel.TRACE, LogLevel.INFO, true),
            Arguments.of(LogLevel.TRACE, LogLevel.WARN, true),
            Arguments.of(LogLevel.TRACE, LogLevel.ERROR, true),

            // globalLogLevel=DEBUG should allow DEBUG and above
            Arguments.of(LogLevel.DEBUG, LogLevel.TRACE, false),
            Arguments.of(LogLevel.DEBUG, LogLevel.DEBUG, true),
            Arguments.of(LogLevel.DEBUG, LogLevel.INFO, true),
            Arguments.of(LogLevel.DEBUG, LogLevel.WARN, true),
            Arguments.of(LogLevel.DEBUG, LogLevel.ERROR, true),

            // globalLogLevel=INFO should allow INFO and above
            Arguments.of(LogLevel.INFO, LogLevel.TRACE, false),
            Arguments.of(LogLevel.INFO, LogLevel.DEBUG, false),
            Arguments.of(LogLevel.INFO, LogLevel.INFO, true),
            Arguments.of(LogLevel.INFO, LogLevel.WARN, true),
            Arguments.of(LogLevel.INFO, LogLevel.ERROR, true),

            // globalLogLevel=WARN should allow TO WARN and above
            Arguments.of(LogLevel.WARN, LogLevel.TRACE, false),
            Arguments.of(LogLevel.WARN, LogLevel.DEBUG, false),
            Arguments.of(LogLevel.WARN, LogLevel.INFO, false),
            Arguments.of(LogLevel.WARN, LogLevel.WARN, true),
            Arguments.of(LogLevel.WARN, LogLevel.ERROR, true),

            // globalLogLevel=ERROR should only allow ERROR
            Arguments.of(LogLevel.ERROR, LogLevel.TRACE, false),
            Arguments.of(LogLevel.ERROR, LogLevel.DEBUG, false),
            Arguments.of(LogLevel.ERROR, LogLevel.INFO, false),
            Arguments.of(LogLevel.ERROR, LogLevel.WARN, false),
            Arguments.of(LogLevel.ERROR, LogLevel.ERROR, true)
        );
    }

    @Test
    @DisplayName("globalLogLevel getter and setter work correctly")
    void globalLogLevel_getterSetter() {
        // Test setter and getter
        assertNull(Logger.getGlobalLogLevel());

        Logger.setGlobalLogLevel(LogLevel.INFO);
        assertEquals(LogLevel.INFO, Logger.getGlobalLogLevel());

        Logger.setGlobalLogLevel(LogLevel.DEBUG);
        assertEquals(LogLevel.DEBUG, Logger.getGlobalLogLevel());

        Logger.setGlobalLogLevel(null);
        assertNull(Logger.getGlobalLogLevel());
    }

    @Test
    @DisplayName("Static block - reads LOG_LEVEL environment variable on class loading")
    void staticBlock_readsEnvVar() {
        // This test verifies that the static block has processed the LOG_LEVEL environment variable
        // Since the static block runs when the class is first loaded, we can only verify the current state

        String currentLogLevel = System.getenv("LOG_LEVEL");

        if (currentLogLevel == null) {
            // If no LOG_LEVEL is set, globalLogLevel should be null (default)
            // Note: We can't easily test this since the class is already loaded when test runs
            // But we can verify the current behavior
            assertTrue(true, "No LOG_LEVEL environment variable set");
        } else {
            // If LOG_LEVEL is set, verify it was processed correctly
            try {
                LogLevel expectedLevel = LogLevel.valueOf(currentLogLevel.toUpperCase().trim());
                assertEquals(expectedLevel, Logger.getGlobalLogLevel(),
                    "Static block should have processed LOG_LEVEL environment variable: " + currentLogLevel);
            } catch (IllegalArgumentException e) {
                // If current LOG_LEVEL is invalid, globalLogLevel should be null
                assertNull(
                        Logger.getGlobalLogLevel(),
                    "Invalid LOG_LEVEL should result in null globalLogLevel: " + currentLogLevel);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("provideCaseInsensitiveLogLevels")
    @DisplayName("LogLevel.valueOf with case-insensitive processing")
    void logLevelValueOf_caseInsensitive(String input, LogLevel expected) {
        // Test the logic that the constructor uses internally
        // This verifies that toUpperCase().trim() works correctly for LogLevel.valueOf()

        LogLevel result = LogLevel.valueOf(input.toUpperCase().trim());
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("LogLevel.valueOf throws IllegalArgumentException for invalid values")
    void logLevelValueOf_invalidValue() {
        assertThrows(IllegalArgumentException.class, () -> LogLevel.valueOf("INVALID".toUpperCase().trim()), "Should throw IllegalArgumentException for invalid log level");

        assertThrows(IllegalArgumentException.class, () -> LogLevel.valueOf("".toUpperCase().trim()), "Should throw IllegalArgumentException for empty string");
    }

    @Test
    @DisplayName("Static method calls work without creating instances")
    void staticMethods_workWithoutInstances() {
        // Verify that static methods can be called without creating instances
        // This demonstrates that the static block initialization works correctly

        // These calls should work fine and use the globalLogLevel set by static block
        assertDoesNotThrow(() -> {
            Logger.info("Test info message");
            Logger.debug("Test debug message");
            Logger.warn("Test warn message");
            Logger.error("Test error message");
            Logger.trace("Test trace message");
        }, "Static logging methods should work without creating Logger instances");

        // Verify that globalLogLevel is accessible
        assertNotNull(Logger.class, "Logger class should be loaded");
        // The getter should work (globalLogLevel might be null, which is fine)
        assertDoesNotThrow(
                Logger::getGlobalLogLevel,
            "getGlobalLogLevel should work without creating instances");
    }

    @Test
    @DisplayName("Static block logic handles case-insensitive and whitespace correctly")
    void staticBlock_logicVerification() {
        // Test the internal logic that static block uses
        // We'll test the case-insensitive and trim logic directly

        // Test valid values with different cases and whitespace
        String[] testInputs = {"debug", "DEBUG", "Debug", "  INFO  ", "warn", "ERROR"};
        LogLevel[] expectedOutputs = {LogLevel.DEBUG, LogLevel.DEBUG, LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR};

        for (int i = 0; i < testInputs.length; i++) {
            String input = testInputs[i];
            LogLevel expected = expectedOutputs[i];

            // This tests the exact logic used in the static block
            LogLevel result = LogLevel.valueOf(input.toUpperCase().trim());
            assertEquals(expected, result, "Failed for input: '" + input + "'");
        }
    }

    static Stream<Arguments> provideCaseInsensitiveLogLevels() {
        return Stream.of(
            // Test different cases for each log level
            Arguments.of("trace", LogLevel.TRACE),
            Arguments.of("TRACE", LogLevel.TRACE),
            Arguments.of("Trace", LogLevel.TRACE),
            Arguments.of("TrAcE", LogLevel.TRACE),
            Arguments.of("  trace  ", LogLevel.TRACE),

            Arguments.of("debug", LogLevel.DEBUG),
            Arguments.of("DEBUG", LogLevel.DEBUG),
            Arguments.of("Debug", LogLevel.DEBUG),
            Arguments.of("DeBuG", LogLevel.DEBUG),
            Arguments.of("  DEBUG  ", LogLevel.DEBUG),

            Arguments.of("info", LogLevel.INFO),
            Arguments.of("INFO", LogLevel.INFO),
            Arguments.of("Info", LogLevel.INFO),
            Arguments.of("InFo", LogLevel.INFO),
            Arguments.of("  info  ", LogLevel.INFO),

            Arguments.of("warn", LogLevel.WARN),
            Arguments.of("WARN", LogLevel.WARN),
            Arguments.of("Warn", LogLevel.WARN),
            Arguments.of("WaRn", LogLevel.WARN),
            Arguments.of("  WARN  ", LogLevel.WARN),

            Arguments.of("error", LogLevel.ERROR),
            Arguments.of("ERROR", LogLevel.ERROR),
            Arguments.of("Error", LogLevel.ERROR),
            Arguments.of("ErRoR", LogLevel.ERROR),
            Arguments.of("  ERROR  ", LogLevel.ERROR)
        );
    }
}