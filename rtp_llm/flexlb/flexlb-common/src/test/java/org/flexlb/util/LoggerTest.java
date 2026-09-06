package org.flexlb.util;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LoggerTest {

    private ch.qos.logback.classic.Logger backendLogger;
    private Level originalLevel;
    private ListAppender<ILoggingEvent> appender;

    @BeforeEach
    void setUp() {
        backendLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        originalLevel = backendLogger.getLevel();
        appender = new ListAppender<>();
        appender.start();
        backendLogger.addAppender(appender);
    }

    @AfterEach
    void tearDown() {
        Logger.setLevel(null);
    }

    // ---- setLevel / getLevel sync tests ----

    @Test
    @DisplayName("setLevel(null) resets logback level to INFO")
    void setLevel_null_resetsLogbackToInfo() {
        Logger.setLevel(LogLevel.DEBUG);
        assertEquals(Level.DEBUG, logbackLevel());

        Logger.setLevel(null);

        assertEquals(LogLevel.INFO, Logger.getLevel());
        assertEquals(Level.INFO, logbackLevel(),
                "logback level should reset to INFO when passed null");
    }

    @ParameterizedTest
    @MethodSource("provideLogLevelToLogbackMappings")
    @DisplayName("setLevel syncs logback level correctly")
    void setLevel_syncsLogbackLevel(LogLevel inputLevel, Level expectedLogbackLevel) {
        Logger.setLevel(inputLevel);

        assertEquals(inputLevel, Logger.getLevel());
        assertEquals(expectedLogbackLevel, logbackLevel(),
                "logback level should match after setLevel(" + inputLevel + ")");
    }

    static Stream<Arguments> provideLogLevelToLogbackMappings() {
        return Stream.of(
                Arguments.of(LogLevel.TRACE, Level.TRACE),
                Arguments.of(LogLevel.DEBUG, Level.DEBUG),
                Arguments.of(LogLevel.INFO, Level.INFO),
                Arguments.of(LogLevel.WARN, Level.WARN),
                Arguments.of(LogLevel.ERROR, Level.ERROR)
        );
    }

    @Test
    void delegatesDebugFilteringToLoggingBackend() {
        backendLogger.setLevel(Level.INFO);
        assertFalse(Logger.isDebugEnabled());
        Logger.debug("hidden debug");
        Logger.info("visible info");
        assertEquals(1, appender.list.size());
        assertEquals("visible info", appender.list.getFirst().getFormattedMessage());

        backendLogger.setLevel(Level.DEBUG);
        assertTrue(Logger.isDebugEnabled());
        Logger.debug("visible debug");
        assertEquals(2, appender.list.size());
        assertEquals("visible debug", appender.list.get(1).getFormattedMessage());
    }

    @Test
    void formatsFixedArityOverloads() {
        backendLogger.setLevel(Level.DEBUG);

        Logger.debug("one={}", 1);
        Logger.debug("one={}, two={}", 1, 2);
        Logger.debug("{}{}{}", 1, 2, 3);
        Logger.debug("{}{}{}{}", 1, 2, 3, 4);
        Logger.debug("{}{}{}{}{}", 1, 2, 3, 4, 5);
        Logger.debug("{}{}{}{}{}{}", 1, 2, 3, 4, 5, 6);
        Logger.info("message={}", "value");

        assertEquals("one=1", appender.list.get(0).getFormattedMessage());
        assertEquals("one=1, two=2", appender.list.get(1).getFormattedMessage());
        assertEquals("123", appender.list.get(2).getFormattedMessage());
        assertEquals("1234", appender.list.get(3).getFormattedMessage());
        assertEquals("12345", appender.list.get(4).getFormattedMessage());
        assertEquals("123456", appender.list.get(5).getFormattedMessage());
        assertEquals("message=value", appender.list.get(6).getFormattedMessage());
    }

    @Test
    void doesNotExposeVariableArityMethods() {
        boolean hasVariableArityMethod = Arrays.stream(Logger.class.getDeclaredMethods())
                .anyMatch(Method::isVarArgs);

        assertFalse(hasVariableArityMethod);
    }
    @Test
    void setLevelNullRestoresInfoDefault() {
        Logger.setLevel(LogLevel.DEBUG);
        assertEquals(LogLevel.DEBUG, Logger.getLevel());

        Logger.setLevel(null);

        assertEquals(LogLevel.INFO, Logger.getLevel());
        assertEquals(Level.INFO, backendLogger.getLevel());
    }

    @Test
    void runtimeLevelRoundTripsEverySupportedLevel() {
        for (LogLevel level : LogLevel.values()) {
            Logger.setLevel(level);
            assertEquals(level, Logger.getLevel());
            assertEquals(Level.toLevel(level.name()), backendLogger.getLevel());
        }
    }

    @Test
    void isDebugEnabledFollowsRuntimeLevel() {
        Logger.setLevel(LogLevel.DEBUG);
        assertTrue(Logger.isDebugEnabled());

        Logger.setLevel(LogLevel.INFO);
        assertFalse(Logger.isDebugEnabled());
    }

    @Test
    @DisplayName("Static block - reads LOG_LEVEL environment variable on class loading")
    void staticBlock_readsEnvVar() {
        String currentLogLevel = System.getenv("LOG_LEVEL");

        if (currentLogLevel == null) {
            assertTrue(true, "No LOG_LEVEL environment variable set");
        } else {
            try {
                LogLevel expectedLevel = LogLevel.valueOf(currentLogLevel.toUpperCase().trim());
                assertEquals(expectedLevel, Logger.getLevel(),
                        "Static block should have processed LOG_LEVEL: " + currentLogLevel);
            } catch (IllegalArgumentException e) {
                assertNull(Logger.getLevel(),
                        "Invalid LOG_LEVEL should result in null: " + currentLogLevel);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("provideCaseInsensitiveLogLevels")
    @DisplayName("LogLevel.valueOf with case-insensitive processing")
    void logLevelValueOf_caseInsensitive(String input, LogLevel expected) {
        LogLevel result = LogLevel.valueOf(input.toUpperCase().trim());
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("LogLevel.valueOf throws IllegalArgumentException for invalid values")
    void logLevelValueOf_invalidValue() {
        assertThrows(IllegalArgumentException.class,
                () -> LogLevel.valueOf("INVALID".toUpperCase().trim()),
                "Should throw for invalid log level");

        assertThrows(IllegalArgumentException.class,
                () -> LogLevel.valueOf("".toUpperCase().trim()),
                "Should throw for empty string");
    }

    @Test
    @DisplayName("Static method calls work without creating instances")
    void staticMethods_workWithoutInstances() {
        assertDoesNotThrow(() -> {
            Logger.info("Test info message");
            Logger.debug("Test debug message");
            Logger.warn("Test warn message");
            Logger.error("Test error message");
            Logger.trace("Test trace message");
        }, "Static logging methods should work");

        assertNotNull(Logger.class, "Logger class should be loaded");
        assertDoesNotThrow(Logger::getLevel, "getLevel should work");
    }

    @Test
    @DisplayName("isDebugEnabled follows the runtime logback level")
    void isDebugEnabled_followsRuntimeLevel() {
        Logger.setLevel(LogLevel.INFO);
        assertFalse(Logger.isDebugEnabled());

        Logger.setLevel(LogLevel.DEBUG);
        assertTrue(Logger.isDebugEnabled());

        Logger.setLevel(LogLevel.WARN);
        assertFalse(Logger.isDebugEnabled());
    }

    @Test
    @DisplayName("isTraceEnabled follows the runtime logback level")
    void isTraceEnabled_followsRuntimeLevel() {
        Logger.setLevel(LogLevel.TRACE);
        assertTrue(Logger.isTraceEnabled());

        Logger.setLevel(LogLevel.DEBUG);
        assertFalse(Logger.isTraceEnabled());

        Logger.setLevel(LogLevel.WARN);
        assertFalse(Logger.isTraceEnabled());
    }

    @Test
    @DisplayName("Static block logic handles case-insensitive and whitespace correctly")
    void staticBlock_logicVerification() {
        String[] testInputs = {"debug", "DEBUG", "Debug", "  INFO  ", "warn", "ERROR"};
        LogLevel[] expectedOutputs = {LogLevel.DEBUG, LogLevel.DEBUG, LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR};

        for (int i = 0; i < testInputs.length; i++) {
            String input = testInputs[i];
            LogLevel expected = expectedOutputs[i];
            LogLevel result = LogLevel.valueOf(input.toUpperCase().trim());
            assertEquals(expected, result, "Failed for input: '" + input + "'");
        }
    }

    static Stream<Arguments> provideCaseInsensitiveLogLevels() {
        return Stream.of(
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

    private static ch.qos.logback.classic.Level logbackLevel() {
        ch.qos.logback.classic.Logger lbLogger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        return lbLogger.getLevel();
    }
}
