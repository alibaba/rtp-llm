package org.flexlb.util;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
        backendLogger.detachAppender(appender);
        backendLogger.setLevel(originalLevel);
        appender.stop();
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
}
