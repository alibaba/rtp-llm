package org.flexlb.util;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
        Logger.debug("hidden debug");
        Logger.info("visible info");
        assertEquals(1, appender.list.size());
        assertEquals("visible info", appender.list.getFirst().getFormattedMessage());

        backendLogger.setLevel(Level.DEBUG);
        Logger.debug("visible debug");
        assertEquals(2, appender.list.size());
        assertEquals("visible debug", appender.list.get(1).getFormattedMessage());
    }
}
