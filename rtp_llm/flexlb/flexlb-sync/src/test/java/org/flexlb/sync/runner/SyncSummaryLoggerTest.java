package org.flexlb.sync.runner;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link SyncSummaryLogger}.
 *
 * <p>Verifies that the summary log is emitted exactly every 50 sync rounds,
 * that the log message contains the correct worker counts, and that
 * edge cases (zero workers, multiple intervals) behave as expected.
 *
 * <p>Uses a Logback {@link ListAppender} attached to the "syncLogger"
 * logger to capture log events without reflection on the static final field.
 */
class SyncSummaryLoggerTest {

    private SyncSummaryLogger syncSummaryLogger;
    private ListAppender<ILoggingEvent> appender;
    private Logger syncLogger;
    private Level originalLevel;

    @BeforeEach
    void setUp() {
        syncSummaryLogger = new SyncSummaryLogger();
        syncLogger = (Logger) LoggerFactory.getLogger("syncLogger");
        originalLevel = syncLogger.getLevel();
        syncLogger.setLevel(Level.INFO);
        appender = new ListAppender<>();
        appender.start();
        syncLogger.addAppender(appender);
    }

    @AfterEach
    void tearDown() {
        syncLogger.detachAppender(appender);
        syncLogger.setLevel(originalLevel);
    }

    @Test
    @DisplayName("First 49 rounds should not produce any summary log")
    void noSummaryLogBeforeRound50() {
        for (int i = 0; i < 49; i++) {
            syncSummaryLogger.recordSyncRound(10, 8, 2);
        }
        assertTrue(appender.list.isEmpty(),
            "Expected no summary log before round 50, but got " + appender.list.size());
    }

    @Test
    @DisplayName("Round 50 should produce exactly one summary log")
    void summaryLogAtRound50() {
        for (int i = 0; i < 50; i++) {
            syncSummaryLogger.recordSyncRound(10, 8, 2);
        }
        assertEquals(1, appender.list.size(),
            "Expected exactly 1 summary log at round 50, but got " + appender.list.size());
    }

    @Test
    @DisplayName("Summary log should contain correct total, alive, and failed counts")
    void summaryLogContainsCorrectCounts() {
        for (int i = 0; i < 50; i++) {
            syncSummaryLogger.recordSyncRound(10, 8, 2);
        }
        assertEquals(1, appender.list.size());
        String message = appender.list.get(0).getFormattedMessage();
        assertTrue(message.contains("10 workers total, 8 alive, 2 failed"),
            "Expected message to contain worker counts, but was: " + message);
    }

    @Test
    @DisplayName("Rounds 50 and 100 should each produce a summary log")
    void summaryLogAtRound50And100() {
        for (int i = 0; i < 100; i++) {
            syncSummaryLogger.recordSyncRound(10, 8, 2);
        }
        assertEquals(2, appender.list.size(),
            "Expected 2 summary logs (at rounds 50 and 100), but got " + appender.list.size());
        String firstMessage = appender.list.get(0).getFormattedMessage();
        assertTrue(firstMessage.contains("round=50"),
            "First summary should be at round 50, but was: " + firstMessage);
        String secondMessage = appender.list.get(1).getFormattedMessage();
        assertTrue(secondMessage.contains("round=100"),
            "Second summary should be at round 100, but was: " + secondMessage);
    }

    @Test
    @DisplayName("Zero workers scenario should not crash and should log correctly")
    void zeroWorkersScenario() {
        for (int i = 0; i < 50; i++) {
            syncSummaryLogger.recordSyncRound(0, 0, 0);
        }
        assertEquals(1, appender.list.size());
        String message = appender.list.get(0).getFormattedMessage();
        assertTrue(message.contains("0 workers total, 0 alive, 0 failed"),
            "Expected message to contain zero counts, but was: " + message);
    }
}
