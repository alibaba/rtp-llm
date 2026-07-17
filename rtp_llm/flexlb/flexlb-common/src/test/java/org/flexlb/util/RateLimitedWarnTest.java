package org.flexlb.util;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the limiter every failure-storm hot path leans on (FE outage, master unreachable,
 * discovery gap): first call emits, the window suppresses, and the suppressed count rides on the
 * next line so outage magnitude stays visible. The clock is injected because the semantics are
 * defined in nanoTime deltas — and because {@link System#nanoTime()} has an arbitrary,
 * possibly-negative origin the implementation must not bake in "starts at zero".
 */
class RateLimitedWarnTest {

    private static final String MARKER = "rate-limited-warn-test";

    private ListAppender<ILoggingEvent> appender;
    private ch.qos.logback.classic.Logger flexlbLogger;

    @BeforeEach
    void setUp() {
        Logger.setGlobalLogLevel(null);
        flexlbLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        appender = new ListAppender<>();
        appender.start();
        flexlbLogger.addAppender(appender);
    }

    @AfterEach
    void tearDown() {
        flexlbLogger.detachAppender(appender);
    }

    private List<String> emitted() {
        return appender.list.stream()
                .map(ILoggingEvent::getFormattedMessage)
                .filter(m -> m.contains(MARKER))
                .collect(Collectors.toList());
    }

    @Test
    void firstCallEmitsEvenWhenTheNanoClockOriginIsNegative() {
        // A negative nanoTime is JVM-legal; a limiter whose "last warned" state started at zero
        // would see now - 0 < interval forever and silence every operationally critical WARN.
        AtomicLong clock = new AtomicLong(-TimeUnit.HOURS.toNanos(5));
        RateLimitedWarn warn = new RateLimitedWarn(1, TimeUnit.SECONDS, clock::get);

        warn.warn(MARKER + " first");

        assertEquals(1, emitted().size(), "the first call must always emit, whatever the clock origin");
        assertTrue(emitted().get(0).contains("suppressed=0"));
    }

    @Test
    void callsInsideTheWindowAreSuppressedAndRideOnTheNextEmission() {
        AtomicLong clock = new AtomicLong(0);
        RateLimitedWarn warn = new RateLimitedWarn(1, TimeUnit.SECONDS, clock::get);

        warn.warn(MARKER + " emit");
        warn.warn(MARKER + " swallowed");
        warn.warn(MARKER + " swallowed");
        warn.warn(MARKER + " swallowed");
        assertEquals(1, emitted().size(), "calls inside the window must not emit");

        clock.addAndGet(TimeUnit.SECONDS.toNanos(1));
        warn.warn(MARKER + " next");

        assertEquals(2, emitted().size());
        assertTrue(emitted().get(1).contains("suppressed=3"),
                "the next emission must carry how many calls the window swallowed");
    }

    @Test
    void suppressedCounterResetsAfterEachEmission() {
        AtomicLong clock = new AtomicLong(0);
        RateLimitedWarn warn = new RateLimitedWarn(1, TimeUnit.SECONDS, clock::get);

        warn.warn(MARKER + " emit");
        warn.warn(MARKER + " swallowed");
        clock.addAndGet(TimeUnit.SECONDS.toNanos(1));
        warn.warn(MARKER + " carries count");
        clock.addAndGet(TimeUnit.SECONDS.toNanos(1));
        warn.warn(MARKER + " clean window");

        List<String> lines = emitted();
        assertEquals(3, lines.size());
        assertTrue(lines.get(1).contains("suppressed=1"));
        assertTrue(lines.get(2).contains("suppressed=0"),
                "the suppressed count must reset once it has ridden on an emitted line");
    }

    @Test
    void concurrentBurstAtAFrozenClockEmitsExactlyOnce() throws Exception {
        AtomicLong clock = new AtomicLong(0);
        RateLimitedWarn warn = new RateLimitedWarn(1, TimeUnit.SECONDS, clock::get);
        int threads = 8;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        try {
            for (int i = 0; i < threads; i++) {
                pool.submit(() -> {
                    try {
                        start.await(5, TimeUnit.SECONDS);
                        warn.warn(MARKER + " burst");
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        done.countDown();
                    }
                });
            }
            start.countDown();
            assertTrue(done.await(5, TimeUnit.SECONDS));
        } finally {
            pool.shutdownNow();
        }

        assertEquals(1, emitted().size(),
                "a concurrent burst inside one window must produce exactly one line — the CAS, not "
                        + "luck, decides the winner");
    }
}
