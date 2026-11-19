package org.flexlb.engine.grpc.core;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.ManagedChannel;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks recorded local close reasons and forced shutdown without allowing diagnostics to interrupt cleanup.
 */
class GrpcChannelCloseDiagnosticsTest {

    @Test
    void discoveryRemovalAndClientShutdownKeepTheirCloseReasons() {
        var pool = new GrpcChannelPool<String>(key -> mock(ManagedChannel.class));
        var removed = pool.getOrCreate("removed");
        pool.removeStaleChannels(List.of());
        assertEquals(
                GrpcChannelPool.CloseReason.DISCOVERY_REMOVED,
                removed.getCloseObservation().reason());
        verify(removed.getChannel()).shutdown();
        var stopped = pool.getOrCreate("stopped");
        pool.shutdown();
        assertEquals(
                GrpcChannelPool.CloseReason.CLIENT_SHUTDOWN,
                stopped.getCloseObservation().reason());
        verify(stopped.getChannel()).shutdown();
    }

    @Test
    void replacementClosesOldAndLosingChannels() {
        var channels = new java.util.ArrayList<ManagedChannel>();
        var pool =
                new GrpcChannelPool<String>(
                        key -> {
                            var channel = mock(ManagedChannel.class);
                            channels.add(channel);
                            return channel;
                        });
        var first = pool.getOrCreate("worker");
        var current = pool.replace("worker", first);
        assertSame(current, pool.replace("worker", first));
        assertEquals(GrpcChannelPool.CloseReason.REPLACED, first.getCloseObservation().reason());
        verify(channels.get(0)).shutdown();
        verify(channels.get(2)).shutdown();
        verify(channels.get(1), never()).shutdown();
        assertNull(current.getCloseObservation());
    }

    @Test
    void first_close_observation_is_preserved_and_graceful_close_does_not_force() throws Exception {
        ManagedChannel channel = mock(ManagedChannel.class);
        when(channel.awaitTermination(1000, TimeUnit.MILLISECONDS)).thenReturn(true);
        var invoker = new GrpcChannelPool<String>(key -> channel).getOrCreate("worker");
        long before = System.nanoTime();

        invoker.shutdown(GrpcChannelPool.CloseReason.DISCOVERY_REMOVED);
        var first = invoker.getCloseObservation();
        long after = System.nanoTime();
        invoker.shutdown(GrpcChannelPool.CloseReason.CLIENT_SHUTDOWN);

        assertSame(first, invoker.getCloseObservation());
        assertEquals(GrpcChannelPool.CloseReason.DISCOVERY_REMOVED, first.reason());
        assertTrue(!first.forced());
        assertTrue(first.nanoTime() - before >= 0);
        assertTrue(after - first.nanoTime() >= 0);
        verify(channel, never()).shutdownNow();
    }

    @Test
    void gracefulTimeoutRecordsTheForcedCloseBeforeCancellingCalls() {
        ManagedChannel channel = mock(ManagedChannel.class);
        var invoker = new GrpcChannelPool<String>(key -> channel).getOrCreate("worker");
        org.mockito.Mockito.doAnswer(
                        ignored -> {
                            assertTrue(invoker.getCloseObservation().forced());
                            assertEquals(
                                    GrpcChannelPool.CloseReason.DISCOVERY_REMOVED,
                                    invoker.getCloseObservation().reason());
                            return channel;
                        })
                .when(channel)
                .shutdownNow();

        invoker.shutdown(GrpcChannelPool.CloseReason.DISCOVERY_REMOVED);

        verify(channel).shutdownNow();
    }

    @Test
    void close_diagnostics_cannot_prevent_graceful_or_forceful_shutdown() {
        ManagedChannel channel = mock(ManagedChannel.class);
        var invoker = new GrpcChannelPool<String>(key -> channel).getOrCreate("worker");
        var logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        Level previousLevel = logger.getLevel();
        AtomicInteger attempts = new AtomicInteger();
        var failingAppender =
                new AppenderBase<ILoggingEvent>() {
                    @Override
                    protected void append(ILoggingEvent event) {
                        if (event.getFormattedMessage()
                                .startsWith("event=engine_grpc_channel_close")) {
                            attempts.incrementAndGet();
                            throw new OutOfMemoryError("diagnostic failure");
                        }
                    }
                };
        failingAppender.start();
        logger.setLevel(Level.INFO);
        logger.addAppender(failingAppender);
        try {
            invoker.shutdown(GrpcChannelPool.CloseReason.CLIENT_SHUTDOWN);
            verify(channel).shutdown();
            verify(channel).shutdownNow();
            assertEquals(2, attempts.get());
        } finally {
            logger.detachAppender(failingAppender);
            failingAppender.stop();
            logger.setLevel(previousLevel);
        }
    }

    @Test
    void interrupted_graceful_close_preserves_interrupt_and_forces_shutdown() throws Exception {
        ManagedChannel channel = mock(ManagedChannel.class);
        when(channel.awaitTermination(1000, TimeUnit.MILLISECONDS))
                .thenThrow(new InterruptedException());
        var invoker = new GrpcChannelPool<String>(key -> channel).getOrCreate("worker");
        try {
            invoker.shutdown(GrpcChannelPool.CloseReason.CLIENT_SHUTDOWN);
            assertTrue(Thread.currentThread().isInterrupted());
            verify(channel).shutdownNow();
        } finally {
            Thread.interrupted();
        }
    }

    /**
     * Captures channel shutdown diagnostics and restores logger state when the assertion scope closes.
     */
    private static final class CloseLogs extends ListAppender<ILoggingEvent>
            implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        private final Level previousLevel = logger.getLevel();

        private CloseLogs() {
            start();
            logger.setLevel(Level.INFO);
            logger.addAppender(this);
        }

        @Override
        public void close() {
            logger.detachAppender(this);
            stop();
            logger.setLevel(previousLevel);
        }
    }
}
