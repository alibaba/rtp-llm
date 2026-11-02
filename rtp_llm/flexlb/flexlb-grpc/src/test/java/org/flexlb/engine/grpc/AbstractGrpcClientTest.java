package org.flexlb.engine.grpc;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.ManagedChannel;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
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

class AbstractGrpcClientTest {

    @Test
    void address_update_manages_all_rpc_service_types() {
        TestGrpcClient client = new TestGrpcClient();

        client.onAddressUpdate(List.of("10.0.0.1:8080"));

        for (AbstractGrpcClient.ServiceType serviceType : AbstractGrpcClient.ServiceType.values()) {
            assertTrue(client.hasChannel("10.0.0.1", 8081, serviceType));
        }
        assertEquals(AbstractGrpcClient.ServiceType.values().length, client.channelCount());

        client.onAddressUpdate(List.of("10.0.0.1:8080"));

        assertEquals(AbstractGrpcClient.ServiceType.values().length, client.channelCount());
    }

    @Test
    void discovery_removal_and_client_shutdown_keep_their_actual_close_reasons() {
        TestGrpcClient client = new TestGrpcClient();
        client.onAddressUpdate(List.of("10.0.0.1:8080"));
        var removed = List.copyOf(client.channelPool.values());

        client.onAddressUpdate(List.of());

        assertEquals(0, client.channelCount());
        for (var invoker : removed) {
            assertEquals(AbstractGrpcClient.CloseReason.DISCOVERY_REMOVED,
                    invoker.getCloseObservation().reason());
            verify(invoker.getChannel()).shutdown();
        }

        client.onAddressUpdate(List.of("10.0.0.1:8080"));
        var stopped = List.copyOf(client.channelPool.values());
        client.shutdownChannelPool();

        assertEquals(0, client.channelCount());
        for (var invoker : stopped) {
            assertEquals(AbstractGrpcClient.CloseReason.CLIENT_SHUTDOWN,
                    invoker.getCloseObservation().reason());
            verify(invoker.getChannel()).shutdown();
        }
    }

    @Test
    void replacement_closes_old_channel_and_discards_only_competing_new_channels() {
        TestGrpcClient client = new TestGrpcClient();
        String key = AbstractGrpcClient.createKey("10.0.0.1", 8081, AbstractGrpcClient.ServiceType.WORKER_STATUS);
        ManagedChannel original = mock(ManagedChannel.class);
        var first = client.putInvokerIfAbsent(key, original);
        ManagedChannel duplicate = mock(ManagedChannel.class);
        ManagedChannel replacement = mock(ManagedChannel.class);
        ManagedChannel lostReplacement = mock(ManagedChannel.class);

        try (var logs = new CloseLogs()) {
            assertSame(first, client.putInvokerIfAbsent(key, duplicate));
            var current = client.replaceInvoker(key, first, replacement);
            assertSame(current, client.replaceInvoker(key, first, lostReplacement));

            assertEquals(AbstractGrpcClient.CloseReason.REPLACED, first.getCloseObservation().reason());
            assertNull(current.getCloseObservation());
            assertEquals(2, logs.list.stream()
                    .filter(event -> event.getFormattedMessage().contains("reason=DUPLICATE_DISCARDED"))
                    .filter(event -> event.getFormattedMessage().contains("force=false"))
                    .count());
            verify(original).shutdown();
            verify(duplicate).shutdown();
            verify(lostReplacement).shutdown();
            verify(replacement, never()).shutdown();
        }
    }

    @Test
    void first_close_observation_is_preserved_and_graceful_close_does_not_force() throws Exception {
        TestGrpcClient client = new TestGrpcClient();
        ManagedChannel channel = mock(ManagedChannel.class);
        when(channel.awaitTermination(1000, TimeUnit.MILLISECONDS)).thenReturn(true);
        var invoker = client.new Invoker("worker", channel);
        long before = System.nanoTime();

        invoker.shutdown(AbstractGrpcClient.CloseReason.DISCOVERY_REMOVED);
        var first = invoker.getCloseObservation();
        long after = System.nanoTime();
        invoker.shutdown(AbstractGrpcClient.CloseReason.CLIENT_SHUTDOWN);

        assertSame(first, invoker.getCloseObservation());
        assertEquals(AbstractGrpcClient.CloseReason.DISCOVERY_REMOVED, first.reason());
        assertTrue(!first.forced());
        assertTrue(first.nanoTime() - before >= 0);
        assertTrue(after - first.nanoTime() >= 0);
        verify(channel, never()).shutdownNow();
    }

    @Test
    void gracefulTimeoutRecordsTheForcedCloseBeforeCancellingCalls() {
        TestGrpcClient client = new TestGrpcClient();
        ManagedChannel channel = mock(ManagedChannel.class);
        var invoker = client.new Invoker("worker", channel);
        org.mockito.Mockito.doAnswer(ignored -> {
            assertTrue(invoker.getCloseObservation().forced());
            assertEquals(AbstractGrpcClient.CloseReason.DISCOVERY_REMOVED,
                    invoker.getCloseObservation().reason());
            return channel;
        }).when(channel).shutdownNow();

        invoker.shutdown(AbstractGrpcClient.CloseReason.DISCOVERY_REMOVED);

        verify(channel).shutdownNow();
    }

    @Test
    void close_diagnostics_cannot_prevent_graceful_or_forceful_shutdown() {
        TestGrpcClient client = new TestGrpcClient();
        ManagedChannel channel = mock(ManagedChannel.class);
        var invoker = client.new Invoker("worker", channel);
        var logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        Level previousLevel = logger.getLevel();
        AtomicInteger attempts = new AtomicInteger();
        var failingAppender = new AppenderBase<ILoggingEvent>() {
            @Override
            protected void append(ILoggingEvent event) {
                if (event.getFormattedMessage().startsWith("event=engine_grpc_channel_close")) {
                    attempts.incrementAndGet();
                    throw new OutOfMemoryError("diagnostic failure");
                }
            }
        };
        failingAppender.start();
        logger.setLevel(Level.INFO);
        logger.addAppender(failingAppender);
        try {
            invoker.shutdown(AbstractGrpcClient.CloseReason.CLIENT_SHUTDOWN);
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
        TestGrpcClient client = new TestGrpcClient();
        ManagedChannel channel = mock(ManagedChannel.class);
        when(channel.awaitTermination(1000, TimeUnit.MILLISECONDS)).thenThrow(new InterruptedException());
        var invoker = client.new Invoker("worker", channel);
        try {
            invoker.shutdown(AbstractGrpcClient.CloseReason.CLIENT_SHUTDOWN);
            assertTrue(Thread.currentThread().isInterrupted());
            verify(channel).shutdownNow();
        } finally {
            Thread.interrupted();
        }
    }

    private static final class CloseLogs extends ListAppender<ILoggingEvent> implements AutoCloseable {
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

    private static final class TestGrpcClient extends AbstractGrpcClient {

        private TestGrpcClient() {
            super(mock(GrpcReporter.class));
        }

        @Override
        protected ManagedChannel createChannel(String channelKey) {
            return mock(ManagedChannel.class);
        }

        private boolean hasChannel(String ip, int port, ServiceType serviceType) {
            return channelPool.containsKey(createKey(ip, port, serviceType));
        }

        private int channelCount() {
            return channelPool.size();
        }
    }
}
