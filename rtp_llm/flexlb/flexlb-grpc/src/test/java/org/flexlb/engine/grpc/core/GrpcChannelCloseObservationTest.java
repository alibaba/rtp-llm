package org.flexlb.engine.grpc.core;

import io.grpc.ManagedChannel;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcChannelCloseObservationTest {
    @Test
    void discoveryRemovalAndClientShutdownKeepTheirCloseReasons() throws Exception {
        ManagedChannel channel = mock(ManagedChannel.class);
        when(channel.awaitTermination(1, TimeUnit.SECONDS)).thenReturn(true);
        var pool = new GrpcChannelPool<String>(ignored -> channel);
        var removed = pool.getOrCreate("removed");
        pool.removeStaleChannels(List.of());
        assertEquals(GrpcChannelPool.CloseReason.DISCOVERY_REMOVED, removed.getCloseObservation().reason());
        assertFalse(removed.getCloseObservation().forced());
        var stopped = pool.getOrCreate("stopped");
        pool.shutdown();
        assertEquals(GrpcChannelPool.CloseReason.CLIENT_SHUTDOWN, stopped.getCloseObservation().reason());
        assertEquals(0, pool.size());
    }

    @Test
    void firstCloseObservationIsPreserved() throws Exception {
        ManagedChannel channel = mock(ManagedChannel.class);
        when(channel.awaitTermination(1, TimeUnit.SECONDS)).thenReturn(true);
        var pool = new GrpcChannelPool<String>(ignored -> channel);
        var pooled = pool.getOrCreate("worker");
        pooled.shutdown(GrpcChannelPool.CloseReason.DISCOVERY_REMOVED);
        var first = pooled.getCloseObservation();
        pooled.shutdown(GrpcChannelPool.CloseReason.CLIENT_SHUTDOWN);
        assertSame(first, pooled.getCloseObservation());
        verify(channel, never()).shutdownNow();
    }

    @Test
    void forcedCloseIsRecordedBeforeCancellingCalls() {
        ManagedChannel channel = mock(ManagedChannel.class);
        var pool = new GrpcChannelPool<String>(ignored -> channel);
        var pooled = pool.getOrCreate("worker");
        doAnswer(ignored -> {
            assertTrue(pooled.getCloseObservation().forced());
            assertEquals(GrpcChannelPool.CloseReason.DISCOVERY_REMOVED, pooled.getCloseObservation().reason());
            return channel;
        }).when(channel).shutdownNow();
        pooled.shutdown(GrpcChannelPool.CloseReason.DISCOVERY_REMOVED);
        verify(channel).shutdownNow();
    }

    @Test
    void interruptedClosePreservesInterruptAndForcesShutdown() throws Exception {
        ManagedChannel channel = mock(ManagedChannel.class);
        when(channel.awaitTermination(1, TimeUnit.SECONDS)).thenThrow(new InterruptedException());
        var pool = new GrpcChannelPool<String>(ignored -> channel);
        var pooled = pool.getOrCreate("worker");
        try {
            pooled.shutdown(GrpcChannelPool.CloseReason.CLIENT_SHUTDOWN);
            assertTrue(Thread.currentThread().isInterrupted());
            assertTrue(pooled.getCloseObservation().forced());
            verify(channel).shutdownNow();
        } finally {
            Thread.interrupted();
        }
    }
}
