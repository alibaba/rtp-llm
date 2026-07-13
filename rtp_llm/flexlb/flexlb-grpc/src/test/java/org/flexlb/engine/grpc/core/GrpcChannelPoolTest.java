package org.flexlb.engine.grpc.core;

import io.grpc.ManagedChannel;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcChannelPoolTest {

    @Test
    void reusesUsableChannel() {
        ManagedChannel channel = mock(ManagedChannel.class);
        GrpcChannelPool<String> pool = new GrpcChannelPool<>(ignored -> channel);

        GrpcChannelPool.PooledChannel first = pool.getOrCreate("target");
        GrpcChannelPool.PooledChannel second = pool.getOrCreate("target");

        assertSame(first, second);
        assertEquals(1, pool.size());
    }

    @Test
    void createsOnlyOneChannelForConcurrentRequests() {
        ManagedChannel channel = mock(ManagedChannel.class);
        AtomicInteger created = new AtomicInteger();
        GrpcChannelPool<String> pool = new GrpcChannelPool<>(ignored -> {
            created.incrementAndGet();
            return channel;
        });

        List<GrpcChannelPool.PooledChannel> results = IntStream.range(0, 100)
                .parallel()
                .mapToObj(ignored -> pool.getOrCreate("target"))
                .toList();

        assertEquals(1, created.get());
        for (GrpcChannelPool.PooledChannel result : results) {
            assertSame(results.getFirst(), result);
        }
    }

    @Test
    void recreatesTerminatedChannel() {
        ManagedChannel firstChannel = mock(ManagedChannel.class);
        ManagedChannel secondChannel = mock(ManagedChannel.class);
        when(firstChannel.isTerminated()).thenReturn(true);
        AtomicInteger created = new AtomicInteger();
        GrpcChannelPool<String> pool = new GrpcChannelPool<>(ignored ->
                created.getAndIncrement() == 0 ? firstChannel : secondChannel);

        GrpcChannelPool.PooledChannel first = pool.getOrCreate("target");
        GrpcChannelPool.PooledChannel second = pool.getOrCreate("target");

        assertNotSame(first, second);
        assertSame(secondChannel, second.getChannel());
        verify(firstChannel).shutdown();
    }

    @Test
    void removesStaleChannels() {
        ManagedChannel firstChannel = mock(ManagedChannel.class);
        ManagedChannel secondChannel = mock(ManagedChannel.class);
        GrpcChannelPool<String> pool = new GrpcChannelPool<>(key ->
                "first".equals(key) ? firstChannel : secondChannel);
        pool.getOrCreate("first");
        pool.getOrCreate("second");

        pool.removeStaleChannels(List.of("first"));

        assertEquals(1, pool.size());
        verify(firstChannel, never()).shutdown();
        verify(secondChannel).shutdown();
    }

    @Test
    void retainsAllPortsForActiveGroup() {
        ManagedChannel firstPort = mock(ManagedChannel.class);
        ManagedChannel secondPort = mock(ManagedChannel.class);
        ManagedChannel inactiveHost = mock(ManagedChannel.class);
        GrpcChannelPool<String> pool = new GrpcChannelPool<>(key -> switch (key) {
            case "10.0.0.1:8081" -> firstPort;
            case "10.0.0.1:18002" -> secondPort;
            default -> inactiveHost;
        });
        pool.getOrCreate("10.0.0.1:8081");
        pool.getOrCreate("10.0.0.1:18002");
        pool.getOrCreate("10.0.0.2:8081");

        pool.removeChannelsForInactiveGroups(
                List.of("10.0.0.1"), key -> key.substring(0, key.indexOf(':')));

        assertEquals(2, pool.size());
        verify(firstPort, never()).shutdown();
        verify(secondPort, never()).shutdown();
        verify(inactiveHost).shutdown();
    }

    @Test
    void replacesExpectedChannel() {
        ManagedChannel firstChannel = mock(ManagedChannel.class);
        ManagedChannel secondChannel = mock(ManagedChannel.class);
        AtomicInteger created = new AtomicInteger();
        GrpcChannelPool<String> pool = new GrpcChannelPool<>(ignored ->
                created.getAndIncrement() == 0 ? firstChannel : secondChannel);
        GrpcChannelPool.PooledChannel first = pool.getOrCreate("target");

        GrpcChannelPool.PooledChannel replacement = pool.replace("target", first);

        assertSame(secondChannel, replacement.getChannel());
        assertSame(replacement, pool.getOrCreate("target"));
        verify(firstChannel).shutdown();
    }
}
