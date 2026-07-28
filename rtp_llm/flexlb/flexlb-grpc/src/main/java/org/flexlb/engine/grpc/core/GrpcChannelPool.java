package org.flexlb.engine.grpc.core;

import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Thread-safe keyed cache for reusable gRPC channels.
 */
@Slf4j
public class GrpcChannelPool<K> {

    private final ConcurrentMap<K, PooledChannel> channels = new ConcurrentHashMap<>();
    private final Function<K, ManagedChannel> channelFactory;

    public GrpcChannelPool(Function<K, ManagedChannel> channelFactory) {
        this.channelFactory = Objects.requireNonNull(channelFactory, "channelFactory");
    }

    public PooledChannel getOrCreate(K key) {
        PooledChannel current = channels.get(key);
        if (current != null && current.isUsable()) {
            return current;
        }
        return channels.compute(key, (ignored, existing) -> {
            if (existing != null && existing.isUsable()) {
                return existing;
            }
            if (existing != null) {
                existing.shutdown();
            }
            return create(key);
        });
    }

    public PooledChannel replace(K key, PooledChannel old) {
        PooledChannel replacement = create(key);
        boolean replaced;
        if (old == null) {
            PooledChannel existing = channels.putIfAbsent(key, replacement);
            replaced = existing == null;
        } else {
            replaced = channels.replace(key, old, replacement);
        }

        if (replaced) {
            if (old != null) {
                old.shutdown();
            }
            return replacement;
        }

        replacement.shutdown();
        return getOrCreate(key);
    }

    /**
     * Removes and shuts down channels whose keys are absent from the active set.
     */
    public void removeStaleChannels(Collection<K> activeChannelKeys) {
        removeChannelsForInactiveGroups(activeChannelKeys, Function.identity());
    }

    /**
     * Removes all channels belonging to groups that are no longer active. Channels with different
     * keys in an active group are retained.
     */
    public <G> void removeChannelsForInactiveGroups(
            Collection<G> activeGroupKeys,
            Function<K, G> groupKeyExtractor) {
        Set<G> activeGroupKeySet = new HashSet<>(activeGroupKeys);
        for (Map.Entry<K, PooledChannel> entry : channels.entrySet()) {
            if (!activeGroupKeySet.contains(groupKeyExtractor.apply(entry.getKey()))
                    && channels.remove(entry.getKey(), entry.getValue())) {
                entry.getValue().shutdown();
            }
        }
    }

    public int size() {
        return channels.size();
    }

    public void shutdown() {
        for (Map.Entry<K, PooledChannel> entry : channels.entrySet()) {
            if (channels.remove(entry.getKey(), entry.getValue())) {
                entry.getValue().shutdown();
            }
        }
    }

    private PooledChannel create(K key) {
        ManagedChannel channel = Objects.requireNonNull(
                channelFactory.apply(key), "channelFactory returned null");
        return new PooledChannel(String.valueOf(key), channel);
    }

    public static final class PooledChannel {

        private final String description;
        @Getter
        private final ManagedChannel channel;
        private final long createTimeUs;
        private final AtomicBoolean shutdown = new AtomicBoolean();
        @Getter
        private volatile long lastUsedTimeUs;
        private volatile long expireTimeUs;

        private PooledChannel(String description, ManagedChannel channel) {
            this.description = description;
            this.channel = channel;
            this.createTimeUs = System.nanoTime() / 1000;
            this.lastUsedTimeUs = createTimeUs;
        }

        public boolean isUsable() {
            if (shutdown.get() || channel.isShutdown() || channel.isTerminated()) {
                return false;
            }
            ConnectivityState state = channel.getState(true);
            return state != ConnectivityState.SHUTDOWN && state != ConnectivityState.TRANSIENT_FAILURE;
        }

        public void markUsed() {
            lastUsedTimeUs = System.nanoTime() / 1000;
        }

        public void markExpired() {
            expireTimeUs = System.nanoTime() / 1000;
        }

        public long getConnectionDurationUs() {
            long endTimeUs = expireTimeUs > 0 ? expireTimeUs : System.nanoTime() / 1000;
            return endTimeUs - createTimeUs;
        }

        public void shutdown() {
            if (!shutdown.compareAndSet(false, true)) {
                return;
            }
            try {
                channel.shutdown();
            } catch (Exception e) {
                log.warn("Failed to shut down gRPC channel: {}", description, e);
            }
        }
    }
}
