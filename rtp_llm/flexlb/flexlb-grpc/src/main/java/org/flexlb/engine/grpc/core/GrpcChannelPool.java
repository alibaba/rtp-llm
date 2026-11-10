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
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Thread-safe keyed cache for reusable gRPC channels.
 *
 * <p>Lookups reuse or create a keyed channel. Replacement and discovery cleanup retire the exact
 * old channels; shutdown closes all retained channels and waits according to the supplied
 * deadline.
 */
@Slf4j
public class GrpcChannelPool<K> {
    private final ConcurrentMap<K, PooledChannel> channels = new ConcurrentHashMap<>();
    private final Function<K, ManagedChannel> channelFactory;

    public GrpcChannelPool(Function<K, ManagedChannel> channelFactory) {
        this.channelFactory = Objects.requireNonNull(channelFactory, "channelFactory");
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

    public PooledChannel getOrCreate(K key) {
        PooledChannel current = channels.get(key);
        if (current != null && current.isUsable()) {
            return current;
        }
        return channels.compute(
                key,
                (ignored, existing) -> {
                    if (existing != null && existing.isUsable()) {
                        return existing;
                    }
                    if (existing != null) {
                        existing.shutdown();
                    }
                    return create(key);
                });
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
            Collection<G> activeGroupKeys, Function<K, G> groupKeyExtractor) {
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

    public void shutdown(long timeout, TimeUnit unit) {
        for (Map.Entry<K, PooledChannel> entry : channels.entrySet()) {
            if (channels.remove(entry.getKey(), entry.getValue())) {
                entry.getValue().shutdown(timeout, unit);
            }
        }
    }

    private PooledChannel create(K key) {
        ManagedChannel channel =
                Objects.requireNonNull(channelFactory.apply(key), "channelFactory returned null");
        return new PooledChannel(String.valueOf(key), channel);
    }

    /**
     * Closes a cached channel once and bounds its termination wait before forcing shutdown.
     */
    public static final class PooledChannel {

        private final String description;
        @Getter private final ManagedChannel channel;
        private final AtomicBoolean shutdown = new AtomicBoolean();

        private PooledChannel(String description, ManagedChannel channel) {
            this.description = description;
            this.channel = channel;
        }

        public boolean isUsable() {
            if (shutdown.get() || channel.isShutdown() || channel.isTerminated()) {
                return false;
            }
            ConnectivityState state = channel.getState(true);
            return state != ConnectivityState.SHUTDOWN
                    && state != ConnectivityState.TRANSIENT_FAILURE;
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

        public void shutdown(long timeout, TimeUnit unit) {
            if (!shutdown.compareAndSet(false, true)) {
                return;
            }
            try {
                channel.shutdown();
                if (!channel.awaitTermination(timeout, unit)) {
                    channel.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                channel.shutdownNow();
            } catch (Exception e) {
                log.warn("Failed to shut down gRPC channel: {}", description, e);
                channel.shutdownNow();
            }
        }
    }
}
