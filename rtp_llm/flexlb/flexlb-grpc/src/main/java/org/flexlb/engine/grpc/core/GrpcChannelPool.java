package org.flexlb.engine.grpc.core;

import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.util.Logger;

import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
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
                old.shutdown(CloseReason.REPLACED);
            }
            return replacement;
        }

        replacement.shutdown(CloseReason.DUPLICATE_DISCARDED);
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
                        existing.shutdown(CloseReason.REPLACED);
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
                entry.getValue().shutdown(CloseReason.DISCOVERY_REMOVED);
            }
        }
    }

    public int size() {
        return channels.size();
    }

    public void shutdown() {
        for (Map.Entry<K, PooledChannel> entry : channels.entrySet()) {
            if (channels.remove(entry.getKey(), entry.getValue())) {
                entry.getValue().shutdown(CloseReason.CLIENT_SHUTDOWN);
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

    public enum CloseReason {
        DISCOVERY_REMOVED,
        REPLACED,
        CLIENT_SHUTDOWN,
        DUPLICATE_DISCARDED
    }

    public record CloseObservation(CloseReason reason, long nanoTime, boolean forced) {}

    /**
     * Closes a cached channel once and bounds its termination wait before forcing shutdown.
     */
    public static final class PooledChannel {
        private final String description;
        private final AtomicReference<CloseObservation> closeObservation = new AtomicReference<>();
        @Getter private final ManagedChannel channel;

        @Getter private final long createTimeUs;

        private final AtomicBoolean shutdown = new AtomicBoolean();

        @Getter private volatile long lastUsedTimeUs;
        private volatile long expireTimeUs;

        private PooledChannel(String description, ManagedChannel channel) {
            this.description = description;
            this.channel = channel;
            this.createTimeUs = System.nanoTime() / 1000;
            this.lastUsedTimeUs = createTimeUs;
        }

        public String getChannelKey() {
            return description;
        }

        public boolean isUsable() {
            if (shutdown.get() || channel.isShutdown() || channel.isTerminated()) {
                return false;
            }
            ConnectivityState state = channel.getState(true);
            return state != ConnectivityState.SHUTDOWN
                    && state != ConnectivityState.TRANSIENT_FAILURE;
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

        public CloseObservation getCloseObservation() {
            return closeObservation.get();
        }

        public void shutdown() {
            shutdown(CloseReason.CLIENT_SHUTDOWN);
        }

        public void shutdown(CloseReason reason) {
            shutdown(reason, 1000, TimeUnit.MILLISECONDS);
        }

        public void shutdown(long timeout, TimeUnit unit) {
            shutdown(CloseReason.CLIENT_SHUTDOWN, timeout, unit);
        }

        private void shutdown(CloseReason reason, long timeout, TimeUnit unit) {
            if (!shutdown.compareAndSet(false, true)) {
                return;
            }
            if (channel == null) {
                return;
            }
            try {
                closeObservation.compareAndSet(
                        null, new CloseObservation(reason, System.nanoTime(), false));
            } catch (RuntimeException | Error ignored) {
                // Recording diagnostics must not prevent closing a channel under memory pressure.
            }
            logCloseSafely(reason, false, "requested", null);
            try {
                channel.shutdown();
                if (reason == CloseReason.REPLACED || reason == CloseReason.DUPLICATE_DISCARDED) {
                    // These paths can run from a saturated gRPC callback on its Netty eventloop.
                    // Waiting here stalls the same loop needed to finish channel termination.
                    if (!channel.isTerminated()) {
                        forceShutdown(reason);
                    }
                    return;
                }
                if (!channel.awaitTermination(timeout, unit)) {
                    logCloseSafely(reason, true, "graceful_timeout", null);
                    forceShutdown(reason);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logCloseSafely(reason, true, "interrupted", e);
                forceShutdown(reason);
            } catch (Exception e) {
                logCloseSafely(reason, true, "shutdown_failed", e);
                forceShutdown(reason);
            }
        }

        private void logCloseSafely(
                CloseReason reason, boolean force, String trigger, Throwable failure) {
            try {
                String format =
                        "event=engine_grpc_channel_close reason={} channelKey={} createTimeUs={} force={} trigger={}";
                if (force) {
                    Logger.warn(format, reason, description, createTimeUs, true, trigger, failure);
                } else {
                    Logger.info(format, reason, description, createTimeUs, false, trigger);
                }
            } catch (RuntimeException | Error ignored) {
                // Logging failures cannot prevent shutdown or its forceful fallback.
            }
        }

        private void forceShutdown(CloseReason reason) {
            try {
                // A graceful request does not cancel active calls. Record the distinct forced
                // action.
                closeObservation.updateAndGet(
                        previous ->
                                previous != null && previous.forced()
                                        ? previous
                                        : new CloseObservation(reason, System.nanoTime(), true));
            } catch (RuntimeException | Error ignored) {
                // Diagnostics must not prevent releasing the channel.
            }
            channel.shutdownNow();
        }
    }
}
