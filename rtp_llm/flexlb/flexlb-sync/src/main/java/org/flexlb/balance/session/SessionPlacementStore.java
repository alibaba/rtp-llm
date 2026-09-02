package org.flexlb.balance.session;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.config.RoutingConfig;
import org.springframework.stereotype.Component;

import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

@Component
public final class SessionPlacementStore {
    private static final long DEFAULT_MAXIMUM_SIZE = 1_000_000L;
    static final long MAX_IDLE_RETENTION_MS =
            RoutingConfig.SessionAffinityConfig.MAX_TTL_MS;
    private static final int MAX_SESSION_ID_LENGTH = 256;

    private final Cache<Key, State> placements;
    private final LongSupplier clock;
    private final AtomicLong epochs = new AtomicLong();

    public SessionPlacementStore() {
        this(DEFAULT_MAXIMUM_SIZE, System::currentTimeMillis, System::nanoTime);
    }

    SessionPlacementStore(long maximumSize, LongSupplier clock) {
        this(maximumSize, clock,
                () -> TimeUnit.MILLISECONDS.toNanos(clock.getAsLong()));
    }

    private SessionPlacementStore(long maximumSize, LongSupplier clock,
                                  LongSupplier ticker) {
        this.placements = Caffeine.newBuilder()
                .maximumSize(maximumSize)
                .expireAfterAccess(MAX_IDLE_RETENTION_MS, TimeUnit.MILLISECONDS)
                .ticker(ticker::getAsLong)
                .build();
        this.clock = clock;
    }

    public Optional<Placement> find(String model, String sessionId, long ttlMs) {
        if (!valid(sessionId)) {
            return Optional.empty();
        }
        Key key = new Key(model, sessionId);
        State state = placements.getIfPresent(key);
        if (state == null || state.placement() == null) {
            return Optional.empty();
        }
        Placement placement = state.placement();
        if (clock.getAsLong() - placement.storedAtMs() > ttlMs) {
            return Optional.empty();
        }
        return Optional.of(placement);
    }

    public void record(String model, String sessionId, String ipPort,
                       long requestId, long expectedEpoch) {
        if (!valid(sessionId) || ipPort == null || ipPort.isBlank()) {
            return;
        }
        Key key = new Key(model, sessionId);
        Placement placement = new Placement(ipPort, requestId, clock.getAsLong());
        placements.asMap().compute(key, (ignored, state) -> {
            if (state == null) {
                return null;
            }
            if (state.epoch() != expectedEpoch) {
                return state;
            }
            return new State(state.epoch(), placement);
        });
    }

    public long currentEpoch(String model, String sessionId) {
        if (!valid(sessionId)) {
            return -1L;
        }
        State state = placements.asMap().computeIfAbsent(
                new Key(model, sessionId), ignored -> new State(nextEpoch(), null));
        return state.epoch();
    }

    public long reset(String model, String sessionId) {
        if (!valid(sessionId)) {
            return -1L;
        }
        State state = placements.asMap().compute(new Key(model, sessionId),
                (ignored, current) -> new State(nextEpoch(), null));
        return state.epoch();
    }

    public long resetIfPresent(String model, String sessionId) {
        if (!valid(sessionId)) {
            return -1L;
        }
        State state = placements.asMap().computeIfPresent(
                new Key(model, sessionId), (ignored, current) -> new State(nextEpoch(), null));
        return state == null ? -1L : state.epoch();
    }

    long estimatedSize() {
        return placements.estimatedSize();
    }

    void cleanUp() {
        placements.cleanUp();
    }

    private static boolean valid(String sessionId) {
        return sessionId != null && !sessionId.isBlank()
                && sessionId.length() <= MAX_SESSION_ID_LENGTH;
    }

    private long nextEpoch() {
        long epoch = epochs.incrementAndGet();
        return epoch == 0L ? epochs.incrementAndGet() : epoch;
    }

    private record Key(String model, String sessionId) {
    }

    public record Placement(String ipPort, long requestId, long storedAtMs) {
    }

    private record State(long epoch, Placement placement) {
    }
}
