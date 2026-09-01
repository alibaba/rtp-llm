package org.flexlb.balance.session;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.springframework.stereotype.Component;

import java.util.Optional;
import java.util.function.LongSupplier;

@Component
public final class SessionPlacementStore {
    private static final long DEFAULT_MAXIMUM_SIZE = 1_000_000L;
    private static final int MAX_SESSION_ID_LENGTH = 256;

    private final Cache<Key, State> placements;
    private final LongSupplier clock;

    public SessionPlacementStore() {
        this(DEFAULT_MAXIMUM_SIZE, System::currentTimeMillis);
    }

    SessionPlacementStore(long maximumSize, LongSupplier clock) {
        this.placements = Caffeine.newBuilder().maximumSize(maximumSize).build();
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

    public void record(String model, String sessionId, String ipPort, long requestId) {
        record(model, sessionId, ipPort, requestId, currentEpoch(model, sessionId));
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
                return expectedEpoch == 0L ? new State(0L, placement) : null;
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
        State state = placements.getIfPresent(new Key(model, sessionId));
        return state == null ? 0L : state.epoch();
    }

    public long reset(String model, String sessionId) {
        if (!valid(sessionId)) {
            return -1L;
        }
        State state = placements.asMap().compute(new Key(model, sessionId),
                (ignored, current) -> new State(
                        current == null ? 1L : current.epoch() + 1L, null));
        return state.epoch();
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

    private record Key(String model, String sessionId) {
    }

    public record Placement(String ipPort, long requestId, long storedAtMs) {
    }

    private record State(long epoch, Placement placement) {
    }
}
