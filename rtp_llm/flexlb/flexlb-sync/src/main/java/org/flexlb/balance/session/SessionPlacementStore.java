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

    private final Cache<Key, Placement> placements;
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
        Placement placement = placements.getIfPresent(key);
        if (placement == null) {
            return Optional.empty();
        }
        if (clock.getAsLong() - placement.storedAtMs() > ttlMs) {
            placements.asMap().remove(key, placement);
            return Optional.empty();
        }
        return Optional.of(placement);
    }

    public void record(String model, String sessionId, String ipPort, long requestId) {
        if (!valid(sessionId) || ipPort == null || ipPort.isBlank()) {
            return;
        }
        placements.put(new Key(model, sessionId),
                new Placement(ipPort, requestId, clock.getAsLong()));
    }

    public void invalidate(String model, String sessionId) {
        if (valid(sessionId)) {
            placements.invalidate(new Key(model, sessionId));
        }
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
}
