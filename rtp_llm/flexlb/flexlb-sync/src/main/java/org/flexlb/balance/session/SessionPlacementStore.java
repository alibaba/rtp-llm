package org.flexlb.balance.session;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.config.ConfigService;
import org.flexlb.config.RoutingConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

@Component
public final class SessionPlacementStore {
    private static final int MAX_SESSION_ID_LENGTH = 256;

    private final Cache<Key, Placement> placements;
    private final LongSupplier clock;

    public SessionPlacementStore() {
        this(RoutingConfig.SessionAffinityConfig.DEFAULT_MAX_ENTRIES,
                RoutingConfig.SessionAffinityConfig.MAX_TTL_MS,
                System::currentTimeMillis,
                System::nanoTime);
    }

    SessionPlacementStore(long maximumSize, LongSupplier clock) {
        this(maximumSize, RoutingConfig.SessionAffinityConfig.MAX_TTL_MS, clock,
                () -> TimeUnit.MILLISECONDS.toNanos(clock.getAsLong()));
    }

    @Autowired
    public SessionPlacementStore(ConfigService configService) {
        this(maximumSize(configService), retentionMs(configService),
                System::currentTimeMillis, System::nanoTime);
    }

    private SessionPlacementStore(long maximumSize, long retentionMs, LongSupplier clock,
                                  LongSupplier ticker) {
        this.placements = Caffeine.newBuilder()
                .maximumSize(maximumSize)
                .expireAfterAccess(retentionMs, TimeUnit.MILLISECONDS)
                .ticker(ticker::getAsLong)
                .build();
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
            return Optional.empty();
        }
        return Optional.of(placement);
    }

    public void record(String model, String sessionId, String ipPort) {
        if (!valid(sessionId) || ipPort == null || ipPort.isBlank()) {
            return;
        }
        placements.put(new Key(model, sessionId), new Placement(ipPort, clock.getAsLong()));
    }

    public void invalidate(String model, String sessionId) {
        if (valid(sessionId)) {
            placements.invalidate(new Key(model, sessionId));
        }
    }

    public long estimatedSize() {
        placements.cleanUp();
        return placements.estimatedSize();
    }

    void cleanUp() {
        placements.cleanUp();
    }

    private static boolean valid(String sessionId) {
        if (sessionId == null || sessionId.isEmpty()
                || sessionId.length() > MAX_SESSION_ID_LENGTH) {
            return false;
        }
        return sessionId.chars().allMatch(character -> character >= 0x21 && character <= 0x7e);
    }

    private static long maximumSize(ConfigService configService) {
        RoutingConfig.SessionAffinityConfig config = affinityConfig(configService);
        return config == null
                ? RoutingConfig.SessionAffinityConfig.DEFAULT_MAX_ENTRIES
                : config.getMaxEntries();
    }

    private static long retentionMs(ConfigService configService) {
        RoutingConfig.SessionAffinityConfig config = affinityConfig(configService);
        return config == null
                ? RoutingConfig.SessionAffinityConfig.MAX_TTL_MS
                : config.getTtlMs();
    }

    private static RoutingConfig.SessionAffinityConfig affinityConfig(
            ConfigService configService) {
        return configService.loadBalanceConfig().getRouter().getRoles()
                .getPrefill().getSessionAffinity();
    }

    private record Key(String model, String sessionId) {
    }

    public record Placement(String ipPort, long storedAtMs) {
    }

}
