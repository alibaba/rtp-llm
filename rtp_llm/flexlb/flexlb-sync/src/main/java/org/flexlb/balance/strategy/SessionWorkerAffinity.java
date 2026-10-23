package org.flexlb.balance.strategy;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;

import java.time.Duration;

/** Bounded scheduling hints, never evidence of reusable KV. Endpoint identity fences restarts. */
final class SessionWorkerAffinity {
    record Key(String tenant, String model, RoleType role, String group, String session) { }

    private final Cache<Key, PrefillEndpoint> workers = Caffeine.newBuilder()
            .maximumSize(100_000).expireAfterWrite(Duration.ofMinutes(20)).build();

    static Key key(Request request, RoleType role, String group) {
        String session = request.getSessionId();
        if (session == null || session.isBlank()) { return null; }
        return new Key(request.getApiKey(), request.getModel(), role, group, session);
    }

    PrefillEndpoint get(Key key) { return key == null ? null : workers.getIfPresent(key); }

    void remember(Key key, PrefillEndpoint endpoint) {
        if (key != null) { workers.put(key, endpoint); }
    }

    void forget(Key key, PrefillEndpoint endpoint) {
        if (key != null) { workers.asMap().remove(key, endpoint); }
    }
}
