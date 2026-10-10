package org.flexlb.constraint;

import org.springframework.stereotype.Component;

import java.net.URI;
import java.time.Duration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.LongSupplier;

/** Short-lived control-plane addresses, never input to the inference scheduler. */
@Component
public class ConstraintTreeBootstrapRegistry {
    private record Lease(URI uri, long renewedAt) { }

    private final Map<String, Map<String, Lease>> pending = new HashMap<>();
    private final Map<String, Set<String>> discovered = new HashMap<>();
    private final LongSupplier clock;
    private final long leaseNanos;

    public ConstraintTreeBootstrapRegistry() {
        this(System::nanoTime, Duration.ofMinutes(2));
    }

    ConstraintTreeBootstrapRegistry(LongSupplier clock, Duration lease) {
        this.clock = clock;
        this.leaseNanos = lease.toNanos();
    }

    /** True only after ordinary discovery has observed this address. */
    public synchronized boolean register(String model, URI uri) {
        expire();
        String address = uri.getRawAuthority();
        if (discovered.getOrDefault(model, Set.of()).contains(address)) { return true; }
        var entries = pending.computeIfAbsent(model, ignored -> new LinkedHashMap<>());
        if (!entries.containsKey(address) && pending.values().stream().mapToInt(Map::size).sum() >= 10000) {
            throw new IllegalStateException("too many pending constraint-tree workers");
        }
        entries.put(address, new Lease(uri, clock.getAsLong()));
        return false;
    }

    public synchronized Map<String, URI> merge(String model, Map<String, URI> ordinary) {
        expire();
        discovered.put(model, Set.copyOf(ordinary.keySet()));
        var entries = pending.get(model);
        Map<String, URI> result = new LinkedHashMap<>(ordinary);
        if (entries != null) {
            entries.keySet().removeAll(ordinary.keySet());
            entries.forEach((address, lease) -> result.putIfAbsent(address, lease.uri()));
        }
        return result;
    }

    public synchronized boolean isPending(String model, URI uri) {
        expire();
        return pending.containsKey(model) && pending.get(model).containsKey(uri.getRawAuthority());
    }

    public synchronized Set<String> pendingModels() {
        expire();
        return Set.copyOf(pending.keySet());
    }

    private void expire() {
        long now = clock.getAsLong();
        pending.values().forEach(entries -> entries.values().removeIf(lease -> now - lease.renewedAt() >= leaseNanos));
        pending.values().removeIf(Map::isEmpty);
    }
}
