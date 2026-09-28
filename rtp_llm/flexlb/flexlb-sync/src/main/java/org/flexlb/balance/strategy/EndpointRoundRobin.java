package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.IntFunction;
import java.util.function.IntPredicate;

/** Continues an address ring within each routing group, even as eligibility changes. */
final class EndpointRoundRobin {
    private record Scope(RoleType role, String group) { }
    private static final class Cursor { private String lastSelectedAddress; }
    private final ConcurrentHashMap<Scope, Cursor> cursors = new ConcurrentHashMap<>();

    /** Select consecutive entries from a snapshot of eligible, unique addresses. */
    <T> List<T> nextBatch(RoleType role, String group, Map<String, T> candidates, int count) {
        if (count <= 0 || candidates.isEmpty()) { return List.of(); }
        // Sort the snapshot outside the cursor lock, not once per selected item.
        List<String> addresses = new ArrayList<>(candidates.keySet());
        addresses.sort(null);
        Cursor cursor = cursors.computeIfAbsent(new Scope(role, group), ignored -> new Cursor());
        int start;
        synchronized (cursor) {
            // Same address cursor as next(), with one atomic advancement for the whole batch.
            int previous = cursor.lastSelectedAddress == null ? -1 : Collections.binarySearch(addresses, cursor.lastSelectedAddress);
            // If the previous worker disappeared, continue at its insertion point in the new snapshot.
            start = (previous >= 0 ? previous + 1 : -previous - 1) % addresses.size();
            cursor.lastSelectedAddress = addresses.get((start + count - 1) % addresses.size());
        }
        List<T> selected = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            selected.add(candidates.get(addresses.get((start + i) % addresses.size())));
        }
        return selected;
    }

    int next(RoleType role, String group, int size, IntPredicate eligible, IntFunction<String> address) {
        Cursor cursor = cursors.computeIfAbsent(new Scope(role, group), ignored -> new Cursor());
        synchronized (cursor) {
            int firstIndex = -1;
            int nextIndex = -1;
            String firstAddress = null;
            String nextAddress = null;
            for (int i = 0; i < size; i++) {
                if (!eligible.test(i)) { continue; }
                String candidate = address.apply(i);
                if (firstAddress == null || candidate.compareTo(firstAddress) < 0) {
                    firstIndex = i;
                    firstAddress = candidate;
                }
                if (cursor.lastSelectedAddress != null && candidate.compareTo(cursor.lastSelectedAddress) > 0
                        && (nextAddress == null || candidate.compareTo(nextAddress) < 0)) {
                    nextIndex = i;
                    nextAddress = candidate;
                }
            }
            if (nextIndex >= 0) {
                cursor.lastSelectedAddress = nextAddress;
                return nextIndex;
            }
            if (firstIndex >= 0) { cursor.lastSelectedAddress = firstAddress; }
            return firstIndex;
        }
    }
}
