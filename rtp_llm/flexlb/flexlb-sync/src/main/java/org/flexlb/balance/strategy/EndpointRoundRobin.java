package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;

import java.util.concurrent.ConcurrentHashMap;
import java.util.function.IntFunction;
import java.util.function.IntPredicate;

/** Continues an address ring within each routing group, even as eligibility changes. */
final class EndpointRoundRobin {
    private record Scope(RoleType role, String group) { }
    private static final class Cursor { private String lastSelectedAddress; }
    private final ConcurrentHashMap<Scope, Cursor> cursors = new ConcurrentHashMap<>();

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
