package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;

import java.util.Arrays;
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
        // Eligibility belongs to this routing snapshot. Capture it before the shared cursor lock.
        String[] eligibleAddresses = new String[size];
        String[] orderedAddresses = new String[size];
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (eligible.test(i)) {
                String candidate = address.apply(i);
                eligibleAddresses[i] = candidate;
                orderedAddresses[count++] = candidate;
            }
        }
        if (count == 0) { return -1; }
        Arrays.sort(orderedAddresses, 0, count);
        String selected;
        synchronized (cursor) {
            int index = cursor.lastSelectedAddress == null ? 0
                    : Arrays.binarySearch(orderedAddresses, 0, count, cursor.lastSelectedAddress);
            if (cursor.lastSelectedAddress != null) {
                if (index < 0) { index = -index - 1; }
                else {
                    // Advance past the address, including duplicate entries in a captured snapshot.
                    while (index < count && orderedAddresses[index].equals(cursor.lastSelectedAddress)) { index++; }
                }
            }
            selected = orderedAddresses[index == count ? 0 : index];
            cursor.lastSelectedAddress = selected;
        }
        // Keep the original index so callers can recover their exact captured endpoint.
        return Arrays.asList(eligibleAddresses).indexOf(selected);
    }
}
